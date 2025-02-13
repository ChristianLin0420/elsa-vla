"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import copy

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on

class QNetwork(torch.nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + action_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states, actions):
        actions = actions.reshape(hidden_states.shape[0], -1)
        print(f"hidden_states: {type(hidden_states)}, actions: {type(actions)}")
        print(f"hidden_states shape: {hidden_states.shape}, actions shape: {actions.shape}")
        x = torch.cat([hidden_states, actions], dim=-1)
        print(f"x type: {type(x)}, shape: {x.shape}")
        # Check data types
        print(f"q_net parameters dtype: {next(self.q_net.parameters()).dtype}")
        print(f"x dtype: {x.dtype}")
        
        # Convert to float32 if needed
        if x.dtype != torch.float32:
            x = x.float()
        return self.q_net(x)


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # SAC Parameters
    q_lr: float = 3e-4                                             # Q-network learning rate
    alpha: float = 0.2                                             # Temperature parameter for entropy
    gamma: float = 0.99                                            # Discount factor
    tau: float = 0.005                                             # Soft update coefficient
    use_sac: bool = True                                           # Whether to use SAC
    hidden_size: int = 4096                                        # Hidden size for networks
    action_dim: int = 7                                            # Dimension of continuous actions
    target_update_interval: int = 1                                # How often to update target networks
    automatic_entropy_tuning: bool = True                          # Whether to automatically tune entropy

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+sac"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        # print(f"Trainable parameters w/ LoRA: \n {vla.print_trainable_parameters()}")
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    # print(f"Trainable parameters for optimizer: \n {trainable_params}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_q_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_policy_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_entropies = deque(maxlen=cfg.grad_accumulation_steps)

    # Initialize networks for SAC
    q_net1 = QNetwork(cfg.hidden_size, cfg.action_dim).to(device_id)
    q_net2 = QNetwork(cfg.hidden_size, cfg.action_dim).to(device_id)
    q_net1_target = QNetwork(cfg.hidden_size, cfg.action_dim).to(device_id)
    q_net2_target = QNetwork(cfg.hidden_size, cfg.action_dim).to(device_id)

    # Initialize target networks with same weights
    for target_param, param in zip(q_net1_target.parameters(), q_net1.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(q_net2_target.parameters(), q_net2.parameters()):
        target_param.data.copy_(param.data)

    q_net1 = DDP(q_net1, device_ids=[device_id])
    q_net2 = DDP(q_net2, device_ids=[device_id])
    q_net1_target = DDP(q_net1_target, device_ids=[device_id])
    q_net2_target = DDP(q_net2_target, device_ids=[device_id])

    # Create optimizers for SAC
    q1_optimizer = AdamW(q_net1.parameters(), lr=cfg.q_lr)
    q2_optimizer = AdamW(q_net2.parameters(), lr=cfg.q_lr)

    # Automatic entropy tuning
    if cfg.automatic_entropy_tuning:
        target_entropy = -torch.prod(torch.Tensor([cfg.action_dim])).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device_id)
        alpha_optimizer = AdamW([log_alpha], lr=cfg.q_lr)
        alpha = log_alpha.exp()
    else:
        alpha = cfg.alpha

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: {v.shape}")
                    else:
                        print(f"{k}")
                    if k in ["input_ids", "attention_mask", "labels"]:
                        print(f"{k}, shape: {v[0].shape}, value: {v[0]}")

                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_hidden_states=True
                )

                print(f"output.hidden_states: {len(list(output.hidden_states))}")
                print(f"original hidden states shape: {output.hidden_states[-1].shape}")

                # Extract hidden states and action predictions
                hidden_states = output.hidden_states[-1][:, 0]  # Use CLS token
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                print(f"action_logits shape: {action_logits.shape}")
                action_probs = torch.softmax(action_logits, dim=-1)
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                print(f"action_preds shape: {action_preds.shape}, action_gt shape: {action_gt.shape}")
                print(f"mask shape: {mask.shape}")
                # print(f"mask: {mask}")
                print(f"action_preds w/o mask shape: {action_preds.shape}")
                # print(f"action_preds w/o mask: {action_preds}")
                print(f"action_preds w/ mask shape: {action_preds[mask].shape}")
                # print(f"action_preds w/ mask: {action_preds[mask]}")
                
                # Convert discrete actions to continuous
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                ).to(device_id)
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                ).to(device_id)

                print(f"continuous_actions_pred shape: {continuous_actions_pred.shape}")
                print(f"continuous_actions_gt shape: {continuous_actions_gt.shape}")
                # print(f"continuous_actions_pred: {continuous_actions_pred}")
                # print(f"continuous_actions_gt: {continuous_actions_gt}")

                # Compute entropy of action distribution
                entropy = -(action_probs * torch.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()

                print(f"entropy: {entropy.shape}")
                # print(f"entropy: {entropy}")
                # SAC updates
                # 1. Get current Q estimates and compute Q loss
                current_q1 = q_net1(hidden_states, continuous_actions_pred.reshape(cfg.batch_size, -1))
                current_q2 = q_net2(hidden_states, continuous_actions_pred.reshape(cfg.batch_size, -1))

                # Get next action and Q values for TD target
                with torch.no_grad():
                    next_state_log_pi = torch.log_softmax(action_logits, dim=-1)
                    next_state_actions = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    ).to(device_id)
                    
                    # Target Q-values
                    target_q1 = q_net1_target(hidden_states, next_state_actions)
                    target_q2 = q_net2_target(hidden_states, next_state_actions)
                    target_q = torch.min(target_q1, target_q2)
                    
                    # Compute rewards (negative L1 loss in this case)
                    rewards = -torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt, reduction='none')
                    
                    # TD target
                    print(f"rewards shape: {rewards.shape}")
                    print(f"target_q shape: {target_q.shape}")
                    print(f"next_state_log_pi shape: {next_state_log_pi.shape}")
                    print(f"next_state_log_pi mean: {next_state_log_pi.mean(dim=-1)}")
                    target_q_value = rewards + cfg.gamma * (target_q - alpha * next_state_log_pi.mean(dim=-1))

                # Q-function loss
                q1_loss = torch.nn.functional.mse_loss(current_q1, target_q_value.detach())
                q2_loss = torch.nn.functional.mse_loss(current_q2, target_q_value.detach())
                q_loss = q1_loss + q2_loss

                # Policy loss
                policy_loss = (alpha * next_state_log_pi - torch.min(current_q1, current_q2)).mean()

                # Optional: Update temperature parameter alpha
                if cfg.automatic_entropy_tuning:
                    alpha_loss = -(log_alpha * (next_state_log_pi.mean() + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp()

                # Total loss
                total_loss = policy_loss + q_loss

                # Update metrics
                recent_losses.append(total_loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_q_losses.append(q_loss.item())
                recent_policy_losses.append(policy_loss.item())
                recent_entropies.append(-next_state_log_pi.mean().item())

                # Backward pass and optimization
                total_loss.backward()

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    q1_optimizer.step()
                    q2_optimizer.step()
                    
                    optimizer.zero_grad()
                    q1_optimizer.zero_grad()
                    q2_optimizer.zero_grad()

                    # Update target networks
                    if batch_idx % cfg.target_update_interval == 0:
                        for target_param, param in zip(q_net1_target.parameters(), q_net1.parameters()):
                            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                        for target_param, param in zip(q_net2_target.parameters(), q_net2.parameters()):
                            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

                    progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if batch_idx > 0 and batch_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {batch_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {batch_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{batch_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {batch_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if batch_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

            # Update smoothened metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_q_loss = sum(recent_q_losses) / len(recent_q_losses)
            smoothened_policy_loss = sum(recent_policy_losses) / len(recent_policy_losses)
            smoothened_entropy = sum(recent_entropies) / len(recent_entropies)

            # Push metrics to W&B
            if distributed_state.is_main_process and batch_idx % 10 == 0:
                wandb.log(
                    {
                        "total_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "q_loss": smoothened_q_loss,
                        "policy_loss": smoothened_policy_loss,
                        "entropy": smoothened_entropy,
                    },
                    step=batch_idx,
                )


if __name__ == "__main__":
    finetune()
