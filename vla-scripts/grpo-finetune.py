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

    # GRPO Arguments
    group_size: int = 8                                             # Number of outputs to sample per input (G in paper)
    clip_param: float = 0.2                                         # PPO clip parameter epsilon
    entropy_coef: float = 0.01                                      # Entropy coefficient for exploration
    grpo_iterations: int = 1                                        # Number of GRPO iterations (μ in paper)
    beta: float = 0.2                                               # KL divergence coefficient (β in paper)

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
        f"+grpo"
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
    recent_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    recent_kl_divs = deque(maxlen=cfg.grad_accumulation_steps)
    recent_entropies = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Initialize reference model with current policy
            reference_model = copy.deepcopy(vla.module)
            reference_model.eval()
            reference_model = reference_model.to(device_id)

            # Sample G outputs for each input
            all_outputs = []
            all_action_preds = []
            all_log_probs = []
            
            for _ in range(cfg.group_size):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                        output_hidden_states=True
                    )
                    
                    print(f"output.logits shape: {output.logits.shape}")
                    # Extract action logits and compute predictions
                    action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                    print(f"action_logits shape: {action_logits.shape}")
                    action_preds = action_logits.argmax(dim=2)
                    print(f"action_preds shape: {action_preds.shape}")
                    log_probs = torch.log_softmax(action_logits, dim=-1).gather(dim=2, index=action_preds.unsqueeze(-1)).squeeze(-1)
                    print(f"log_probs shape: {log_probs.shape}")
                    
                    all_outputs.append(output)
                    all_action_preds.append(action_preds)
                    all_log_probs.append(log_probs)
            
            # Stack all predictions and log probs
            action_preds = torch.stack(all_action_preds)  # [G, B, Seq]
            print(f"stacked action_preds shape: {action_preds.shape}")
            log_probs = torch.stack(all_log_probs)  # [G, B, Seq]
            print(f"stacked log_probs shape: {log_probs.shape}")
            
            # Get ground truth actions
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            print(f"action_gt shape: {action_gt.shape}")
            mask = action_gt > action_tokenizer.action_token_begin_idx
            print(f"mask shape: {mask.shape}")
            
            # Compute rewards for each sampled output
            rewards = []
            for i in range(cfg.group_size):
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[i][mask].cpu().numpy())
                )
                print(f"continuous_actions_pred shape: {continuous_actions_pred.shape}")
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                print(f"continuous_actions_gt shape: {continuous_actions_gt.shape}")
                l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                rewards.append(-l1_loss)  # Negative L1 loss as reward
            
            rewards = torch.stack(rewards)  # [G]
            print(f"rewards shape: {rewards.shape}")
            
            # Compute advantages using group computation
            advantages = rewards - rewards.mean() / (rewards.std() + 1e-8)
            advantages = advantages.to(device_id)
            print(f"advantages shape: {advantages.shape}")
            
            # GRPO policy update
            accumulated_loss = 0
            for iter_idx in range(cfg.grpo_iterations):
                optimizer.zero_grad()  # Clear gradients at start of each iteration
                
                # Get log probs from reference model
                with torch.no_grad():
                    ref_output = reference_model(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"]
                    )
                    print(f"ref_output.logits shape: {ref_output.logits.shape}")
                    ref_action_logits = ref_output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                    print(f"ref_action_logits shape: {ref_action_logits.shape}")
                    ref_action_preds = ref_action_logits.argmax(dim=2)
                    print(f"ref_action_preds shape: {ref_action_preds.shape}")
                    ref_log_probs = torch.log_softmax(ref_action_logits, dim=-1).gather(dim=2, index=ref_action_preds.unsqueeze(-1)).squeeze(-1)
                    print(f"ref_log_probs shape: {ref_log_probs.shape}")
                    
                # Compute policy ratio and clipped objective
                ratios = torch.exp(log_probs - ref_log_probs)  # [G, B, Seq]
                print(f"ratios shape: {ratios.shape}")
                surr1 = ratios * advantages.unsqueeze(-1).unsqueeze(-1)
                print(f"surr1 shape: {surr1.shape}")
                surr2 = torch.clamp(ratios, 1 - cfg.clip_param, 1 + cfg.clip_param) * advantages.unsqueeze(-1).unsqueeze(-1)
                print(f"surr2 shape: {surr2.shape}")
                policy_loss = -torch.min(surr1, surr2).mean()
                print(f"policy_loss shape: {policy_loss.shape}")
                
                # Compute KL divergence loss
                kl_div = (ref_log_probs - log_probs).mean()
                print(f"kl_div shape: {kl_div.shape}")
                
                # Compute entropy bonus
                entropy = -(torch.softmax(action_logits, dim=-1) * torch.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
                print(f"entropy shape: {entropy.shape}")
                
                # Total loss for this iteration
                total_loss = policy_loss + cfg.beta * kl_div - cfg.entropy_coef * entropy
                print(f"total_loss shape: {total_loss.shape}")
                
                # Backward pass
                total_loss.backward()
                
                # Optimizer step after each iteration
                optimizer.step()
                
                # Store the loss for metrics
                if iter_idx == cfg.grpo_iterations - 1:  # Store only the last iteration's loss
                    accumulated_loss = total_loss.item()
            
            # Compute metrics for logging
            correct_preds = (action_preds == action_gt.unsqueeze(0)) & mask.unsqueeze(0)
            action_accuracy = correct_preds.sum().float() / mask.sum().float()
            
            # Store recent metrics
            recent_losses.append(accumulated_loss)
            recent_action_accuracies.append(action_accuracy.item())
            recent_rewards.append(rewards.mean().item())
            recent_kl_divs.append(kl_div.item())
            recent_entropies.append(entropy.item())
            
            # Compute smoothened metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_reward = sum(recent_rewards) / len(recent_rewards)
            smoothened_kl_div = sum(recent_kl_divs) / len(recent_kl_divs)
            smoothened_entropy = sum(recent_entropies) / len(recent_entropies)
            
            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and batch_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "reward": smoothened_reward,
                        "kl_divergence": smoothened_kl_div,
                        "entropy": smoothened_entropy,
                    },
                    step=batch_idx,
                )
            
            # Update progress
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                progress.update()

            # Save Model Checkpoint
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
                            # Save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{batch_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(f"Saved Model Checkpoint for Step {batch_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if batch_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
