#!/bin/bash

# Default values
MODEL_FAMILY="openvla"
CHECKPOINT_PATH="/home/crlc112358/elsa-vla/runs/openvla-7b+bridge_orig+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug"
TASK_SUITE="libero_spatial"
CENTER_CROP="true"
RUN_ID_NOTE=""
USE_WANDB="false"
WANDB_PROJECT="libero_lora_finetune_eval"
WANDB_ENTITY="elsa-vla"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --task_suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --center_crop)
            CENTER_CROP="$2"
            shift 2
            ;;
        --run_id_note)
            RUN_ID_NOTE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint parameter is required"
    exit 1
fi

# Validate task suite name
valid_tasks=("libero_spatial" "libero_object" "libero_goal" "libero_10")
if [[ ! " ${valid_tasks[@]} " =~ " ${TASK_SUITE} " ]]; then
    echo "Error: Invalid task suite. Must be one of: ${valid_tasks[*]}"
    exit 1
fi

# Run the evaluation
python experiments/robot/libero/run_libero_eval.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$CHECKPOINT_PATH" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_wandb "$USE_WANDB" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" 