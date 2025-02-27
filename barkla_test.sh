#!/bin/bash -l

# job name
#SBATCH -J elsaval

# output file name
#SBATCH -o elsaval.out

# running time
#SBATCH -t 2:00:00  

# partition
#SBATCH -p gpulowsmall

#SBATCH -N 1
#SBATCH --gres=gpu:2

# CPU Core
#SBATCH -n 48

# mail notice
#SBATCH --mail-user=crlc112358@gmail.com
#SBATCH --mail-type=ALL 

# Load necessary modules and set up environment
module purge
module load libs/nvidia-cuda/12.1.1/bin

conda activate elsavla

# Set environment variables for distributed training
# Use a random port with retries
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
    export MASTER_PORT=$(shuf -i 29500-65000 -n 1)
    nc -z $HOSTNAME $MASTER_PORT || break
    if [ $i -eq $MAX_RETRIES ]; then
        echo "Could not find an available port after $MAX_RETRIES attempts"
        exit 1
    fi
done

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

echo "Distributed training configuration:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "RANK: $RANK"

# Run the finetuning script
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    vla-scripts/grpo-finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "/users/sgyson10/volatile/elsa-vla/data" \
    --dataset_name bridge_orig \
    --run_root_dir "runs" \
    --adapter_tmp_dir "adapters" \
    --max_steps 20000 \
    --lora_rank 32 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "testing" \
    --wandb_entity "elsa-vla" \
    --save_steps 500 
