#!/bin/bash
#SBATCH --account=MST113264                 # (-A) iService Project ID
#SBATCH --job-name=openvla_ppo_ft           # (-J) Job name
#SBATCH --partition=normal                  # (-p) Slurm partition for H100 nodes
#SBATCH --nodes=2                           # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=8                   # Gpus per node
#SBATCH --cpus-per-task=12                  # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=8                 # Maximum number of tasks on each node
#SBATCH --time=48:00:00                     # (-t) Wall time limit (days-hrs:min:sec)
#SBATCH --output=openvla-ppo.out            # (-o) Path to the standard output file
#SBATCH --error=openvla-ppo.err             # (-e) Path to the standard error file
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=crlc112358@gmail.com    # Where to send mail.  Set this to your email address

# Load necessary modules and set up environment
module purge
module load cuda/12.2

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate vla-rl

# Check if conda env is activated
echo "Checking conda environment activation..."
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "Conda environment is activated: $CONDA_DEFAULT_ENV"
else
    echo "Conda environment is NOT activated"
    exit 1
fi

# Verify CUDA is available
echo -e "\nChecking CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# List all installed packages
echo -e "\nListing installed packages:"
pip list

# Print Python path to verify we're using the correct interpreter
echo -e "\nPython interpreter path:"
which python
echo -e "\nPython version:"
python --version

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
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    vla-scripts/ppo-finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "/work/crlc112358/datasets/" \
    --dataset_name bridge_orig \
    --run_root_dir "runs" \
    --adapter_tmp_dir "adapters" \
    --max_steps 100000 \
    --lora_rank 32 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "bridge_orig_finetune" \
    --wandb_entity "elsa-vla" \
    --save_steps 500 