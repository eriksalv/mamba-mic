#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=SegMamba
#SBATCH --time=0-48:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --constraint=gpu32g
#SBATCH --cpus-per-task=4

#SBATCH --mem=32G                 # Asking for 32GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=jespee@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# Setup
module purge
module load Anaconda3/2024.02-1
conda create -n vim python=3.10.13
conda activate vim
module load CUDA/11.8.0
nvcc --version
nvidia-smi
export TORCH_CUDA_ARCH_LIST="6.0"
export CUDA_LAUNCH_BLOCKING=1
# Create and activate a virtual environment
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.o>pip install -r vim_requirements.txt
pip install -e causal-conv1d
pip install -e mamba_3d

# Run SegMamba
srun python main.py fit --model configs/models/segmamba.yaml --wandb.name 'SegMamba-test'