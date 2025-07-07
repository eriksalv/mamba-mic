#!/bin/sh
#SBATCH --account=share-ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=SegMamba
#SBATCH --time=0-00:30:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the numbe
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # Asking for 32GB RAM
#SBATCH --nodes=1
#SBATCH --output=idun_logs/segmamba-focal-eval.txt      # Specifying 'stdout'
#SBATCH --error=idun_logs/segmamba-focal-eval.err        # Specifying 'stderr'
#SBATCH --mail-user=eriksalv@stud.ntnu.no
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
conda activate segmamba
module load CUDA/11.8.0
nvcc --version
nvidia-smi
export TORCH_CUDA_ARCH_LIST="6.0"
export CUDA_LAUNCH_BLOCKING=1

# Run SegMamba
srun $1
