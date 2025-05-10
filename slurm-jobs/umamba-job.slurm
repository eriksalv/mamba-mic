#!/bin/sh
#SBATCH --account=share-ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=swinumamba-camus
#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                 # Asking for 32GB RAM
#SBATCH --nodes=1
#SBATCH --output=idun_logs/%j.txt      # Specifying 'stdout'
#SBATCH --error=idun_logs/%j.err        # Specifying 'stderr'
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
conda activate mamba
module load CUDA/11.8.0
module load GCCcore/11.3.0
nvcc --version
nvidia-smi

# Run SegMamba
srun $1
