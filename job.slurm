#!/bin/sh
#SBATCH --account=ie-idi          # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=swin-unetr
#SBATCH --time=4-00:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G                 # Amount of memory
#SBATCH --cpus-per-task=8         # Num. of CPU cores
#SBATCH --nodes=1                 # Num. of nodes
#SBATCH --output=output.txt       # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=you@email.com
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv/bin/activate

# Running your python file
srun $1
