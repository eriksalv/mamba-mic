#!/bin/sh
#SBATCH --account=share-ie-idi          # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=dynunet
#SBATCH --time=2-00:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --mem=128G                 # Amount of memory
#SBATCH --cpus-per-task=16       # Num. of CPU cores
#SBATCH --nodes=1                 # Num. of nodes
#SBATCH --output=idun_logs/%j.txt       # Specifying 'stdout'
#SBATCH --error=idun_logs/%j.err        # Specifying 'stderr'

#SBATCH --mail-user=eriksalv@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source ../mamba-mic/.venv/bin/activate

# Running your python file
srun $1
