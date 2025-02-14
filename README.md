# Mamba MIC project

## Prerequisites

- Python 3.11

## Set up environment

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Login to wandb

`wandb login`

## Specific setups

#### UMamba

1. Install main deps:

```shell
module purge
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load GCCcore/11.3.0
conda create -n umamba python=3.11 -y
conda activate umamba
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

2. Verify that nvcc and pytorch have the same cuda version:

```shell
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
``` 

2. Install `causal-conv1d` and `mamba`:

  - Clone the `causal-conv1d` and `mamba` repos
  - `pip install -e causal-conv1d`
  - `pip install --no-build-isolation -e mamba`

3. Run job with `sbatch umamba-job.slurm "python main.py ..."`

## Running on IDUN

On a login node, after following the prerequisites and setting up environment, simply run

```shell
sbatch job.slurm "python main.py [fit/validate/test/predict] [--subcommands...]"
```

For example start a training run with a specific model config and log to wandb project by running

```shell
sbatch job.slurm "python main.py fit --wandb.project my-project --wandb.name my-run --model path/to/model-config.yaml"
```

## Training

This project uses [lightning cli](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), so all training configuration can be set through cli args or through config files.

To run the cli for training:

`python main.py fit`

Running the "fit" command will use the "configs/default.yaml" file by default. You can change the default configuration with your own config file, or override specific parts of the default config file. For example change the model with a specific model config file by running 

`python main.py fit --model path/to/model.yaml`

Or change specific arguments from cli, like the name of the run with

`python main.py fit --wandb.name my-run`

#### Some parameters that might be useful to change

- `trainer.max_time` depending on time limit of slurm job
- `data.batch_size` and `data.num_workers` for efficiency depending on gpu memory and cpu count
- `trainer.fast_dev_run true` or `trainer.limit_train_batches 1` for debugging/testing

#### Train UNet on Decathlon BraTS dataset (example without lightning cli)

`python train_example.py`

## Inference

#### Run standard evaluation loop on the test set

`python main.py test --ckpt_path path/to/ckpt_file.ckpt`

This will measure loss, dice score and HD95 using standard MONAI metrics.

#### Save predictions on the test set and evaluate using BraTS 2024 metrics

Submissions to BraTS 2024 are evaluated with custom "lesion-wise" dice and HD95 metrics. To save predictions on all test cases as nifti files and evaluate performance using these custom metrics run:

`python brats2024_submission.py --name ... --model_ckpt ...`

*Note:* this will require a few additional dependencies that can be installed with:

`pip install -r brats_2024_metrics/requirements.txt`
