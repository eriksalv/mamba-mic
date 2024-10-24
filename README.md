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

`python main.py fit --wandb.name 'my-run'`

#### Some parameters that might be useful to change

- `trainer.max_time` depending on time limit of slurm job
- `data.batch_size` and `data.num_workers` for efficiency depending on gpu memory and cpu count
- `trainer.fast_dev_run true` or `trainer.limit_train_batches 1` for debugging/testing

#### Train UNet on Decathlon BraTS dataset (example without lightning cli)

`python train_example.py`

## Run inference

`python inference.py --model-ckpt=<model-ckpt> --local={true,false}`

If local is true, then it will look for model ckpt in a local folder, else it will download ckpt from wandb artifact