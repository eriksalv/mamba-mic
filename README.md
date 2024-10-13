# Mamba MIC project

## Prerequisites

- Python 3.11

## Set up environment

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

This project uses [lightning cli](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), so all training configuration can be set through cli args or through config files. Will use `config/default.yaml` by default.

To run the cli for training:

`python main.py fit`

You can change the default configuration with your own config file, or override specific parts of the default config file. For example change the model with a specific model config file by running 

`python main.py fit --model path/to/model.yaml`

Or change specific arguments from cli, like the name of the run with

`python main.py fit --wandb.name 'my-run'`

#### Train UNet on Decathlon BraTS dataset (example without lightning cli)

`python train_example.py`

## Run inference

`python inference.py --model-ckpt=<model-ckpt> --local={true,false}`

If local is true, then it will look for model ckpt in a local folder, else it will download ckpt from wandb artifact