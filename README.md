# Mamba MIC project

## Set up environment

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train UNet on Decathlon BraTS dataset

`python train.py`

## Run inference

`python inference.py --model-ckpt=<model-ckpt> --local={true,false}`

If local is true, then it will look for model ckpt in a local folder, else it will download ckpt from wandb artifact