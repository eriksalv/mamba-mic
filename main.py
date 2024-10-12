from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything


def cli_main():
    cli = LightningCLI(
        parser_kwargs={"fit": {"default_config_files": ["configs/default.yaml"]}},
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
