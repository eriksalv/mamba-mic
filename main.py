from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
import wandb
import logging
import torch


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb.project", default="monai-runs")
        parser.add_argument("--wandb.name")

        parser.link_arguments("wandb.name", "data.init_args.name")

    def before_fit(self):
        wandb_config = self.config_dump["wandb"]
        wandb.init(
            job_type="train",
            config=self.config_dump,
            project=wandb_config["project"],
            name=wandb_config["name"],
        )
        wandb.watch(self.model, log_freq=100)

    def before_test(self):
        wandb_config = self.config_dump["wandb"]
        wandb.init(
            job_type="eval",
            config=self.config_dump,
            project=wandb_config["project"],
            name=wandb_config["name"],
        )


def cli_main():
    cli = MyLightningCLI(
        parser_kwargs={
            "fit": {"default_config_files": ["configs/default.yaml"]},
            "test": {"default_config_files": ["configs/default_eval.yaml"]},
        },
        save_config_kwargs={"config_filename": "run_config.yaml", "overwrite": "true"},
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")

    cli_main()
