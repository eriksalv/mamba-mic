from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import SaveConfigCallback
import lightning.pytorch as pl
import wandb
import logging
import torch


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb.project", required=True)
        parser.add_argument("--wandb.name", required=True)
        parser.add_argument("--wandb.group", required=False)
        parser.add_argument("--wandb.watch_model", default=False)

        parser.link_arguments("wandb.name", "data.init_args.name")

    def before_fit(self):
        wandb_config = self.config_dump["wandb"]
        wandb.init(
            job_type="train",
            config=self.config_dump,
            project=wandb_config["project"],
            name=wandb_config["name"],
            group=wandb_config["group"],
        )
        if wandb_config["watch_model"]:
            wandb.watch(self.model, log_freq=100)

    def before_test(self):
        wandb_config = self.config_dump["wandb"]
        wandb.init(
            job_type="eval",
            config=self.config_dump,
            project=wandb_config["project"],
            name=wandb_config["name"],
        )


class MySaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        self.parser.save(
            self.config,
            "run_config.yaml",
            skip_none=False,
            overwrite=True,
            multifile=False,
        )
        wandb.save("run_config.yaml", policy="now")


def cli_main():
    cli = MyLightningCLI(  # noqa: F841
        parser_kwargs={
            "fit": {"default_config_files": ["configs/default.yaml"]},
            "validate": {"default_config_files": ["configs/default_eval.yaml"]},
            "test": {"default_config_files": ["configs/default_eval.yaml"]},
        },
        save_config_callback=MySaveConfigCallback,
        save_config_kwargs={"overwrite": "true"},
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")

    cli_main()
