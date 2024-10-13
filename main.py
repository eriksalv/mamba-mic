from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
import wandb


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb.project", default="monai-runs")
        parser.add_argument("--wandb.name")

    def before_fit(self):
        wandb_config = self.config_dump["wandb"]
        wandb.init(
            job_type="train",
            config=self.config_dump,
            project=wandb_config["project"],
            name=wandb_config["name"],
        )
        wandb.watch(self.model, log_freq=100)


def cli_main():
    cli = MyLightningCLI(
        parser_kwargs={"fit": {"default_config_files": ["configs/default.yaml"]}},
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
