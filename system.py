import torch
import lightning.pytorch as pl
from monai.metrics import DiceMetric
from monai.losses.dice import DiceCELoss
from monai.inferers import Inferer
from monai.transforms import Compose, Activations, AsDiscrete


class System(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer,
        lr=0.001,
    ) -> None:
        super().__init__()
        self.net = net
        self.val_inferer = val_inferer
        self.lr = lr

        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True)
        self.mean_dice = DiceMetric(include_background=True, reduction="mean")
        self.channel_dice = DiceMetric(
            include_background=True, reduction="mean_channel"
        )

        self.postprocess = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ]
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def infer_batch(self, batch, val=False):
        x, y = batch["image"], batch["label"]
        if val is False:
            y_hat = self(x)
        else:
            y_hat = self.val_inferer(inputs=x, network=self.net)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch, val=True)
        loss = self.criterion(y_hat, y)

        y_hat_binarized = self.postprocess(y_hat)

        self.mean_dice(y_hat_binarized, y)
        self.channel_dice(y_hat_binarized, y)

        self.log("val_loss", loss, sync_dist=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        mean_dice = self.mean_dice.aggregate()
        per_channel_dice = self.channel_dice.aggregate()

        self.mean_dice.reset()
        self.channel_dice.reset()

        self.log_dict(
            {"val_dice": mean_dice},
            sync_dist=True,
        )

        num_channels = len(per_channel_dice)
        for i in range(num_channels):
            self.log_dict(
                {
                    f"channel{i}/val_dice": per_channel_dice[i],
                },
                sync_dist=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
