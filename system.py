import torch
import lightning.pytorch as pl
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True,
            distance_metric="euclidean",
            percentile=95,
            reduction="none",
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

        self.dice_metric(y_hat_binarized, y)

        if batch_idx % 10 == 0:
            self.hd95_metric(y_hat_binarized, y)

        self.log("val_loss", loss, sync_dist=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        per_channel_dice = self.dice_metric.aggregate().nanmean(dim=0)
        per_channel_hd95 = self.hd95_metric.aggregate().nanmean(dim=0)

        self.dice_metric.reset()
        self.hd95_metric.reset()

        self.log_dict(
            {"val_dice": per_channel_dice.mean(), "val_hd95": per_channel_hd95.mean()},
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
