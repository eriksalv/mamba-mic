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
        softmax=False,
        include_background=True,
        num_output_channels=None,
        log_hd95=True
    ) -> None:
        super().__init__()
        self.net = net
        self.val_inferer = val_inferer
        self.lr = lr
        self.softmax = softmax
        self.include_background = include_background
        self.log_hd95 = log_hd95

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"

        self.criterion = DiceCELoss(
            include_background=include_background,
            sigmoid=not softmax,
            softmax=softmax,
            squared_pred=True,
        )
        self.dice_metric = DiceMetric(
            include_background=include_background, reduction="none"
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=include_background,
            distance_metric="euclidean",
            percentile=95,
            reduction="none",
        )

        self.post_pred = (
            Compose([AsDiscrete(argmax=True, dim=1, to_onehot=num_output_channels)])
            if softmax
            else Compose(
                [
                    Activations(
                        sigmoid=True if not softmax else False, softmax=softmax
                    ),
                    AsDiscrete(threshold=0.5),
                ]
            )
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def infer_batch(self, batch, val=False):
        if isinstance(batch, list):
            batch = batch[0]

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

        y_hat_binarized = self.post_pred(y_hat)

        self.dice_metric(y_hat_binarized, y)
        
        if self.log_hd95:
            self.hd95_metric(y_hat_binarized, y)

        self.log("val_loss", loss, sync_dist=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        per_channel_dice = self.dice_metric.aggregate().nanmean(dim=0)

        self.dice_metric.reset()

        self.log_dict(
            {"val_dice": per_channel_dice.mean()},
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

        if self.log_hd95:
            per_channel_hd95 = self.hd95_metric.aggregate().nanmean(dim=0)

            self.hd95_metric.reset()

            self.log_dict(
                {"val_hd95": per_channel_hd95.mean()},
                sync_dist=True,
            )

            num_channels = len(per_channel_hd95)
            for i in range(num_channels):
                self.log_dict(
                    {
                        f"channel{i}/val_hd95": per_channel_hd95[i],
                    },
                    sync_dist=True,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
