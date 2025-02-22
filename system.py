import torch
import lightning.pytorch as pl
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.losses.dice import DiceFocalLoss
from monai.inferers import Inferer, SliceInferer
from monai.transforms import Compose, Activations, AsDiscrete


class System(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer = None,
        lr=0.001,
        softmax=False,
        include_background=True,
        num_output_channels=None,
        log_hd95=True,
        do_slice_inference=False,
        slice_shape=None,
        slice_dim=None,
        slice_batch_size=None,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.softmax = softmax
        self.include_background = include_background
        self.log_hd95 = log_hd95

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"

        self.criterion = DiceFocalLoss(
            include_background=include_background,
            sigmoid=not softmax,
            softmax=softmax,
            squared_pred=True,
            gamma=0.5,
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
        self.precision_metric = ConfusionMatrixMetric(
            metric_name="precision", reduction="mean"
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

        self.do_slice_inference = do_slice_inference
        if do_slice_inference:
            assert (
                slice_dim is not None
                and slice_shape is not None
                and slice_batch_size is not None
            ), (
                "slice_dim, slice_shape and slice_batch_size must be specified if do_slice_inference is True"
            )
            self.slice_inferer = SliceInferer(
                roi_size=slice_shape,
                sw_batch_size=slice_batch_size,
                spatial_dim=slice_dim,
            )
        else:
            assert val_inferer is not None, (
                "val_inferer must be specified if do_slice_inference is False"
            )
            self.val_inferer = val_inferer

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def infer_batch(self, batch, val=False):
        if isinstance(batch, list):
            batch = batch[0]

        x, y = batch["image"], batch["label"]

        if self.do_slice_inference:
            return self.slice_inferer(inputs=x, network=self.net), y

        if val is False:
            return self(x), y

        return self.val_inferer(inputs=x, network=self.net), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch, val=True)
        loss = self.criterion(y_hat, y)

        y_hat_binarized = self.post_pred(y_hat)

        self.dice_metric(y_hat_binarized, y)

        if self.log_hd95:
            self.hd95_metric(y_hat_binarized, y)

        self.precision_metric(y_hat_binarized, y)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return {"test_loss": loss}

    def _shared_on_epoch_end(self, stage="val"):
        per_channel_dice = self.dice_metric.aggregate().nanmean(dim=0)
        self.dice_metric.reset()

        metrics = {
            f"{stage}_dice": per_channel_dice.mean(),
            f"{stage}_precision": self.precision_metric.aggregate()[0].item(),
        }
        self.precision_metric.reset()

        num_channels = len(per_channel_dice)
        for i in range(num_channels):
            metrics[f"channel{i}/{stage}_dice"] = per_channel_dice[i]

        if self.log_hd95:
            per_channel_hd95 = self.hd95_metric.aggregate().nanmean(dim=0)
            self.hd95_metric.reset()

            metrics[f"{stage}_hd95"] = per_channel_hd95.mean()

            num_channels = len(per_channel_hd95)
            for i in range(num_channels):
                metrics[f"channel{i}/{stage}_hd95"] = per_channel_hd95[i]

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
