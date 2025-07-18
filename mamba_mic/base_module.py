import random

import lightning.pytorch as pl
import torch
from monai.inferers import Inferer, SliceInferer
from monai.losses.dice import DiceCELoss, FocalLoss
from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    HausdorffDistanceMetric,
)
from monai.transforms import Activations, AsDiscrete, Compose
from torch.nn.modules.loss import _Loss


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer = None,
        criterion: _Loss = None,
        lr=0.001,
        softmax=False,
        include_background=True,
        num_output_channels=None,
        do_slice_inference=False,
        slice_shape=None,
        slice_dim=None,
        slice_batch_size=None,
        save_output=False,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.softmax = softmax
        self.include_background = include_background
        self.save_output = save_output
        self.val_inferer = val_inferer

        self.criterion = criterion if criterion else FocalLoss(include_background=include_background, gamma=2)

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"

        self.metrics = {
            "dice": DiceMetric(include_background=include_background, reduction="mean_batch"),
            "hd95": HausdorffDistanceMetric(
                include_background=include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
            "confusion_matrix": ConfusionMatrixMetric(
                metric_name=["precision", "sensitivity", "f1 score"],
                reduction="mean_batch",
            ),
        }

        self.post_pred = (
            Compose([AsDiscrete(argmax=True, dim=1, to_onehot=num_output_channels)])
            if softmax
            else Compose(
                [
                    Activations(sigmoid=True, softmax=False),
                    AsDiscrete(threshold=0.5),
                ]
            )
        )

        self.do_slice_inference = do_slice_inference
        if do_slice_inference:
            assert slice_dim is not None and slice_shape is not None and slice_batch_size is not None, (
                "slice_dim, slice_shape and slice_batch_size must be specified if do_slice_inference is True"
            )
            self.slice_inferer = SliceInferer(
                roi_size=slice_shape,
                sw_batch_size=slice_batch_size,
                spatial_dim=slice_dim,
            )

        self.save_hyperparameters(ignore=["net"])

    def forward(self, x):
        return self.net(x)

    def infer_batch(self, batch, val=False):
        if isinstance(batch, list):
            batch = batch[0]

        x, y = batch["image"], batch["label"]

        if self.do_slice_inference:
            return self.slice_inferer(inputs=x, network=self.net), y

        if val and self.val_inferer:
            return self.val_inferer(inputs=x, network=self.net), y

        return self(x), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch, val=True)

        loss = self.criterion(y_hat, y)

        y_hat_binarized = self.post_pred(y_hat)

        batch["image"] = batch["image"].squeeze(0)
        batch["label"] = batch["label"].squeeze(0)
        batch["pred"] = Activations(sigmoid=True, softmax=False)(y_hat).squeeze(0)

        if self.trainer.state.fn in ["validate", "test", "predict"] and self.save_output:
            self.trainer.datamodule.invert_and_save(batch)

        for metric in self.metrics.values():
            metric(y_hat_binarized, y)

        return loss, batch

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return {"test_loss": loss}

    def _shared_on_epoch_end(self, stage="val"):
        metrics = {}

        for name, metric in self.metrics.items():
            per_channel = metric.aggregate()

            if name == "confusion_matrix":
                for conf_name, conf_metric in zip(metric.metric_name, per_channel):
                    metrics[f"{stage}_{conf_name}"] = conf_metric.mean()
                    metric.reset()
            else:
                for i in range(len(per_channel)):
                    metrics[f"channel{i}/{stage}_{name}"] = per_channel[i]

                metrics[f"{stage}_{name}"] = per_channel.mean()
                metric.reset()

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
