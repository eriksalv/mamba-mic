import torch
import lightning.pytorch as pl
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.inferers import Inferer, SliceInferer
from monai.transforms import Compose, Activations, AsDiscrete
from monai.losses.dice import FocalLoss
from torch.nn.modules.loss import _Loss


class System(pl.LightningModule):
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

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"

        self.criterion = criterion if criterion else FocalLoss(include_background=include_background, gamma=2)

        self.metrics_ed = {
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

        self.metrics_es = {
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
        batch_ed = {"image": batch["ED"], "label": batch["ED_gt"]}
        batch_es = {"image": batch["ES"], "label": batch["ES_gt"]}

        y_hat_ed, y_ed = self.infer_batch(batch_ed, val=True)
        y_hat_es, y_es = self.infer_batch(batch_es, val=True)

        loss_ed = self.criterion(y_hat_ed, y_ed)
        loss_es = self.criterion(y_hat_es, y_es)

        y_hat_binarized_ed = self.post_pred(y_hat_ed)
        y_hat_binarized_es = self.post_pred(y_hat_es)

        for metric in self.metrics_ed.values():
            metric(y_hat_binarized_ed, y_ed)

        for metric in self.metrics_es.values():
            metric(y_hat_binarized_es, y_es)

        return (loss_ed + loss_es) / 2

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return {"test_loss": loss}

    def _shared_on_epoch_end(self, stage="val"):
        metrics = {}
        for name, metric in self.metrics_ed.items():
            per_channel = metric.aggregate()

            if name == "confusion_matrix":
                for conf_name, conf_metric in zip(metric.metric_name, per_channel):
                    metrics[f"{stage}_{conf_name}_ed"] = conf_metric.mean()
                    metric.reset()
            else:
                for i in range(len(per_channel)):
                    metrics[f"channel{i}/{stage}_{name}_ed"] = per_channel[i]

                metrics[f"{stage}_{name}_ed"] = per_channel.mean()
                metric.reset()

        for name, metric in self.metrics_es.items():
            per_channel = metric.aggregate()

            if name == "confusion_matrix":
                for conf_name, conf_metric in zip(metric.metric_name, per_channel):
                    metrics[f"{stage}_{conf_name}_es"] = conf_metric.mean()
                    metric.reset()
            else:
                for i in range(len(per_channel)):
                    metrics[f"channel{i}/{stage}_{name}_es"] = per_channel[i]

                metrics[f"{stage}_{name}_es"] = per_channel.mean()
                metric.reset()

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
