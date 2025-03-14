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
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from data_modules.pi_caiv2 import PICAIV2DataModule, evaluate_cases


class System(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer = None,
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

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"

        self.criterion = FocalLoss(
            include_background=include_background, gamma=1.0, alpha=0.9
        )

        self.metrics = {
            "dice": DiceMetric(
                include_background=include_background, reduction="mean_batch"
            ),
            "hd95": HausdorffDistanceMetric(
                include_background=include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
            "confusion_matrix": ConfusionMatrixMetric(
                metric_name=["accuracy", "precision", "sensitivity", "f1 score"],
                reduction="mean_batch",
            ),
        }
        self.extra_metrics = {"picai": []}

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

        if val:
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

        if isinstance(self.trainer.datamodule, PICAIV2DataModule):
            eval = evaluate_case(
                y_true=batch["label"].squeeze().cpu().numpy(),
                y_det=batch["pred"].squeeze().cpu().numpy(),
                y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
            )
            self.extra_metrics["picai"].append(eval)

        if (
            self.trainer.state.fn in ["validate", "test", "predict"]
            and self.save_output
        ):
            self.trainer.datamodule.invert_and_save(batch)

        for metric in self.metrics.values():
            metric(y_hat_binarized, y)

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

        if isinstance(self.trainer.datamodule, PICAIV2DataModule):
            metrics |= evaluate_cases(self.extra_metrics["picai"])
            self.extra_metrics["picai"] = []

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
