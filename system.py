import torch
import lightning.pytorch as pl
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.inferers import Inferer, SliceInferer
from monai.transforms import Compose, Activations, AsDiscrete
from monai.losses.dice import FocalLoss, DiceCELoss
from ocelot_util import normalize_crop_coords_batch

class System(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer = None,
        lr=0.001,
        softmax=False,
        include_background=True,
        num_output_channels=None,
        use_bce = False,
        use_focal = True,
        do_slice_inference=False,
        slice_shape=None,
        slice_dim=None,
        slice_batch_size=None,
        save_output=False,
        is_ocelot = False,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.softmax = softmax
        self.include_background = include_background
        self.save_output = save_output
        self.use_bce = use_bce
        self.use_focal = use_focal
        self.is_ocelot = is_ocelot

        assert (num_output_channels is None and softmax is False) or (
            num_output_channels is not None and softmax is True
        ), "num_output_channels should only be set if softmax is True"
        if self.use_focal:
            self.criterion = FocalLoss(
               include_background=include_background,
               gamma=1.0,
               alpha=0.90
            )
        else:
            self.criterion = DiceCELoss(
              include_background=include_background,
              sigmoid=not softmax,
              softmax=softmax,
              squared_pred=True,
            )
        self.empty_criterion = torch.nn.BCEWithLogitsLoss()
        
        if self.is_ocelot:
            self.tissue_criterion = DiceCELoss(
              include_background=include_background,
              sigmoid=not softmax,
              softmax=softmax,
              squared_pred=True,
            )
            self.cell_criterion = DiceCELoss(
              include_background=include_background,
              sigmoid=not softmax,
              softmax=softmax,
              squared_pred=True,
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

        if self.is_ocelot:
            x_t, y_t = batch["img_tissue"],  batch["label_tissue"]
            x_c, y_c = batch["img_cell"], batch["label_cropped_tissue"]
            meta_batch = batch["meta"]
        
            cropped_coords = normalize_crop_coords_batch(meta_batch)
            
            if val:
                return self.val_inferer(inputs=(x_t, x_c, cropped_coords), network=self.net), y_t, y_c

            t_pred, c_pred = self.net.forward(x_t, x_c, cropped_coords, train_mode=True)

            return t_pred, c_pred, y_t, y_c

        else:
            x, y = batch["image"], batch["label"]

            if self.do_slice_inference:
                return self.slice_inferer(inputs=x, network=self.net), y

            if val:
                return self.val_inferer(inputs=x, network=self.net), y

            return self(x), y

    def training_step(self, batch, batch_idx):
        if self.is_ocelot:
            y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch)
            tissue_loss = self.tissue_criterion(y_t_hat, y_t)
            cell_loss = self.cell_criterion(y_c_hat, y_c)
            
            total_loss = tissue_loss + cell_loss
            self.log("train_loss", total_loss, prog_bar=True)
            return total_loss
        else:   
            y_hat, y = self.infer_batch(batch)
            if self.use_bce:
                empty_labels_mask = torch.all(y == 0, dim=(1, 2, 3, 4))  # Shape: [batch_size]

                # Compute loss for non-empty labels
                masked_y_hat = y_hat[~empty_labels_mask, ...]
                masked_y = y[~empty_labels_mask, ...]

                if masked_y_hat.numel() > 0 and masked_y.numel() > 0:
                    non_empty_loss = self.criterion(masked_y_hat, masked_y)
                else:
                    non_empty_loss = torch.tensor(0.0, device=y_hat.device)

                # Compute loss for empty labels (all zeros)
                empty_masked_y_hat = y_hat[empty_labels_mask, ...]
                empty_masked_y = y[empty_labels_mask, ...]

                if empty_masked_y_hat.numel() > 0 and empty_masked_y.numel() > 0:
                    empty_loss = self.empty_criterion(empty_masked_y_hat, empty_masked_y)
                else:
                    empty_loss = torch.tensor(0.0, device=y_hat.device)

                # Total loss
                loss = non_empty_loss + empty_loss
            else:
                loss = self.criterion(y_hat, y)

            self.log("train_loss", loss, prog_bar=True)

            return loss

    def _shared_eval_step(self, batch, batch_idx):
        if self.is_ocelot:
            y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch)
            tissue_loss = self.tissue_criterion(y_t_hat, y_t)
            cell_loss = self.cell_criterion(y_c_hat, y_c)
            
            total_loss = tissue_loss + cell_loss

            y_t_hat_binarized = self.post_pred(y_t_hat)
            y_c_hat_binarized = self.post_pred(y_c_hat)

            for metric in self.metrics.values():
                metric(y_t_hat_binarized, y_t)
                metric(y_c_hat_binarized, y_c)


            return total_loss

        else:
            y_hat, y = self.infer_batch(batch, val=True)
            loss = self.criterion(y_hat, y)

            y_hat_binarized = self.post_pred(y_hat)

            if (
                self.trainer.state.fn in ["validate", "test", "predict"]
                and self.save_output
            ):
                batch["image"] = batch["image"].squeeze(0)
                batch["label"] = batch["label"].squeeze(0)
                batch["pred"] = y_hat_binarized.squeeze(0)
                # batch["softmax"] = Activations(sigmoid=True, softmax=False)(y_hat_binarized).squeeze(0)
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

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
