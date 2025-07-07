import random

import torch
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
)
from monai.transforms import Activations

from mamba_mic.base_module import BaseModule
from mamba_mic.data_modules.util.ocelot_util import (
    MaskedBCEAndDiceLoss,
    calculate_metrics,
    cell_detection_postprocessing_batch,
    compute_class_weights,
    evaluate_cell_detection_batch,
    load_ground_truth,
    normalize_crop_coords_batch,
    weighted_mse_loss,
)


class OcelotModule(BaseModule):
    def __init__(
        self,
        use_bce=False,
        use_focal=True,
        tissue_only=False,
        is_pretrained=False,
        use_mse=False,
        masked_loss=False,
        prob_custom_label=1.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.use_bce = use_bce
        self.use_focal = use_focal
        self.tissue_only = tissue_only
        self.is_pretrained = is_pretrained
        self.use_mse = use_mse
        self.masked_loss = masked_loss
        self.prob_custom_label = prob_custom_label

        if self.use_focal:
            self.criterion = FocalLoss(include_background=self.include_background, gamma=1.0, alpha=0.90)
        else:
            self.criterion = DiceCELoss(
                include_background=self.include_background,
                sigmoid=not self.softmax,
                softmax=self.softmax,
                squared_pred=True,
            )

        self.empty_criterion = torch.nn.BCEWithLogitsLoss()

        self.tissue_criterion = DiceCELoss(
            include_background=True,
            sigmoid=True,
            softmax=False,
            squared_pred=True,
        )
        self.masked_criterion = MaskedBCEAndDiceLoss()
        self.tissue_metrics = {
            "dice": DiceMetric(include_background=True, reduction="mean_batch"),
            "hd95": HausdorffDistanceMetric(
                include_background=self.include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
        }
        self.masked_metrics = {
            "dice_masked": DiceMetric(include_background=True, reduction="mean_batch"),
            "hd95_masked": HausdorffDistanceMetric(
                include_background=self.include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
        }
        self.cell_criterion = DiceCELoss(
            include_background=self.include_background,
            sigmoid=False,
            softmax=True,
            squared_pred=True,
        )

        self.tp_tc_count = 0
        self.fp_tc_count = 0
        self.fn_tc_count = 0

        self.tp_bc_count = 0
        self.fp_bc_count = 0
        self.fn_bc_count = 0

        self.cell_post_transform = Activations(softmax=True, dim=1)

    def infer_batch(self, batch, val=False):
        if isinstance(batch, list):
            batch = batch[0]

        y_t = batch["label_tissue"]

        if not val:
            if random.random() < self.prob_custom_label:
                cropped_label = batch["custom_label"]
            else:
                cropped_label = batch["label_cropped_tissue"]

        y_c = batch["soft_is_mask"]
        meta_batch = batch["meta"]

        cropped_coords = normalize_crop_coords_batch(meta_batch)
        if self.tissue_only:
            if val:
                return self.val_inferer(inputs=batch["img_tissue"], network=self.net), y_t
            return self(batch["img_tissue"]), y_t

        if val:
            t_pred, c_pred = self.net.forward(batch, cropped_coords, train_mode=False)
            return t_pred, c_pred, y_t, y_c

        t_pred, c_pred = self.net.forward(
            batch, cropped_coords, tissue_label=y_t, cell_tissue_label=cropped_label, train_mode=True
        )

        return t_pred, c_pred, y_t, y_c

    def training_step(self, batch, batch_idx):
        if self.tissue_only:
            y_t_hat, y_t = self.infer_batch(batch)
            if isinstance(batch, list):
                batch = batch[0]
            mask = batch["unknown_mask"]
            if self.masked_loss:
                tissue_loss = self.masked_criterion(y_t_hat, y_t, mask=mask)
            else:
                tissue_loss = self.tissue_criterion(y_t_hat, y_t)
            self.log("train_tissue_loss", tissue_loss, prog_bar=True)
            return tissue_loss

        y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch)
        tissue_loss = self.tissue_criterion(y_t_hat, y_t)

        if self.use_mse:
            class_weights = compute_class_weights(y_c)  # Compute per batch
            cell_loss = weighted_mse_loss(y_c_hat, y_c, class_weights)  # Apply weighted MS
        else:
            cell_loss = self.cell_criterion(y_c_hat, y_c)
        total_loss = tissue_loss + cell_loss
        self.log("train_cell_loss", cell_loss, prog_bar=True)

        if not self.is_pretrained:
            self.log("train_tissue_loss", tissue_loss, prog_bar=True)
            return total_loss

        return cell_loss

    def _shared_eval_step(self, batch, batch_idx):
        if self.tissue_only:
            y_t_hat, y_t = self.infer_batch(batch, val=True)
        else:
            y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch, val=True)

        if isinstance(batch, list):
            batch = batch[0]

        mask = batch["unknown_mask"]

        # Masking should happen after sigmoid and thresholding
        y_t_hat_binarized = self.post_pred(y_t_hat)  # Apply the thresholding on the logits

        # Use the masked label to compute loss and metrics
        # y_t_masked = y_t * mask  # Apply mask to ground truth labels as well
        if self.masked_loss:
            tissue_loss = self.masked_criterion(y_t_hat, y_t, mask=mask)
        else:
            tissue_loss = self.tissue_criterion(y_t_hat, y_t)

        if self.tissue_only:
            for name, metric in self.tissue_metrics.items():
                metric(y_t_hat_binarized, y_t)
            y_t_hat_masked_binarized = y_t_hat_binarized * mask
            for name, metric in self.masked_metrics.items():
                metric(y_t_hat_masked_binarized, y_t)
            return tissue_loss

        # class_weights = compute_class_weights(y_c)
        # cell_loss = weighted_mse_loss(y_c_hat, y_c, class_weights)
        cell_loss = self.cell_criterion(y_c_hat, y_c)
        total_loss = tissue_loss + cell_loss

        y_c_hat = self.cell_post_transform(y_c_hat)

        gt_cells_list, gt_classes_list = load_ground_truth(batch)

        # Get predicted cells and their classes from the model outputs
        pred_cells_list, pred_classes_list, confidences_list = cell_detection_postprocessing_batch(y_c_hat)

        # Compute TP, FP, FN
        total_tp_bc, total_fp_bc, total_fn_bc, total_tp_tc, total_fp_tc, total_fn_tc = evaluate_cell_detection_batch(
            pred_cells_list,
            pred_classes_list,
            gt_cells_list,
            gt_classes_list,
        )

        self.tp_bc_count += total_tp_bc
        self.fp_bc_count += total_fp_bc
        self.fn_bc_count += total_fn_bc

        self.tp_tc_count += total_tp_tc
        self.fp_tc_count += total_fp_tc
        self.fn_tc_count += total_fn_tc

        if not self.is_pretrained:
            for name, metric in self.tissue_metrics.items():
                metric(y_t_hat_binarized, y_t)

        return total_loss, tissue_loss, cell_loss

    def validation_step(self, batch, batch_idx):
        if self.tissue_only:
            tissue_loss = self._shared_eval_step(batch, batch_idx)
            self.log("val_loss", tissue_loss, sync_dist=True)
            return {"val_loss": tissue_loss}

        loss, tissue_loss, cell_loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", cell_loss, sync_dist=True)
        if not self.is_pretrained:
            self.log("val_tissue_loss", tissue_loss, sync_dist=True)
        self.log("val_cell_loss", cell_loss, sync_dist=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        if self.tissue_only:
            tissue_loss = self._shared_eval_step(batch, batch_idx)
            self.log("test_loss", tissue_loss, sync_dist=True)
            return {"test_loss": tissue_loss}

        loss, tissue_loss, cell_loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", cell_loss, sync_dist=True)
        self.log("test_tissue_loss", tissue_loss, sync_dist=True)
        self.log("test_cell_loss", cell_loss, sync_dist=True)

        return {"test_loss": cell_loss}

    def _shared_on_epoch_end(self, stage="val"):
        metrics = {}
        if not self.is_pretrained:
            for name, metric in self.tissue_metrics.items():
                per_channel = metric.aggregate()
                if name == "confusion_matrix":
                    for conf_name, conf_metric in zip(metric.metric_name, per_channel):
                        metrics[f"{stage}_tissue_{conf_name}"] = conf_metric.mean()
                        metric.reset()
                else:
                    for i in range(len(per_channel)):
                        metrics[f"tissue_channel{i}/{stage}_{name}"] = per_channel[i]
                    metrics[f"{stage}_tissue_{name}"] = per_channel.mean()
                    metric.reset()

            for name, metric in self.masked_metrics.items():
                per_channel = metric.aggregate()
                for i in range(len(per_channel)):
                    metrics[f"tissue_channel{i}/{stage}_{name}"] = per_channel[i]
                metrics[f"{stage}_tissue_{name}"] = per_channel.mean()
                metric.reset()

        if not self.tissue_only:
            precision_bc, recall_bc, f1_bc = calculate_metrics(self.tp_bc_count, self.fp_bc_count, self.fn_bc_count)
            precision_tc, recall_tc, f1_tc = calculate_metrics(self.tp_tc_count, self.fp_tc_count, self.fn_tc_count)
            mean_f1 = (f1_bc + f1_tc) / 2

            metrics[f"{stage}_cell_bc_precision"] = precision_bc
            metrics[f"{stage}_cell_bc_recall"] = recall_bc
            metrics[f"{stage}_cell_bc_f1"] = f1_bc

            metrics[f"{stage}_cell_tc_precision"] = precision_tc
            metrics[f"{stage}_cell_tc_recall"] = recall_tc
            metrics[f"{stage}_cell_tc_f1"] = f1_tc

            metrics[f"{stage}_cell_mf1"] = mean_f1

            self.tp_tc_count = 0
            self.fp_tc_count = 0
            self.fn_tc_count = 0

            self.tp_bc_count = 0
            self.fp_bc_count = 0
            self.fn_bc_count = 0
        self.log_dict(metrics, sync_dist=True)
