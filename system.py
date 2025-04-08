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
from ocelot_util import (
    normalize_crop_coords_batch,
    evaluate_cell_detection_batch, 
    cell_detection_postprocessing_batch, 
    calculate_metrics, 
    load_ground_truth,
    compute_class_weights,
    weighted_mse_loss,
)


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
        tissue_only = False,
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
        self.tissue_only = tissue_only

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
            sigmoid=True,
            softmax=False,
            squared_pred=True,
            )
            self.tissue_metrics = {
            "dice": DiceMetric(include_background=include_background, reduction="mean_batch"),
            "hd95": HausdorffDistanceMetric(
                include_background=include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
            }

            #self.cell_criterion = torch.nn.MSELoss()
            self.cell_criterion = DiceCELoss(
            include_background=include_background,
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
            y_t = batch["label_tissue"]
            cropped_label = batch["label_cropped_tissue"]
            y_c = batch['soft_is_mask']
            meta_batch = batch["meta"]
        
            cropped_coords = normalize_crop_coords_batch(meta_batch)
            if self.tissue_only:
                if val:
                    return self.val_inferer(inputs=batch['img_tissue'], network=self.net), y_t
                return self(batch['img_tissue']), y_t

            if val:
                t_pred, c_pred =  self.net.forward(batch,cropped_coords, train_mode = False)
                return t_pred, c_pred, y_t, y_c

            t_pred, c_pred = self.net.forward(batch, cropped_coords,
             tissue_label = y_t, cell_tissue_label = cropped_label, train_mode=True)

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
            if self.tissue_only:
                y_t_hat, y_t = self.infer_batch(batch)
                tissue_loss = self.tissue_criterion(y_t_hat, y_t)
                self.log("train_tissue_loss", tissue_loss, prog_bar=True)
                return tissue_loss

            y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch)
            tissue_loss = self.tissue_criterion(y_t_hat, y_t)
            # Compute class weights per batch for cell segmentation
            #class_weights = compute_class_weights(y_c)  # Compute per batch
            #cell_loss = weighted_mse_loss(y_c_hat, y_c, class_weights)  # Apply weighted MS
            cell_loss = self.cell_criterion(y_c_hat, y_c)
            total_loss = tissue_loss + cell_loss
            self.log("train_cell_loss", cell_loss, prog_bar=True)

            if not self.net.is_pretrained:
                self.log("train_tissue_loss", tissue_loss, prog_bar=True)

            return total_loss
        else:   
            y_hat, y = self.infer_batch(batch)
            loss = self.criterion(y_hat, y)

            self.log("train_loss", loss, prog_bar=True)

            return loss

    def _shared_eval_step(self, batch, batch_idx):

        if self.is_ocelot:
            if self.tissue_only:
                y_t_hat, y_t = self.infer_batch(batch, val=True)
            else:
                y_t_hat, y_c_hat, y_t, y_c = self.infer_batch(batch, val=True)

            y_t_hat_binarized = self.post_pred(y_t_hat)
            tissue_loss = self.tissue_criterion(y_t_hat, y_t)

            if self.tissue_only:
                for name, metric in self.tissue_metrics.items():
                    metric(y_t_hat_binarized, y_t)
                return tissue_loss

            #class_weights = compute_class_weights(y_c)  
            #cell_loss = weighted_mse_loss(y_c_hat, y_c, class_weights) 
            cell_loss = self.cell_criterion(y_c_hat, y_c)
            total_loss = tissue_loss + cell_loss


            gt_cells_list, gt_classes_list = load_ground_truth(batch)

            # Get predicted cells and their classes from the model outputs
            pred_cells_list, pred_classes_list, confidences_list = cell_detection_postprocessing_batch(y_c_hat)

            # Compute TP, FP, FN
            total_tp_bc, total_fp_bc, total_fn_bc, total_tp_tc, total_fp_tc, total_fn_tc = evaluate_cell_detection_batch(
                pred_cells_list, pred_classes_list, gt_cells_list, gt_classes_list,
                )

            self.tp_bc_count += total_tp_bc
            self.fp_bc_count += total_fp_bc
            self.fn_bc_count += total_fn_bc

            self.tp_tc_count += total_tp_tc
            self.fp_tc_count += total_fp_tc
            self.fn_tc_count += total_fn_tc

            

            if not self.net.is_pretrained:
                for name, metric in self.tissue_metrics.items():
                    metric(y_t_hat_binarized, y_t)

            

            return total_loss, tissue_loss, cell_loss

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
        if self.is_ocelot:
            if self.tissue_only:
                tissue_loss = self._shared_eval_step(batch, batch_idx)
                self.log("val_loss", tissue_loss, sync_dist=True)
                return {"val_loss": tissue_loss}

            loss, tissue_loss, cell_loss = self._shared_eval_step(batch, batch_idx)
            self.log("val_loss", cell_loss, sync_dist=True)
            if not self.net.is_pretrained:
                self.log("val_tissue_loss", tissue_loss, sync_dist=True)
            self.log("val_cell_loss", cell_loss, sync_dist=True)

            return {"val_loss": loss}
        else:
            loss = self._shared_eval_step(batch, batch_idx)
            self.log("val_loss", loss, sync_dist=True)
            return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return {"test_loss": loss}

    def _shared_on_epoch_end(self, stage="val"):
        metrics = {}

        if self.is_ocelot:
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

        else:

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
