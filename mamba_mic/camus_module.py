from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
)

from mamba_mic.base_module import BaseModule


class CAMUSModule(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.metrics_ed = {
            "dice": DiceMetric(include_background=self.include_background, reduction="mean_batch"),
            "hd95": HausdorffDistanceMetric(
                include_background=self.include_background,
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
            "dice": DiceMetric(include_background=self.include_background, reduction="mean_batch"),
            "hd95": HausdorffDistanceMetric(
                include_background=self.include_background,
                distance_metric="euclidean",
                percentile=95,
                reduction="mean_batch",
            ),
            "confusion_matrix": ConfusionMatrixMetric(
                metric_name=["precision", "sensitivity", "f1 score"],
                reduction="mean_batch",
            ),
        }

    def _shared_eval_step(self, batch, batch_idx):
        batch_ed = {"image": batch["ED"], "label": batch["ED_gt"]}
        batch_es = {"image": batch["ES"], "label": batch["ES_gt"]}

        y_hat_ed, y_ed = self.infer_batch(batch_ed, val=True)
        y_hat_es, y_es = self.infer_batch(batch_es, val=True)

        loss_ed = self.criterion(y_hat_ed, y_ed)
        loss_es = self.criterion(y_hat_es, y_es)

        y_hat_binarized_ed = self.post_pred(y_hat_ed)
        y_hat_binarized_es = self.post_pred(y_hat_es)

        if self.trainer.state.fn in ["validate", "test", "predict"] and self.save_output:
            batch["pred_ed"] = y_hat_binarized_ed
            batch["pred_es"] = y_hat_binarized_es
            self.trainer.datamodule.invert_and_save(batch)

        for metric in self.metrics_ed.values():
            metric(y_hat_binarized_ed, y_ed)

        for metric in self.metrics_es.values():
            metric(y_hat_binarized_es, y_es)

        return (loss_ed + loss_es) / 2, None

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
