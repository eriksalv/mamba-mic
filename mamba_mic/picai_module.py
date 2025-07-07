import wandb
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates

from mamba_mic.data_modules.pi_caiv2 import PICAIV2DataModule, evaluate_cases
from mamba_mic.base_module import BaseModule


class PICAIModule(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extra_metrics = {"picai": []}

    def _shared_eval_step(self, batch, batch_idx):
        loss, batch = super()._shared_eval_step(batch, batch_idx)

        if isinstance(self.trainer.datamodule, PICAIV2DataModule):
            _eval = evaluate_case(
                y_true=batch["label"].squeeze().cpu().numpy(),
                y_det=batch["pred"].squeeze().cpu().numpy(),
                y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
            )
            self.extra_metrics["picai"].append(_eval)

        return loss, batch

    def _shared_on_epoch_end(self, stage="val"):
        super()._shared_on_epoch_end(stage)

        if isinstance(self.trainer.datamodule, PICAIV2DataModule):
            metrics = self.log_picai_metrics(stage)
            self.log_dict(metrics, sync_dist=True)

    def log_picai_metrics(self, stage):
        picai_metrics = evaluate_cases(self.extra_metrics["picai"])
        self.extra_metrics["picai"] = []

        roc_data = [
            [fpr, tpr]
            for (fpr, tpr) in zip(picai_metrics.case_FPR, picai_metrics.case_TPR)
        ]
        pr_data = [
            [recall, precision]
            for (recall, precision) in zip(
                picai_metrics.recall, picai_metrics.precision
            )
        ]
        roc_table = wandb.Table(data=roc_data, columns=["FPR", "TPR"])
        pr_table = wandb.Table(data=pr_data, columns=["Recall", "Precision"])

        wandb.log(
            {f"{stage}/ROC": wandb.plot.line(roc_table, "FPR", "TPR", title="ROC")}
        )
        wandb.log(
            {
                f"{stage}/PR": wandb.plot.line(
                    pr_table, "Recall", "Precision", title="PR"
                )
            }
        )

        return {
            "AP": picai_metrics.AP,
            "AUROC": picai_metrics.auroc,
            "PICAI-Score": picai_metrics.score,
        }
