import torch
import wandb
from pathlib import Path
from tqdm import tqdm
import re
import nibabel as nib
from brats_2024_metrics.metrics_GLI import get_LesionWiseResults
from lightning.pytorch import seed_everything
from monai.inferers import sliding_window_inference

from mamba_mic.base_module import System
from mamba_mic.data_modules.brats2024 import (
    BraTS2024DataModule,
)


def run(args):
    if args.local:
        checkpoint_path = args.model_ckpt
    else:
        run = wandb.init(
            project="brats2024",
            name=args.name,
            job_type="submission",
        )
        artifact = run.use_artifact(args.model_ckpt, type="model")
        artifact_dir = artifact.download()
        checkpoint_path = Path(artifact_dir) / "model.ckpt"

    data_module = BraTS2024DataModule(batch_size=1, data_dir="./data/BRATS2024")
    data_module.prepare_data()
    data_module.setup(stage="test")
    test_dataset = data_module.test_set

    test_paths = [data["label"] for data in test_dataset.data]
    submission_paths = [
        re.search("BraTS-GLI-\d{5}-\d{3}", path).group(0) for path in test_paths
    ]

    if args.generate:
        generate_submission(
            args,
            test_dataset,
            checkpoint_path,
            submission_paths,
            data_module.postprocess,
        )

    evaluate_submission(args, test_paths, submission_paths)


def generate_submission(
    args, test_dataset, checkpoint_path, submission_paths, postprocess
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = System.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    model.to(device)

    with torch.no_grad():
        num_samples = len(test_dataset)
        assert num_samples == len(submission_paths)
        progress_bar = tqdm(
            test_dataset,
            total=num_samples,
            desc="Generating Predictions:",
        )

        for test_data, sub_path in zip(progress_bar, submission_paths):
            progress_bar.set_description("Generating Predictions: " + sub_path)
            x, y = (
                test_data["image"].to(device).unsqueeze(0),
                test_data["label"].to(device).unsqueeze(0),
            )
            test_data["pred"] = sliding_window_inference(
                x,
                roi_size=[128, 128, 128],
                overlap=0.5,
                sw_batch_size=2,
                predictor=model,
            ).squeeze(0)

            postprocessed = postprocess(test_data)
            y_pred, y = postprocessed["pred"], postprocessed["label"]

            nib.save(
                nib.Nifti1Image(
                    y_pred.type(torch.float).numpy(),
                    affine=test_data["image"].meta["original_affine"],
                ),
                f"./data/BRATS2024/submissions/pred/{args.name}_{sub_path}.nii.gz",
            )


def evaluate_submission(args, test_paths, submission_paths):
    progress_bar = tqdm(
        submission_paths,
        total=len(submission_paths),
        desc="Evaluating Predictions:",
    )
    for test_path, sub_path in zip(test_paths, progress_bar):
        progress_bar.set_description("Evaluating Predictions: " + sub_path)
        full_sub_path = (
            f"./data/BRATS2024/submissions/pred/{args.name}_{sub_path}.nii.gz"
        )
        get_LesionWiseResults(
            full_sub_path,
            test_path,
            challenge_name="BraTS-GLI",
            output=f"./data/BRATS2024/submissions/scores/{args.name}_{sub_path}.csv",
        )


if __name__ == "__main__":
    import argparse

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument(
        "--generate", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    run(args)
