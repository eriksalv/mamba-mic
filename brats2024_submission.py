import torch
from system import System
import wandb
from pathlib import Path
from data_modules.brats2024 import (
    BraTS2024DataModule,
    ConvertToMultiChannelBasedOnBratsClassesd,
)
from tqdm import tqdm
import re
import nibabel as nib


def generate_submission(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project="brats2024",
        name=args.name,
        job_type="submission",
    )

    if args.local:
        checkpoint_path = args.model_ckpt
    else:
        artifact = run.use_artifact(args.model_ckpt, type="model")
        artifact_dir = artifact.download()
        checkpoint_path = Path(artifact_dir) / "model.ckpt"

    model = System.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    model.to(device)

    data_module = BraTS2024DataModule(batch_size=1, data_dir="./data/BRATS2024")
    data_module.prepare_data()
    data_module.setup(stage="test")
    test_dataset = data_module.test_set

    t1c_paths = [path["t1c"] for path in data_module.subjects_without_ground_truth]
    submission_paths = [
        re.search("BraTS-GLI-\d{5}-\d{3}", path).group(0) for path in t1c_paths
    ]

    with torch.no_grad():
        num_samples = len(test_dataset)
        assert num_samples == len(submission_paths)
        progress_bar = tqdm(
            test_dataset,
            total=num_samples,
            desc="Generating Predictions:",
        )

        for sample, sub_path in zip(progress_bar, submission_paths):
            progress_bar.set_description(sub_path)
            test_input = sample["image"].unsqueeze(0).to(device)
            test_output = model.val_inferer(inputs=test_input, network=model)
            test_output = data_module.postprocess(test_output[0])
            test_output = ConvertToMultiChannelBasedOnBratsClassesd(
                keys="label"
            ).inverse({"label": test_output})

            nib.save(
                nib.Nifti1Image(
                    test_output["label"].type(torch.float).cpu().numpy(),
                    affine=sample["image"].meta["affine"],
                ),
                f"./data/BRATS2024/{sub_path}.nii.gz",
            )

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-ckpt", required=True)
    args = parser.parse_args()

    generate_submission(args)
