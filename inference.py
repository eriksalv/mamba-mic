import wandb
import torch
from pathlib import Path
from models.unet import UNetModel
from tqdm.auto import tqdm
import numpy as np
import random
from data_modules.decathlon import DecathlonDataModule


def log_predictions_into_tables(
    sample_image: np.array,
    sample_label: np.array,
    predicted_label: np.array,
    split: str = None,
    data_idx: int = None,
    table: wandb.Table = None,
    max_slices=10,
):
    num_channels, _, _, num_slices = sample_image.shape
    total = min(num_slices, max_slices)
    with tqdm(total=total, leave=False) as progress_bar:
        slice_indices = list(range(num_slices))
        random.shuffle(slice_indices)
        for slice_idx in slice_indices[:max_slices]:
            wandb_images = []
            for channel_idx in range(num_channels):
                wandb_images += [
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Tumor-Core": {
                                "mask_data": sample_label[0, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Tumor Core"},
                            },
                            "prediction/Tumor-Core": {
                                "mask_data": predicted_label[0, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Tumor Core"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Whole-Tumor": {
                                "mask_data": sample_label[1, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Whole Tumor"},
                            },
                            "prediction/Whole-Tumor": {
                                "mask_data": predicted_label[1, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Whole Tumor"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Enhancing-Tumor": {
                                "mask_data": sample_label[2, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Enhancing Tumor"},
                            },
                            "prediction/Enhancing-Tumor": {
                                "mask_data": predicted_label[2, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Enhancing Tumor"},
                            },
                        },
                    ),
                ]
            table.add_data(split, data_idx, slice_idx, *wandb_images)
            progress_bar.update(1)
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--project", default="evals")
    parser.add_argument("--name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=args.project, name=args.name, job_type="eval")

    if args.local:
        checkpoint_path = args.model_ckpt
    else:
        artifact = run.use_artifact(args.model_ckpt, type="model")
        artifact_dir = artifact.download()
        checkpoint_path = Path(artifact_dir) / "model.ckpt"

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = UNetModel(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    data_module = DecathlonDataModule(batch_size=4)
    data_module.setup()
    val_dataset = data_module.val_set

    # create the prediction table
    prediction_table = wandb.Table(
        columns=[
            "Split",
            "Data Index",
            "Slice Index",
            "Image-Channel-0/Tumor-Core",
            "Image-Channel-1/Tumor-Core",
            "Image-Channel-2/Tumor-Core",
            "Image-Channel-3/Tumor-Core",
            "Image-Channel-0/Whole-Tumor",
            "Image-Channel-1/Whole-Tumor",
            "Image-Channel-2/Whole-Tumor",
            "Image-Channel-3/Whole-Tumor",
            "Image-Channel-0/Enhancing-Tumor",
            "Image-Channel-1/Enhancing-Tumor",
            "Image-Channel-2/Enhancing-Tumor",
            "Image-Channel-3/Enhancing-Tumor",
        ]
    )

    # Perform inference and visualization
    with torch.no_grad():
        max_samples = 10
        progress_bar = tqdm(
            enumerate(val_dataset[:max_samples]),
            total=max_samples,
            desc="Generating Predictions:",
        )
        for data_idx, sample in progress_bar:
            val_input = sample["image"].unsqueeze(0).to(device)
            val_output = model(val_input)
            val_output = data_module.post_trans(val_output[0])
            prediction_table = log_predictions_into_tables(
                sample_image=sample["image"].cpu().numpy(),
                sample_label=sample["label"].cpu().numpy(),
                predicted_label=val_output.cpu().numpy(),
                data_idx=data_idx,
                split="validation",
                table=prediction_table,
            )

        wandb.log({"Predictions/Tumor-Segmentation-Data": prediction_table})

    wandb.finish()
