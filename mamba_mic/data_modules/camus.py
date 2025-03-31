import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, Any
import SimpleITK as sitk
import numpy as np
import torch
from mamba_mic.data_modules.util.kfold import kfold
from mamba_mic.data_modules.util.select_random_slice import SelectRandomSliced


class CAMUSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir="./data/CAMUS/database_nifti",
        cv_folds=10,
        fold_index=0,
        views=None,
        train_frac=None,
        filter_poor_image_quality_for_training=False,
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
        name="camus",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.cv_folds = cv_folds
        self.fold_index = fold_index
        self.train_frac = train_frac
        self.filter_poor_image_quality_for_training = filter_poor_image_quality_for_training
        self.name = name
        self.views = views or ["2CH", "4CH"]

        default_preprocess = T.Compose(
            [
                T.Lambdad(keys=["image", "label"], func=sitk_load),
                # Adds metadata from `Info_<view>.cfg` files
                T.ToMetaTensord(keys=["image", "label"]),
                # Scale intensities to [0, 1]
                T.Lambdad(keys=["image"], func=lambda x: x / 255.0),
                # Select a random slice from the time dimension
                T.Resized(keys=["image", "label"], spatial_size=[512, 512], mode=("area", "nearest")),
                T.Lambdad(keys=["label"], func=lambda x: x.unsqueeze(0)),
                T.AsDiscreted(keys=["label"], to_onehot=4),
                T.Lambdad(keys=["label"], func=lambda x: x.permute(1, 0, 2, 3)),
            ]
        )

        self.preprocess_val = T.Lambda(get_es_ed_from_sequence)

        default_augment = T.Compose(
            [
                SelectRandomSliced(keys=["image", "label"]),
                T.SqueezeDimd(keys=["label"], dim=0),
                T.RandFlipd(
                    keys=["image", "label"],
                    prob=0.4,
                    spatial_axis=1,
                ),
                T.RandRotated(
                    keys=["image", "label"],
                    prob=0.4,
                    range_x=[-0.2, 0.2],
                    range_y=[-0.2, 0.2],
                    range_z=[-0.2, 0.2],
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                ),
                T.RandScaleIntensityd(
                    keys=["image"],
                    prob=0.4,
                    factors=0.1,
                ),
                T.RandShiftIntensityd(
                    keys=["image"],
                    prob=0.4,
                    offsets=0.1,
                ),
            ]
        )
        self.preprocess = preprocess or default_preprocess
        self.augment = augment or default_augment

        self.subjects = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        patient_ids = list(range(1, 501))
        patient_folders = [Path(self.data_dir) / f"patient{patient_id:04d}" for patient_id in patient_ids]
        self.subjects = [
            {
                # Full sequences
                "image": patient_folder / f"patient{patient_id:04d}_{view}_half_sequence.nii.gz",
                "label": patient_folder / f"patient{patient_id:04d}_{view}_half_sequence_gt.nii.gz",
                # Info cfg files
                "image_meta_dict": load_meta(patient_folder, view),
            }
            for patient_folder, patient_id in zip(patient_folders, patient_ids)
            for view in self.views
        ]

    def setup(self, stage=None):
        train_subjects, test_subjects = kfold(self.subjects, self.cv_folds, shuffle=True, seed=42)[self.fold_index]

        if self.train_frac and self.train_frac < 1.0:
            train_subjects = train_subjects[: int(len(train_subjects) * self.train_frac)]

        if self.filter_poor_image_quality_for_training:
            train_subjects = [
                subject for subject in train_subjects if subject["image_meta_dict"]["ImageQuality"] != "Poor"
            ]

        # Sanity check. Should be 450 * 2 training subjects
        # And 50 * 2 test subjects with default arguments
        print("Training subjects:", len(train_subjects))
        print("Test subjects:", len(test_subjects))

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = CacheDataset(
                train_subjects,
                transform=T.Compose(
                    [
                        self.preprocess,
                        T.EnsureTyped(keys=["image", "label"], track_meta=False),
                        self.augment,
                    ]
                ),
                cache_rate=self.cache_rate,
            )
            self.val_set = CacheDataset(
                test_subjects,
                transform=[self.preprocess, self.preprocess_val],
                cache_rate=self.cache_rate,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                test_subjects,
                transform=[self.preprocess, self.preprocess_val],
                cache_rate=self.cache_rate,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)


def sitk_load(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))

    # Extract numpy array from the SimpleITK image object
    im_array = torch.Tensor(np.squeeze(sitk.GetArrayFromImage(image)))

    return im_array


def load_meta(patient_path, view):
    cfg = {"view": view, "patient_id": patient_path.name}
    with open(patient_path / f"Info_{view}.cfg", "r") as f:
        for line in f:
            key, value = line.split(": ")
            cfg[key.strip()] = value.strip()
    return cfg


def get_es_ed_from_sequence(sequence):
    """
    Get the ED and ES frames from a sequence.
    """
    img = sequence["image"]
    ED_frame = int(img.meta["ED"]) - 1
    ES_frame = int(img.meta["ES"]) - 1

    label = sequence["label"]

    return {
        "ED": img[ED_frame].unsqueeze(0),
        "ES": img[ES_frame].unsqueeze(0),
        "ED_gt": label[ED_frame],
        "ES_gt": label[ES_frame],
    }
