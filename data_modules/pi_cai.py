import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
from glob import glob
import os
import torch
import numpy as np


class PICAIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir="/cluster/projects/vc/data/mic/open/Prostate/PI-CAI/preped-images/images/",
        val_frac=0.1,
        test_frac=0.1,
        use_test_for_val=False,
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
        filter_empty_labels=False,
        lowest_positive_isup_grade=2
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.use_test_for_val = use_test_for_val
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.filter_empty_labels = filter_empty_labels

        default_preprocess = T.Compose(
            [
                T.LoadImaged(keys=["image", "label"]),
                T.EnsureChannelFirstd(keys=["image", "label"]),
                T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                T.Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.5, 0.5, 3),
                    mode=("bilinear", "nearest"),
                ),
                T.CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    channel_indices=0,
                    allow_smaller=False,
                ),
                T.CenterSpatialCropd(keys=["image", "label"], roi_size=[384, 384, 24]),
                T.Resized(
                    keys=["image", "label"],
                    spatial_size=[384, 384, 24],
                    mode=("area", "nearest"),
                ),
                T.ClipIntensityPercentilesd(
                    keys="image", lower=None, upper=99, channel_wise=True
                ),
                T.NormalizeIntensityd(keys="image", channel_wise=True),
                ConvertToBinaryLabeld(keys="label", lowest_positive=lowest_positive_isup_grade),
            ]
        )

        self.preprocess = preprocess if preprocess is not None else default_preprocess

        augment_spatial = T.Compose(
            [
                T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                T.Rand3DElasticd(
                    keys=["image", "label"],
                    prob=0.6,
                    magnitude_range=(50, 80),
                    sigma_range=(5, 7),
                    mode=("bilinear", "nearest"),
                ),
                T.RandRotate90d(
                    keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)
                ),
                T.RandRotated(
                    keys=["image", "label"],
                    prob=0.4,
                    range_x=[-0.25, 0.25],
                    range_y=[-0.25, 0.25],
                    range_z=[-0.25, 0.25],
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                T.RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[256, 256, 24],
                    num_classes=2,
                    num_samples=1,
                    ratios=[0.6, 1],
                ),
            ]
        )

        augment_intensity = T.Compose(
            [
                T.RandGaussianNoised(keys="image", prob=0.3, std=0.1),
                T.RandAdjustContrastd(
                    keys="image",
                    prob=0.4,
                    gamma=[0.7, 1.5],
                    retain_stats=True,
                ),
                T.RandScaleIntensityd(
                    keys="image", prob=0.6, factors=0.1, channel_wise=True
                ),
                T.RandShiftIntensityd(
                    keys="image", prob=0.5, offsets=0.1, channel_wise=True
                ),
            ]
        )

        self.augment = (
            augment
            if augment is not None
            else T.Compose([augment_spatial, augment_intensity])
        )

        self.train_subjects = None
        self.test_subjects = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        image_files = sorted(glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")))
        label_files = sorted(glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))

        if len(image_files) == 0 or len(label_files) == 0:
            raise RuntimeError(f"No images or labels found in {self.data_dir}")

        if len(image_files) != len(label_files):
            raise RuntimeError("Mismatch between number of images and labels.")

        # Store subject dictionaries
        self.subjects_with_ground_truth = [
            {"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)
        ]

        if self.filter_empty_labels:
            print(f"Subjects before filtering: {len(self.subjects_with_ground_truth)}")
            self.subjects_with_ground_truth = [
                s
                for s in self.subjects_with_ground_truth
                if np.any(T.LoadImage()(s["label"]) >= 2)
            ]
            print(f"Subjects after filtering: {len(self.subjects_with_ground_truth)}")

    def setup(self, stage=None):
        train_subjects, val_subjects, test_subjects = random_split(
            self.subjects_with_ground_truth,
            [1 - self.val_frac - self.test_frac, self.val_frac, self.test_frac],
            generator=torch.Generator().manual_seed(42),
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = CacheDataset(
                train_subjects,
                transform=T.Compose([self.preprocess, self.augment]),
                cache_rate=self.cache_rate,
            )
            self.val_set = CacheDataset(
                val_subjects,
                transform=self.preprocess,
                cache_rate=0.0,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                test_subjects,
                transform=self.preprocess,
                cache_rate=0.0,
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class ConvertToMultiChanneld(T.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in data:
                label = d[key]  # Extract label tensor

                # Create a one-hot encoding with 3 channels (excluding background)
                result = [
                    (label == 1).float(),  # Class 1
                    (label == 2).float(),  # Class 2
                    (label == 3).float(),  # Class 3
                    (label == 4).float(),  # Class 4
                    (label == 5).float(),  # Class 5
                ]

                d[key] = torch.stack(
                    result, axis=0
                ).squeeze()  # Stack along channel axis
        return d


class ConvertToBinaryLabeld(T.MapTransform):
    def __init__(self, keys, lowest_positive):
        super().__init__(keys)
        self.lowest_positive = lowest_positive

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in data:
                label = d[key]  # Extract label tensor

                # Convert to binary: 0 for ISUP ≤1, 1 for ISUP ≥2
                d[key] = (label >= self.lowest_positive).float()

        return d
