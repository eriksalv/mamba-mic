import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
from glob import glob
import os
from typing import Literal


class HNTSMRGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        task: Literal["preRT", "midRT"],
        data_dir="/cluster/projects/vc/data/mic/open/HNTS-MRG",
        val_frac=0.15,
        use_test_for_val=True,
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.use_test_for_val = use_test_for_val
        self.num_workers = num_workers
        self.cache_rate = cache_rate

        self.preprocess = (
            preprocess
            if preprocess is not None
            else T.Compose(
                [
                    T.LoadImaged(keys=["image", "label"]),
                    T.EnsureChannelFirstd(keys=["image", "label"]),
                    T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                    T.Spacingd(
                        keys=["image", "label"],
                        pixdim=(0.5, 0.5, 1.2),
                        mode=("bilinear", "nearest"),
                    ),
                    T.Resized(keys=["image", "label"], spatial_size=[512, 512, 124]),
                    T.AsDiscreted(keys="label", to_onehot=3),
                ]
            )
        )
        self.augment = (
            augment
            if augment is not None
            else T.Compose(
                [
                    T.RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=[192, 192, 48],
                        ratios=[1, 1, 1],
                        num_classes=3,
                        image_key="image",
                        image_threshold=0,
                        num_samples=1,
                    ),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    T.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                    T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ]
            )
        )

        self.train_subjects = None
        self.test_subjects = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        if not (
            os.path.isdir(os.path.join(self.data_dir, "train"))
            and os.path.isdir(os.path.join(self.data_dir, "test"))
        ):
            raise FileNotFoundError(f"No train and test set found in {self.data_dir}")

        # TODO: consider adding registered preRT if task is midRT
        train_t2_paths = sorted(
            glob(os.path.join(self.data_dir, "train", "*", self.task, "*_T2.nii.gz"))
        )
        test_t2_paths = sorted(
            glob(os.path.join(self.data_dir, "test", "*", self.task, "*_T2.nii.gz"))
        )
        train_mask_paths = sorted(
            glob(os.path.join(self.data_dir, "train", "*", self.task, "*_mask.nii.gz"))
        )
        test_mask_paths = sorted(
            glob(os.path.join(self.data_dir, "test", "*", self.task, "*_mask.nii.gz"))
        )

        self.train_subjects = []
        self.test_subjects = []

        for t2, mask in zip(train_t2_paths, train_mask_paths, strict=True):
            self.train_subjects.append({"image": t2, "label": mask})

        for t2, mask in zip(test_t2_paths, test_mask_paths, strict=True):
            self.test_subjects.append({"image": t2, "label": mask})

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.use_test_for_val:
                train_subjects = self.train_subjects
                val_subjects = self.test_subjects
            else:
                train_subjects, val_subjects = random_split(
                    self.train_subjects, [1 - self.val_frac, self.val_frac]
                )

            self.train_set = CacheDataset(
                train_subjects,
                transform=T.Compose([self.preprocess, self.augment]),
                cache_rate=self.cache_rate,
            )
            self.val_set = CacheDataset(
                val_subjects, 
                transform=T.Compose(
                    [
                        self.preprocess,
                        T.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                    ]
                ),
                cache_rate=0
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                self.test_subjects,
                transform=T.Compose(
                    [
                        self.preprocess,
                        T.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                    ]
                ),
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
