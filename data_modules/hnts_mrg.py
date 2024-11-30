import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
from glob import glob
import os
from typing import Literal
import torch


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

        default_preprocess = (
            T.Compose(
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
            ) if task == 'preRT'
            else T.Compose(
                [
                    T.LoadImaged(keys=["t2", "t2_registered", "mask_registered", "label"]),
                    T.EnsureChannelFirstd(keys=["t2", "t2_registered", "mask_registered", "label"]),
                    T.ConcatItemsd(keys=["t2", "t2_registered"], name="image"),
                    T.SelectItemsd(keys=["image", "mask_registered", "label"]),
                    T.Orientationd(keys=["image", "mask_registered", "label"], axcodes="RAS"),
                    T.Spacingd(
                        keys=["image", "mask_registered", "label"],
                        pixdim=(0.5, 0.5, 1.2),
                        mode=("bilinear", "nearest", "nearest"),
                    ),
                    T.Resized(keys=["image", "mask_registered", "label"], spatial_size=[512, 512, 124]),
                    ConvertToMultiChannelBasedOnHNTSMRGClassesd(keys="mask_registered"),
                    T.AsDiscreted(keys="label", to_onehot=3),
                ]
            )
        )

        self.preprocess = (
            preprocess
            if preprocess is not None
            else default_preprocess
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

        self.val_transform = [
                        self.preprocess,
                        T.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                    ]
        
        if task == 'midRT':
            self.val_transform.append(T.ConcatItemsd(keys=["image", "mask_registered"], name="image"))
        
        self.val_transform = T.Compose(self.val_transform)

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

        if self.task == "preRT":
            for t2, mask in zip(train_t2_paths, train_mask_paths, strict=True):
                self.train_subjects.append({"image": t2, "label": mask})

            for t2, mask in zip(test_t2_paths, test_mask_paths, strict=True):
                self.test_subjects.append({"image": t2, "label": mask})

        elif self.task == "midRT":
            train_t2_registered_paths = sorted(
                glob(os.path.join(self.data_dir, "train", "*", self.task, "*_T2_registered.nii.gz"))
            )
            test_t2_registered_paths = sorted(
                glob(os.path.join(self.data_dir, "test", "*", self.task, "*_T2_registered.nii.gz"))
            )
            train_mask_registered_paths = sorted(
                glob(os.path.join(self.data_dir, "train", "*", self.task, "*_mask_registered.nii.gz"))
            )
            test_mask_registered_paths = sorted(
                glob(os.path.join(self.data_dir, "test", "*", self.task, "*_mask_registered.nii.gz"))
            )
            train_t2_pre_paths = sorted(
                glob(os.path.join(self.data_dir, "train", "*", "preRT", "*_T2.nii.gz"))
            )
            test_t2_pre_paths = sorted(
                glob(os.path.join(self.data_dir, "test", "*", "preRT", "*_T2.nii.gz"))
            )
            train_mask_pre_paths = sorted(
                glob(os.path.join(self.data_dir, "train", "*", "preRT", "*_mask.nii.gz"))
            )
            test_mask_pre_paths = sorted(
                glob(os.path.join(self.data_dir, "test", "*", "preRT", "*_mask.nii.gz"))
            )

            for t2, mask, t2_registered, mask_registered, t2_pre, mask_pre in zip(
                train_t2_paths,
                train_mask_paths,
                train_t2_registered_paths,
                train_mask_registered_paths,
                train_t2_pre_paths,
                train_mask_pre_paths,
                strict=True,
            ):
                self.train_subjects.append(
                    {
                        "t2": t2,
                        "label": mask,
                        "t2_registered": t2_registered,
                        "mask_registered": mask_registered,
                        "t2_pre": t2_pre,
                        "mask_pre": mask_pre
                    }
                )

            for t2, mask, t2_registered, mask_registered, t2_pre, mask_pre in zip(
                test_t2_paths,
                test_mask_paths,
                test_t2_registered_paths,
                test_mask_registered_paths,
                test_t2_pre_paths,
                test_mask_pre_paths,
                strict=True,
            ):
                self.test_subjects.append(
                    {
                        "t2": t2,
                        "label": mask,
                        "t2_registered": t2_registered,
                        "mask_registered": mask_registered,
                        "t2_pre": t2_pre,
                        "mask_pre": mask_pre
                    }
                )

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
                transform=self.val_transform,
                cache_rate=0
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                self.test_subjects,
                transform=self.val_transform,
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

class ConvertToMultiChannelBasedOnHNTSMRGClassesd(T.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in data:
                result = []
                result.append(d[key] == 1)
                result.append(d[key] == 2)
                d[key] = torch.stack(result, axis=0).squeeze().float()
        return d
