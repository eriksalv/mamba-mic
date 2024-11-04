import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
import torch
from glob import glob
import os


class BraTS2024DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir="./data/BRATS2024",
        val_frac=0.15,
        num_workers=4,
        cache_rate=0.1,
        preprocess=None,
        augment=None,
        postprocess=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers
        self.cache_rate = cache_rate

        self.preprocess = (
            preprocess
            if preprocess is not None
            else T.Compose(
                [
                    T.LoadImaged(keys=["t1c", "t1n", "t2f", "t2w", "label"]),
                    T.EnsureChannelFirstd(
                        keys=["t1c", "t1n", "t2f", "t2w"],
                        channel_dim="no_channel",
                    ),
                    T.ConcatItemsd(keys=["t1c", "t1n", "t2f", "t2w"], name="image"),
                    T.SelectItemsd(keys=["image", "label"]),
                    T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                    T.CropForegroundd(keys=["image", "label"], source_key="image"),
                    T.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                ]
            )
        )
        self.augment = (
            augment
            if augment is not None
            else T.Compose(
                [
                    T.RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=[128, 128, 128],
                        random_size=False,
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
        self.postprocess = (
            postprocess
            if postprocess is not None
            else T.Compose(
                [
                    T.Activations(sigmoid=True),
                    T.AsDiscrete(threshold=0.5),
                ]
            )
        )

        self.subjects_with_ground_truth = None
        self.subjects_without_ground_truth = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        if not (
            os.path.isdir(os.path.join(self.data_dir, "training_data1_v2"))
            and os.path.isdir(os.path.join(self.data_dir, "validation_data"))
        ):
            raise FileNotFoundError(
                f"Please download BraTS2024 train and val set and extract it into {self.data_dir}"
            )

        seg_train_paths = sorted(
            glob(os.path.join(self.data_dir, "training_data1_v2", "*", "*-seg.nii.gz"))
        )
        t1c_train_paths = sorted(
            glob(os.path.join(self.data_dir, "training_data1_v2", "*", "*-t1c.nii.gz"))
        )
        t1n_train_paths = sorted(
            glob(os.path.join(self.data_dir, "training_data1_v2", "*", "*-t1n.nii.gz"))
        )
        t2f_train_paths = sorted(
            glob(os.path.join(self.data_dir, "training_data1_v2", "*", "*-t2f.nii.gz"))
        )
        t2w_train_paths = sorted(
            glob(os.path.join(self.data_dir, "training_data1_v2", "*", "*-t2w.nii.gz"))
        )

        t1c_val_paths = sorted(
            glob(os.path.join(self.data_dir, "validation_data", "*", "*-t1c.nii.gz"))
        )
        t1n_val_paths = sorted(
            glob(os.path.join(self.data_dir, "validation_data", "*", "*-t1n.nii.gz"))
        )
        t2f_val_paths = sorted(
            glob(os.path.join(self.data_dir, "validation_data", "*", "*-t2f.nii.gz"))
        )
        t2w_val_paths = sorted(
            glob(os.path.join(self.data_dir, "validation_data", "*", "*-t2w.nii.gz"))
        )

        self.subjects_with_ground_truth = []
        self.subjects_without_ground_truth = []

        for t1c, t1n, t2f, t2w, seg in zip(
            t1c_train_paths,
            t1n_train_paths,
            t2f_train_paths,
            t2w_train_paths,
            seg_train_paths,
            strict=True,
        ):
            self.subjects_with_ground_truth.append(
                {"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w, "label": seg}
            )

        for t1c, t1n, t2f, t2w in zip(
            t1c_val_paths,
            t1n_val_paths,
            t2f_val_paths,
            t2w_val_paths,
            strict=True,
        ):
            self.subjects_without_ground_truth.append(
                {"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w}
            )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_subjects, val_subjects = random_split(
                self.subjects_with_ground_truth, [1 - self.val_frac, self.val_frac]
            )
            self.train_set = CacheDataset(
                train_subjects,
                transform=T.Compose([self.preprocess, self.augment]),
                cache_rate=self.cache_rate,
            )
            self.val_set = CacheDataset(
                val_subjects, transform=self.preprocess, cache_rate=0
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                self.subjects_without_ground_truth,
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
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class ConvertToMultiChannelBasedOnBratsClassesd(T.MapTransform):
    """
    Converts non-overlapping BraTS 2024 labels (NETC, SNFH, ET, RC)
    into partially overlapping multi-channels (ET, TC, WT, RC)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Enhancing Tumour (ET) is label 3
            result.append(d[key] == 3)
            # Tumour Core (TC) is ET + NETC (label 3 + 1)
            result.append(torch.logical_or(d[key] == 3, d[key] == 1))
            # Whole Tumour (WT) is ET + SNFH + NETC (label 3 + 2 + 1)
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 3, d[key] == 2), d[key] == 1
                )
            )
            # Resection Cavity (RC) is label 4
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).squeeze().float()
        return d
