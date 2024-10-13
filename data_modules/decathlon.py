import lightning.pytorch as pl
from monai.apps.datasets import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    Activations,
    AsDiscrete,
)
from torch.utils.data import DataLoader
import torch


class DecathlonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir="./data",
        task="Task01_BrainTumour",
        val_frac=0.2,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.task = task
        self.val_frac = val_frac
        self.num_workers = num_workers

        self.preprocess = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityd(keys="image", minv=0, maxv=1, channel_wise=True),
            ]
        )
        self.augment = Compose(
            [
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=[128, 128, 128], random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        self.post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        # Download dataset (will download all sections, not just training)
        DecathlonDataset(
            self.data_dir,
            task=self.task,
            section="training",
            download=True,
            cache_rate=0.0,
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = DecathlonDataset(
                self.data_dir,
                task=self.task,
                section="training",
                transform=Compose([self.preprocess, self.augment]),
                cache_rate=0.0,
                val_frac=self.val_frac,
            )
            self.val_set = DecathlonDataset(
                self.data_dir,
                task=self.task,
                section="validation",
                transform=self.preprocess,
                cache_rate=0.0,
                val_frac=self.val_frac,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = DecathlonDataset(
                self.data_dir,
                task=self.task,
                section="test",
                transform=self.preprocess,
                cache_rate=0.0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).squeeze().float()
        return d
