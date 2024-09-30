import lightning.pytorch as pl
from monai.apps.datasets import DecathlonDataset
from monai.transforms.compose import Compose
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord
)
from torch.utils.data import DataLoader


class DecathlonDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir='./data', task='Task01_BrainTumour', val_frac=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.task = task
        self.val_frac = val_frac

        # TODO: resize all images to same height, width, and slices, or do a random fixed size crop
        self.transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        # Download dataset (will download all sections, not just training)
        DecathlonDataset(self.data_dir, task=self.task,
                         section='training', download=True, cache_rate=0.0)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = DecathlonDataset(
                self.data_dir, task=self.task, section='training', transform=self.transform, cache_rate=0.0, val_frac=self.val_frac)
            self.val_set = DecathlonDataset(
                self.data_dir, task=self.task, section='validation', transform=self.transform, cache_rate=0.0, val_frac=self.val_frac)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = DecathlonDataset(
                self.data_dir, task=self.task, section='test', transform=self.transform, cache_rate=0.0)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
