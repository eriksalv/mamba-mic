import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader
from glob import glob
import os
import torch
import numpy as np

class OcelotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        image_dir="/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data/images",
        label_dir="/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data/annotations",
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
        name="ocelot"
    ):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.name = name
    
        default_preprocess = T.Compose(
                [
                    T.LoadImaged(keys=["img_tissue", "img_cells", "label_tissues", "label_cropped_tissues"]),
                    T.EnsureChannelFirstd(keys=["img_tissue", "img_cells", "label_tissues", "label_cropped_tissues"]), 
                    T.NormalizeIntensityd(keys=["img_tissue", "img_cells"]),  # Normalize intensity
                    T.ConcatItemsd(keys=["img_tissue", "img_cells"], name="image", dim=0),  # Stack images as multi-channel input
                    T.ConcatItemsd(keys=["label_tissues", "label_cropped_tissues"], name="label", dim=0),  # Stack labels as multi-channel target
                    T.ToTensord(keys=["image", "label"]),  # Convert to PyTorch tensors
                ]
            )
        self.preprocess = preprocess if preprocess is not None else default_preprocess
        self.augment = (
            augment
            if augment is not None
            else T.Compose(
                [
                T.RandZoomd(keys=["image", "label"], prob=0.7, min_zoom=0.9, max_zoom=1.1),  # Zoom with ±10% resizing
                    T.RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=[896, 896], num_classes=4, 
                                            num_samples=1, ratios=[1, 1, 1, 1]),  # Random crop to 896x896
                    T.RandFlipd(keys=["image", "label"], prob=0.7, spatial_axis=[0, 1]),  # Random flip (horizontal or vertical)
                    T.RandRotate90d(keys=["image", "label"], prob=0.7),  # Random rotation by 90°, 180°, or 270°
                    T.RandAdjustContrastd(keys=["image"], prob=0.7, gamma=(0.8, 1.2)),  # Random brightness and contrast adjustment
                ]
            )
        )

    def collect_pairs(self, split):
        img_tissues = sorted(glob(os.path.join(self.image_dir, split, "tissue", "*.jpg")))
        img_cells = sorted(glob(os.path.join(self.image_dir, split, "cell", "*.jpg")))
        
        label_tissues = sorted(glob(os.path.join(self.label_dir, split, "tissue", "*.png")))
        label_cropped_tissues = sorted(glob(os.path.join(self.label_dir, split, "cropped_tissue", "*.png")))
        label_cell_masks = sorted(glob(os.path.join(self.label_dir, split, "cell_mask_images", "*.png")))
        label_cells = sorted(glob(os.path.join(self.label_dir, split, "cell", "*.csv")))

        return [{"img_tissue": i_t, "img_cells": i_c, 
                "label_tissues": l_t, "label_cropped_tissues": l_ct,
                "label_cell_masks": l_cm, "label_cells": l_c}
                for i_t, i_c, l_t, l_ct, l_cm, l_c in zip(img_tissues, img_cells, label_tissues, label_cropped_tissues, label_cell_masks, label_cells)] 

    def prepare_data(self):
        directiories = ['train', 'val', 'test']
        self.train_files = self.collect_pairs('train')
        self.val_files = self.collect_pairs('val')
        self.test_files = self.collect_pairs('test')  

        print(f"Train: {len(self.train_files)} | Val: {len(self.val_files)} | Test: {len(self.test_files)}")
   


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = CacheDataset(
                self.train_files,
                transform=T.Compose([self.preprocess, self.augment]),
                cache_rate=self.cache_rate,
            )
            self.val_set = CacheDataset(
                self.val_files,
                transform=self.preprocess,
                cache_rate=0.0,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_files = CacheDataset(
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
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1)