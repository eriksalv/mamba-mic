import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader
from glob import glob
import os
import torch
import numpy as np
import json

class OcelotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        image_dir="/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data/images",
        label_dir="/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data/annotations",
        metadata_path="/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data/metadata.json",
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
        self.metadata = load_metadata(metadata_path)

        
      
        default_preprocess = T.Compose(
                [
                    T.LoadImaged(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"]),
                    T.EnsureChannelFirstd(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"]), 
                    T.Lambdad(keys=["label_tissue"], func=standardize_label),
                    T.AsDiscreted(keys=["label_tissue"], to_onehot=2, threshold=0.5),
                    T.Lambdad(keys=["label_cropped_tissue"], func=lambda x: x[:-1]),  # Remove last channel
                    T.NormalizeIntensityd(keys=["img_tissue", "img_cell"]),  # Normalize intensity
                    T.ToTensord(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"]),  # Convert to PyTorch tensors
                ]
            )
       
        self.preprocess = preprocess if preprocess is not None else default_preprocess

        
        self.augment = (
            augment
            if augment is not None
            else T.Compose([
                    T.RandAdjustContrastd(keys=["img_tissue", "img_cell",], prob=0.7, gamma=(0.8, 1.2)),  # Random brightness and contrast adjustment
                ])
        )

    
    def collect_pairs_with_metadata(self, split):
        """
        Collects image-label pairs and retrieves metadata for tissue-cell alignment.
        """
        img_tissues = sorted(glob(os.path.join(self.image_dir, split, "tissue", "*.jpg")))
        img_cells = sorted(glob(os.path.join(self.image_dir, split, "cell", "*.jpg")))
        
        label_tissues = sorted(glob(os.path.join(self.label_dir, split, "tissue", "*.png")))
        label_cropped_tissues = sorted(glob(os.path.join(self.label_dir, split, "cropped_tissue", "*.png")))
        label_cell_masks = sorted(glob(os.path.join(self.label_dir, split, "cell_mask_images", "*.png")))
        label_cells = sorted(glob(os.path.join(self.label_dir, split, "cell", "*.csv")))

        # Extract metadata per sample
        pairs = []
        for i_t, i_c, l_t, l_ct, l_cm, l_c in zip(img_tissues, img_cells, label_tissues, label_cropped_tissues, label_cell_masks, label_cells):
            sample_id = os.path.basename(i_t).split(".")[0]  # Extract sample ID from filename
            if sample_id in self.metadata:
                meta = self.metadata[sample_id]
                pairs.append({
                    "img_tissue": i_t,
                    "img_cell": i_c, 
                    "label_tissue": l_t,
                    "label_cropped_tissue": l_ct,
                    "label_cell_mask": l_cm,
                    "label_cell": l_c,
                    "meta": meta  # Attach metadata for cropping & alignment
                })

        return pairs

    def prepare_data(self):
        directiories = ['train', 'val', 'test']
        self.train_files = self.collect_pairs_with_metadata('train')
        self.val_files = self.collect_pairs_with_metadata('val')
        self.test_files = self.collect_pairs_with_metadata('test')  

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

def standardize_label(label):
    """
    Standardize `label_tissues`:
    - Normalize values from [0, 255] to [0, 1] only for valid labels.
    - Convert unknown label (255) to 0 (background).
    """
    label = label.astype(np.float32)  # Ensure correct data type

    # Set unknown pixels (255) to background (0)
    label[label == 255] = 0

    # Normalize if max value is greater than 1
    if label.max() > 1:
        label = label / label.max()  # Normalize to [0,1]
    
    return label

def load_metadata(metadata_path):
    """
    Load metadata.json and return a dictionary mapping sample IDs to metadata.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    sample_pairs = metadata["sample_pairs"]
    
    metadata_dict = {}
    for sample_id, data in sample_pairs.items():
        metadata_dict[sample_id] = {
            "slide_name": data["slide_name"],
            "cell_patch": {
                "x_start": data["cell"]["x_start"],
                "y_start": data["cell"]["y_start"],
                "x_end": data["cell"]["x_end"],
                "y_end": data["cell"]["y_end"],
                "resized_mpp_x": data["cell"]["resized_mpp_x"],
                "resized_mpp_y": data["cell"]["resized_mpp_y"],
            },
            "tissue_patch": {
                "x_start": data["tissue"]["x_start"],
                "y_start": data["tissue"]["y_start"],
                "x_end": data["tissue"]["x_end"],
                "y_end": data["tissue"]["y_end"],
                "resized_mpp_x": data["tissue"]["resized_mpp_x"],
                "resized_mpp_y": data["tissue"]["resized_mpp_y"],
            },
            "organ": data["organ"],
            "subset": data["subset"],
        }
    
    return metadata_dict