import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader
from glob import glob
import os
import torch
import numpy as np
import json
from PIL import Image, ImageOps
from monai.data import MetaTensor
import pandas as pd
from scipy.ndimage import gaussian_filter


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

        
      
        default_preprocess = T.Compose([
            CorrectOrientationd(
                keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"],
                label_keys=["label_tissue"]
            ),
            #T.LoadImaged(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"]),
            #T.EnsureChannelFirstd(keys = ["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue"]),
            T.Lambdad(keys=["label_tissue"], func=onehot_tissue_label),
            T.Lambdad(keys=["label_tissue", "label_cropped_tissue"], func=exclude_bg),
            T.NormalizeIntensityd(keys=["img_tissue", "img_cell"]),
            SoftISMaskd(keys=["label_cell"]), 
            T.ToTensord(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue", "soft_is_mask"])
        ])
       
        self.preprocess = preprocess if preprocess is not None else default_preprocess

        
        self.augment = (
            augment
            if augment is not None
            else T.Compose(
                        [   T.RandSpatialCropd(keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue",
                                 "soft_is_mask"], roi_size=(512, 512)),
                            T.RandZoomd(
                                keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue",
                                 "soft_is_mask"], prob=0.7, min_zoom=0.9, max_zoom=1.1
                            ),  
                            T.RandFlipd(
                                keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue",
                                 "soft_is_mask"], prob=1, spatial_axis=[0, 1]
                            ),  # Random flip (horizontal or vertical)
                            T.RandRotate90d(
                                keys=["img_tissue", "img_cell", "label_tissue", "label_cropped_tissue",
                                 "soft_is_mask"], prob=1
                            ),  # Random rotation by 90°, 180°, or 270°
                            T.RandAdjustContrastd(
                                keys=["img_tissue", "img_cell"], prob=0.7, gamma=(0.8, 1.2)
                            ),  # Random brightness and contrast adjustment
                        ]
                    )
        )

    def invert_tissue_pred(self, batch, parallelized=False):
        transform = T.Invertd(
            keys="pred_tissue",
            transform=self.train_set.transform,  
            orig_keys="label_tissue",  
            meta_keys="pred_tissue_meta_dict",
            orig_meta_keys="img_tissue_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        )
        tissue_pred_to_crop_list = []
        batch["pred_tissue"] = MetaTensor(batch["pred_tissue"])
        
        if parallelized:
            batch_inverter = T.BatchInverseTransform(self.train_set.transform, self.train_dataloader())
            segs_dict = {"pred_tissue": batch["pred_tissue"]}
            
            with T.allow_missing_keys_mode(self.train_set.transform):
                inverted_batch = batch_inverter(segs_dict)
                tissue_pred_to_crop_list  = [entry['pred_tissue'] for entry in inverted_batch]
        else:
            # Loop through the batch and apply the transform to each row (each entry in batch)
            for i in range(len(batch["pred_tissue"])):
                # Create a copy of the batch to prevent modifying the original during inversion
                batch_copy = batch.copy()
                batch_copy["pred_tissue"] = batch["pred_tissue"][i]  
                batch_copy["label_tissue"] = batch["label_tissue"][i]  
                batch_copy["img_tissue"] = batch["img_tissue"][i]
                    
                
                inverted_entry = transform(batch_copy)
                tissue_pred_to_crop_list.append(inverted_entry["pred_tissue"])
        
        # After inversion: store the results back in the batch
        batch['pred_tissue'] = tissue_pred_to_crop_list
    
        return batch
    
     
        
    def reapply_augmentations(self, batch):
        transform = T.ResampleToMatchd(
            keys ='tissue_crop_resized',
            key_dst = 'label_cropped_tissue',
            mode = 'bilinear'
        )
        aligned_crop = transform(batch)
        return aligned_crop

    def collect_pairs_with_metadata(self, split):
        """
        Collects image-label pairs and retrieves metadata for tissue-cell alignment.
        Ensures lists have the same length to prevent mismatches.
        """
        img_tissues = sorted(glob(os.path.join(self.image_dir, split, "tissue", "*.jpg")))
        img_cells = sorted(glob(os.path.join(self.image_dir, split, "cell", "*.jpg")))
        
        label_tissues = sorted(glob(os.path.join(self.label_dir, split, "tissue", "*.png")))
        label_cropped_tissues = sorted(glob(os.path.join(self.label_dir, split, "cropped_tissue", "*.png")))
        #label_cell_masks = sorted(glob(os.path.join(self.label_dir, split, "cell_mask_images", "*.png")))
        label_cells = sorted(glob(os.path.join(self.label_dir, split, "cell", "*.csv")))

        # Check if all lists have the same length
        #list_lengths = [len(img_tissues), len(img_cells), len(label_tissues), len(label_cropped_tissues), len(label_cell_masks), len(label_cells)]
        #if len(set(list_lengths)) > 1:
        #    raise ValueError(f"Mismatch in file counts: {dict(zip(['img_tissues', 'img_cells', 'label_tissues', 'label_cropped_tissues', 'label_cell_masks', 'label_cells'], list_lengths))}")

        # Extract metadata per sample
        pairs = []
        for i_t, i_c, l_t, l_ct, l_c in zip(img_tissues, img_cells, label_tissues, label_cropped_tissues, label_cells):
            sample_id = os.path.basename(i_t).split(".")[0]  # Extract sample ID from filename
            if sample_id in self.metadata:
                meta = self.metadata[sample_id]
                pairs.append({
                    "img_tissue": i_t,
                    "img_cell": i_c, 
                    "label_tissue": l_t,
                    "label_cropped_tissue": l_ct,
                    #"label_cell_mask": l_cm,
                    "label_cell": l_c,
                    "meta": meta  # Attach metadata for cropping & alignment
                })
            else:
                print(f"Warning: Sample ID {sample_id} not found in metadata.")

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
            self.test_set = CacheDataset(
                self.test_files,
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

def standardize_label(label):
    """
    Standardize label_tissues to be in [0, 1] range.
    - If values are in [0, 255], normalize.
    - If values are categorical (e.g., {0,1,2}), leave as is.
    """
    if label.max() > 1:  # If the label has intensity values 0-255, normalize
        label = label / 255.0
    return label  # Keep as is if already normalized or categorical


def onehot_tissue_label(label):

    label[label > 2] = 3
    return T.AsDiscrete(to_onehot=3)(label - 1)


def exclude_bg(label):
    return label[1].unsqueeze(0)

class CorrectOrientationd(T.MapTransform):
    def __init__(self, keys, label_keys=None, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.label_keys = label_keys if label_keys else []  # Keys for labels that should remain grayscale
        self.load_image = T.LoadImaged(keys=self.keys, allow_missing_keys=True, ensure_channel_first=True)
    def inverse(self, data):
        """ Prevent inversion by returning data unchanged. """
        return data

    def __call__(self, data):
        loaded_data = self.load_image(data)

        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                # Already processed, skip conversion
                continue

            
            img = Image.open(data[key])
            if isinstance(img, Image.Image):  
                if hasattr(img, '_getexif'):
                    exif = img._getexif()
                    if exif is not None:
                        orientation = exif.get(274)  # EXIF tag for orientation
                        if orientation == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation == 8:
                            img = img.rotate(90, expand=True)
                # Convert based on whether it's an image or label
                if key in self.label_keys:
                    img = img.convert("L") 
                else:
                    img = img.convert("RGB")  

                img = np.array(img)

                if key in self.label_keys:
                    img = torch.tensor(img).unsqueeze(0)  # Shape: (1, H, W)
                else:
                    img = torch.tensor(img).permute(2, 0, 1)  # Shape: (3, H, W)
                
                img_meta_tensor = MetaTensor(
                    img, 
                    affine=torch.eye(4),  # Default identity affine matrix
                    meta={"original_size": img.shape[:2], "orientation_fixed": True}
                )
                

                data[key] = img_meta_tensor

                # Ensure metadata tracking
                data[f"{key}_meta_dict"] = img_meta_tensor.meta

        return data

class SoftISMaskd(T.MapTransform):
    """
    A MONAI transform that converts cell centroid annotations from a CSV file into
    a Soft Instance Segmentation (Soft IS) probability map using Gaussian kernels.

    Outputs a 3-channel probability map: (background, tumor cells, background cells)
    """

    def __init__(self, keys, sigma=2, radius_in_microns = 1.4, image_size=(1024, 1024)):
        """
        Args:
            keys (list of str): Keys that contain the CSV file paths (e.g., ["label_cell"]).
            sigma (int): Standard deviation of the Gaussian kernel in pixels (~15 pixels = 3µm at 0.2mpp).
            image_size (tuple): The size of the output probability map (height, width).
        """
        super().__init__(keys)
        self.sigma = sigma
        self.image_size = image_size
        self.radius_in_microns = radius_in_microns

    def __call__(self, data):
        d = dict(data)
        key = self.keys[0] 
        
        csv_path = d[key]
        df = pd.read_csv(csv_path, header=None, names=["x", "y", "class"])
        # Initialize probability maps (H, W)
        tumor_mask = np.zeros(self.image_size, dtype=np.float32)
        background_cell_mask = np.zeros(self.image_size, dtype=np.float32)
        total_fg = np.zeros(self.image_size, dtype=np.float32)
        soft_is_mask = np.zeros((3, *self.image_size), dtype=np.float32)
        # Process each centroid to create a soft instance segmentation - paper used nuclick, but this might be good enough
        Y, X = np.ogrid[:self.image_size[0], :self.image_size[1]]
        radius_in_px = int(self.radius_in_microns/0.2)
        expected_area = np.pi * (radius_in_px ** 2)

        for _, row in df.iterrows():
            x, y, cell_class = int(row["x"]), int(row["y"]), int(row["class"])
            
            if 0 <= x < self.image_size[1] and 0 <= y < self.image_size[0]:
                mask = (X - x) ** 2 + (Y - y) ** 2 <= radius_in_px ** 2
                y_min, y_max = max(0, y - radius_in_px), min(self.image_size[0], y + radius_in_px + 1)
                x_min, x_max = max(0, x - radius_in_px), min(self.image_size[1], x + radius_in_px + 1)
                
                mask_cropped = mask[y_min:y_max, x_min:x_max]

                # Apply the mask to the corresponding area in the full image
                if cell_class == 1:
                    background_cell_mask[y_min:y_max, x_min:x_max] = np.maximum(background_cell_mask[y_min:y_max, x_min:x_max], mask_cropped)
                elif cell_class == 2:
                    tumor_mask[y_min:y_max, x_min:x_max] = np.maximum(tumor_mask[y_min:y_max, x_min:x_max], mask_cropped)

                # Calculate the sum of the mask (i.e., the number of pixels inside the mask)
                mask_sum = np.sum(mask_cropped)
               
                if cell_class == 1:
                    background_cell_mask[mask] = 1
                elif cell_class == 2:
                    tumor_mask[mask] = 1

        
        tumor_mask = gaussian_filter(tumor_mask, sigma=self.sigma)
        background_cell_mask = gaussian_filter(background_cell_mask, sigma=self.sigma)

        total_fg = tumor_mask + background_cell_mask
        background_mask = 1 - np.clip(total_fg, 0, 1)  
        
        soft_is_mask = np.stack([background_mask, background_cell_mask, tumor_mask], axis=0)  

        # Convert to PyTorch tensor
        d["soft_is_mask"] = torch.tensor(soft_is_mask, dtype=torch.float32)
        return d