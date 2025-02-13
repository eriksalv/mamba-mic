import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
from glob import glob
import os
import torch
import numpy as np
from monai.data import ITKReader

class PICAIV2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        image_dir = "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0/images",
        label_dir = "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0/picai_labels/csPCa_lesion_delineations/human_expert/resampled",
        val_frac=0.1,
        test_frac=0.1,
        use_test_for_val=False,
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.use_test_for_val = use_test_for_val
        self.num_workers = num_workers
        self.cache_rate = cache_rate

        default_preprocess = T.Compose(
                [
                    T.LoadImaged(keys=["t2w", "adc", "hbv", "label"], reader = ITKReader),
                    T.EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "label"]),
            
                    T.ResampleToMatchd(
                        keys=["adc", "hbv"], 
                        key_dst = 't2w',
                        mode="bilinear", 
                        padding_mode="border",  
                        ),
                    
                    T.Orientationd(keys=["t2w", "adc", "hbv", "label"], axcodes="RAS"),
                    
                    T.Spacingd(
                        keys=["t2w", "adc", "hbv", "label"],
                        pixdim=(0.5, 0.5, 3.0),
                        mode=("bilinear", "bilinear", "bilinear", "nearest"),
                    ),
                    
                    T.NormalizeIntensityd(keys=["adc"], channel_wise = False),
                    ZScoreNormalizeD(keys = ['t2w', 'hbv']),
                    T.Resized(keys=["t2w", "adc", "hbv", "label"], spatial_size=[512, 512, 32], mode=("area","area", "area", "nearest")),
                    ConvertToBinaryLabeld(keys=["label"]),
            ]) 

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

                    T.RandCropByLabelClassesd(keys=["t2w", "adc", "hbv", "label"], label_key = "label", spatial_size = [256, 256, 32],
                                              num_classes = 2, num_samples = 1, ratios = [0.5, 0.5]),
                    StackImages(keys=["t2w", "adc", "hbv"]),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                ]
            )
        )

        self.val_transform = [
                        self.preprocess,
                    ]
        
        self.val_transform = T.Compose(self.val_transform)

        self.train_subjects = None
        self.test_subjects = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        patient_folders = sorted(glob(os.path.join(self.image_dir, "*")))
        patient_ids = [os.path.basename(folder) for folder in patient_folders]


        data = []
        for patient_id in patient_ids:
            patient_folder = os.path.join(self.image_dir, patient_id)
            
            # Find all file paths
            label_paths = glob(os.path.join(self.label_dir, f"{patient_id}_*.nii.gz"))
            t2w_paths = glob(os.path.join(patient_folder, "*_t2w.mha"))
            adc_paths = glob(os.path.join(patient_folder, "*_adc.mha"))
            hbv_paths = glob(os.path.join(patient_folder, "*_hbv.mha"))
            
            # Identify the case IDs by extracting the common identifier part from the filenames
            case_ids = set()
            for path in t2w_paths + adc_paths + hbv_paths:
                # Extract the case ID (e.g., '10131_1000132' from '10131_1000132_t2w.mha')
                case_id = '_'.join(os.path.basename(path).split('_')[:2])
                case_ids.add(case_id)
            
            # Process each case separately
            for case_id in case_ids:
                # Filter files by case_id
                case_t2w = [path for path in t2w_paths if case_id in path]
                case_adc = [path for path in adc_paths if case_id in path]
                case_hbv = [path for path in hbv_paths if case_id in path]
                case_label = [path for path in label_paths if case_id in path]
                
                # Ensure all files are found for this case before adding it to the data
                if case_t2w and case_adc and case_hbv and case_label:
                    data_entry = {
                        "t2w": case_t2w,
                        "adc": case_adc,
                        "hbv": case_hbv,
                        "label": case_label
                    }
                    data.append(data_entry)


        # Store subject dictionaries
        self.subjects_with_ground_truth = data

    def setup(self, stage=None):
        train_subjects, val_subjects, test_subjects = random_split(
            self.subjects_with_ground_truth,
            [1 - self.val_frac - self.test_frac, self.val_frac, self.test_frac], 
            generator=torch.Generator().manual_seed(42)
        )
        # Sanity check
        print(train_subjects)
        print(val_subjects)
        print(test_subjects)
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
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
                        StackImages(keys=["t2w", "adc", "hbv"]),
                    ]
                ), 
                cache_rate=0.0
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CacheDataset(
                test_subjects,
                transform=T.Compose(
                    [
                        self.preprocess,
                        StackImages(keys=["t2w", "adc", "hbv"]),
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
                
                d[key] = torch.stack(result, axis=0).squeeze()  # Stack along channel axis
        return d

class ConvertToBinaryLabeld(T.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in data:
                label = d[key]  # Extract label tensor
                
                # Convert to binary: 0 for ISUP ≤1, 1 for ISUP ≥2
                d[key] = (label >= 1).float()
                
        return d
    
class ZScoreNormalizeD(T.MapTransform):
    """
    Custom MONAI transform for instance-wise Z-score normalization.
    This normalizes each image by subtracting the mean and dividing by the standard deviation.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]  
                
                mean = torch.mean(img)
                std = torch.std(img)

          
                if std > 0:
                    d[key] = (img - mean) / std
                else:
                    d[key] = img  
                    
        return d
    
class StackImages(T.MapTransform):
    
    def __init__(self, keys: list):
        super().__init__(keys)
        
    def __call__(self, data: dict):
        
        # Retrieve the images for t2w, adc, hbv
        t2w = data["t2w"]
        adc = data["adc"]
        hbv = data["hbv"]
        # Stack them into a single 3D tensor (channels last)
        stacked_image = torch.stack([t2w, adc, hbv], dim=0)
        # Add the stacked image to the dictionary
        data["image"] = stacked_image.squeeze()
        return data
