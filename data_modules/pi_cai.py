import lightning.pytorch as pl
from monai.data import CacheDataset
from monai import transforms as T
from torch.utils.data import DataLoader, random_split
from glob import glob
import os
import torch


class PICAIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir='/cluster/projects/vc/data/mic/open/Prostate/PI-CAI/preped-images/images/',
        val_frac=0.1,
        test_frac=0.1,
        use_test_for_val=False,
        num_workers=4,
        cache_rate=0.0,
        preprocess=None,
        augment=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.use_test_for_val = use_test_for_val
        self.num_workers = num_workers
        self.cache_rate = cache_rate

        default_preprocess = T.Compose(
                [
                    T.LoadImaged(keys=["image", "label"]),
                    T.EnsureChannelFirstd(keys=["image", "label"]),
                    T.Lambdad(keys="image", func=lambda x: x[:1]),
                    T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                    T.Spacingd(
                        keys=["image", "label"],
                        pixdim=(0.5, 0.5, 1.2),
                        mode=("bilinear", "nearest"),
                    ),
                    T.NormalizeIntensityd(keys="image", channel_wise=True),
                    T.Resized(keys=["image", "label"], spatial_size=[512, 512, 32]),
                    ConvertToMultiChanneld(keys=["label"]),
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
                    T.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                    T.RandCropByLabelClassesd(keys=["image", "label"], label_key = "label", spatial_size = [256, 256, 32],
                                              num_classes = 5, num_samples = 1, ratios = [1, 1, 1, 1, 1], allow_missing_keys=True),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                ]
            )
        )

        self.val_transform = [
                        self.preprocess,
                        T.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                    ]
        
        self.val_transform = T.Compose(self.val_transform)

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
                        T.NormalizeIntensityd(keys="image", channel_wise=True, nonzero=True)
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
                        T.NormalizeIntensityd(keys="image", channel_wise=True, nonzero=True)
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

