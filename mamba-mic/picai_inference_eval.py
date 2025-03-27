import torch
from system import System
from data_modules.pi_caiv2 import PICAIV2DataModule
from monai.inferers import sliding_window_inference
import lightning.pytorch as pl
from monai.transforms import Compose, Activations, Invertd
import monai.transforms as T
from lightning.pytorch import seed_everything
from tqdm import tqdm
import re
import nibabel as nib
import numpy as np
import os
import wandb
from picai_eval import evaluate
from scipy.ndimage import label
from pathlib import Path


def run(args):

    run = wandb.init(
            project="picai_eval",
            name=args.name,
            job_type="inference_and_eval",
            )
    if args.local:
        checkpoint_path = args.model_ckpt
    else:
        artifact = run.use_artifact(args.model_ckpt, type="model")
        artifact_dir = artifact.download()
        checkpoint_files = list(Path(artifact_dir).glob("*.ckpt"))
        if len(checkpoint_files) > 0:
            checkpoint_path = checkpoint_files[0]  
        else:
            raise FileNotFoundError("No .ckpt file found in the artifact directory.")
   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = System.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    model.to(device)

    data_module = PICAIV2DataModule(batch_size=1, include_empty_eval=True)
    data_module.prepare_data()
    data_module.setup()
    val_set = data_module.val_set
    test_set = data_module.test_set
    print(val_set)
    print(test_set)
    infer(args = args, model=model, test_set=test_set, device=device)
    eval(args=args)

def infer(args, model, test_set, device):

    model.eval()
    with torch.no_grad():
        for test_data in tqdm(test_set):
            path = test_data['label'].meta['filename_or_obj']
            filename = os.path.basename(path)  
            case_id = '_'.join(filename.split('_')[:2])


            case_label_dir = f"./data/PICCAIv2/labels/test/{case_id}"
            case_pred_dir = f"./data/PICCAIv2/predictions/test/{case_id}"

            # Define paths to label and prediction files
            pred_path = f"{case_pred_dir}/{args.name}.nii.gz"
            label_path = f"{case_label_dir}/test.nii.gz"

            # Check if the files already exist to avoid reprocessing
            #if os.path.exists(label_path) and os.path.exists(pred_path):
            #    print(f"Skipping case {case_id} as predictions already exist.")
            #    continue
            
            x, y = test_data["image"].to(device).unsqueeze(0), test_data["label"].to(device).unsqueeze(0)
            test_data['pred'] = sliding_window_inference(
                x, roi_size=[256, 256, 24], overlap=0.5, sw_batch_size=3, predictor=model
            ).squeeze(0)
            
            
            postprocessed = post_transforms(test_data, test_set)
            y_pred = torch.tensor(postprocessed['pred'])  
            y = torch.tensor(postprocessed['label'])

            os.makedirs(case_label_dir, exist_ok=True)
            os.makedirs(case_pred_dir, exist_ok=True)
        
            nib.save(
                nib.Nifti1Image(
                    y_pred.type(torch.float).numpy(),
                    affine=test_data['image'].meta['original_affine'],
                ),
                f"{case_pred_dir}/{args.name}.nii.gz",
            )


            label_file_path = f"{case_label_dir}/test.nii.gz"
            if not os.path.exists(label_file_path):
                nib.save(
                    nib.Nifti1Image(
                        y.type(torch.float).numpy(),
                        affine=test_data['image'].meta['original_affine'],
                    ),
                    label_file_path,
                )

def eval(args):
    ground_truth_dir = "./data/PICCAIv2/labels/test"
    predictions_dir = "./data/PICCAIv2/predictions/test"

    # Get sorted case IDs
    case_ids = sorted(os.listdir(ground_truth_dir))

    ground_truths = []
    processed_predictions = []

    processor = ProcessPredictions(keys=["pred"])

    for case_id in tqdm(case_ids, desc="Processing Predictions"):
        
        case_label_dir = os.path.join(ground_truth_dir, case_id)
        case_pred_dir = os.path.join(predictions_dir, case_id)

        # Find the correct file in each case directory
        gt_files = sorted([f for f in os.listdir(case_label_dir) if f.endswith(".nii.gz")])
        pred_files = sorted([
            f for f in os.listdir(case_pred_dir) 
            if f.endswith(".nii.gz") and f.startswith(f"{args.name}")
        ])

        # Ensure each case has exactly one label and one prediction file
        if len(gt_files) != 1 or len(pred_files) != 1:
            print(f"Skipping case {case_id} due to missing or multiple files.")
            continue

        gt_path = os.path.join(case_label_dir, gt_files[0])
        pred_path = os.path.join(case_pred_dir, pred_files[0])

        # Load ground truth
        gt_nifti = nib.load(gt_path)
        ground_truth = gt_nifti.get_fdata().astype(np.uint8)  # Ensure binary format
        ground_truths.append(ground_truth.squeeze())

        # Load prediction
        pred_nifti = nib.load(pred_path)
        pred = pred_nifti.get_fdata().astype(np.float32)  # Ensure float format

        # Apply processing
        processed_pred = processor({"pred": pred})["pred"]
        processed_predictions.append(processed_pred.squeeze())

    print(f"Loaded {len(ground_truths)} ground truth masks and {len(processed_predictions)} processed predictions.")

    piccai_score = evaluate(
    y_det = processed_predictions,
    y_true = ground_truths
    )
    print(piccai_score)
    auroc_match = re.search(r"auroc=(\d+\.\d+)%", str(piccai_score))
    ap_match = re.search(r"AP=(\d+\.\d+)%", str(piccai_score))

    if auroc_match and ap_match:
        auroc = float(auroc_match.group(1))
        ap = float(ap_match.group(1))
        mean_score = (auroc + ap) / 2

        print(f"AUROC: {auroc:.2f}%")
        print(f"AP: {ap:.2f}%")
        print(f"Overall Score (AUROC + AP) / 2: {mean_score:.2f}%")

        # Log metrics to Weights & Biases
      
        wandb.log({
            "AUROC": auroc,
            "AP": ap,
            "Overall Score": mean_score
        })

        
class ConvertToBinaryLabeld(T.MapTransform):
    def __init__(self, keys: list, invertible=True, allow_missing_keys=True):
        self.keys = keys
        self.invertible = invertible
        self.allow_missing_keys = allow_missing_keys
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in data:
                label = d[key]  # Extract label tensor

                if self.invertible:
                    # Store the original label tensor for later inversion
                    d[f"original_{key}"] = label.clone()

                # Convert to binary: 0 for ISUP ≤1, 1 for ISUP ≥2
                d[key] = (label >= 1).float()

        return d
        
def post_transforms(val_data, test_set):
    transform = Compose([
        
        Invertd(
            keys="pred",
            transform=test_set.transform,
            orig_keys="label",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
            device="cpu",
        ),

        Invertd(
            keys="label",
            transform=test_set.transform,  
            orig_keys="label",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        

        ConvertToBinaryLabeld(keys = ["label"]),
        
    ])

    # Apply transformations
    val_data = transform(val_data)
    sigmoid = Activations(sigmoid=True)
    val_data['pred'] = sigmoid(val_data['pred'])
    return val_data


class ProcessPredictions(T.MapTransform):
    """Ensure predictions are non-overlapping 3D connected components with a single confidence score per lesion."""
    def __init__(self, keys, threshold=0.1):
        super().__init__(keys)
        self.threshold = threshold

    def __call__(self, data):
        for key in self.keys:
            pred = data[key]
            processed_pred = self.process_predictions(pred)
            data[key] = processed_pred 
        return data

    def process_predictions(self, pred):
        """Processes prediction to match evaluation format."""
        # Binarize the prediction based on the threshold
        binary_pred = (pred > self.threshold).astype(np.uint8)
        
        if binary_pred.ndim > 3:
            binary_pred = np.squeeze(binary_pred)
            pred = np.squeeze(pred)

        # Label connected components using 26-connectivity
        labeled_pred, num_components = label(binary_pred, structure=np.ones((3, 3, 3)))

        # Create a new array to store the processed prediction
        processed_pred = np.zeros_like(pred, dtype=np.float32)

        # Iterate over each connected component (lesion)
        for i in range(1, num_components + 1):
            lesion_mask = (labeled_pred == i) 
            
            # Extract the prediction values of the lesion region
            lesion_values = pred[lesion_mask]
            
            # Calculate the median value for the lesion region
            lesion_median = np.median(lesion_values)
            
            # Assign the median value to all voxels in the lesion region
            processed_pred[lesion_mask] = lesion_median
       
        processed_pred = np.expand_dims(processed_pred, axis=0)  # Add an extra dimension for consistency (as a 3D volume)
        return processed_pred

if __name__ == "__main__":
    import argparse
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_ckpt", required=True)
    args = parser.parse_args()

    run(args)
