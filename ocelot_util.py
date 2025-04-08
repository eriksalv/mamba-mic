import torch
from skimage.feature import peak_local_max
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import cv2
def normalize_crop_coords_batch(meta_batch):
    """
    Convert whole-slide image coordinates to downsampled tissue patch coordinates for a batch.

    Args:
        meta_batch (list[dict] or dict): List of metadata dictionaries or a single metadata dictionary.

    Returns:
        list[tuple]: List of (x1, y1, x2, y2) coordinates for each sample in the batch.
    """
    # Batch size
    batch_size = len(meta_batch["slide_name"])

    # If meta_batch is a single metadata dictionary, handle it directly
    if batch_size == 1:
        return [normalize_crop_coords(meta_batch)]  # Return as a list for compatibility

    

    # Extracting tensors from the meta_batch for efficient batch processing
    x_start = torch.stack([v for v in meta_batch["cell_patch"]["x_start"]], dim=0)
    y_start = torch.stack([v for v in meta_batch["cell_patch"]["y_start"]], dim=0)
    x_end = torch.stack([v for v in meta_batch["cell_patch"]["x_end"]], dim=0)
    y_end = torch.stack([v for v in meta_batch["cell_patch"]["y_end"]], dim=0)
    
    tissue_x_start = torch.stack([v for v in meta_batch["tissue_patch"]["x_start"]], dim=0)
    tissue_y_start = torch.stack([v for v in meta_batch["tissue_patch"]["y_start"]], dim=0)
    tissue_x_end = torch.stack([v for v in meta_batch["tissue_patch"]["x_end"]], dim=0)
    tissue_y_end = torch.stack([v for v in meta_batch["tissue_patch"]["y_end"]], dim=0)

    # Compute scaling factors (scale_x, scale_y) for each sample in the batch
    scale_x = 1024 / (tissue_x_end - tissue_x_start)
    scale_y = 1024 / (tissue_y_end - tissue_y_start)

    # Normalize coordinates for the entire batch using vectorized operations
    x1 = ((x_start - tissue_x_start) * scale_x).int()
    y1 = ((y_start - tissue_y_start) * scale_y).int()
    x2 = ((x_end - tissue_x_start) * scale_x).int()
    y2 = ((y_end - tissue_y_start) * scale_y).int()

    # Collect the results as tuples for each sample in the batch
    batch_crop_coords = list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))
    
    return batch_crop_coords

def normalize_crop_coords(meta):
    """
    Convert whole-slide image coordinates to downsampled tissue patch coordinates.
    """
    cell_patch = meta["cell_patch"]
    tissue_patch = meta["tissue_patch"]
   
    # Helper function to extract tensor values
    def extract_value(v):
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else [i.item() for i in v]  # Handle single and multiple elements
        return v
    
    # Compute scaling factor from whole-slide coordinates to patch-level coordinates
    scale_x = 1024 / (extract_value(tissue_patch["x_end"]) - extract_value(tissue_patch["x_start"]))
    scale_y = 1024 / (extract_value(tissue_patch["y_end"]) - extract_value(tissue_patch["y_start"]))

    # Normalize `cell_patch` coordinates relative to the tissue patch
    x1 = int((extract_value(cell_patch["x_start"]) - extract_value(tissue_patch["x_start"])) * scale_x)
    y1 = int((extract_value(cell_patch["y_start"]) - extract_value(tissue_patch["y_start"])) * scale_y)
    x2 = int((extract_value(cell_patch["x_end"]) - extract_value(tissue_patch["x_start"])) * scale_x)
    y2 = int((extract_value(cell_patch["y_end"]) - extract_value(tissue_patch["y_start"])) * scale_y)

    return (x1, y1, x2, y2)

def cell_detection_postprocessing(y_tc, y_bc, y_bg, min_distance=3):
    if isinstance(y_tc, torch.Tensor):
        y_tc = y_tc.cpu().numpy()
    if isinstance(y_bc, torch.Tensor):
        y_bc = y_bc.cpu().numpy()
    if isinstance(y_bg, torch.Tensor):
        y_bg = y_bg.cpu().numpy()
    # Compute foreground probability
    foreground = 1 - y_bg
    foreground = cv2.GaussianBlur(foreground, (0, 0), sigmaX=3)
    # Detect peaks (local maxima)
    cell_candidates = peak_local_max(foreground, min_distance=min_distance, exclude_border=0, threshold_abs=0.0)

    # Store valid cells, classes, and confidence scores
    valid_cells = []
    valid_classes = []
    confidence_scores = []
    
    for x, y in cell_candidates:
        if y_tc[x, y] > y_bg[x, y] or y_bc[x, y] > y_bg[x, y]:  
            valid_cells.append((x, y))
            
            # Determine the predicted class (1 = background cell, 2 = tumor cell)
            if y_bc[x, y] > y_tc[x, y]:
                cell_class = 1  # Background cell
            else:
                cell_class = 2  # Tumor cell
            
            valid_classes.append(cell_class)

            # Confidence score is the max probability of either class
            confidence_score = max(y_tc[x, y], y_bc[x, y])
            confidence_scores.append(confidence_score)

    # Convert to numpy arrays
    valid_cells = np.array(valid_cells)
    valid_classes = np.array(valid_classes)
    confidence_scores = np.array(confidence_scores)

    # Sort by confidence score (descending order)
    sorted_indices = np.argsort(-confidence_scores)
    return valid_cells[sorted_indices], valid_classes[sorted_indices], confidence_scores[sorted_indices]

def cell_detection_postprocessing_batch(y_c_batch, min_distance=3):

    cells_list = []
    classes_list = []
    confidences_list = []

    for y_c in y_c_batch:
        # Extract individual channels
        y_bg, y_bc, y_tc = y_c[0], y_c[1], y_c[2]

        # Process each batch item individually
        cells, classes, confidences = cell_detection_postprocessing(y_tc, y_bc, y_bg, min_distance=min_distance)
        cells_list.append(cells)
        classes_list.append(classes)
        confidences_list.append(confidences)

    return cells_list, classes_list, confidences_list

def evaluate_cell_detection(pred_cells, pred_classes, gt_cells, gt_classes, max_distance=15):
    tp_bc_count, fp_bc_count, fn_bc_count = 0, 0, 0
    tp_tc_count, fp_tc_count, fn_tc_count = 0, 0, 0
  
    for class_val in [1, 2]:
        class_gt_cells = gt_cells[gt_classes == class_val]
        class_pred_cells = pred_cells[pred_classes == class_val]

        if len(class_gt_cells) == 0:
            if class_val == 1:
                fp_bc_count += len(class_pred_cells)
            else:
                fp_tc_count += len(class_pred_cells)
            continue
        
        gt_tree = cKDTree(class_gt_cells)
        matched_gt_indices = set()  
        matched_pred_indices = set() 
        for i, pred in enumerate(class_pred_cells):
            if i in matched_pred_indices:  
                continue  
           
            neighbors = gt_tree.query_ball_point(pred, max_distance)
            # Find unmatched GT cells
            available_neighbors = [idx for idx in neighbors if idx not in matched_gt_indices]

            if not available_neighbors:
                # No matching ground-truth cell found, so it's a false positive
                if class_val == 1:
                    fp_bc_count += 1
                else:
                    fp_tc_count += 1
                continue

            # Find the closest unmatched GT cell
            closest_idx = min(available_neighbors, key=lambda idx: np.linalg.norm(pred - class_gt_cells[idx]))

           
            if class_val == 1:
                tp_bc_count += 1
            else:
                tp_tc_count += 1
            matched_gt_indices.add(closest_idx)  # Mark this ground-truth cell as matched
            # Mark all predictions within this GT's radius as matched (avoid duplicates)
            #for j, other_pred in enumerate(class_pred_cells):
            #    if np.linalg.norm(other_pred - class_gt_cells[closest_idx]) <= max_distance:
            #        #print(f"Matching {other_pred} with {class_gt_cells[closest_idx]}")
            #        matched_pred_indices.add(j)
          

        # Any remaining unmatched ground-truth cells are false negatives
        if class_val == 1:
            fn_bc_count = len(class_gt_cells) - len(matched_gt_indices)
        else:
            fn_tc_count = len(class_gt_cells) - len(matched_gt_indices)

    return tp_bc_count, fp_bc_count, fn_bc_count, tp_tc_count, fp_tc_count, fn_tc_count
    
def evaluate_cell_detection_batch(pred_cells_list, pred_classes_list, gt_cells_list, gt_classes_list, max_distance=15):
    total_tp_bc, total_fp_bc, total_fn_bc = 0, 0, 0
    total_tp_tc, total_fp_tc, total_fn_tc = 0, 0, 0

  

    for pred_cells, pred_classes, gt_cells, gt_classes in zip(pred_cells_list, pred_classes_list, gt_cells_list, gt_classes_list):
        tp_bc, fp_bc, fn_bc, tp_tc, fp_tc, fn_tc = evaluate_cell_detection(pred_cells, pred_classes, gt_cells, gt_classes, max_distance = max_distance)
        
        total_tp_bc += tp_bc
        total_fp_bc += fp_bc
        total_fn_bc += fn_bc

        total_tp_tc += tp_tc
        total_fp_tc += fp_tc
        total_fn_tc += fn_tc


    return total_tp_bc, total_fp_bc, total_fn_bc, total_tp_tc, total_fp_tc, total_fn_tc

def load_ground_truth(batch):
    gt_cells_list = []
    gt_classes_list = []
 
    for csv_path in batch['label_cell']:  # Iterate over CSV file paths
        df = pd.read_csv(csv_path, header=None, names=["x", "y", "class"])
        gt_cells_list.append(df[['y', 'x']].values)  # Store (x, y) coords
        gt_classes_list.append(df['class'].values)   # Store class labels

    return gt_cells_list, gt_classes_list

def calculate_metrics(tp, fp, fn):

    # Precision, Recall, and F1 score calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_class_weights(target):
    """
    target: (batch, 3, H, W) - Ground truth (soft values per class)
    Returns: (3,) - Normalized weights for each class
    """
    total_per_class = target.sum(dim=(0, 2, 3))  # Sum over batch, height, width
    total_pixels = total_per_class.sum()  # Total sum across all classes

    weights = total_per_class / total_pixels  # Normalize
    class_weights = 1 / (weights + 1e-6)
    class_weights /= class_weights.sum() 
    return class_weights

def weighted_mse_loss(pred, target, class_weights):
    mse_per_channel = (pred - target) ** 2  
    # Apply class weights: (batch, 3, H, W) * (3, 1, 1)
    weighted_mse = mse_per_channel * class_weights.view(3, 1, 1)

    return weighted_mse.mean()
