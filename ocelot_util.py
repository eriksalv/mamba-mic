import torch

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