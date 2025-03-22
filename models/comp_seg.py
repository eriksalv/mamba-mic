import torch
import torch.nn as nn

class CompositeSegmentationModel(nn.Module):
    def __init__(self, ts_net, cs_net, device='cuda'):  
        super(CompositeSegmentationModel, self).__init__()
        self.tissue_segmentation = ts_net
        self.cell_segmentation = cs_net
        self.device = device  # Store device
        self.to(self.device)


    def forward(self, batch, data_module, crop_coords, tissue_label=None, train_mode=False, parallelize_invert = False, only_tissue_to_crop = False):
        large_fov = batch["img_tissue"].to(self.device)
        small_fov = batch["img_cell"].to(self.device)  

        batch_size = large_fov.shape[0]
        tissue_pred = self.tissue_segmentation(large_fov).to(self.device)  

        if train_mode and tissue_label is not None:
            tissue_pred = tissue_label.to(self.device) 

        batch['pred_tissue'] = tissue_pred
        inverted = data_module.invert_tissue_pred(batch, parallelized=parallelize_invert)
        tissue_pred_to_crop_list = inverted['pred_tissue']

        tissue_pred_to_crop = torch.stack([t.to(self.device) for t in tissue_pred_to_crop_list], dim=0)

        if only_tissue_to_crop:
            return tissue_pred_to_crop

        tissue_crops_resized_list = []
        for i in range(batch_size):
            if crop_coords is not None:
                x1, y1, x2, y2 = crop_coords[i]
                tissue_crop = tissue_pred_to_crop[i:i + 1, :, y1:y2, x1:x2]
            else:
                tissue_crop = tissue_pred_to_crop[i:i + 1]

            tissue_crop_resized = nn.functional.interpolate(tissue_crop, size=small_fov.shape[2:], mode="bilinear")
            tissue_crops_resized_list.append(tissue_crop_resized.to(self.device))  

        batch['tissue_crop_resized'] = torch.cat(tissue_crops_resized_list, dim=0).to(self.device)
        resized_crops = data_module.reapply_augmentations(batch)['tissue_crop_resized'].to(self.device)  

        combined_input = torch.cat([small_fov, tissue_pred_to_crop, resized_crops], dim=1).to(self.device)  
        cell_pred = self.cell_segmentation(combined_input)

        return tissue_pred, cell_pred #, resized_crops, tissue_pred_to_crop

    
