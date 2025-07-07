import monai.transforms as T
import torch
import torch.nn as nn
from mamba_mic.ocelot_module import OcelotModule
from monai.inferers import SlidingWindowInferer


class CompositeSegmentationModel(nn.Module):
    def __init__(
        self, cs_net, ts_net=None, ts_pretrained_path: str = None, device="cuda", ts_patch_size=896, cs_patch_size=896
    ):
        super(CompositeSegmentationModel, self).__init__()
        assert (ts_net and not ts_pretrained_path) or (not ts_net and ts_pretrained_path), (
            "Provide either a nn.Module for tissue segmentation or path to a pretrained one"
        )
        if ts_net:
            self.tissue_segmentation = ts_net
            self.is_pretrained = False
        else:
            self.tissue_segmentation = OcelotModule.load_from_checkpoint(checkpoint_path=ts_pretrained_path)
            self.is_pretrained = True
        self.cell_segmentation = cs_net
        if device is not None:
            self.device = device  # Store device
            self.to(self.device)
        self.ts_patch_size = ts_patch_size
        self.cs_patch_size = cs_patch_size

        self.post_process_tissue = T.Compose(
            [
                T.Activations(sigmoid=True, softmax=False),
                T.AsDiscrete(threshold=0.5),
            ]
        )
        self.tissue_inferer = SlidingWindowInferer(
            roi_size=(self.ts_patch_size, self.ts_patch_size), overlap=0.5, sw_batch_size=1, mode="gaussian"
        )
        self.cell_inferer = SlidingWindowInferer(
            roi_size=(self.cs_patch_size, self.cs_patch_size), overlap=0.5, sw_batch_size=1, mode="gaussian"
        )

    def forward(
        self,
        batch,
        crop_coords=None,
        tissue_label=None,
        cell_tissue_label=None,
        train_mode=False,
        parallelize_invert=False,
    ):
        large_fov = batch["img_tissue"].to(self.device)
        small_fov = batch["img_cell"].to(self.device)

        batch_size = large_fov.shape[0]

        if train_mode:
            tissue_pred = self.tissue_segmentation(large_fov)
        else:
            tissue_pred = self.tissue_inferer(large_fov, self.tissue_segmentation)

        if train_mode and tissue_label is not None and cell_tissue_label is not None:
            tissue_label = tissue_label.to(self.device)
            tissue_crop = cell_tissue_label
            combined_input = torch.cat([small_fov, tissue_crop], dim=1).to(self.device)
            cell_pred = self.cell_segmentation(combined_input)
            return tissue_pred, cell_pred

        else:
            tissue_crops_resized_list = []
            post_tissue_pred = self.post_process_tissue(tissue_pred)
            for i in range(batch_size):
                if crop_coords is not None:
                    x1, y1, x2, y2 = crop_coords[i]
                    tissue_crop = post_tissue_pred[i : i + 1, :, y1:y2, x1:x2]
                else:
                    tissue_crop = post_tissue_pred[i : i + 1]

                tissue_crop_resized = nn.functional.interpolate(tissue_crop, size=small_fov.shape[2:], mode="bilinear")
                tissue_crops_resized_list.append(tissue_crop_resized.to(self.device))

            resized_crops = torch.cat(tissue_crops_resized_list, dim=0).to(self.device)

            combined_input = torch.cat([small_fov, resized_crops], dim=1).to(self.device)
            cell_pred = self.cell_inferer(combined_input, self.cell_segmentation)

            return tissue_pred, cell_pred
