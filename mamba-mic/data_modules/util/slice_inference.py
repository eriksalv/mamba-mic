import torch
import torch.nn.functional as F
import monai.transforms as T
from einops import rearrange
from monai.config.type_definitions import KeysCollection


class ConcatAdjacentSlicesToChannelsd(T.MapTransform):
    def __init__(self, keys: KeysCollection, n_adjacent_slices=1):
        super().__init__(keys)
        self.n_adjacent_slices = n_adjacent_slices

    def __call__(self, data: dict):
        d = dict(data)
        for key in self.keys:
            if key in data:
                image = d[key]
                C, H, W, D = image.shape
                N = self.n_adjacent_slices
                result = torch.zeros(
                    (C + 2 * N * C, H, W, D), device=image.device, dtype=image.dtype
                )

                # Pad depth (last) dimension with n_adjacent_slices on both sides
                padded = F.pad(image, (N, N))

                for i in range(N, D + N):
                    result[:, :, :, i - N] = rearrange(
                        padded[
                            :,
                            :,
                            :,
                            i - N : i + N + 1,
                        ],
                        "c h w d -> (c d) h w",
                    )

                d[key] = result

        return d
