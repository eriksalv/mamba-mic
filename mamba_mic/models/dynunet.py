import torch
import lightning.pytorch as pl
from monai.networks.nets import DynUNet
from monai.metrics import DiceHelper
from monai.losses.dice import DiceCELoss

class DynUNetModel(pl.LightningModule):
    def __init__(self, lr: float, spatial_dims: int, in_channels: int, out_channels: int,
                 kernel_size, strides, upsample_kernel_size, filters=None, dropout=None,
                 norm_name=('INSTANCE', {'affine': True}), act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
                 deep_supervision=False, deep_supr_num=1, res_block=False, trans_bias=False) -> None:
        super().__init__()
        
        self.lr = lr
        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True)
        self.dice_metric = DiceHelper(sigmoid=True)

        # Initialize DynUNet with the given parameters
        self.dynunet = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            trans_bias=trans_bias
        )

        self.save_hyperparameters()

    def forward(self, x):
        # Forward pass
        return self.dynunet(x)

    def infer_batch(self, batch):
        # Extract images and labels
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        dice_score, _ = self.dice_metric(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/dice', dice_score.mean())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
