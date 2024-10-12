import torch
import lightning.pytorch as pl
from monai.networks.nets.unet import UNet
from monai.metrics import DiceHelper
from monai.losses.dice import DiceCELoss


class UNetModel(pl.LightningModule):
    def __init__(
        self,
        # spatial_dims: int,
        # in_channels: int,
        # out_channels: int,
        # channels=(8, 16, 32, 64),
        # strides=(2, 2, 2),
        unet: UNet,
        lr=0.001,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True)

        self.dice_metric = DiceHelper(sigmoid=True)

        # self.unet = UNet(
        #     spatial_dims=spatial_dims,
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     channels=channels,
        #     strides=strides,
        # )
        self.unet = unet

        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)

    def infer_batch(self, batch):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        dice_score, _ = self.dice_metric(y_hat, y)

        self.log("val/loss", loss)
        self.log("val/dice", dice_score.mean())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
