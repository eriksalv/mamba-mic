import torch
import lightning.pytorch as pl
from monai.metrics import DiceHelper
from monai.losses.dice import DiceCELoss
from monai.inferers import Inferer


class System(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        val_inferer: Inferer,
        lr=0.001,
    ) -> None:
        super().__init__()
        self.net = net
        self.val_inferer = val_inferer
        self.lr = lr

        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True)
        self.dice_metric = DiceHelper(sigmoid=True)

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def infer_batch(self, batch, val=False):
        x, y = batch["image"], batch["label"]
        if val is False:
            y_hat = self(x)
        else:
            y_hat = self.val_inferer(inputs=x, network=self.net)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch, val=True)
        loss = self.criterion(y_hat, y)

        dice_score, _ = self.dice_metric(y_hat, y)
        val_dice = dice_score.mean()

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_dice", val_dice, sync_dist=True)

        return {"val_loss": loss, "val_dice": val_dice}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
