from data_modules.decathlon import DecathlonDataModule
from models.unet import UNetModel
import lightning.pytorch as pl
from monai.networks.nets.unet import UNet
import wandb

from pathlib import Path


def train():
    data_module = DecathlonDataModule(batch_size=4)
    if not Path("data/Task01_BrainTumour").exists():
        data_module.prepare_data()
    data_module.setup()

    unet = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
    )

    model = UNetModel(unet)

    wandb_logger = pl.loggers.WandbLogger(project="unet-brats-test", log_model="all")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=3, dirpath="checkpoints"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss", mode="min", patience=5, min_delta=0.001, verbose=True
    )

    # TODO: add debug option to limit training batches to 1
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    # TODO: add cli arg to run wandb experiment or not
    wandb.login()
    train()
    wandb.finish()
