import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)


from birdclef.data_modeling.loader import BirdclefDataset
from birdclef.data_modeling.model_handlers import BirdCLEFCNN
from birdclef.data_modeling.lightning import LightModel


KAGGLE_PATH = "/Users/taa/Documents/kaggle/kaggle_birdclef25"
LOGS_PATH = "/Users/taa/Documents/kaggle/kaggle_birdclef25/logs"
SPECTOGRAM_STYLE = "mel"  # "mel" or "stft"


INPUT_DATA_MODEL = {"stft": (1025, 313), "mel": (128, 313)}


DATA_PATH = os.path.join(KAGGLE_PATH, "data", "processed", SPECTOGRAM_STYLE)


def train():
    exisiting_files = os.listdir(DATA_PATH)
    print("Number of FILES/FOLDERS", len(exisiting_files))

    train_ds = BirdclefDataset(
        data_path=os.path.join(DATA_PATH, "train"),
        set_type="train",
    )
    val_ds = BirdclefDataset(
        data_path=os.path.join(DATA_PATH, "val"),
        set_type="val",
        label2id=train_ds.label2id,
        id2label=train_ds.id2label,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=14,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=14,
        persistent_workers=True,
    )

    logger = TensorBoardLogger(
        f"{KAGGLE_PATH}/lightning_logs/", name="birdclef_specaugment", default_hp_metric=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.2f}"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=5,
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    model = BirdCLEFCNN(num_classes=len(train_ds.label2id))
    light_model = LightModel(model=model, n_classes=len(train_ds.id2label), learning_rate=1e-3)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        check_val_every_n_epoch=2,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(light_model, train_dl, val_dl)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    train()
