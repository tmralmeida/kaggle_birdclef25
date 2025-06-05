import os
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch.optim as optim

from birdclef.data_modeling.model_handlers import SpecAugment, FocalLoss
from birdclef.io import dump_json_file


class LightModel(pl.LightningModule):
    def __init__(self, model, n_classes: int, learning_rate=1e-3):
        super(LightModel, self).__init__()
        self.spec_augment = SpecAugment()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = FocalLoss()
        self.n_classes = n_classes

    def forward(self, x):
        x = self.spec_augment(x)
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, min_lr=1e-6)
        return [opt], [dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = val_batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y.long())
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        final_preds = torch.nn.functional.softmax(y_hat, dim=-1)
        self.update_metrics(final_preds, y.long())

        return val_loss

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            accuracy=Accuracy(
                task="multiclass",
                num_classes=self.n_classes,
                average="weighted",
            ).to(self.device),
            f1_score=F1Score(
                task="multiclass",
                num_classes=self.n_classes,
                average="weighted",
            ).to(self.device),
            precision=Precision(
                task="multiclass",
                num_classes=self.n_classes,
                average="weighted",
            ).to(self.device),
            recall=Recall(
                task="multiclass",
                num_classes=self.n_classes,
                average="weighted",
            ).to(self.device),
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def update_metrics(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ):
        self.eval_metrics["accuracy"].update(preds=pred, target=gt)
        self.eval_metrics["f1_score"].update(preds=pred, target=gt)
        self.eval_metrics["precision"].update(preds=pred, target=gt)
        self.eval_metrics["recall"].update(preds=pred, target=gt)

    def compute_metrics(self) -> dict:
        final_metrics = {
            met_name: met.compute().item() for met_name, met in self.eval_metrics.items()
        }
        return final_metrics

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
