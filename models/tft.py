"""TFT model building and training."""
import os
import numpy as np
import time
import json

# Suppress Lightning tips about litlogger/litmodels
os.environ.setdefault("LIGHTNING_VERBOSE", "0")
from config import (
    MAX_EPOCHS, BATCH, HIDDEN, HEADS, SEED, PATIENCE,
    MAX_ENCODER, MAX_PRED, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, TARGET_TRANSFORM
)
from utils import enable_determinism, determinism_report
from training.callbacks import make_epoch_reporter
from training.logger import get_logger
from training.trainer import create_trainer

try:
    # Imported at MODULE level on purpose: Lightning checkpoints pickle the loss
    # object, and a class defined inside a factory function is not picklable.
    from pytorch_forecasting import QuantileLoss
except ImportError:
    try:
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError:
        class QuantileLoss:  # stub: keeps `--mode prep` working without pf
            def __init__(self, *args, **kwargs):
                raise ImportError("pytorch-forecasting is required for training")


class WeightedQuantileLoss(QuantileLoss):
    """Quantile loss with per-quantile emphasis.

    We report the MEDIAN as the point forecast, so the median term gets the most
    weight; the 0.02/0.98 tails only shape the prediction interval and are
    down-weighted - letting them dominate the shared representation was pushing
    the model towards extreme months, which inflated the pooled MAE.
    Weights are re-normalised to mean 1 so the loss/gradient scale is unchanged.
    """

    WEIGHTS = [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]

    def loss(self, y_pred, target):
        import torch
        losses = super().loss(y_pred, target)          # (batch, time, quantiles)
        w = self.WEIGHTS[: losses.shape[-1]]
        if len(w) != losses.shape[-1]:
            w = [1.0] * losses.shape[-1]
        w = torch.tensor(w, device=losses.device, dtype=losses.dtype)
        w = w / w.mean()
        return losses * w


def make_loss():
    return WeightedQuantileLoss()


def train_tft(training, validation, out_dir, max_epochs=MAX_EPOCHS,
              batch=BATCH, hidden=HIDDEN, patience=PATIENCE, ref_median=None):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    try:
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError:
        from pytorch_forecasting import TemporalFusionTransformer

    pl.seed_everything(SEED, workers=True)
    enable_determinism()
    print(determinism_report(), flush=True)
    train_dl = training.to_dataloader(train=True, batch_size=batch, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch * 2, num_workers=0)

    loss = make_loss()
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden,
        lstm_layers=2,
        attention_head_size=HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=32,
        output_size=7,                    # 7 quantiles [0.02..0.98], median = index 3
        loss=loss,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        log_interval=20,
        reduce_on_plateau_patience=4,
    )
    print(f"\nTFT params: hidden={hidden} heads={HEADS} dropout={DROPOUT} "
          f"lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY} | "
          f"target={TARGET_TRANSFORM or 'raw'} | "
          f"quantiles={tft.loss.quantiles if hasattr(tft.loss, 'quantiles') else 7}")

    print("\n" + "=" * 70)
    print("LIVE EPOCH REPORT - one line printed after every epoch")
    print("=" * 70)
    print(f"  train windows: {len(training)} | batches/epoch: {int(np.ceil(len(training) / batch))}"
          f" | val windows: {len(validation)} | batch: {batch}")
    print(f"  history file : {os.path.join(out_dir, 'training_history.csv')} (written at end)")
    print("=" * 70, flush=True)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    reporter = make_epoch_reporter(pl, out_dir, max_epochs, ref_median, patience)
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    lr_monitor = LearningRateMonitor()
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", save_top_k=1, mode="min")
    callbacks = [reporter, early_stop, lr_monitor, ckpt_cb]

    logger = get_logger(out_dir)
    trainer = create_trainer(max_epochs, callbacks, logger)
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    history = reporter.save()
    try:
        from reporting.plots import plot_training_curves
        plot_training_curves(history, out_dir)
    except Exception as e:
        print(f"(training curves skipped: {e})")
    summary = reporter.summary()
    if summary:
        print(f"\nTRAINING SUMMARY: {summary}", flush=True)

    best_path = ckpt_cb.best_model_path
    print(f"\nbest checkpoint: {best_path}")
    best = TemporalFusionTransformer.load_from_checkpoint(best_path)
    return best, trainer, history