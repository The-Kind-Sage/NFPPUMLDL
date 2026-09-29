"""Trainer factory."""
import os

os.environ.setdefault("LIGHTNING_VERBOSE", "0")

def create_trainer(max_epochs, callbacks, logger):
    """Create Lightning Trainer with proper accelerator."""
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="gpu" if has_gpu else "cpu", devices=1,
        gradient_clip_val=0.1, callbacks=callbacks, enable_model_summary=True,
        enable_progress_bar=True, logger=logger,
    )
    return trainer