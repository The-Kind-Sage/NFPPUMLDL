"""Run short, validation-selected fine-tuning experiments from the latest TFT."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from config import BATCH, SEED, TEST_START, VAL_START
from data import build_datasets, chronological_split, find_csv, load_and_report, preprocess
from evaluation import evaluate
from models.tft import WeightedQuantileLoss
from training.callbacks import make_epoch_reporter
from training.trainer import create_trainer
from utils import accelerator_name, denorm, enable_determinism, unpack_predict_output


EPOCHS = 2
TRIALS = {
    "low_lr": {"learning_rate": 0.00005, "weight_decay": 0.0001},
    "regularized": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0005,
        "dropout": 0.4,
    },
    "median_focused": {
        "learning_rate": 0.00005,
        "weight_decay": 0.0001,
        "quantile_weights": [0.25, 0.5, 1.0, 4.0, 1.0, 0.5, 0.25],
    },
}


def validation_mae(model, validation, ref_median):
    dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH * 2, num_workers=0
    )
    result = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        return_y=True,
        trainer_kwargs={"accelerator": accelerator_name(), "devices": 1, "logger": False},
    )
    output = unpack_predict_output(result)
    predictions = denorm(output["raw"], output["x"]["target_scale"], ref_median)
    actual = denorm(output["y"], output["x"]["target_scale"], ref_median)
    median = np.clip(predictions[:, :, 3], 0, None)
    return float(np.mean(np.abs(actual - median)))


def tune_trial(name, settings, checkpoint, training, validation, ref_median, run_dir):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer

    pl.seed_everything(SEED, workers=True)
    enable_determinism()
    trial_dir = run_dir / name
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.hparams.learning_rate = settings["learning_rate"]
    model.hparams.weight_decay = settings["weight_decay"]

    if "dropout" in settings:
        import torch

        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = settings["dropout"]
    if "quantile_weights" in settings:
        if not isinstance(model.loss, WeightedQuantileLoss):
            raise TypeError("Checkpoint loss is not WeightedQuantileLoss")
        model.loss.WEIGHTS = settings["quantile_weights"]

    train_dl = training.to_dataloader(train=True, batch_size=BATCH, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=0)
    reporter = make_epoch_reporter(pl, str(trial_dir), EPOCHS, ref_median=ref_median)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="epoch{epoch:02d}",
        save_top_k=-1, every_n_epochs=1,
    )
    trainer = create_trainer(EPOCHS, [reporter, checkpoint_callback], logger=False)
    print(f"\n=== Fine-tuning {name}: {settings} ===", flush=True)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    scored = []
    for candidate_path in sorted(checkpoint_dir.glob("*.ckpt")):
        candidate = TemporalFusionTransformer.load_from_checkpoint(
            str(candidate_path), map_location="cpu"
        )
        score = validation_mae(candidate, validation, ref_median)
        scored.append({"checkpoint": str(candidate_path), "validation_MAE": score})
        print(f"{candidate_path.name}: validation MAE={score:.4f}", flush=True)
    if not scored:
        raise RuntimeError(f"No checkpoints saved for {name}")
    best = min(scored, key=lambda row: row["validation_MAE"])
    return {"settings": settings, "epochs": EPOCHS, "checkpoints": scored, "best": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    try:
        from lightning.pytorch import seed_everything
    except ImportError:
        from pytorch_lightning import seed_everything
    from pytorch_forecasting import TemporalFusionTransformer

    seed_everything(SEED, workers=True)
    enable_determinism()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = max(Path("tft_results/checkpoints").glob("*.ckpt"),
                         key=lambda path: path.stat().st_mtime)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = args.out_dir or Path("tft_results") / (
        "finetune_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting checkpoint: {checkpoint}")
    raw = load_and_report(find_csv())
    df = preprocess(raw)
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(
        df, VAL_START, TEST_START
    )
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx
    )
    ref_median = float(df["price"].median())

    results = {}
    for name, settings in TRIALS.items():
        results[name] = tune_trial(
            name, settings, checkpoint, training, validation, ref_median, run_dir
        )

    winner_name, winner = min(
        results.items(), key=lambda item: item[1]["best"]["validation_MAE"]
    )
    summary = {
        "starting_checkpoint": str(checkpoint),
        "selection_metric": "2024 validation median-forecast MAE (NPR)",
        "trials": results,
        "winner": winner_name,
    }
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    best_model = TemporalFusionTransformer.load_from_checkpoint(
        winner["best"]["checkpoint"], map_location="cpu"
    )
    print(f"\nSelected on validation: {winner_name} "
          f"(MAE={winner['best']['validation_MAE']:.4f} NPR)")
    test_metrics = evaluate(best_model, test, df, str(run_dir))
    summary["winner_test_metrics"] = test_metrics
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nExperiment outputs: {run_dir}")


if __name__ == "__main__":
    main()