"""Eval mode - regenerate all evaluation outputs from the newest checkpoint (NO training)."""
import json
import os
from pathlib import Path

from config import OUT_DIR, VAL_START, TEST_START
from data import (
    find_csv, load_and_report, preprocess, chronological_split, build_datasets
)
from evaluation import evaluate, per_commodity_breakdown
from reporting import print_final_summary, write_final_report
from reporting.plots import plot_training_curves


def run_eval():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_dir = Path(OUT_DIR) / "checkpoints"
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {ckpt_dir}")
    # newest file in the checkpoint dir = best checkpoint of the latest training run
    best_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"EVAL MODE (no training) - checkpoint: {best_path}")

    df_raw = load_and_report(find_csv())
    df = preprocess(df_raw)
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(
        df, VAL_START, TEST_START)
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx)

    try:
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError:
        from pytorch_forecasting import TemporalFusionTransformer
    best = TemporalFusionTransformer.load_from_checkpoint(
        str(best_path), map_location="cpu")

    metrics = evaluate(best, test, df, OUT_DIR)
    commodity_rows = per_commodity_breakdown(
        best, training, test_hist, test_idx, OUT_DIR)

    history = []
    hist_csv = os.path.join(OUT_DIR, "training_history.csv")
    if os.path.isfile(hist_csv):
        import pandas as pd
        history = json.loads(pd.read_csv(hist_csv).to_json(orient="records"))
        try:
            plot_training_curves(history, OUT_DIR)
        except Exception as e:
            print(f"(training curves skipped: {e})")

    path = write_final_report(OUT_DIR, "eval", history, metrics, commodity_rows)
    print_final_summary(metrics)
    print(f"\nEVAL DONE (no training performed). All outputs in ./{OUT_DIR}/")
    print(f"Final report : {path}")


if __name__ == "__main__":
    run_eval()
