"""Train mode - FULL run: train TFT + evaluate + forecast."""
import os
from data import find_csv, load_and_report, preprocess, chronological_split, build_datasets
from models.tft import train_tft
from evaluation import evaluate, future_forecast, per_commodity_breakdown
from reporting import write_final_report, print_final_summary
from config import (
    OUT_DIR, VAL_START, TEST_START, MAX_ENCODER, MAX_PRED,
    BATCH, HIDDEN, PATIENCE, SEED
)


def run_train():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_raw = load_and_report(find_csv())
    df = preprocess(df_raw)
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(df, VAL_START, TEST_START)

    # FULL TRAIN
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx)
    ref_med = float(df["price"].median())
    best, _, history = train_tft(training, validation, OUT_DIR, ref_median=ref_med)
    metrics = evaluate(best, test, df, OUT_DIR)
    commodity_rows = per_commodity_breakdown(best, training, test_hist, test_idx, OUT_DIR)
    future_forecast(best, training, df, OUT_DIR)
    path = write_final_report(OUT_DIR, "train", history, metrics, commodity_rows)
    print_final_summary(metrics)
    print(f"\nDONE. All outputs in ./{OUT_DIR}/")
    print(f"Final report : {path}")
    print("Best-model comparison for your report: TFT overall_pooled vs naive references in metrics.json")


if __name__ == "__main__":
    run_train()