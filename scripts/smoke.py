"""Smoke mode - tiny 2-epoch test that TFT runs end-to-end."""
import os
from data import find_csv, load_and_report, preprocess, chronological_split, build_datasets
from models.tft import train_tft
from evaluation import evaluate
from reporting import write_final_report, print_final_summary
from config import (
    OUT_DIR, VAL_START, TEST_START, SMOKE_SERIES, SMOKE_EPOCHS,
    MAX_ENCODER, MAX_PRED, BATCH, HIDDEN, PATIENCE, SEED
)


def run_smoke():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_raw = load_and_report(find_csv())
    df = preprocess(df_raw)

    print("\n*** SMOKE MODE: tiny subset, 2 epochs, just to verify TFT runs ***")
    top_series = df.groupby("series_id").size().sort_values(ascending=False).head(SMOKE_SERIES).index
    df = df[df["series_id"].isin(top_series)].copy()
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(df, VAL_START, TEST_START)
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx, max_encoder=12, max_pred=2)
    ref_med = float(df["price"].median())
    best, _, history = train_tft(training, validation, OUT_DIR, max_epochs=SMOKE_EPOCHS,
                                 batch=256, hidden=16, patience=2, ref_median=ref_med)
    metrics = evaluate(best, test, df, OUT_DIR)
    path = write_final_report(
        OUT_DIR, "smoke", history, metrics,
        config=dict(lookback=12, horizon=2, batch=256, hidden=16,
                    max_epochs=SMOKE_EPOCHS, patience=2))
    print_final_summary(metrics)
    print(f"\nsaved {path}")
    print("\nSMOKE OK. Next: python -m scripts.train")


if __name__ == "__main__":
    run_smoke()