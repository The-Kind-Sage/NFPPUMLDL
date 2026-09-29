"""Prep mode - EDA + preprocessing check (no torch needed)."""
import os
import json
from data import find_csv, load_and_report, preprocess, chronological_split
from evaluation.metrics import naive_baselines
from config import OUT_DIR, VAL_START, TEST_START


def run_prep():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_raw = load_and_report(find_csv())
    df = preprocess(df_raw)
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(df, VAL_START, TEST_START)

    df.to_csv(os.path.join(OUT_DIR, "processed_monthly_panel.csv"), index=False)
    print(f"\nsaved {OUT_DIR}/processed_monthly_panel.csv")
    print("Naive one-step references on test rows:")
    print(json.dumps(naive_baselines(df, TEST_START), indent=2))
    print("\nPREP OK. Next: python -m scripts.smoke")


if __name__ == "__main__":
    run_prep()