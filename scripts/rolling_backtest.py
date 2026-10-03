"""Nested rolling-origin test of persistence versus residual boosting."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED
from data import find_csv, load_and_report, preprocess
from evaluation.metrics import reg_metrics
from scripts.benchmark_forecasts import (
    CATEGORICAL,
    NUMERIC,
    add_model_features,
    feature_row,
)


def examples_before(df, cutoff):
    rows, residuals = [], []
    for _, group in df.groupby("series_id", sort=False):
        group = group.sort_values("time_idx").reset_index(drop=True)
        for origin_idx in range(12, len(group) - 1):
            origin = group.iloc[origin_idx]
            for horizon in range(1, 4):
                target_idx = origin_idx + horizon
                if target_idx >= len(group):
                    break
                target = group.iloc[target_idx]
                if target["date"] >= cutoff:
                    continue
                rows.append(feature_row(origin, horizon))
                residuals.append(float(target["price"] - origin["price"]))
    return pd.DataFrame(rows), np.asarray(residuals)


def make_windows(df, start, end=None):
    start = pd.Timestamp(start)
    start_idx = int(df.loc[df["date"] >= start, "time_idx"].min())
    end_idx = (
        int(df.loc[df["date"] >= pd.Timestamp(end), "time_idx"].min())
        if end is not None
        else int(df["time_idx"].max()) + 1
    )
    feature_rows, actual_rows, persistence_rows, window_keys = [], [], [], []
    for series_id, group in df.groupby("series_id", sort=False):
        group = group.sort_values("time_idx").reset_index(drop=True)
        by_time = {
            int(row["time_idx"]): row for _, row in group.iterrows()
        }
        first_idx, last_idx = int(group["time_idx"].min()), int(group["time_idx"].max())
        for first_target in range(max(start_idx, first_idx + 12), min(end_idx, last_idx - 1)):
            origin = by_time.get(first_target - 1)
            targets = [by_time.get(first_target + step) for step in range(3)]
            if origin is None or any(target is None for target in targets):
                continue
            if end is not None and any(target.date >= pd.Timestamp(end) for target in targets):
                continue
            feature_rows.extend(feature_row(origin, horizon) for horizon in range(1, 4))
            actual_rows.append([float(target.price) for target in targets])
            persistence_rows.append([float(origin.price)] * 3)
            window_keys.append((str(series_id), first_target))
    if not actual_rows:
        raise ValueError(f"No complete forecast windows for {start} to {end}")
    return (
        pd.DataFrame(feature_rows),
        np.asarray(actual_rows),
        np.asarray(persistence_rows),
        window_keys,
    )


def fit_residual_model(df, cutoff, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    features, residuals = examples_before(df, pd.Timestamp(cutoff))
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    model = make_pipeline(
        encoder,
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=seed,
        ),
    )
    model.fit(features[CATEGORICAL + NUMERIC], residuals)
    return model


def fold_predictions(model, features, actual, persistence):
    residual = model.predict(features[CATEGORICAL + NUMERIC]).reshape(actual.shape)
    return np.clip(persistence + residual, 0, None)


def choose_by_horizon(actual, persistence, residual):
    choices, prediction = {}, np.empty_like(actual)
    for horizon in range(actual.shape[1]):
        persist_mae = reg_metrics(actual[:, horizon], persistence[:, horizon])["MAE"]
        residual_mae = reg_metrics(actual[:, horizon], residual[:, horizon])["MAE"]
        selected = "residual_boosting" if residual_mae < persist_mae else "persistence"
        choices[f"horizon_h{horizon + 1}"] = selected
        prediction[:, horizon] = residual[:, horizon] if selected == "residual_boosting" else persistence[:, horizon]
    return choices, prediction


def apply_choices(choices, persistence, residual):
    prediction = np.empty_like(persistence)
    for horizon in range(persistence.shape[1]):
        selected = choices[f"horizon_h{horizon + 1}"]
        prediction[:, horizon] = (
            residual[:, horizon]
            if selected == "residual_boosting"
            else persistence[:, horizon]
        )
    return prediction


def summarize(actual, persistence, hybrid):
    return {
        "windows": int(actual.shape[0]),
        "persistence": {
            "overall_pooled": reg_metrics(actual, persistence),
            **{
                f"horizon_h{h + 1}": reg_metrics(actual[:, h], persistence[:, h])
                for h in range(actual.shape[1])
            },
        },
        "validation_selected_hybrid": {
            "overall_pooled": reg_metrics(actual, hybrid),
            **{
                f"horizon_h{h + 1}": reg_metrics(actual[:, h], hybrid[:, h])
                for h in range(actual.shape[1])
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--test-years", nargs="+", type=int,
        default=[2022, 2023, 2024, 2025, 2026],
    )
    args = parser.parse_args()

    try:
        from lightning.pytorch import seed_everything
    except ImportError:
        from pytorch_lightning import seed_everything

    seed_everything(SEED, workers=True)
    raw = load_and_report(find_csv())
    df = add_model_features(preprocess(raw))
    run_dir = args.out_dir or Path("tft_results") / (
        "rolling_backtest_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    folds, pooled_actual, pooled_persistence, pooled_hybrid = [], [], [], []
    latest_date = df["date"].max()
    for test_year in args.test_years:
        test_start = pd.Timestamp(year=test_year, month=1, day=15)
        validation_start = pd.Timestamp(year=test_year - 1, month=1, day=15)
        test_end = (
            pd.Timestamp(year=test_year + 1, month=1, day=15)
            if test_year < latest_date.year
            else None
        )
        validation_features, validation_actual, validation_persistence, _ = make_windows(
            df, validation_start, test_start
        )
        validation_model = fit_residual_model(df, validation_start, SEED)
        validation_residual = fold_predictions(
            validation_model,
            validation_features,
            validation_actual,
            validation_persistence,
        )
        choices, _ = choose_by_horizon(
            validation_actual, validation_persistence, validation_residual
        )

        # Refit after model selection, using every observation available by test start.
        test_model = fit_residual_model(df, test_start, SEED)
        test_features, test_actual, test_persistence, window_keys = make_windows(
            df, test_start, test_end
        )
        test_residual = fold_predictions(
            test_model, test_features, test_actual, test_persistence
        )
        test_hybrid = apply_choices(choices, test_persistence, test_residual)
        fold = {
            "test_period": [test_start.date().isoformat(),
                            test_end.date().isoformat() if test_end is not None else latest_date.date().isoformat()],
            "validation_period": [validation_start.date().isoformat(), test_start.date().isoformat()],
            "selected_by_horizon": choices,
            "validation": summarize(validation_actual, validation_persistence,
                                     choose_by_horizon(validation_actual, validation_persistence,
                                                       validation_residual)[1]),
            "test": summarize(test_actual, test_persistence, test_hybrid),
            "window_keys": len(window_keys),
        }
        folds.append(fold)
        pooled_actual.append(test_actual)
        pooled_persistence.append(test_persistence)
        pooled_hybrid.append(test_hybrid)
        print(f"{test_year}: persistence MAE="
              f"{fold['test']['persistence']['overall_pooled']['MAE']:.3f}; "
              f"hybrid MAE={fold['test']['validation_selected_hybrid']['overall_pooled']['MAE']:.3f}; "
              f"choices={choices}", flush=True)

    actual = np.concatenate(pooled_actual, axis=0)
    persistence = np.concatenate(pooled_persistence, axis=0)
    hybrid = np.concatenate(pooled_hybrid, axis=0)
    summary = {
        "method": "nested expanding-window residual-vs-persistence selection by horizon",
        "selection_uses_each_test_period": False,
        "folds": folds,
        "pooled_non_overlapping_test_periods": summarize(actual, persistence, hybrid),
        "caveat": (
            "The 2025 and 2026 periods overlapped earlier aggregate holdout scores, and the strategy "
            "was designed after those scores were inspected. Treat all folds as exploratory, not as "
            "an independent final confirmation; new future data is needed for that."
        ),
    }
    path = run_dir / "rolling_backtest_metrics.json"
    path.write_text(json.dumps(summary, indent=2))
    print("\nPooled non-overlapping test periods:")
    for name in ("persistence", "validation_selected_hybrid"):
        values = summary["pooled_non_overlapping_test_periods"][name]
        print(f"  {name}: MAE={values['overall_pooled']['MAE']:.3f} "
              f"RMSE={values['overall_pooled']['RMSE']:.3f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()