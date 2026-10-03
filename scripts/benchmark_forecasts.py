"""Compare five three-month forecasting strategies without replacing the TFT."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED, TEST_START, VAL_START
from data import build_datasets, chronological_split, find_csv, load_and_report, preprocess
from evaluation.metrics import aligned_baselines, reg_metrics
from models.tft import WeightedQuantileLoss
from utils import accelerator_name, denorm, enable_determinism, unpack_predict_output

CATEGORICAL = ["market", "commodity", "category", "unit"]
NUMERIC = [
    "horizon", "price", "lag_1", "lag_3", "lag_12", "roll_mean_3",
    "roll_mean_12", "roll_std_12", "momentum_1", "momentum_3",
    "momentum_12", "slope_6", "month_sin", "month_cos", "year_norm",
]


def add_model_features(df):
    df = df.sort_values(["series_id", "time_idx"]).copy()
    grouped = df.groupby("series_id")["price"]
    df["lag_5"] = grouped.shift(5)
    df["momentum_1"] = df["price"] - df["lag_1"]
    df["momentum_3"] = df["price"] - df["lag_3"]
    df["momentum_12"] = df["price"] - df["lag_12"]
    df["slope_6"] = (df["price"] - df["lag_5"]) / 5.0
    return df


def feature_row(origin, horizon):
    forecast_date = origin["date"] + pd.DateOffset(months=horizon)
    month_angle = 2 * np.pi * forecast_date.month / 12.0
    return {
        "market": origin["market"],
        "commodity": origin["commodity"],
        "category": origin["category"],
        "unit": origin["unit"],
        "horizon": horizon,
        "price": origin["price"],
        "lag_1": origin["lag_1"],
        "lag_3": origin["lag_3"],
        "lag_12": origin["lag_12"],
        "roll_mean_3": origin["roll_mean_3"],
        "roll_mean_12": origin["roll_mean_12"],
        "roll_std_12": origin["roll_std_12"],
        "momentum_1": origin["momentum_1"],
        "momentum_3": origin["momentum_3"],
        "momentum_12": origin["momentum_12"],
        "slope_6": origin["slope_6"],
        "month_sin": np.sin(month_angle),
        "month_cos": np.cos(month_angle),
        "year_norm": (forecast_date.year - 2001) / 25.0,
    }


def make_training_examples(df):
    rows, targets = [], []
    for _, group in df.groupby("series_id", sort=False):
        group = group.sort_values("time_idx").reset_index(drop=True)
        for origin_idx in range(12, len(group) - 1):
            origin = group.iloc[origin_idx]
            for horizon in range(1, 4):
                target_idx = origin_idx + horizon
                if target_idx >= len(group):
                    break
                target = group.iloc[target_idx]
                if target["date"] >= pd.Timestamp(VAL_START):
                    continue
                rows.append(feature_row(origin, horizon))
                targets.append(float(target["price"] - origin["price"]))
    return pd.DataFrame(rows), np.asarray(targets)


def make_window_features(df, index):
    lookup = df.set_index(["series_id", "time_idx"], drop=False)
    rows = []
    for sid, first_time in zip(index["series_id"].astype(str), index["time_idx"]):
        origin = lookup.loc[(sid, int(first_time) - 1)]
        rows.extend(feature_row(origin, horizon) for horizon in range(1, 4))
    return pd.DataFrame(rows)


def predict_tft(model, dataset):
    dataloader = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
    result = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        return_y=True,
        return_index=True,
        trainer_kwargs={"accelerator": accelerator_name(), "devices": 1, "logger": False},
    )
    output = unpack_predict_output(result)
    median = np.clip(
        denorm(output["raw"], output["x"]["target_scale"], float(model._benchmark_median))[:, :, 3],
        0,
        None,
    )
    actual = denorm(output["y"], output["x"]["target_scale"], float(model._benchmark_median))
    return actual, median, output["index"]


def metric_block(actual, predicted):
    result = {"overall_pooled": reg_metrics(actual, predicted)}
    for horizon in range(actual.shape[1]):
        result[f"horizon_h{horizon + 1}"] = reg_metrics(actual[:, horizon], predicted[:, horizon])
    return result


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
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    seed_everything(SEED, workers=True)
    enable_determinism()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = max(Path("tft_results/checkpoints").glob("*.ckpt"),
                         key=lambda path: path.stat().st_mtime)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = args.out_dir or Path("tft_results") / (
        "benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = load_and_report(find_csv())
    df = add_model_features(preprocess(raw))
    train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(
        df, VAL_START, TEST_START
    )
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx
    )
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), map_location="cpu")
    if not isinstance(model.loss, WeightedQuantileLoss):
        raise TypeError("Checkpoint loss is not WeightedQuantileLoss")
    model._benchmark_median = float(df["price"].median())

    val_actual, val_tft, val_index = predict_tft(model, validation)
    test_actual, test_tft, test_index = predict_tft(model, test)
    val_baselines = aligned_baselines(df, val_index, val_tft.shape[1])
    test_baselines = aligned_baselines(df, test_index, test_tft.shape[1])
    val_persistence = val_baselines["persistence"]
    test_persistence = test_baselines["persistence"]
    val_seasonal = val_baselines["seasonal_naive"]
    test_seasonal = test_baselines["seasonal_naive"]
    val_features = make_window_features(df, val_index)[CATEGORICAL + NUMERIC]
    test_features = make_window_features(df, test_index)[CATEGORICAL + NUMERIC]
    val_slope = val_features["slope_6"].to_numpy().reshape(val_actual.shape)

    blend_weights = np.linspace(0.0, 1.0, 101)
    blend_weight = min(
        blend_weights,
        key=lambda weight: reg_metrics(
            val_actual, weight * val_tft + (1 - weight) * val_persistence
        )["MAE"],
    )
    trend_alpha = min(
        np.linspace(0.0, 1.0, 101),
        key=lambda alpha: reg_metrics(
            val_actual,
            val_persistence + alpha * val_slope * np.arange(1, 4),
        )["MAE"],
    )

    train_features, train_residuals = make_training_examples(df)
    categorical = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    residual_model = make_pipeline(
        categorical,
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=SEED,
        ),
    )
    residual_model.fit(train_features[CATEGORICAL + NUMERIC], train_residuals)

    val_residual = residual_model.predict(val_features).reshape(val_actual.shape)
    test_residual = residual_model.predict(test_features).reshape(test_actual.shape)

    val_predictions = {
        "raw_tft": val_tft,
        "persistence": val_persistence,
        "tft_persistence_blend": blend_weight * val_tft + (1 - blend_weight) * val_persistence,
        "damped_trend": val_persistence + trend_alpha * val_slope * np.arange(1, 4),
        "residual_boosting": np.clip(val_persistence + val_residual, 0, None),
    }
    test_predictions = {
        "raw_tft": test_tft,
        "persistence": test_persistence,
        "tft_persistence_blend": blend_weight * test_tft + (1 - blend_weight) * test_persistence,
        "damped_trend": test_persistence + trend_alpha * test_features["slope_6"].to_numpy().reshape(test_actual.shape) * np.arange(1, 4),
        "residual_boosting": np.clip(test_persistence + test_residual, 0, None),
    }
    selected_by_horizon = {}
    val_horizon_switch = np.empty_like(val_actual)
    test_horizon_switch = np.empty_like(test_actual)
    for horizon_idx in range(val_actual.shape[1]):
        selected = min(
            ("persistence", "residual_boosting"),
            key=lambda name: reg_metrics(
                val_actual[:, horizon_idx], val_predictions[name][:, horizon_idx]
            )["MAE"],
        )
        selected_by_horizon[f"horizon_h{horizon_idx + 1}"] = selected
        val_horizon_switch[:, horizon_idx] = val_predictions[selected][:, horizon_idx]
        test_horizon_switch[:, horizon_idx] = test_predictions[selected][:, horizon_idx]
    summary = {
        "checkpoint": str(checkpoint),
        "holdout_used_for_selection": False,
        "tuning": {
            "blend_tft_share": float(blend_weight),
            "damped_trend_alpha": float(trend_alpha),
            "residual_training_target": "future price minus origin persistence price",
            "residual_training_targets_before": VAL_START,
        },
        "validation": {
            name: metric_block(val_actual, prediction)
            for name, prediction in val_predictions.items()
        },
        "test": {
            name: metric_block(test_actual, prediction)
            for name, prediction in test_predictions.items()
        },
        "test_reference_seasonal_naive": metric_block(test_actual, test_seasonal),
        "validation_selected_horizon_switch": {
            "selection_rule": "Choose persistence or residual boosting by validation MAE, separately per horizon",
            "selected_by_horizon": selected_by_horizon,
            "validation": metric_block(val_actual, val_horizon_switch),
            "test": metric_block(test_actual, test_horizon_switch),
        },
        "best_validation_strategy": min(
            val_predictions,
            key=lambda name: reg_metrics(val_actual, val_predictions[name])["MAE"],
        ),
        "best_test_strategy_for_diagnostic_only": min(
            test_predictions,
            key=lambda name: reg_metrics(test_actual, test_predictions[name])["MAE"],
        ),
    }
    output_path = run_dir / "benchmark_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nTuning: TFT share={blend_weight:.2f}; trend alpha={trend_alpha:.2f}")
    print("\nValidation pooled MAE (selection period):")
    for name, result in summary["validation"].items():
        print(f"  {name:24s} {result['overall_pooled']['MAE']:.3f}")
    print("\nTest pooled MAE (final holdout):")
    for name, result in summary["test"].items():
        print(f"  {name:24s} {result['overall_pooled']['MAE']:.3f}")
    print(f"  {'seasonal_naive':24s} {summary['test_reference_seasonal_naive']['overall_pooled']['MAE']:.3f}")
    print("\nValidation-selected horizon switch:")
    print(f"  choices: {selected_by_horizon}")
    print(f"  test pooled MAE: {summary['validation_selected_horizon_switch']['test']['overall_pooled']['MAE']:.3f}")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()