"""Model evaluation on test set."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import BATCH, TEST_START
from utils import unpack_predict_output, denorm, accelerator_name
from evaluation.metrics import reg_metrics, naive_baselines, aligned_baselines
from reporting.plots import (
    plot_actual_vs_predicted, plot_sample_forecasts, plot_interval_calibration,
    plot_residual_distribution, plot_interpretations,
)
from reporting.report_writer import _metrics_lines


def evaluate(best, test, df, out_dir):
    test_dl = test.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=0)
    try:
        res = best.predict(test_dl, mode="raw", return_x=True, return_y=True,
                           return_index=True,
                           trainer_kwargs={"accelerator": accelerator_name(), "devices": 1, "logger": False})
    except TypeError:
        res = best.predict(test_dl, mode="raw", return_x=True, return_y=True,
                           return_index=True)
    d = unpack_predict_output(res)
    raw, x, y, idx = d["raw"], d["x"], d["y"], d["index"]
    if y is None and x is not None and "decoder_target" in x:
        y = x["decoder_target"]  # fallback: targets straight from the batch
    assert raw is not None and y is not None and x is not None, "unexpected predict() output"

    ref_med = float(df.loc[df["date"] >= TEST_START, "price"].median())
    scale = x["target_scale"]
    pred_q = denorm(raw, scale, ref_med)          # (N, P, 7)
    y_true = denorm(y, scale, ref_med)            # (N, P)
    pred_med = pred_q[:, :, 3]                    # median quantile
    pred_med = np.clip(pred_med, 0, None)

    P = pred_med.shape[1]
    metrics = {"overall_pooled": reg_metrics(y_true, pred_med)}
    for h in range(P):
        metrics[f"horizon_h{h+1}"] = reg_metrics(y_true[:, h], pred_med[:, h])

    # ---- fair baselines: SAME windows, SAME rows, SAME information set ----
    if idx is not None and len(idx) == len(y_true):
        for name, arr in aligned_baselines(df, idx, P).items():
            metrics[f"baseline_{name}"] = {
                **{f"horizon_h{h+1}": reg_metrics(y_true[:, h], arr[:, h])
                   for h in range(P)},
                "overall_pooled": reg_metrics(y_true, arr),
            }
        pers = metrics["baseline_persistence"]
        if pers["overall_pooled"]["MAE"] > 0:
            skill = {f"horizon_h{h+1}": 1.0 - metrics[f"horizon_h{h+1}"]["MAE"]
                     / pers[f"horizon_h{h+1}"]["MAE"] for h in range(P)}
            skill["overall_pooled"] = (1.0 - metrics["overall_pooled"]["MAE"]
                                       / pers["overall_pooled"]["MAE"])
            metrics["skill_vs_persistence"] = skill
    else:
        print("(warning: window index unavailable - aligned baselines skipped)")
    metrics["naive_row_level_reference"] = naive_baselines(df, TEST_START)  # legacy: different n
    # quantile-band quality: coverage vs nominal 80%, width, crossings -> also plots
    metrics["interval_calibration"] = plot_interval_calibration(pred_q, y_true, out_dir)

    print("\n" + "=" * 78)
    print("TEST METRICS (2025-01 .. 2026-08 holdout, prices in NPR)")
    print("=" * 78)
    for line in _metrics_lines(metrics):
        print(line)
    print("=" * 78)
    print("skill > 0 -> TFT beats persistence on that horizon (same rows, same info).")
    print("rolling_one_step peeks at actuals for h>1 - it is an upper bar, not a rival.")
    print(f"Overall_pooled covers all {P} horizons, so it is naturally a bit higher.")

    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"),
                        y_true=y_true, pred_median=pred_med, pred_q10=pred_q[:, :, 1],
                        pred_q90=pred_q[:, :, 5],
                        time_idx=np.asarray(idx["time_idx"]) if idx is not None else [],
                        series_id=np.asarray(idx["series_id"].astype(str))
                        if idx is not None else [])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- plots ---
    plot_actual_vs_predicted(y_true, pred_med, metrics, out_dir)
    plot_residual_distribution(y_true, pred_med, metrics, out_dir)
    plot_sample_forecasts(best, x, d["out"], raw, out_dir)
    plot_interpretations(best, d["out"], out_dir)
    return metrics