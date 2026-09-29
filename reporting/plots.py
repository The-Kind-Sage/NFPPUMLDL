"""Plotting utilities."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_true, pred_med, metrics, out_dir):
    """Actual vs predicted scatter plot."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true.ravel(), pred_med.ravel(), s=3, alpha=0.3)
    lo, hi = float(y_true.min()), float(y_true.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
    m = metrics["overall_pooled"]
    plt.title(f"TFT test: actual vs predicted (R2={m['R2']:.3f}, MAE={m['MAE']:.1f} NPR)")
    plt.xlabel("Actual price (NPR)")
    plt.ylabel("Predicted price (NPR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()


def plot_sample_forecasts(best, x, out, raw, out_dir):
    """Sample forecast windows (history + median + intervals)."""
    try:
        for i in [0, len(x["decoder_target"]) // 2, len(x["decoder_target"]) - 1]:
            try:
                fig = best.plot_prediction(x, out, idx=i, add_loss_to_title=True)
            except Exception:
                fig = best.plot_prediction(x, raw, idx=i, add_loss_to_title=True)
            fig.savefig(os.path.join(out_dir, f"sample_forecast_{i}.png"), dpi=150)
            plt.close(fig)
        print("saved sample forecast plots")
    except Exception as e:
        print(f"(sample plots skipped: {e})")


def _f(v):
    """Finite float or None (tolerates CSV NaN / empty cells / strings)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def plot_training_curves(history, out_dir):
    """Per-epoch train/val loss, validation error (NPR) and learning rate."""
    rows = [r for r in history if _f(r.get("epoch")) is not None]
    if not rows:
        return
    epochs = [int(_f(r["epoch"])) for r in rows]

    def series(key):
        out = []
        for r in rows:
            v = _f(r.get(key))
            out.append(np.nan if v is None else v)
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, series("train_loss"), "o-", label="train_loss")
    ax1.plot(epochs, series("val_loss"), "s-", label="val_loss")
    scored = [r for r in rows if _f(r.get("val_loss")) is not None]
    if scored:
        b = min(scored, key=lambda r: _f(r["val_loss"]))
        be, bv = int(_f(b["epoch"])), _f(b["val_loss"])
        ax1.axvline(be, color="tab:green", ls="--", lw=1, alpha=0.8)
        ax1.annotate(f"best @{be}", (be, bv), xytext=(6, -14),
                     textcoords="offset points", fontsize=9, color="tab:green")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("quantile loss")
    ax1.set_title("Train / validation loss")
    ax1.set_xticks(epochs)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, series("val_MAE"), "o-", label="val MAE")
    ax2.plot(epochs, series("val_RMSE"), "s-", label="val RMSE")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("error (NPR)")
    ax2.set_title("Validation error")
    ax2.set_xticks(epochs)
    ax2.grid(alpha=0.3)
    ax2.legend()
    lrs = series("lr")
    if np.isfinite(lrs).any():
        ax3 = ax2.twinx()
        ax3.plot(epochs, lrs, "k:", lw=1, alpha=0.7, label="lr")
        ax3.set_ylabel("learning rate")
        ax3.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_interval_calibration(pred_q, y_true, out_dir, nominal=0.8):
    """q10-q90 coverage, interval width and quantile crossings.

    Returns the metrics dict (caller merges it into metrics.json).
    """
    pred_q = np.asarray(pred_q, dtype=float)
    y = np.asarray(y_true, dtype=float)
    P = y.shape[1]
    q10, q90 = pred_q[:, :, 1], pred_q[:, :, 5]
    inside = (y >= q10) & (y <= q90)
    width = q90 - q10
    crossing = (pred_q[:, :, :-1] > pred_q[:, :, 1:]).any(axis=-1)
    m = {
        "coverage_80_overall": float(inside.mean()),
        "mean_width_overall": float(width.mean()),
        "quantile_crossing_cells": int(crossing.sum()),
        "quantile_crossing_pct": float(100.0 * crossing.mean()),
    }
    for h in range(P):
        m[f"coverage_80_h{h+1}"] = float(inside[:, h].mean())
        m[f"mean_width_h{h+1}"] = float(width[:, h].mean())

    labels = [f"h{h + 1}" for h in range(P)] + ["overall"]
    cov = [m[f"coverage_80_h{h + 1}"] for h in range(P)] + [m["coverage_80_overall"]]
    wid = [m[f"mean_width_h{h + 1}"] for h in range(P)] + [m["mean_width_overall"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bars = ax1.bar(labels, cov, color="tab:blue", alpha=0.8)
    ax1.axhline(nominal, color="r", ls="--", lw=1.2, label=f"nominal {nominal:.0%}")
    ax1.bar_label(bars, fmt="%.3f", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("empirical coverage")
    ax1.set_title(f"q10-q90 coverage (n={y.size})")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    bars2 = ax2.bar(labels, wid, color="tab:orange", alpha=0.8)
    ax2.bar_label(bars2, fmt="%.1f", fontsize=9)
    ax2.set_ylabel("mean q90 - q10 width (NPR)")
    ax2.set_title("interval width | crossings: "
                  f"{m['quantile_crossing_cells']} cells ({m['quantile_crossing_pct']:.3f}%)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interval_calibration.png"), dpi=150)
    plt.close(fig)
    return m


def plot_interpretations(best, out, out_dir):
    """Aggregate attention + variable-selection importance figures (pf 1.8 API)."""
    try:
        interp = best.interpret_output(out, reduction="mean")
        figs = best.plot_interpretation(interp)
        names = {
            "attention": "interpretation_attention.png",
            "static_variables": "interpretation_static_importance.png",
            "encoder_variables": "interpretation_encoder_importance.png",
            "decoder_variables": "interpretation_decoder_importance.png",
        }
        for key, fig in figs.items():
            fig.savefig(os.path.join(out_dir, names.get(key, f"interpretation_{key}.png")),
                        dpi=150)
            plt.close(fig)
        print("saved interpretation plots (attention + variable importance)")
    except Exception as e:
        print(f"(interpretation plots skipped: {e})")


def plot_per_commodity(rows, out_dir):
    """Grouped MAE/RMSE horizontal bars per commodity."""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["MAE"]))
    names = [str(r["commodity"]) for r in rows]
    mae = [float(r["MAE"]) for r in rows]
    rmse = [float(r["RMSE"]) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    h = 0.38
    ax.barh(y + h / 2, mae, height=h, label="MAE", color="tab:blue", alpha=0.85)
    ax.barh(y - h / 2, rmse, height=h, label="RMSE", color="tab:orange", alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    xmax = max(rmse + mae) or 1.0
    for i, v in enumerate(mae):
        ax.text(v + 0.015 * xmax, i + h / 2, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("error (NPR)")
    ax.set_title(f"Per-commodity test error ({len(rows)} commodities)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_commodity_mae.png"), dpi=150)
    plt.close(fig)


def plot_residual_distribution(y_true, pred, metrics, out_dir):
    """Signed-error histogram + absolute-error distribution (log count)."""
    err = (np.asarray(pred, dtype=float) - np.asarray(y_true, dtype=float)).ravel()
    err = err[np.isfinite(err)]
    if err.size == 0:
        return
    m = (metrics or {}).get("overall_pooled")
    mae = float(m["MAE"]) if m else float(np.mean(np.abs(err)))
    rmse = float(m["RMSE"]) if m else float(np.sqrt(np.mean(err ** 2)))

    lo, hi = np.percentile(err, [1, 99])
    outside = int(np.sum((err < lo) | (err > hi)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(err, bins=80, range=(lo, hi), color="tab:blue", alpha=0.8)
    ax1.axvline(0, color="k", lw=1.2)
    ax1.set_xlabel("prediction - actual (NPR)")
    ax1.set_ylabel("count")
    note = f" | x-axis: 1-99 pct ({outside} outside)" if outside else ""
    ax1.set_title(f"Signed error (median forecast){note}")
    ax1.grid(alpha=0.3)

    ax2.hist(np.abs(err), bins=80, color="tab:green", alpha=0.8)
    ax2.set_yscale("log")
    ax2.axvline(mae, color="r", ls="--", lw=1.2, label=f"MAE {mae:.1f}")
    ax2.axvline(rmse, color="k", ls="--", lw=1.2, label=f"RMSE {rmse:.1f}")
    ax2.set_xlabel("|error| (NPR)")
    ax2.set_ylabel("count (log scale)")
    ax2.set_title("Absolute error")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_distribution.png"), dpi=150)
    plt.close(fig)