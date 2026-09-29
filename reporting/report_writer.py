"""Final report writer."""
import os
import json
from datetime import datetime
from config import (
    MAX_ENCODER, MAX_PRED, BATCH, HIDDEN, HEADS, MAX_EPOCHS, PATIENCE, SEED,
    VAL_START, TEST_START, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, TARGET_TRANSFORM
)
from utils import _num


def _metric_row(name, m):
    return (f"  {name:26s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
            f"MAPE={m['MAPE_%']:.2f}%  R2={m['R2']:.4f}  n={m['n']}")


def _is_metric(v):
    return isinstance(v, dict) and "MAE" in v


def _metrics_lines(metrics):
    """Render a (possibly nested) metrics dict as printable lines.

    Handles: metric dict -> one row | dict of metric dicts -> section |
    dict of floats (skill scores) -> simple lines.
    """
    lines = []
    for k, v in metrics.items():
        if _is_metric(v):
            lines.append(_metric_row(k, v))
        elif isinstance(v, dict) and v:
            lines.append(f"\n  {k}:")
            for nk, nv in v.items():
                if _is_metric(nv):
                    lines.append(_metric_row(nk, nv))
                elif isinstance(nv, dict) and nv:
                    lines.append(f"    {nk}:")
                    for nnk, nnv in nv.items():
                        if _is_metric(nnv):
                            lines.append(_metric_row("  " + nnk, nnv))
                        else:
                            lines.append(f"      {nnk}: {nnv}")
                else:
                    try:
                        lines.append(f"    {nk:24s} {float(nv):+.3f}")
                    except (TypeError, ValueError):
                        lines.append(f"    {nk}: {nv}")
        else:
            lines.append(f"  {k}: {v}")
    return lines


def print_final_summary(metrics):
    """Compact final block - printed LAST so the run ends with the evaluated metrics."""
    print("\n" + "=" * 78)
    print("FINAL EVALUATED METRICS (test holdout 2025-01 .. 2026-08, prices in NPR)")
    print("=" * 78)
    for line in _metrics_lines(metrics):
        print(line)
    print("=" * 78, flush=True)


def write_final_report(out_dir, mode, history, metrics, commodity_rows=None, config=None):
    """Write final_report.txt: config + per-epoch table + all final metrics + file list."""
    cfg = dict(lookback=MAX_ENCODER, horizon=MAX_PRED, batch=BATCH, hidden=HIDDEN,
               heads=HEADS, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
               dropout=DROPOUT, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
               target_transform=TARGET_TRANSFORM or "none", loss="WeightedQuantileLoss",
               lookback_note="", epochs_run=(history[-1]["epoch"] if history else 0))
    if config:
        cfg.update(config)
    L = []
    add = L.append
    add("=" * 78)
    add("NEPAL FOOD PRICE FORECASTING - TFT FINAL REPORT")
    add(f"generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"mode      : {mode}")
    add(f"output dir: {out_dir}")
    add("=" * 78)

    add("")
    add("[CONFIG]")
    add(f"  lookback={cfg['lookback']} mo | horizon={cfg['horizon']} mo | batch={cfg['batch']} "
        f"| hidden={cfg['hidden']} | heads={cfg['heads']}")
    add(f"  max_epochs={cfg['max_epochs']} | epochs_run={cfg['epochs_run']} | "
        f"early_stop_patience={cfg['patience']} | seed={cfg['seed']}")
    add(f"  split: train < {VAL_START} | val = 2024 | test >= {TEST_START} (holdout)")
    add(f"  dropout={cfg['dropout']} | lr={cfg['lr']} | weight_decay={cfg['weight_decay']}")
    add(f"  target = {cfg['target_transform']} | loss = {cfg['loss']} "
        f"over 7 quantiles; reported prediction = median")

    add("")
    add(f"[1] PER-EPOCH TRAINING REPORT ({len(history)} epochs, live-printed during run)")
    if history:
        hdr = (f"{'epoch':>5} {'train_loss':>11} {'val_loss':>10} {'val_MAE':>10} "
               f"{'val_RMSE':>10} {'lr':>9} {'sec':>7}  note")
        add("  " + hdr)
        add("  " + "-" * (len(hdr) + 2))
        for r in history:
            note = []
            if r.get("is_best"):
                note.append("*new best*")
            if r.get("best_epoch"):
                note.append(f"best@{r['best_epoch']}")
                note.append(f"stale={r.get('stale_epochs', 0)}/{cfg['patience']}")
            add(f"  {r['epoch']:>5} {_num(r.get('train_loss')):>11} "
                f"{_num(r.get('val_loss')):>10} {_num(r.get('val_MAE'), 1):>10} "
                f"{_num(r.get('val_RMSE'), 1):>10} {_num(r.get('lr'), 5):>9} "
                f"{_num(r.get('seconds'), 1):>7}  {' '.join(note)}".rstrip())
        rows = [r for r in history if r.get("val_loss") is not None]
        if rows:
            best = min(rows, key=lambda r: r["val_loss"])
            add("")
            add(f"  best epoch {best['epoch']}: val_loss={best['val_loss']:.4f} "
                f"val_MAE={_num(best.get('val_MAE'), 1)} "
                f"val_RMSE={_num(best.get('val_RMSE'), 1)} NPR")
    else:
        add("  (no epochs recorded - prep mode)")

    add("")
    add("[2] FINAL TEST METRICS (holdout 2025-01 .. 2026-08, prices in NPR)")
    if metrics:
        for line in _metrics_lines(metrics):
            add(line)
        add("")
        add("  baseline_* = same windows / same rows / same information as the TFT")
        add("              (fixes the old naive-vs-TFT row mismatch).")
        add("  skill_vs_persistence > 0 -> TFT beats persistence on that horizon.")
        add("  rolling_one_step peeks at actuals for h>1: an upper bar, not a rival.")
        add("  naive_row_level_reference = legacy row-level h1 numbers (different n).")
    else:
        add("  (no metrics - prep mode)")

    if commodity_rows:
        add("")
        add(f"[3] PER-COMMODITY BREAKDOWN ({len(commodity_rows)} commodities)")
        add(f"  {'commodity':22s} {'MAE':>9} {'RMSE':>9} {'MAPE%':>8} {'R2':>9} {'n':>7}")
        for r in commodity_rows:
            add(f"  {str(r['commodity'])[:22]:22s} {r['MAE']:>9.2f} {r['RMSE']:>9.2f} "
                f"{r['MAPE_%']:>8.2f} {r['R2']:>9.4f} {r['n']:>7d}")

    add("")
    add("[4] FILES IN THIS RUN")
    try:
        for f in sorted(os.listdir(out_dir)):
            add(f"  {f}")
    except OSError:
        add("  (unreadable)")

    path = os.path.join(out_dir, "final_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path