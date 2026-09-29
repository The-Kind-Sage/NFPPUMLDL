"""Formatting helpers and callback metric extractors."""
import numpy as np


def _num(v, nd=4):
    """Format a metric for the report ('-' when missing / non-finite)."""
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


def _cm_first(cm, *names):
    """First matching value in trainer.callback_metrics (exact, then prefix)."""
    for n in names:
        if n in cm:
            try:
                return float(cm[n])
            except (TypeError, ValueError):
                pass
    for n in names:
        for k, v in cm.items():
            if k.startswith(n):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None


def _cm_lr(cm):
    for k, v in cm.items():
        if k.startswith("lr") or "learning_rate" in k:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _epoch_line(r, max_epochs, patience):
    parts = [f"[EPOCH {r['epoch']:02d}/{max_epochs:02d}]"]
    parts.append(f"train_loss={_num(r.get('train_loss'))}")
    parts.append(f"val_loss={_num(r.get('val_loss'))}")
    parts.append(f"val_MAE={_num(r.get('val_MAE'), 1)}")
    parts.append(f"val_RMSE={_num(r.get('val_RMSE'), 1)}")
    if r.get("lr") is not None:
        parts.append(f"lr={r['lr']:.5f}")
    parts.append(f"{r.get('seconds', 0):.1f}s")
    if r.get("is_best"):
        parts.append("*new best*")
    if r.get("best_epoch"):
        parts.append(f"best@{r['best_epoch']} stale={r.get('stale_epochs', 0)}/{patience}")
    return "  ".join(parts)


def accelerator_name():
    """Prefer the NVIDIA GPU (RTX 4050 etc.), fall back to CPU. Enables TF32 on GPU."""
    import torch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")  # TF32 tensor cores, big speedup
        return "gpu"
    return "cpu"


def enable_determinism():
    """Make training reproducible: same seed -> same weights -> same metrics.

    Without this, GPU backward passes use atomic reductions that vary run to run;
    two identical 4-epoch runs gave pooled MAE 17.66 and 20.29 (seed fixed).
    warn_only=True: pf's TimeDistributedInterpolation (F.interpolate, linear) has no
    deterministic backward on CUDA - strict mode raises, so we accept a small residual
    run-to-run spread (~0.5 pooled MAE instead of ~2.6 without these flags).
    """
    import os
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 kept: it is deterministic per op
    torch.backends.cudnn.allow_tf32 = True


def determinism_report():
    """One-line summary of the determinism flags actually in effect."""
    import os
    import torch
    return (f"determinism: algorithms={torch.are_deterministic_algorithms_enabled()} "
            f"cudnn_deterministic={torch.backends.cudnn.deterministic} "
            f"benchmark={torch.backends.cudnn.benchmark} "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")