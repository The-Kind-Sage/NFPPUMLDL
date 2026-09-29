"""Version compatibility helpers for pytorch-forecasting / lightning."""
import torch
import numpy as np
import pandas as pd


def unpack_predict_output(res):
    """Unpack model.predict(...) robustly across library versions.

    Newer versions return Prediction(output, x, index, decoder_lengths, y)
    where output is an Output(...) namedtuple holding `.prediction`
    of shape (N, horizon, n_quantiles). Returns dict with keys:
    out (full model output, for plotting), raw (pred tensor), x, y, index.
    """
    out = raw = x = y = idx = None
    if hasattr(res, "_fields"):  # Prediction namedtuple (current versions)
        out = res.output
        raw = out.prediction if hasattr(out, "prediction") else out
        x = res.x
        y = res.y
        if isinstance(y, (list, tuple)):
            y = y[0] if len(y) > 0 else None
        idx = res.index
    else:  # fallback: plain tuple of mixed items (older versions)
        items = list(res) if isinstance(res, (list, tuple)) else [res]
        for it in items:
            if isinstance(it, dict):
                x = it
            elif isinstance(it, pd.DataFrame):
                idx = it
            elif hasattr(it, "prediction"):
                out, raw = it, it.prediction
            elif hasattr(it, "ndim") and int(it.ndim) == 3:
                raw = it
            elif hasattr(it, "ndim") and int(it.ndim) == 2 and y is None:
                y = it
        if out is None:
            out = raw
    return {"out": out, "raw": raw, "x": x, "y": y, "index": idx}


def _prediction_tensor(out):
    """Pull the prediction tensor out of a model forward output (any pf version)."""
    if out is None:
        return None
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("prediction", "output", "out"):
            if k in out:
                t = _prediction_tensor(out[k])
                if t is not None:
                    return t
        for v in out.values():
            t = _prediction_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(out, (tuple, list)):
        for v in out:
            t = _prediction_tensor(v)
            if t is not None:
                return t
        return None
    for k in ("prediction", "output"):
        v = getattr(out, k, None)
        if v is not None and v is not out:
            t = _prediction_tensor(v)
            if t is not None:
                return t
    return None


def _to_price(values, target_scale, ref_median, apply=None):
    """Inverse GroupNormalizer(z) = center + scale*z -> NPR (undo target transform too).

    apply=None auto-detects from the value magnitude (version-proof):
      * pf >= 1.x predicts in real space already -> no-op
      * older pf predicts in normalized space    -> undo the scaling
    When apply is given (True/False) the decision is forced, which keeps
    prediction and ground truth in the SAME space even if their medians differ.
    """
    from config import TARGET_TRANSFORM
    v = torch.as_tensor(values).detach().float()
    if apply is None:
        # normalized z-values sit near 0 while real prices sit near ref_median;
        # compare RELATIVELY so cheap commodities (ref ~15) are not misdetected
        med = abs(float(v.median().item()))
        apply = ref_median > 0 and med < 0.5 * ref_median
    if apply:
        ts = torch.as_tensor(target_scale).detach().float()
        center = ts[..., 0]
        scale = ts[..., 1].clamp_min(1e-8)
        shape = [v.shape[0]] + [1] * (v.ndim - 1)
        v = v * scale.view(shape) + center.view(shape)
        if TARGET_TRANSFORM == "log":          # z -> log(price) -> price
            v = torch.exp(v)
        elif TARGET_TRANSFORM == "log1p":
            v = torch.expm1(v)
    return v.cpu().numpy()


def denorm(values, target_scale, ref_median):
    """Backwards-compatible wrapper around _to_price (auto-detect mode)."""
    return _to_price(values, target_scale, ref_median, apply=None)