"""Metrics computation - pure numpy/sklearn (no torch needed)."""
import numpy as np
from sklearn.metrics import r2_score


def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2, "n": int(len(y_true))}


def naive_baselines(df, test_start):
    """One-step naive references on TEST rows: last-value and seasonal-12.

    NOTE: this is a ROW-level reference (one row per series-month) while the TFT
    is scored on forecast WINDOWS. Use aligned_baselines() for an apples-to-apples
    comparison - this one is kept for prep mode / historic reports.
    """
    s = df.sort_values(["series_id", "date"]).copy()
    s["naive_last"] = s.groupby("series_id")["price"].shift(1)
    s["naive_seas12"] = s.groupby("series_id")["price"].shift(12)
    t = s[s["date"] >= test_start].dropna(subset=["naive_last", "naive_seas12"])
    out = {
        "naive_last_value": reg_metrics(t["price"], t["naive_last"]),
        "naive_seasonal_12": reg_metrics(t["price"], t["naive_seas12"]),
    }
    return out


def aligned_baselines(df, index, horizon=3):
    """Baselines on EXACTLY the same forecast windows as the TFT.

    index.time_idx is the FIRST FORECAST month T (verified: rolling_one_step at h=1
    reproduces persistence exactly), so for each window (series, T) and step h = 1..H
    the predicted month is T + h - 1:

      persistence      price(T-1)           history only   -> fair, main baseline
      seasonal_naive   price(T+h-13)        history only   -> fair, captures yearly cycle
      rolling_one_step price(T+h-2)         h=1 == persistence; h>1 peeks at actuals
                                             (operational upper bar, not a fair
                                             3-month shot - labelled as such)
    """
    lookup = {(str(s), int(t)): float(p) for s, t, p in
              zip(df["series_id"], df["time_idx"], df["price"])}
    T = np.asarray(index["time_idx"])
    sids = [str(s) for s in index["series_id"]]
    n = len(T)
    pers = np.full((n, horizon), np.nan)
    seas = np.full((n, horizon), np.nan)
    roll = np.full((n, horizon), np.nan)
    for i in range(n):
        sid, t = sids[i], int(T[i])
        for h in range(1, horizon + 1):        # h = 1..H, predicts month (T + h - 1)
            pers[i, h - 1] = lookup.get((sid, t - 1), np.nan)
            seas[i, h - 1] = lookup.get((sid, t + h - 13), np.nan)
            roll[i, h - 1] = lookup.get((sid, t + h - 2), np.nan)
    return {"persistence": pers, "seasonal_naive": seas, "rolling_one_step": roll}