"""Future forecasting beyond last observed date."""
import os
import numpy as np
import pandas as pd
from config import MAX_ENCODER, MAX_PRED, NEEDED
from pytorch_forecasting import TimeSeriesDataSet
from utils import unpack_predict_output, denorm, accelerator_name


def future_forecast(best, training, df, out_dir, max_encoder=MAX_ENCODER, max_pred=MAX_PRED):
    """Forecast the next `max_pred` months beyond the last observed date, per active series.

    A series is active when it is still reported close to the global end date;
    dead segments (split off at a collection gap years ago) are skipped, because
    their "next 3 months" would land in the past.
    """
    tmax = int(df["time_idx"].max())
    last_dates = df.groupby("series_id")["date"].max()
    cutoff = last_dates.max() - pd.DateOffset(months=max_pred)
    active = last_dates[last_dates >= cutoff].index
    if len(active) < len(last_dates):
        print(f"  skipping {len(last_dates) - len(active)} ended series "
              f"(last obs before {cutoff.date()}) -> forecasting {len(active)} active series")
    parts = []
    for sid, g in df[df["series_id"].isin(active)].groupby("series_id"):
        g = g.sort_values("date").tail(max_encoder).copy()
        last = g.iloc[-1]
        fut = []
        for k in range(1, max_pred + 1):
            idx = int(last["time_idx"]) + k
            d = last["date"] + pd.DateOffset(months=k)
            row = {c: last[c] for c in NEEDED}   # defaults: history-only lag/roll/* placeholders
            row.update({
                "series_id": sid, "time_idx": idx, "date": d,
                "price": float(last["price"]),  # placeholder; model predicts these
                # calendar features MUST be recomputed for the future month, not copied
                "month": f"{d.month:02d}",
                "month_sin": float(np.sin(2 * np.pi * d.month / 12)),
                "month_cos": float(np.cos(2 * np.pi * d.month / 12)),
                "year_norm": (d.year - 2001) / 25.0,
                "time_norm": idx / tmax,
                "monsoon": float(d.month in (6, 7, 8, 9)),
                "harvest": float(d.month in (10, 11, 12, 4, 5)),
                "lean_season": float(d.month in (3, 4, 5, 6)),
                "festival": float(d.month in (3, 10, 11)),
                "is_imputed": 0,
            })
            fut.append(row)
        parts.append(pd.concat([g[NEEDED + ["date"]], pd.DataFrame(fut)], ignore_index=True))
    fut_input = pd.concat(parts, ignore_index=True)

    fds = TimeSeriesDataSet.from_dataset(training, fut_input[NEEDED],
                                         predict=True, stop_randomization=True)
    fdl = fds.to_dataloader(train=False, batch_size=512, num_workers=0)
    res = best.predict(fdl, mode="raw", return_x=True, return_index=True,
                       trainer_kwargs={"accelerator": accelerator_name(), "devices": 1, "logger": False})
    dd = unpack_predict_output(res)
    raw, x, index = dd["raw"], dd["x"], dd["index"]
    assert raw is not None and index is not None, "future predict() output unexpected"
    print(f"  future index columns: {list(index.columns)}")

    ref_med = float(df["price"].median())
    q = denorm(raw, x["target_scale"], ref_med)
    id_col = "series_id" if "series_id" in index.columns else index.columns[0]
    last_date = df.groupby("series_id")["date"].max().to_dict()
    meta = df.drop_duplicates("series_id").set_index("series_id")[
        ["market", "commodity", "unit"]].to_dict("index")

    rows = []
    for i in range(q.shape[0]):
        sid = str(index[id_col].iloc[i])
        for k in range(q.shape[1]):
            d = last_date.get(sid, df["date"].max()) + pd.DateOffset(months=k + 1)
            mi = meta.get(sid, {})
            rows.append({
                "series_id": sid, "market": mi.get("market"), "commodity": mi.get("commodity"),
                "unit": mi.get("unit"), "horizon": k + 1, "date": d.date().isoformat(),
                "pred_q10": round(float(q[i, k, 1]), 2),
                "pred_median": round(float(np.clip(q[i, k, 3], 0, None)), 2),
                "pred_q90": round(float(q[i, k, 5]), 2),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "future_forecast.csv"), index=False)
    print(f"saved future_forecast.csv ({len(out)} rows = "
          f"{out['series_id'].nunique()} series x {max_pred} months)")
    print(out.head(6).to_string(index=False))
    return out