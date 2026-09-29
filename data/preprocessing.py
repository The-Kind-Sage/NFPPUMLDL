"""Preprocessing - clean monthly panel per series."""
import numpy as np
import pandas as pd
from config import (
    MIN_SERIES_LEN, STATIC_CAT, STATIC_REAL, KNOWN_CAT, KNOWN_REAL, UNKNOWN_REAL
)


def preprocess(df):
    df = df.sort_values(["market", "commodity", "unit", "date"]).copy()
    df["series_id"] = df["market"] + "__" + df["commodity"] + "__" + df["unit"]

    # --- keep series long enough to train on ---
    counts = df.groupby("series_id").size()
    keep = counts[counts >= MIN_SERIES_LEN].index
    print(f"\nseries total: {len(counts)} | kept (>={MIN_SERIES_LEN} months): {len(keep)} "
          f"| dropped short series: {len(counts) - len(keep)}")
    df = df[df["series_id"].isin(keep)].copy()
    df["ym"] = df["date"].dt.to_period("M")

    # --- complete each series to a contiguous monthly grid ---
    segments, short_filled, n_splits = [], 0, 0
    statics = ["series_id", "market", "commodity", "unit", "category",
               "admin1", "admin2", "latitude", "longitude"]

    def finalize(piece, new_sid):
        piece = piece.copy()
        piece["is_imputed"] = piece["price"].isna().astype(np.int8)
        piece["price"] = piece["price"].interpolate(limit_direction="both")
        piece["series_id"] = new_sid
        return piece.reset_index(drop=True)

    for sid, g in df.groupby("series_id"):
        g = g.sort_values("date")
        full_idx = pd.period_range(g["ym"].min(), g["ym"].max(), freq="M")
        g = g.set_index("ym").reindex(full_idx)
        for c in statics:
            g[c] = g[c].ffill().bfill()
        g["date"] = full_idx.to_timestamp() + pd.offsets.Day(14)  # monthly 15th
        g["ym"] = full_idx
        miss = g["price"].isna().to_numpy()
        # find NaN runs
        runs, i, n = [], 0, len(g)
        while i < n:
            if miss[i]:
                j = i
                while j + 1 < n and miss[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        cuts = [(a, b) for (a, b) in runs if (b - a + 1) > 2]
        short_filled += sum((b - a + 1) for (a, b) in runs if (b - a + 1) <= 2)
        if not cuts:
            seg = finalize(g, sid)
            if len(seg) >= MIN_SERIES_LEN:
                segments.append(seg)
        else:
            n_splits += 1
            bounds, start = [], 0
            for (a, b) in cuts:
                bounds.append((start, a))
                start = b + 1
            bounds.append((start, n))
            k = 0
            for (a, b) in bounds:
                if b - a >= MIN_SERIES_LEN:
                    k += 1
                    segments.append(finalize(g.iloc[a:b], f"{sid}__seg{k}"))
    df = pd.concat(segments, ignore_index=True)
    print(f"short gaps interpolated (<=2 mo): {short_filled} cells | "
          f"series split at long gaps: {n_splits} | resulting segments: {df['series_id'].nunique()}")

    # --- time index (global month counter) ---
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    t0 = df["date"].min()
    df["time_idx"] = ((df["date"].dt.year - t0.year) * 12
                      + (df["date"].dt.month - t0.month)).astype(int)

    # --- known future features (calendar - always knowable ahead) ---
    m = df["date"].dt.month
    df["month"] = m.astype(str).str.zfill(2)          # categorical -> embedding learns seasonality
    df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    df["year_norm"] = (df["date"].dt.year - 2001) / 25.0
    df["time_norm"] = df["time_idx"] / df["time_idx"].max()

    # Nepal-specific calendar (deterministic -> known in the future):
    #   monsoon  Jun-Sep : road damage / transport cost, vegetable supply drops
    #   harvest  Oct-Dec (paddy) + Apr-May (wheat) : market glut, prices ease
    #   lean     Mar-Jun : stocks run down before monsoon, prices firm up
    #   festival Mar/Holi, Oct-Nov (Dashain+Tihar) : demand spike for staples/sweets
    df["monsoon"] = m.isin([6, 7, 8, 9]).astype(np.float32)
    df["harvest"] = m.isin([10, 11, 12, 4, 5]).astype(np.float32)
    df["lean_season"] = m.isin([3, 4, 5, 6]).astype(np.float32)
    df["festival"] = m.isin([3, 10, 11]).astype(np.float32)

    # --- unknown features (derived from past price - encoder side only, no leakage) ---
    grp = df.groupby("series_id")["price"]
    df["lag_1"] = grp.shift(1)
    df["lag_3"] = grp.shift(3)
    df["lag_12"] = grp.shift(12)                      # same month last year (seasonality!)
    df["roll_mean_3"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["roll_mean_12"] = grp.transform(lambda s: s.rolling(12, min_periods=1).mean())
    lag_cols = ["lag_1", "lag_3", "lag_12"]
    df[lag_cols] = df.groupby("series_id")[lag_cols].transform(lambda s: s.bfill().ffill())
    df[lag_cols] = df[lag_cols].fillna(df["price"].median())

    # --- domain-specific engineered features (all history-only) ---
    df["log_price"] = np.log1p(df["price"])           # multiplicative structure of prices
    df["mom_pct"] = df["price"] / df["lag_1"] - 1.0   # 1-month momentum
    df["yoy_pct"] = df["price"] / df["lag_12"] - 1.0  # YoY food inflation of this item
    df["dev_roll12"] = df["price"] / df["roll_mean_12"] - 1.0  # vs 12-mo mean (mean reversion)
    sd = grp.transform(lambda s: s.rolling(3, min_periods=2).std())
    df["roll_std_12"] = grp.transform(lambda s: s.rolling(12, min_periods=3).std())
    eng = ["mom_pct", "yoy_pct", "dev_roll12", "roll_std_12"]
    df[eng] = df.groupby("series_id")[eng].transform(lambda s: s.bfill().ffill())
    df[eng] = df[eng].fillna(0.0)
    df["vol_ratio"] = (sd / df["roll_std_12"].clip(lower=1e-6)).clip(-5, 5).fillna(0.0)

    # cross-series context: national food-price level and own-category level per month
    # (constant across series within a month -> encoder-only macro signal)
    df["nat_index"] = df.groupby("date")["log_price"].transform("mean")
    df["comm_index"] = df.groupby(["date", "category"])["log_price"].transform("mean")

    assert df["price"].isna().sum() == 0, "price still has NaN!"
    print(f"processed rows: {len(df)} | series: {df['series_id'].nunique()} | "
          f"time_idx: 0..{df['time_idx'].max()}")
    return df


def chronological_split(df, val_start, test_start):
    train_df = df[df["date"] < val_start].copy()
    val_hist = df[df["date"] < test_start].copy()   # val windows + their encoder history
    test_hist = df.copy()                            # test windows + their encoder history
    val_start_idx = int(df.loc[df["date"] >= val_start, "time_idx"].min())
    test_start_idx = int(df.loc[df["date"] >= test_start, "time_idx"].min())
    print("\nCHRONOLOGICAL SPLIT (no shuffling, no leakage):")
    print(f"  train : {train_df['date'].min().date()} .. {train_df['date'].max().date()}  ({len(train_df)} rows)")
    print(f"  val   : {val_start} .. 2024-12-15")
    print(f"  test  : {test_start} .. {test_hist['date'].max().date()}  (final holdout)")
    print(f"  val_start_idx={val_start_idx} test_start_idx={test_start_idx}")
    return train_df, val_hist, test_hist, val_start_idx, test_start_idx