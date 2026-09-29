# """
# Nepal Food Price Forecasting with TFT (Temporal Fusion Transformer)
# ====================================================================
# Dataset : WFP Nepal food prices (monthly, 2001-2026)
# Model   : TFT via pytorch-forecasting (multi-horizon, quantile forecasts)
# Target  : price (NPR) -> regression / time-series forecasting
#
# HOW TO RUN
# ----------
# 1) Install (once). On your laptop OR Google Colab (recommended, free GPU):
#
#      pip install -U pytorch-forecasting torch pandas numpy scikit-learn matplotlib
#
#    - Colab: run with GPU runtime (Runtime -> Change runtime type -> T4 GPU).
#      Upload wfp_food_prices_npl.csv, then run this file.
#    - Laptop CPU also works, just slower (use SMOKE test first).
#
# 2) Run modes:
#      python tft_nepal_food_price.py --mode prep    # only EDA + preprocessing check (no torch needed)
#      python tft_nepal_food_price.py --mode smoke   # tiny 2-epoch test that TFT runs end-to-end
#      python tft_nepal_food_price.py --mode train   # FULL run: train TFT + evaluate + forecast (default)
#
# While training, a live report is printed after EVERY epoch (train/val loss, val MAE/RMSE
# in NPR, lr, epoch time, best-checkpoint marker, early-stopping counter).
#
# Outputs go to ./tft_results/ :
#      training_history.csv / .json   per-epoch report (exactly what was printed live)
#      final_report.txt               config + per-epoch table + all final metrics
#      metrics.json                   final test metrics (overall, per horizon, naive baselines)
#      per_commodity_metrics.csv      MAE/RMSE per commodity
#      future_forecast.csv, test_predictions.npz, plots, checkpoints/
# """
#
# import argparse
# import json
# import os
# import sys
# import time
# import warnings
# from datetime import datetime
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
#
# # determinism: cuBLAS reads this only at the FIRST GPU matmul, so set it before torch runs
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
#
# warnings.filterwarnings("ignore")
# import matplotlib
#
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# # ======================================================================================
# # CONFIG - tune these
# # ======================================================================================
# CSV_CANDIDATES = [
#     "Dataset/wfp_food_prices_npl.csv",
#     "wfp_food_prices_npl.csv",
#     "data/wfp_food_prices_npl.csv",
#     "/content/wfp_food_prices_npl.csv",
# ]
# OUT_DIR = "tft_results"
#
# MIN_SERIES_LEN = 48       # keep series with >= 48 monthly obs (4 years). 521 of 648 series qualify.
# MAX_ENCODER = 24          # lookback = 24 months (2 years of history)
# MAX_PRED = 3              # forecast horizon = next 3 months (multi-horizon)
# MIN_ENCODER = 12
#
# VAL_START = "2024-01-15"  # validation year = 2024
# TEST_START = "2025-01-15"  # test = 2025-01 .. 2026-08 (final 20 months, never seen in training)
#
# BATCH = 128
# MAX_EPOCHS = 10
# PATIENCE = 6               # >= reduce_on_plateau patience (4), so the LR decay gets a chance
# HIDDEN = 64               # ablated from 128 (h128 run had worse RMSE/tails)
# HEADS = 4
# SEED = 42
#
# DROPOUT = 0.25            # raised from 0.10 - val loss was blowing up after epoch 3 (overfit)
# LEARNING_RATE = 5e-4      # kept: 1e-3 ablated, worse on every metric (18.92 vs 17.66 pooled MAE)
# WEIGHT_DECAY = 1e-4       # AdamW-style L2 on weights
# TARGET_TRANSFORM = None   # raw NPR (ablated from "log": log-space training inflated MAE)
#
# SMOKE_SERIES = 30         # smoke mode: only 30 longest series
# SMOKE_EPOCHS = 2
#
# # TFT variable groups (single source of truth, used in train + eval + forecast)
# STATIC_CAT = ["market", "commodity", "category", "admin1", "unit"]
# STATIC_REAL = ["latitude", "longitude"]
# KNOWN_CAT = ["month"]
# # calendar features that are knowable for future months (Nepal agriculture/festival calendar)
# KNOWN_REAL = ["time_norm", "month_sin", "month_cos", "year_norm",
#               "monsoon", "harvest", "lean_season", "festival"]
# # history-only features (encoder side - never fed to the decoder, so no leakage)
# UNKNOWN_REAL = ["price", "log_price", "lag_1", "lag_3", "lag_12",
#                 "roll_mean_3", "roll_mean_12", "roll_std_12", "vol_ratio",
#                 "mom_pct", "yoy_pct", "dev_roll12",
#                 "nat_index", "comm_index", "is_imputed"]
# NEEDED = (["series_id", "time_idx"] + STATIC_CAT + STATIC_REAL
#           + KNOWN_CAT + KNOWN_REAL + UNKNOWN_REAL)
#
#
# # ======================================================================================
# # 1. LOAD + EDA
# # ======================================================================================
# def find_csv():
#     here = Path(__file__).resolve().parent
#     for c in CSV_CANDIDATES:
#         for p in (Path(c), here / c):
#             if p.exists():
#                 return str(p)
#     raise FileNotFoundError(f"CSV not found. Tried: {CSV_CANDIDATES}")
#
#
# def load_and_report(csv_path):
#     df = pd.read_csv(csv_path, parse_dates=["date"])
#     print("=" * 70)
#     print("DATASET ANALYSIS - WFP Nepal food prices")
#     print("=" * 70)
#     print(f"file            : {csv_path}")
#     print(f"shape           : {df.shape}")
#     print(f"date range      : {df['date'].min().date()} -> {df['date'].max().date()}")
#     print("frequency       : MONTHLY (observed on the 15th)")
#     print(f"markets         : {df['market'].nunique()}  | commodities: {df['commodity'].nunique()} "
#           f"| units: {sorted(df['unit'].unique().tolist())}")
#     print(f"total series (market x commodity x unit): {df.groupby(['market','commodity','unit']).ngroups}")
#     print(f"price (NPR)     : min {df['price'].min():.1f} | median {df['price'].median():.1f} "
#           f"| mean {df['price'].mean():.1f} | max {df['price'].max():.1f}")
#     print(f"missing values  : {int(df.isna().sum().sum())}")
#     print("\nrows per year (note: dense from 2019, sparse before):")
#     print(df["date"].dt.year.value_counts().sort_index().to_string())
#     print("\nTop commodities by rows:")
#     print(df["commodity"].value_counts().head(10).to_string())
#     return df
#
#
# # ======================================================================================
# # 2. PREPROCESSING - one clean monthly panel per series
# # ======================================================================================
# def preprocess(df):
#     df = df.sort_values(["market", "commodity", "unit", "date"]).copy()
#     df["series_id"] = df["market"] + "__" + df["commodity"] + "__" + df["unit"]
#
#     # --- keep series long enough to train on ---
#     counts = df.groupby("series_id").size()
#     keep = counts[counts >= MIN_SERIES_LEN].index
#     print(f"\nseries total: {len(counts)} | kept (>={MIN_SERIES_LEN} months): {len(keep)} "
#           f"| dropped short series: {len(counts) - len(keep)}")
#     df = df[df["series_id"].isin(keep)].copy()
#     df["ym"] = df["date"].dt.to_period("M")
#
#     # --- complete each series to a contiguous monthly grid ---
#     # Short gaps (1-2 months, e.g. a missed collection) are linearly interpolated.
#     # Long gaps (e.g. the 2017-05..2018-03 collection hole) SPLIT the series into
#     # separate segments instead of inventing a year of fake prices.
#     segments, short_filled, n_splits = [], 0, 0
#     statics = ["series_id", "market", "commodity", "unit", "category",
#                "admin1", "admin2", "latitude", "longitude"]
#
#     def finalize(piece, new_sid):
#         piece = piece.copy()
#         piece["is_imputed"] = piece["price"].isna().astype(np.int8)
#         piece["price"] = piece["price"].interpolate(limit_direction="both")
#         piece["series_id"] = new_sid
#         return piece.reset_index(drop=True)
#
#     for sid, g in df.groupby("series_id"):
#         g = g.sort_values("date")
#         full_idx = pd.period_range(g["ym"].min(), g["ym"].max(), freq="M")
#         g = g.set_index("ym").reindex(full_idx)
#         for c in statics:
#             g[c] = g[c].ffill().bfill()
#         g["date"] = full_idx.to_timestamp() + pd.offsets.Day(14)  # monthly 15th
#         g["ym"] = full_idx
#         miss = g["price"].isna().to_numpy()
#         # find NaN runs
#         runs, i, n = [], 0, len(g)
#         while i < n:
#             if miss[i]:
#                 j = i
#                 while j + 1 < n and miss[j + 1]:
#                     j += 1
#                 runs.append((i, j))
#                 i = j + 1
#             else:
#                 i += 1
#         cuts = [(a, b) for (a, b) in runs if (b - a + 1) > 2]
#         short_filled += sum((b - a + 1) for (a, b) in runs if (b - a + 1) <= 2)
#         if not cuts:
#             seg = finalize(g, sid)
#             if len(seg) >= MIN_SERIES_LEN:
#                 segments.append(seg)
#         else:
#             n_splits += 1
#             bounds, start = [], 0
#             for (a, b) in cuts:
#                 bounds.append((start, a))
#                 start = b + 1
#             bounds.append((start, n))
#             k = 0
#             for (a, b) in bounds:
#                 if b - a >= MIN_SERIES_LEN:
#                     k += 1
#                     segments.append(finalize(g.iloc[a:b], f"{sid}__seg{k}"))
#     df = pd.concat(segments, ignore_index=True)
#     print(f"short gaps interpolated (<=2 mo): {short_filled} cells | "
#           f"series split at long gaps: {n_splits} | resulting segments: {df['series_id'].nunique()}")
#
#     # --- time index (global month counter) ---
#     df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
#     t0 = df["date"].min()
#     df["time_idx"] = ((df["date"].dt.year - t0.year) * 12
#                       + (df["date"].dt.month - t0.month)).astype(int)
#
#     # --- known future features (calendar - always knowable ahead) ---
#     m = df["date"].dt.month
#     df["month"] = m.astype(str).str.zfill(2)          # categorical -> embedding learns seasonality
#     df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
#     df["month_cos"] = np.cos(2 * np.pi * m / 12.0)
#     df["year_norm"] = (df["date"].dt.year - 2001) / 25.0
#     df["time_norm"] = df["time_idx"] / df["time_idx"].max()
#
#     # Nepal-specific calendar (deterministic -> known in the future):
#     #   monsoon  Jun-Sep : road damage / transport cost, vegetable supply drops
#     #   harvest  Oct-Dec (paddy) + Apr-May (wheat) : market glut, prices ease
#     #   lean     Mar-Jun : stocks run down before monsoon, prices firm up
#     #   festival Mar/Holi, Oct-Nov (Dashain+Tihar) : demand spike for staples/sweets
#     df["monsoon"] = m.isin([6, 7, 8, 9]).astype(np.float32)
#     df["harvest"] = m.isin([10, 11, 12, 4, 5]).astype(np.float32)
#     df["lean_season"] = m.isin([3, 4, 5, 6]).astype(np.float32)
#     df["festival"] = m.isin([3, 10, 11]).astype(np.float32)
#
#     # --- unknown features (derived from past price - encoder side only, no leakage) ---
#     grp = df.groupby("series_id")["price"]
#     df["lag_1"] = grp.shift(1)
#     df["lag_3"] = grp.shift(3)
#     df["lag_12"] = grp.shift(12)                      # same month last year (seasonality!)
#     df["roll_mean_3"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean())
#     df["roll_mean_12"] = grp.transform(lambda s: s.rolling(12, min_periods=1).mean())
#     lag_cols = ["lag_1", "lag_3", "lag_12"]
#     df[lag_cols] = df.groupby("series_id")[lag_cols].transform(lambda s: s.bfill().ffill())
#     df[lag_cols] = df[lag_cols].fillna(df["price"].median())
#
#     # --- domain-specific engineered features (all history-only) ---
#     df["log_price"] = np.log1p(df["price"])           # multiplicative structure of prices
#     df["mom_pct"] = df["price"] / df["lag_1"] - 1.0   # 1-month momentum
#     df["yoy_pct"] = df["price"] / df["lag_12"] - 1.0  # YoY food inflation of this item
#     df["dev_roll12"] = df["price"] / df["roll_mean_12"] - 1.0  # vs 12-mo mean (mean reversion)
#     sd = grp.transform(lambda s: s.rolling(3, min_periods=2).std())
#     df["roll_std_12"] = grp.transform(lambda s: s.rolling(12, min_periods=3).std())
#     eng = ["mom_pct", "yoy_pct", "dev_roll12", "roll_std_12"]
#     df[eng] = df.groupby("series_id")[eng].transform(lambda s: s.bfill().ffill())
#     df[eng] = df[eng].fillna(0.0)
#     df["vol_ratio"] = (sd / df["roll_std_12"].clip(lower=1e-6)).clip(-5, 5).fillna(0.0)
#
#     # cross-series context: national food-price level and own-category level per month
#     # (constant across series within a month -> encoder-only macro signal)
#     df["nat_index"] = df.groupby("date")["log_price"].transform("mean")
#     df["comm_index"] = df.groupby(["date", "category"])["log_price"].transform("mean")
#
#     assert df["price"].isna().sum() == 0, "price still has NaN!"
#     print(f"processed rows: {len(df)} | series: {df['series_id'].nunique()} | "
#           f"time_idx: 0..{df['time_idx'].max()}")
#     return df
#
#
# def chronological_split(df):
#     train_df = df[df["date"] < VAL_START].copy()
#     val_hist = df[df["date"] < TEST_START].copy()   # val windows + their encoder history
#     test_hist = df.copy()                            # test windows + their encoder history
#     val_start_idx = int(df.loc[df["date"] >= VAL_START, "time_idx"].min())
#     test_start_idx = int(df.loc[df["date"] >= TEST_START, "time_idx"].min())
#     print("\nCHRONOLOGICAL SPLIT (no shuffling, no leakage):")
#     print(f"  train : {train_df['date'].min().date()} .. {train_df['date'].max().date()}  ({len(train_df)} rows)")
#     print(f"  val   : {VAL_START} .. 2024-12-15")
#     print(f"  test  : {TEST_START} .. {test_hist['date'].max().date()}  (final holdout)")
#     print(f"  val_start_idx={val_start_idx} test_start_idx={test_start_idx}")
#     return train_df, val_hist, test_hist, val_start_idx, test_start_idx
#
#
# # ======================================================================================
# # 3. METRICS (pure numpy/sklearn - no torch needed)
# # ======================================================================================
# def reg_metrics(y_true, y_pred):
#     from sklearn.metrics import r2_score
#     y_true = np.asarray(y_true, dtype=float).ravel()
#     y_pred = np.asarray(y_pred, dtype=float).ravel()
#     mask = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true, y_pred = y_true[mask], y_pred[mask]
#     err = y_pred - y_true
#     mae = float(np.mean(np.abs(err)))
#     rmse = float(np.sqrt(np.mean(err ** 2)))
#     mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100)
#     r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float("nan")
#     return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2, "n": int(len(y_true))}
#
#
# def naive_baselines(df):
#     """One-step naive references on TEST rows: last-value and seasonal-12.
#
#     NOTE: this is a ROW-level reference (one row per series-month, n ~= 8987) while the
#     TFT is scored on forecast WINDOWS (n = 13497). Use aligned_baselines() for an
#     apples-to-apples comparison - this one is kept for prep mode / historic reports.
#     """
#     s = df.sort_values(["series_id", "date"]).copy()
#     s["naive_last"] = s.groupby("series_id")["price"].shift(1)
#     s["naive_seas12"] = s.groupby("series_id")["price"].shift(12)
#     t = s[s["date"] >= TEST_START].dropna(subset=["naive_last", "naive_seas12"])
#     out = {
#         "naive_last_value": reg_metrics(t["price"], t["naive_last"]),
#         "naive_seasonal_12": reg_metrics(t["price"], t["naive_seas12"]),
#     }
#     return out
#
#
# def aligned_baselines(df, index, horizon=MAX_PRED):
#     """Baselines on EXACTLY the same forecast windows as the TFT.
#
#     index.time_idx is the FIRST FORECAST month T (verified: rolling_one_step at h=1
#     reproduces persistence exactly), so for each window (series, T) and step h = 1..H
#     the predicted month is T + h - 1:
#
#       persistence      price(T-1)           history only   -> fair, main baseline
#       seasonal_naive   price(T+h-13)        history only   -> fair, captures yearly cycle
#       rolling_one_step price(T+h-2)         h=1 == persistence; h>1 peeks at actuals
#                                              (operational upper bar, not a fair
#                                              3-month shot - labelled as such)
#
#     Fixes the old discrepancy where naive metrics used a different row set
#     (n=8987, h1 only) than the TFT windows (n=13497, all horizons).
#     """
#     lookup = {(str(s), int(t)): float(p) for s, t, p in
#               zip(df["series_id"], df["time_idx"], df["price"])}
#     T = np.asarray(index["time_idx"])
#     sids = [str(s) for s in index["series_id"]]
#     n = len(T)
#     pers = np.full((n, horizon), np.nan)
#     seas = np.full((n, horizon), np.nan)
#     roll = np.full((n, horizon), np.nan)
#     for i in range(n):
#         sid, t = sids[i], int(T[i])
#         for h in range(1, horizon + 1):        # h = 1..H, predicts month (T + h - 1)
#             pers[i, h - 1] = lookup.get((sid, t - 1), np.nan)
#             seas[i, h - 1] = lookup.get((sid, t + h - 13), np.nan)
#             roll[i, h - 1] = lookup.get((sid, t + h - 2), np.nan)
#     return {"persistence": pers, "seasonal_naive": seas, "rolling_one_step": roll}
#
#
# # ======================================================================================
# # 4. TFT (torch/pf imports are LOCAL so --mode prep works without them)
# # ======================================================================================
# try:
#     # Imported at MODULE level on purpose: Lightning checkpoints pickle the loss
#     # object, and a class defined inside a factory function is not picklable.
#     from pytorch_forecasting import QuantileLoss
# except ImportError:
#     try:
#         from pytorch_forecasting.metrics import QuantileLoss
#     except ImportError:
#         class QuantileLoss:  # stub: keeps `--mode prep` working without pf
#             def __init__(self, *args, **kwargs):
#                 raise ImportError("pytorch-forecasting is required for training")
#
#
# class WeightedQuantileLoss(QuantileLoss):
#     """Quantile loss with per-quantile emphasis.
#
#     We report the MEDIAN as the point forecast, so the median term gets the most
#     weight; the 0.02/0.98 tails only shape the prediction interval and are
#     down-weighted - letting them dominate the shared representation was pushing
#     the model towards extreme months, which inflated the pooled MAE.
#     Weights are re-normalised to mean 1 so the loss/gradient scale is unchanged.
#     """
#
#     WEIGHTS = [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]
#
#     def loss(self, y_pred, target):
#         import torch
#         losses = super().loss(y_pred, target)          # (batch, time, quantiles)
#         w = self.WEIGHTS[: losses.shape[-1]]
#         if len(w) != losses.shape[-1]:
#             w = [1.0] * losses.shape[-1]
#         w = torch.tensor(w, device=losses.device, dtype=losses.dtype)
#         w = w / w.mean()
#         return losses * w
#
#
# def accelerator_name():
#     """Prefer the NVIDIA GPU (RTX 4050 etc.), fall back to CPU. Enables TF32 on GPU."""
#     import torch
#     if torch.cuda.is_available():
#         torch.set_float32_matmul_precision("medium")  # TF32 tensor cores, big speedup
#         return "gpu"
#     return "cpu"
#
#
# def enable_determinism():
#     """Make training reproducible: same seed -> same weights -> same metrics.
#
#     Without this, GPU backward passes use atomic reductions that vary run to run;
#     two identical 4-epoch runs gave pooled MAE 17.66 and 20.29 (seed fixed).
#     warn_only=True: pf's TimeDistributedInterpolation (F.interpolate, linear) has no
#     deterministic backward on CUDA - strict mode raises, so we accept a small residual
#     run-to-run spread (~0.5 pooled MAE instead of ~2.6 without these flags).
#     """
#     import torch
#     torch.use_deterministic_algorithms(True, warn_only=True)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cuda.matmul.allow_tf32 = True   # TF32 kept: it is deterministic per op
#     torch.backends.cudnn.allow_tf32 = True
#
#
# def make_loss():
#     return WeightedQuantileLoss()
#
#
# def build_datasets(train_df, val_hist, test_hist, val_start_idx, test_start_idx,
#                    max_encoder=MAX_ENCODER, max_pred=MAX_PRED):
#     try:
#         from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
#     except ImportError:
#         from pytorch_forecasting import TimeSeriesDataSet
#         from pytorch_forecasting.data import GroupNormalizer
#     try:
#         from pytorch_forecasting.data import RobustScaler
#         scalers = {v: RobustScaler() for v in
#                    [c for c in (KNOWN_REAL + UNKNOWN_REAL + STATIC_REAL) if c != "price"]}
#     except Exception:
#         scalers = {}
#
#     common = dict(
#         time_idx="time_idx", target="price", group_ids=["series_id"],
#         max_encoder_length=max_encoder, max_prediction_length=max_pred,
#         min_encoder_length=min(MIN_ENCODER, max_encoder),
#         min_prediction_length=max_pred,
#         static_categoricals=STATIC_CAT,
#         static_reals=STATIC_REAL,
#         time_varying_known_categoricals=KNOWN_CAT,
#         time_varying_known_reals=KNOWN_REAL,
#         time_varying_unknown_reals=UNKNOWN_REAL,
#         # KEY FOR BEST METRICS: normalize each series separately
#         # (salt ~Rs.20 and chicken ~Rs.500 must not share one scale),
#         # optionally in log space so errors are multiplicative, not additive
#         target_normalizer=GroupNormalizer(groups=["series_id"],
#                                           transformation=TARGET_TRANSFORM),
#         scalers=scalers,
#         add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
#         allow_missing_timesteps=True,
#     )
#     training = TimeSeriesDataSet(train_df[NEEDED], **common)
#
#     def _from(data, start_idx):
#         try:
#             return TimeSeriesDataSet.from_dataset(
#                 training, data[NEEDED], predict=False,
#                 stop_randomization=True, min_prediction_idx=start_idx)
#         except TypeError:  # very old versions without min_prediction_idx
#             print("  (warning: min_prediction_idx unsupported, using sliced data)")
#             return TimeSeriesDataSet.from_dataset(
#                 training, data[NEEDED], predict=False, stop_randomization=True)
#
#     validation = _from(val_hist, val_start_idx)
#     test = _from(test_hist, test_start_idx)
#     print(f"\nwindows -> train: {len(training)} | val: {len(validation)} | test: {len(test)}")
#     return training, validation, test
#
#
# def unpack_predict_output(res):
#     """Unpack model.predict(...) robustly across library versions.
#
#     Newer versions return Prediction(output, x, index, decoder_lengths, y)
#     where output is an Output(...) namedtuple holding `.prediction`
#     of shape (N, horizon, n_quantiles). Returns dict with keys:
#     out (full model output, for plotting), raw (pred tensor), x, y, index.
#     """
#     out = raw = x = y = idx = None
#     if hasattr(res, "_fields"):  # Prediction namedtuple (current versions)
#         out = res.output
#         raw = out.prediction if hasattr(out, "prediction") else out
#         x = res.x
#         y = res.y
#         if isinstance(y, (list, tuple)):
#             y = y[0] if len(y) > 0 else None
#         idx = res.index
#     else:  # fallback: plain tuple of mixed items (older versions)
#         items = list(res) if isinstance(res, (list, tuple)) else [res]
#         for it in items:
#             if isinstance(it, dict):
#                 x = it
#             elif isinstance(it, pd.DataFrame):
#                 idx = it
#             elif hasattr(it, "prediction"):
#                 out, raw = it, it.prediction
#             elif hasattr(it, "ndim") and int(it.ndim) == 3:
#                 raw = it
#             elif hasattr(it, "ndim") and int(it.ndim) == 2 and y is None:
#                 y = it
#         if out is None:
#             out = raw
#     return {"out": out, "raw": raw, "x": x, "y": y, "index": idx}
#
#
# def _to_price(values, target_scale, ref_median, apply=None):
#     """Inverse GroupNormalizer(z) = center + scale*z -> NPR (undo target transform too).
#
#     apply=None auto-detects from the value magnitude (version-proof):
#       * pf >= 1.x predicts in real space already -> no-op
#       * older pf predicts in normalized space    -> undo the scaling
#     When apply is given (True/False) the decision is forced, which keeps
#     prediction and ground truth in the SAME space even if their medians differ.
#     """
#     import torch
#     v = torch.as_tensor(values).detach().float()
#     if apply is None:
#         # normalized z-values sit near 0 while real prices sit near ref_median;
#         # compare RELATIVELY so cheap commodities (ref ~15) are not misdetected
#         med = abs(float(v.median().item()))
#         apply = ref_median > 0 and med < 0.5 * ref_median
#     if apply:
#         ts = torch.as_tensor(target_scale).detach().float()
#         center = ts[..., 0]
#         scale = ts[..., 1].clamp_min(1e-8)
#         shape = [v.shape[0]] + [1] * (v.ndim - 1)
#         v = v * scale.view(shape) + center.view(shape)
#         if TARGET_TRANSFORM == "log":          # z -> log(price) -> price
#             v = torch.exp(v)
#         elif TARGET_TRANSFORM == "log1p":
#             v = torch.expm1(v)
#     return v.cpu().numpy()
#
#
# def denorm(values, target_scale, ref_median):
#     """Backwards-compatible wrapper around _to_price (auto-detect mode)."""
#     return _to_price(values, target_scale, ref_median, apply=None)
#
#
# def _prediction_tensor(out):
#     """Pull the prediction tensor out of a model forward output (any pf version)."""
#     import torch
#     if out is None:
#         return None
#     if torch.is_tensor(out):
#         return out
#     if isinstance(out, dict):
#         for k in ("prediction", "output", "out"):
#             if k in out:
#                 t = _prediction_tensor(out[k])
#                 if t is not None:
#                     return t
#         for v in out.values():
#             t = _prediction_tensor(v)
#             if t is not None:
#                 return t
#         return None
#     if isinstance(out, (tuple, list)):
#         for v in out:
#             t = _prediction_tensor(v)
#             if t is not None:
#                 return t
#         return None
#     for k in ("prediction", "output"):
#         v = getattr(out, k, None)
#         if v is not None and v is not out:
#             t = _prediction_tensor(v)
#             if t is not None:
#                 return t
#     return None
#
#
# def _num(v, nd=4):
#     """Format a metric for the report ('-' when missing / non-finite)."""
#     if v is None:
#         return "-"
#     try:
#         v = float(v)
#     except (TypeError, ValueError):
#         return "-"
#     if not np.isfinite(v):
#         return "-"
#     return f"{v:.{nd}f}"
#
#
# def _cm_first(cm, *names):
#     """First matching value in trainer.callback_metrics (exact, then prefix)."""
#     for n in names:
#         if n in cm:
#             try:
#                 return float(cm[n])
#             except (TypeError, ValueError):
#                 pass
#     for n in names:
#         for k, v in cm.items():
#             if k.startswith(n):
#                 try:
#                     return float(v)
#                 except (TypeError, ValueError):
#                     pass
#     return None
#
#
# def _cm_lr(cm):
#     for k, v in cm.items():
#         if k.startswith("lr") or "learning_rate" in k:
#             try:
#                 return float(v)
#             except (TypeError, ValueError):
#                 pass
#     return None
#
#
# def make_epoch_reporter(pl, out_dir, max_epochs, ref_median=None, patience=PATIENCE):
#     """Lightning callback: live per-epoch report on stdout + training_history.csv."""
#     Callback = pl.Callback
#
#     class EpochReporter(Callback):
#         def __init__(self):
#             super().__init__()
#             self.out_dir = out_dir
#             self.max_epochs = int(max_epochs)
#             self.patience = int(patience)
#             self.ref_median = float(ref_median) if ref_median else 100.0
#             self.history = []
#             self._hook = None
#             self._pred = None
#             self._apply = None
#             self._ae = 0.0
#             self._se = 0.0
#             self._n = 0
#             self._t_epoch = time.time()
#             self._best_path = ""
#             self._best_epoch = None
#             self._warned = False
#
#         # ---- capture the forward pass during validation (costs nothing extra) ----
#         def on_validation_start(self, trainer, pl_module):
#             if trainer.sanity_checking:
#                 return
#             self._ae = self._se = 0.0
#             self._n = 0
#             self._pred = None
#             self._apply = None
#             if self._hook is None:
#                 def _capture(module, inputs, output):
#                     self._pred = _prediction_tensor(output)
#                 try:
#                     self._hook = pl_module.register_forward_hook(_capture)
#                 except Exception:
#                     self._hook = None
#
#         def on_validation_end(self, trainer, pl_module):
#             if self._hook is not None:
#                 try:
#                     self._hook.remove()
#                 except Exception:
#                     pass
#                 self._hook = None
#             self._pred = None
#
#         def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
#                                     batch_idx, dataloader_idx=0):
#             """Accumulate squared/absolute errors -> epoch val MAE + RMSE in NPR."""
#             try:
#                 if trainer.sanity_checking or self._pred is None:
#                     return
#                 if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
#                     return
#                 x, y = batch
#                 if isinstance(y, (tuple, list)):     # pf returns (target, weight)
#                     y = y[0]
#                 if y is None or not hasattr(y, "detach") or not hasattr(x, "get"):
#                     return
#                 scale = x.get("target_scale")
#                 if scale is None:
#                     return
#                 p = self._pred.detach().float().cpu().numpy()
#                 t = y.detach().float().cpu().numpy()
#                 if p.ndim == t.ndim + 1:             # (B,H,Q) -> median quantile
#                     q = p.shape[-1]
#                     p = p[..., 3] if q >= 7 else p[..., q // 2]
#                 if p.ndim != t.ndim or p.shape[0] != t.shape[0]:
#                     return
#                 # one decision for BOTH sides -> same space guaranteed
#                 if self._apply is None:
#                     med = abs(float(np.median(t)))
#                     self._apply = bool(self.ref_median) and med < 0.5 * self.ref_median
#                 e = (_to_price(p, scale, self.ref_median, self._apply)
#                      - _to_price(t, scale, self.ref_median, self._apply)).ravel()
#                 e = e[np.isfinite(e)]
#                 if e.size == 0:
#                     return
#                 self._ae += float(np.abs(e).sum())
#                 self._se += float((e ** 2).sum())
#                 self._n += int(e.size)
#             except Exception:
#                 if not self._warned:
#                     self._warned = True
#                     print("  (note: per-epoch MAE/RMSE unavailable - loss-only report)",
#                           flush=True)
#
#         def on_train_epoch_start(self, trainer, pl_module):
#             self._t_epoch = time.time()
#
#         def on_validation_epoch_end(self, trainer, pl_module):
#             if trainer.sanity_checking:
#                 return
#             cm = trainer.callback_metrics
#             epoch = int(trainer.current_epoch) + 1
#             row = {
#                 "epoch": epoch,
#                 "train_loss": _cm_first(cm, "train_loss", "train_epoch_loss", "training_loss"),
#                 "val_loss": _cm_first(cm, "val_loss", "validation_loss"),
#                 "val_MAE": (self._ae / self._n) if self._n else None,
#                 "val_RMSE": (self._se / self._n) ** 0.5 if self._n else None,
#                 "lr": _cm_lr(cm),
#                 "seconds": round(time.time() - self._t_epoch, 1),
#             }
#             ckpt = getattr(trainer, "checkpoint_callback", None)
#             path = getattr(ckpt, "best_model_path", "") or ""
#             row["is_best"] = bool(path) and path != self._best_path
#             if row["is_best"]:
#                 self._best_path = path
#                 self._best_epoch = epoch
#             row["best_epoch"] = self._best_epoch
#             row["stale_epochs"] = epoch - (self._best_epoch or epoch)
#             self.history.append(row)
#             print(_epoch_line(row, self.max_epochs, self.patience), flush=True)
#             try:
#                 self._write()
#             except Exception:
#                 pass
#
#         # ---- persistence (written after EVERY epoch so the report survives a crash) ----
#         def _write(self):
#             if not self.history:
#                 return
#             pd.DataFrame(self.history).to_csv(
#                 os.path.join(self.out_dir, "training_history.csv"), index=False)
#             with open(os.path.join(self.out_dir, "training_history.json"), "w") as fh:
#                 json.dump(self.history, fh, indent=2)
#
#         def save(self):
#             if not self.history:
#                 print("(no epoch history recorded)")
#                 return self.history
#             self._write()
#             print(f"saved {os.path.join(self.out_dir, 'training_history.csv')} "
#                   f"({len(self.history)} epochs)")
#             return self.history
#
#         def summary(self):
#             rows = [r for r in self.history if r.get("val_loss") is not None]
#             if not rows:
#                 return ""
#             best = min(rows, key=lambda r: r["val_loss"])
#             last = self.history[-1]
#             tail = ""
#             if last["stale_epochs"] >= self.patience:
#                 tail = f" | early stopping after {self.patience} stale epochs"
#             return (f"epochs run: {last['epoch']}/{self.max_epochs} | "
#                     f"best epoch: {best['epoch']} (val_loss={best['val_loss']:.4f}) | "
#                     f"val MAE={_num(best['val_MAE'], 1)} RMSE={_num(best['val_RMSE'], 1)} NPR"
#                     f"{tail}")
#
#     return EpochReporter()
#
#
# def _epoch_line(r, max_epochs, patience):
#     parts = [f"[EPOCH {r['epoch']:02d}/{max_epochs:02d}]"]
#     parts.append(f"train_loss={_num(r.get('train_loss'))}")
#     parts.append(f"val_loss={_num(r.get('val_loss'))}")
#     parts.append(f"val_MAE={_num(r.get('val_MAE'), 1)}")
#     parts.append(f"val_RMSE={_num(r.get('val_RMSE'), 1)}")
#     if r.get("lr") is not None:
#         parts.append(f"lr={r['lr']:.5f}")
#     parts.append(f"{r.get('seconds', 0):.1f}s")
#     if r.get("is_best"):
#         parts.append("*new best*")
#     if r.get("best_epoch"):
#         parts.append(f"best@{r['best_epoch']} stale={r.get('stale_epochs', 0)}/{patience}")
#     return "  ".join(parts)
#
#
# def train_tft(training, validation, out_dir, max_epochs=MAX_EPOCHS,
#               batch=BATCH, hidden=HIDDEN, patience=PATIENCE, ref_median=None):
#     try:
#         import lightning.pytorch as pl
#         from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
#         from lightning.pytorch.loggers import CSVLogger
#     except ImportError:
#         import pytorch_lightning as pl
#         from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
#         from pytorch_lightning.loggers import CSVLogger
#     try:
#         from pytorch_forecasting import TemporalFusionTransformer
#     except ImportError:
#         from pytorch_forecasting import TemporalFusionTransformer
#
#     pl.seed_everything(SEED, workers=True)
#     enable_determinism()
#     import torch as _t
#     print(f"determinism: algorithms={_t.are_deterministic_algorithms_enabled()} "
#           f"cudnn_deterministic={_t.backends.cudnn.deterministic} "
#           f"benchmark={_t.backends.cudnn.benchmark} "
#           f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
#     train_dl = training.to_dataloader(train=True, batch_size=batch, num_workers=0)
#     val_dl = validation.to_dataloader(train=False, batch_size=batch * 2, num_workers=0)
#
#     loss = make_loss()
#     tft = TemporalFusionTransformer.from_dataset(
#         training,
#         hidden_size=hidden,
#         lstm_layers=2,
#         attention_head_size=HEADS,
#         dropout=DROPOUT,
#         hidden_continuous_size=32,
#         output_size=7,                    # 7 quantiles [0.02..0.98], median = index 3
#         loss=loss,
#         learning_rate=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#         log_interval=20,
#         reduce_on_plateau_patience=4,
#     )
#     print(f"\nTFT params: hidden={hidden} heads={HEADS} dropout={DROPOUT} "
#           f"lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY} | "
#           f"target={TARGET_TRANSFORM or 'raw'} | "
#           f"quantiles={tft.loss.quantiles if hasattr(tft.loss, 'quantiles') else 7}")
#
#     print("\n" + "=" * 70)
#     print("LIVE EPOCH REPORT - one line printed after every epoch")
#     print("=" * 70)
#     print(f"  train windows: {len(training)} | batches/epoch: {int(np.ceil(len(training) / batch))}"
#           f" | val windows: {len(validation)} | batch: {batch}")
#     print(f"  history file : {os.path.join(out_dir, 'training_history.csv')} (written at end)")
#     print("=" * 70, flush=True)
#
#     ckpt_dir = os.path.join(out_dir, "checkpoints")
#     os.makedirs(ckpt_dir, exist_ok=True)
#     reporter = make_epoch_reporter(pl, out_dir, max_epochs, ref_median, patience)
#     early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
#     lr_monitor = LearningRateMonitor()
#     ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", save_top_k=1, mode="min")
#     callbacks = [reporter, early_stop, lr_monitor, ckpt_cb]
#     try:
#         import torch
#         has_gpu = torch.cuda.is_available()
#     except Exception:
#         has_gpu = False
#     logger = CSVLogger(save_dir=out_dir, name="training_logs")
#     trainer = pl.Trainer(
#         max_epochs=max_epochs, accelerator="gpu" if has_gpu else "cpu", devices=1,
#         gradient_clip_val=0.1, callbacks=callbacks, enable_model_summary=True,
#         enable_progress_bar=True, logger=logger,
#         # NOTE: Trainer(deterministic=True) is strict here -> crashes on pf's
#         # TimeDistributedInterpolation backward; enable_determinism() covers the flags.
#     )
#     trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
#
#     history = reporter.save()
#     summary = reporter.summary()
#     if summary:
#         print(f"\nTRAINING SUMMARY: {summary}", flush=True)
#
#     best_path = ckpt_cb.best_model_path
#     print(f"\nbest checkpoint: {best_path}")
#     best = TemporalFusionTransformer.load_from_checkpoint(best_path)
#     return best, trainer, history
#
#
# def evaluate(best, test, df, out_dir):
#     test_dl = test.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=0)
#     try:
#         res = best.predict(test_dl, mode="raw", return_x=True, return_y=True,
#                            return_index=True,
#                            trainer_kwargs={"accelerator": accelerator_name(), "devices": 1})
#     except TypeError:
#         res = best.predict(test_dl, mode="raw", return_x=True, return_y=True,
#                            return_index=True)
#     d = unpack_predict_output(res)
#     raw, x, y, idx = d["raw"], d["x"], d["y"], d["index"]
#     if y is None and x is not None and "decoder_target" in x:
#         y = x["decoder_target"]  # fallback: targets straight from the batch
#     assert raw is not None and y is not None and x is not None, "unexpected predict() output"
#
#     ref_med = float(df.loc[df["date"] >= TEST_START, "price"].median())
#     scale = x["target_scale"]
#     pred_q = denorm(raw, scale, ref_med)          # (N, P, 7)
#     y_true = denorm(y, scale, ref_med)            # (N, P)
#     pred_med = pred_q[:, :, 3]                    # median quantile
#     pred_med = np.clip(pred_med, 0, None)
#
#     P = pred_med.shape[1]
#     metrics = {"overall_pooled": reg_metrics(y_true, pred_med)}
#     for h in range(P):
#         metrics[f"horizon_h{h+1}"] = reg_metrics(y_true[:, h], pred_med[:, h])
#
#     # ---- fair baselines: SAME windows, SAME rows, SAME information set ----
#     if idx is not None and len(idx) == len(y_true):
#         for name, arr in aligned_baselines(df, idx, P).items():
#             metrics[f"baseline_{name}"] = {
#                 **{f"horizon_h{h+1}": reg_metrics(y_true[:, h], arr[:, h])
#                    for h in range(P)},
#                 "overall_pooled": reg_metrics(y_true, arr),
#             }
#         pers = metrics["baseline_persistence"]
#         if pers["overall_pooled"]["MAE"] > 0:
#             skill = {f"horizon_h{h+1}": 1.0 - metrics[f"horizon_h{h+1}"]["MAE"]
#                      / pers[f"horizon_h{h+1}"]["MAE"] for h in range(P)}
#             skill["overall_pooled"] = (1.0 - metrics["overall_pooled"]["MAE"]
#                                        / pers["overall_pooled"]["MAE"])
#             metrics["skill_vs_persistence"] = skill
#     else:
#         print("(warning: window index unavailable - aligned baselines skipped)")
#     metrics["naive_row_level_reference"] = naive_baselines(df)  # legacy: different n, h1 only
#
#     print("\n" + "=" * 78)
#     print("TEST METRICS (2025-01 .. 2026-08 holdout, prices in NPR)")
#     print("=" * 78)
#     for line in _metrics_lines(metrics):
#         print(line)
#     print("=" * 78)
#     print("skill > 0 -> TFT beats persistence on that horizon (same rows, same info).")
#     print("rolling_one_step peeks at actuals for h>1 - it is an upper bar, not a rival.")
#     print(f"Overall_pooled covers all {P} horizons, so it is naturally a bit higher.")
#
#     np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"),
#                         y_true=y_true, pred_median=pred_med, pred_q10=pred_q[:, :, 1],
#                         pred_q90=pred_q[:, :, 5],
#                         time_idx=np.asarray(idx["time_idx"]) if idx is not None else [],
#                         series_id=np.asarray(idx["series_id"].astype(str))
#                         if idx is not None else [])
#     with open(os.path.join(out_dir, "metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)
#
#     # --- plots ---
#     # 1) actual vs predicted scatter
#     plt.figure(figsize=(7, 6))
#     plt.scatter(y_true.ravel(), pred_med.ravel(), s=3, alpha=0.3)
#     lo, hi = float(y_true.min()), float(y_true.max())
#     plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
#     m = metrics["overall_pooled"]
#     plt.title(f"TFT test: actual vs predicted (R2={m['R2']:.3f}, MAE={m['MAE']:.1f} NPR)")
#     plt.xlabel("Actual price (NPR)")
#     plt.ylabel("Predicted price (NPR)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=150)
#     plt.close()
#
#     # 2) sample forecast windows (history + median + intervals)
#     try:
#         for i in [0, len(test) // 2, len(test) - 1]:
#             try:
#                 fig = best.plot_prediction(x, d["out"], idx=i, add_loss_to_title=True)
#             except Exception:
#                 fig = best.plot_prediction(x, raw, idx=i, add_loss_to_title=True)
#             fig.savefig(os.path.join(out_dir, f"sample_forecast_{i}.png"), dpi=150)
#             plt.close(fig)
#         print("saved sample forecast plots")
#     except Exception as e:
#         print(f"(sample plots skipped: {e})")
#     return metrics
#
#
# def per_commodity_breakdown(best, training, test_hist, test_start_idx, out_dir, top_n=8):
#     """Evaluate TFT separately per commodity (great for the report)."""
#     from pytorch_forecasting import TimeSeriesDataSet
#     top = test_hist[test_hist["date"] >= TEST_START]["commodity"].value_counts().head(top_n).index.tolist()
#     rows = []
#     print(f"\nPer-commodity test MAE/RMSE (top {top_n}):")
#     for c in top:
#         try:
#             sub = test_hist[test_hist["commodity"] == c]
#             ds = TimeSeriesDataSet.from_dataset(training, sub[NEEDED], predict=False,
#                                                 stop_randomization=True,
#                                                 min_prediction_idx=test_start_idx)
#             if len(ds) == 0:
#                 continue
#             dl = ds.to_dataloader(train=False, batch_size=512, num_workers=0)
#             res = best.predict(dl, mode="raw", return_x=True, return_y=True)
#             dd = unpack_predict_output(res)
#             raw, x, y = dd["raw"], dd["x"], dd["y"]
#             if y is None and x is not None and "decoder_target" in x:
#                 y = x["decoder_target"]
#             ref = float(sub.loc[sub["date"] >= TEST_START, "price"].median())
#             pm = np.clip(denorm(raw, x["target_scale"], ref)[:, :, 3], 0, None)
#             yt = denorm(y, x["target_scale"], ref)
#             mm = reg_metrics(yt, pm)
#             rows.append({"commodity": c, **mm})
#             print(f"  {c:22s} MAE={mm['MAE']:.2f} RMSE={mm['RMSE']:.2f} R2={mm['R2']:.4f} (n={mm['n']})")
#         except Exception as e:
#             print(f"  {c:22s} skipped ({e})")
#     if rows:
#         pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_commodity_metrics.csv"), index=False)
#     return rows
#
#
# def future_forecast(best, training, df, out_dir, max_encoder=MAX_ENCODER, max_pred=MAX_PRED):
#     """Forecast the next `max_pred` months beyond the last observed date, per active series.
#
#     A series is active when it is still reported close to the global end date;
#     dead segments (split off at a collection gap years ago) are skipped, because
#     their "next 3 months" would land in the past.
#     """
#     from pytorch_forecasting import TimeSeriesDataSet
#     tmax = int(df["time_idx"].max())
#     last_dates = df.groupby("series_id")["date"].max()
#     cutoff = last_dates.max() - pd.DateOffset(months=max_pred)
#     active = last_dates[last_dates >= cutoff].index
#     if len(active) < len(last_dates):
#         print(f"  skipping {len(last_dates) - len(active)} ended series "
#               f"(last obs before {cutoff.date()}) -> forecasting {len(active)} active series")
#     parts = []
#     for sid, g in df[df["series_id"].isin(active)].groupby("series_id"):
#         g = g.sort_values("date").tail(max_encoder).copy()
#         last = g.iloc[-1]
#         fut = []
#         for k in range(1, max_pred + 1):
#             idx = int(last["time_idx"]) + k
#             d = last["date"] + pd.DateOffset(months=k)
#             row = {c: last[c] for c in NEEDED}   # defaults: history-only lag/roll/* placeholders
#             row.update({
#                 "series_id": sid, "time_idx": idx, "date": d,
#                 "price": float(last["price"]),  # placeholder; model predicts these
#                 # calendar features MUST be recomputed for the future month, not copied
#                 "month": f"{d.month:02d}",
#                 "month_sin": float(np.sin(2 * np.pi * d.month / 12)),
#                 "month_cos": float(np.cos(2 * np.pi * d.month / 12)),
#                 "year_norm": (d.year - 2001) / 25.0,
#                 "time_norm": idx / tmax,
#                 "monsoon": float(d.month in (6, 7, 8, 9)),
#                 "harvest": float(d.month in (10, 11, 12, 4, 5)),
#                 "lean_season": float(d.month in (3, 4, 5, 6)),
#                 "festival": float(d.month in (3, 10, 11)),
#                 "is_imputed": 0,
#             })
#             fut.append(row)
#         parts.append(pd.concat([g[NEEDED + ["date"]], pd.DataFrame(fut)], ignore_index=True))
#     fut_input = pd.concat(parts, ignore_index=True)
#
#     fds = TimeSeriesDataSet.from_dataset(training, fut_input[NEEDED],
#                                          predict=True, stop_randomization=True)
#     fdl = fds.to_dataloader(train=False, batch_size=512, num_workers=0)
#     res = best.predict(fdl, mode="raw", return_x=True, return_index=True)
#     dd = unpack_predict_output(res)
#     raw, x, index = dd["raw"], dd["x"], dd["index"]
#     assert raw is not None and index is not None, "future predict() output unexpected"
#     print(f"  future index columns: {list(index.columns)}")
#
#     ref_med = float(df["price"].median())
#     q = denorm(raw, x["target_scale"], ref_med)
#     id_col = "series_id" if "series_id" in index.columns else index.columns[0]
#     last_date = df.groupby("series_id")["date"].max().to_dict()
#     meta = df.drop_duplicates("series_id").set_index("series_id")[
#         ["market", "commodity", "unit"]].to_dict("index")
#
#     rows = []
#     for i in range(q.shape[0]):
#         sid = str(index[id_col].iloc[i])
#         for k in range(q.shape[1]):
#             d = last_date.get(sid, df["date"].max()) + pd.DateOffset(months=k + 1)
#             mi = meta.get(sid, {})
#             rows.append({
#                 "series_id": sid, "market": mi.get("market"), "commodity": mi.get("commodity"),
#                 "unit": mi.get("unit"), "horizon": k + 1, "date": d.date().isoformat(),
#                 "pred_q10": round(float(q[i, k, 1]), 2),
#                 "pred_median": round(float(np.clip(q[i, k, 3], 0, None)), 2),
#                 "pred_q90": round(float(q[i, k, 5]), 2),
#             })
#     out = pd.DataFrame(rows)
#     out.to_csv(os.path.join(out_dir, "future_forecast.csv"), index=False)
#     print(f"saved future_forecast.csv ({len(out)} rows = "
#           f"{out['series_id'].nunique()} series x {max_pred} months)")
#     print(out.head(6).to_string(index=False))
#     return out
#
#
# # ======================================================================================
# # 6. REPORTS (everything that was shown live is also written to disk)
# # ======================================================================================
# def _metric_row(name, m):
#     return (f"  {name:26s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
#             f"MAPE={m['MAPE_%']:.2f}%  R2={m['R2']:.4f}  n={m['n']}")
#
#
# def _is_metric(v):
#     return isinstance(v, dict) and "MAE" in v
#
#
# def _metrics_lines(metrics):
#     """Render a (possibly nested) metrics dict as printable lines.
#
#     Handles: metric dict -> one row | dict of metric dicts -> section |
#     dict of floats (skill scores) -> simple lines.
#     """
#     lines = []
#     for k, v in metrics.items():
#         if _is_metric(v):
#             lines.append(_metric_row(k, v))
#         elif isinstance(v, dict) and v:
#             lines.append(f"\n  {k}:")
#             for nk, nv in v.items():
#                 if _is_metric(nv):
#                     lines.append(_metric_row(nk, nv))
#                 elif isinstance(nv, dict) and nv:
#                     lines.append(f"    {nk}:")
#                     for nnk, nnv in nv.items():
#                         if _is_metric(nnv):
#                             lines.append(_metric_row("  " + nnk, nnv))
#                         else:
#                             lines.append(f"      {nnk}: {nnv}")
#                 else:
#                     try:
#                         lines.append(f"    {nk:24s} {float(nv):+.3f}")
#                     except (TypeError, ValueError):
#                         lines.append(f"    {nk}: {nv}")
#         else:
#             lines.append(f"  {k}: {v}")
#     return lines
#
#
# def print_final_summary(metrics):
#     """Compact final block - printed LAST so the run ends with the evaluated metrics."""
#     print("\n" + "=" * 78)
#     print("FINAL EVALUATED METRICS (test holdout 2025-01 .. 2026-08, prices in NPR)")
#     print("=" * 78)
#     for line in _metrics_lines(metrics):
#         print(line)
#     print("=" * 78, flush=True)
#
#
# def write_final_report(out_dir, mode, history, metrics, commodity_rows=None, config=None):
#     """Write final_report.txt: config + per-epoch table + all final metrics + file list."""
#     cfg = dict(lookback=MAX_ENCODER, horizon=MAX_PRED, batch=BATCH, hidden=HIDDEN,
#                heads=HEADS, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
#                dropout=DROPOUT, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
#                target_transform=TARGET_TRANSFORM or "none", loss="WeightedQuantileLoss",
#                lookback_note="", epochs_run=(history[-1]["epoch"] if history else 0))
#     if config:
#         cfg.update(config)
#     L = []
#     add = L.append
#     add("=" * 78)
#     add("NEPAL FOOD PRICE FORECASTING - TFT FINAL REPORT")
#     add(f"generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     add(f"mode      : {mode}")
#     add(f"output dir: {out_dir}")
#     add("=" * 78)
#
#     add("")
#     add("[CONFIG]")
#     add(f"  lookback={cfg['lookback']} mo | horizon={cfg['horizon']} mo | batch={cfg['batch']} "
#         f"| hidden={cfg['hidden']} | heads={cfg['heads']}")
#     add(f"  max_epochs={cfg['max_epochs']} | epochs_run={cfg['epochs_run']} | "
#         f"early_stop_patience={cfg['patience']} | seed={cfg['seed']}")
#     add(f"  split: train < {VAL_START} | val = 2024 | test >= {TEST_START} (holdout)")
#     add(f"  dropout={cfg['dropout']} | lr={cfg['lr']} | weight_decay={cfg['weight_decay']}")
#     add(f"  target = {cfg['target_transform']} | loss = weighted QuantileLoss "
#         f"(median emphasised) over 7 quantiles; reported prediction = median")
#
#     add("")
#     add(f"[1] PER-EPOCH TRAINING REPORT ({len(history)} epochs, live-printed during run)")
#     if history:
#         hdr = (f"{'epoch':>5} {'train_loss':>11} {'val_loss':>10} {'val_MAE':>10} "
#                f"{'val_RMSE':>10} {'lr':>9} {'sec':>7}  note")
#         add("  " + hdr)
#         add("  " + "-" * (len(hdr) + 2))
#         for r in history:
#             note = []
#             if r.get("is_best"):
#                 note.append("*new best*")
#             if r.get("best_epoch"):
#                 note.append(f"best@{r['best_epoch']}")
#                 note.append(f"stale={r.get('stale_epochs', 0)}/{cfg['patience']}")
#             add(f"  {r['epoch']:>5} {_num(r.get('train_loss')):>11} "
#                 f"{_num(r.get('val_loss')):>10} {_num(r.get('val_MAE'), 1):>10} "
#                 f"{_num(r.get('val_RMSE'), 1):>10} {_num(r.get('lr'), 5):>9} "
#                 f"{_num(r.get('seconds'), 1):>7}  {' '.join(note)}".rstrip())
#         rows = [r for r in history if r.get("val_loss") is not None]
#         if rows:
#             best = min(rows, key=lambda r: r["val_loss"])
#             add("")
#             add(f"  best epoch {best['epoch']}: val_loss={best['val_loss']:.4f} "
#                 f"val_MAE={_num(best.get('val_MAE'), 1)} "
#                 f"val_RMSE={_num(best.get('val_RMSE'), 1)} NPR")
#     else:
#         add("  (no epochs recorded - prep mode)")
#
#     add("")
#     add("[2] FINAL TEST METRICS (holdout 2025-01 .. 2026-08, prices in NPR)")
#     if metrics:
#         for line in _metrics_lines(metrics):
#             add(line)
#         add("")
#         add("  baseline_* = same windows / same rows / same information as the TFT")
#         add("              (fixes the old naive-vs-TFT row mismatch).")
#         add("  skill_vs_persistence > 0 -> TFT beats persistence on that horizon.")
#         add("  rolling_one_step peeks at actuals for h>1: an upper bar, not a rival.")
#         add("  naive_row_level_reference = legacy row-level h1 numbers (different n).")
#     else:
#         add("  (no metrics - prep mode)")
#
#     if commodity_rows:
#         add("")
#         add(f"[3] PER-COMMODITY BREAKDOWN ({len(commodity_rows)} commodities)")
#         add(f"  {'commodity':22s} {'MAE':>9} {'RMSE':>9} {'MAPE%':>8} {'R2':>9} {'n':>7}")
#         for r in commodity_rows:
#             add(f"  {str(r['commodity'])[:22]:22s} {r['MAE']:>9.2f} {r['RMSE']:>9.2f} "
#                 f"{r['MAPE_%']:>8.2f} {r['R2']:>9.4f} {r['n']:>7d}")
#
#     add("")
#     add("[4] FILES IN THIS RUN")
#     try:
#         for f in sorted(os.listdir(out_dir)):
#             add(f"  {f}")
#     except OSError:
#         add("  (unreadable)")
#
#     path = os.path.join(out_dir, "final_report.txt")
#     with open(path, "w", encoding="utf-8") as fh:
#         fh.write("\n".join(L) + "\n")
#     return path
#
#
# # ======================================================================================
# # MAIN
# # ======================================================================================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--mode", default="train", choices=["prep", "smoke", "train"])
#     args, _ = ap.parse_known_args()  # known_args so it also works in Colab cells
#
#     os.makedirs(OUT_DIR, exist_ok=True)
#     df_raw = load_and_report(find_csv())
#     df = preprocess(df_raw)
#     train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(df)
#
#     if args.mode == "prep":
#         df.to_csv(os.path.join(OUT_DIR, "processed_monthly_panel.csv"), index=False)
#         print(f"\nsaved {OUT_DIR}/processed_monthly_panel.csv")
#         print("Naive one-step references on test rows:")
#         print(json.dumps(naive_baselines(df), indent=2))
#         print("\nPREP OK. Next: python tft_nepal_food_price.py --mode smoke")
#         return
#
#     smoke = (args.mode == "smoke")
#     ref_med = float(df["price"].median())
#     if smoke:
#         print("\n*** SMOKE MODE: tiny subset, 2 epochs, just to verify TFT runs ***")
#         top_series = df.groupby("series_id").size().sort_values(ascending=False).head(SMOKE_SERIES).index
#         df = df[df["series_id"].isin(top_series)].copy()
#         train_df, val_hist, test_hist, val_idx, test_idx = chronological_split(df)
#         training, validation, test = build_datasets(
#             train_df, val_hist, test_hist, val_idx, test_idx, max_encoder=12, max_pred=2)
#         best, _, history = train_tft(training, validation, OUT_DIR, max_epochs=SMOKE_EPOCHS,
#                                      batch=256, hidden=16, patience=2, ref_median=ref_med)
#         metrics = evaluate(best, test, df, OUT_DIR)
#         path = write_final_report(
#             OUT_DIR, "smoke", history, metrics,
#             config=dict(lookback=12, horizon=2, batch=256, hidden=16,
#                         max_epochs=SMOKE_EPOCHS, patience=2))
#         print_final_summary(metrics)
#         print(f"\nsaved {path}")
#         print("\nSMOKE OK. Next: python tft_nepal_food_price.py --mode train")
#         return
#
#     # FULL TRAIN
#     training, validation, test = build_datasets(
#         train_df, val_hist, test_hist, val_idx, test_idx)
#     best, _, history = train_tft(training, validation, OUT_DIR, ref_median=ref_med)
#     metrics = evaluate(best, test, df, OUT_DIR)
#     commodity_rows = per_commodity_breakdown(best, training, test_hist, test_idx, OUT_DIR)
#     future_forecast(best, training, df, OUT_DIR)
#     path = write_final_report(OUT_DIR, "train", history, metrics, commodity_rows)
#     print_final_summary(metrics)
#     print(f"\nDONE. All outputs in ./{OUT_DIR}/")
#     print(f"Final report : {path}")
#     print("Best-model comparison for your report: metrics.json (skill_vs_persistence = TFT gain over the same-window baseline)")
#
#
# if __name__ == "__main__":
#     main()
#     print("END-OF-SCRIPT-MARKER", flush=True)  # TEMP
#     try:
#         sys.stdout.flush()
#         sys.stderr.flush()
#     except Exception:
#         pass
#     os._exit(0)  # TEMP
