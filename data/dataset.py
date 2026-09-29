"""TimeSeriesDataSet building for TFT."""
from config import (
    MAX_ENCODER, MAX_PRED, MIN_ENCODER, BATCH,
    STATIC_CAT, STATIC_REAL, KNOWN_CAT, KNOWN_REAL, UNKNOWN_REAL, NEEDED,
    TARGET_TRANSFORM
)


def build_datasets(train_df, val_hist, test_hist, val_start_idx, test_start_idx,
                   max_encoder=MAX_ENCODER, max_pred=MAX_PRED):
    try:
        from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
    except ImportError:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    try:
        from pytorch_forecasting.data import RobustScaler
        scalers = {v: RobustScaler() for v in
                   [c for c in (KNOWN_REAL + UNKNOWN_REAL + STATIC_REAL) if c != "price"]}
    except Exception:
        scalers = {}

    common = dict(
        time_idx="time_idx", target="price", group_ids=["series_id"],
        max_encoder_length=max_encoder, max_prediction_length=max_pred,
        min_encoder_length=min(MIN_ENCODER, max_encoder),
        min_prediction_length=max_pred,
        static_categoricals=STATIC_CAT,
        static_reals=STATIC_REAL,
        time_varying_known_categoricals=KNOWN_CAT,
        time_varying_known_reals=KNOWN_REAL,
        time_varying_unknown_reals=UNKNOWN_REAL,
        # KEY FOR BEST METRICS: normalize each series separately
        # (salt ~Rs.20 and chicken ~Rs.500 must not share one scale)
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation=TARGET_TRANSFORM),
        scalers=scalers,
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    training = TimeSeriesDataSet(train_df[NEEDED], **common)

    def _from(data, start_idx):
        try:
            return TimeSeriesDataSet.from_dataset(
                training, data[NEEDED], predict=False,
                stop_randomization=True, min_prediction_idx=start_idx)
        except TypeError:  # very old versions without min_prediction_idx
            print("  (warning: min_prediction_idx unsupported, using sliced data)")
            return TimeSeriesDataSet.from_dataset(
                training, data[NEEDED], predict=False, stop_randomization=True)

    validation = _from(val_hist, val_start_idx)
    test = _from(test_hist, test_start_idx)
    print(f"\nwindows -> train: {len(training)} | val: {len(validation)} | test: {len(test)}")
    return training, validation, test