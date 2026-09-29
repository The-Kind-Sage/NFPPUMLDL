"""Config loader - reads YAML and exposes as module attributes."""
import os
from pathlib import Path

# determinism: cuBLAS reads this only at the FIRST GPU matmul, so set it before
# torch runs (this module is imported before torch anywhere in the pipeline)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml

_config_path = Path(__file__).parent / "base.yaml"
with open(_config_path, "r") as f:
    _cfg = yaml.safe_load(f)

# Flatten for easy access
CSV_CANDIDATES = _cfg["csv_candidates"]
OUT_DIR = _cfg["out_dir"]
MIN_SERIES_LEN = _cfg["min_series_len"]
MAX_ENCODER = _cfg["max_encoder"]
MAX_PRED = _cfg["max_pred"]
MIN_ENCODER = _cfg["min_encoder"]
VAL_START = _cfg["val_start"]
TEST_START = _cfg["test_start"]
BATCH = _cfg["batch"]
MAX_EPOCHS = _cfg["max_epochs"]
PATIENCE = _cfg["patience"]
HIDDEN = _cfg["hidden"]
HEADS = _cfg["heads"]
SEED = _cfg["seed"]
DROPOUT = _cfg["dropout"]
LEARNING_RATE = _cfg["lr"]
WEIGHT_DECAY = _cfg["weight_decay"]
TARGET_TRANSFORM = _cfg["target_transform"]
SMOKE_SERIES = _cfg["smoke_series"]
SMOKE_EPOCHS = _cfg["smoke_epochs"]
STATIC_CAT = _cfg["static_cat"]
STATIC_REAL = _cfg["static_real"]
KNOWN_CAT = _cfg["known_cat"]
KNOWN_REAL = _cfg["known_real"]
UNKNOWN_REAL = _cfg["unknown_real"]

NEEDED = (["series_id", "time_idx"] + STATIC_CAT + STATIC_REAL
          + KNOWN_CAT + KNOWN_REAL + UNKNOWN_REAL)

def get_config():
    """Return full config dict for reporting."""
    return _cfg.copy()