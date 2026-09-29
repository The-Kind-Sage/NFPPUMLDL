"""Per-commodity evaluation breakdown."""
import os
import numpy as np
import pandas as pd
from config import TEST_START, NEEDED
from pytorch_forecasting import TimeSeriesDataSet
from reporting.plots import plot_per_commodity
from utils import unpack_predict_output, denorm, accelerator_name


def per_commodity_breakdown(best, training, test_hist, test_start_idx, out_dir, top_n=8):
    """Evaluate TFT separately per commodity (great for the report)."""
    top = test_hist[test_hist["date"] >= TEST_START]["commodity"].value_counts().head(top_n).index.tolist()
    rows = []
    print(f"\nPer-commodity test MAE/RMSE (top {top_n}):")
    for c in top:
        try:
            sub = test_hist[test_hist["commodity"] == c]
            ds = TimeSeriesDataSet.from_dataset(training, sub[NEEDED], predict=False,
                                                stop_randomization=True,
                                                min_prediction_idx=test_start_idx)
            if len(ds) == 0:
                continue
            dl = ds.to_dataloader(train=False, batch_size=512, num_workers=0)
            res = best.predict(dl, mode="raw", return_x=True, return_y=True,
                           trainer_kwargs={"accelerator": accelerator_name(), "devices": 1, "logger": False})
            dd = unpack_predict_output(res)
            raw, x, y = dd["raw"], dd["x"], dd["y"]
            if y is None and x is not None and "decoder_target" in x:
                y = x["decoder_target"]
            ref = float(sub.loc[sub["date"] >= TEST_START, "price"].median())
            pm = np.clip(denorm(raw, x["target_scale"], ref)[:, :, 3], 0, None)
            yt = denorm(y, x["target_scale"], ref)
            from evaluation.metrics import reg_metrics
            mm = reg_metrics(yt, pm)
            rows.append({"commodity": c, **mm})
            print(f"  {c:22s} MAE={mm['MAE']:.2f} RMSE={mm['RMSE']:.2f} R2={mm['R2']:.4f} (n={mm['n']})")
        except Exception as e:
            print(f"  {c:22s} skipped ({e})")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_commodity_metrics.csv"), index=False)
        plot_per_commodity(rows, out_dir)
    return rows