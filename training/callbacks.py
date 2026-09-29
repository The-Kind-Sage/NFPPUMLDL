"""Training callbacks - EpochReporter for live per-epoch reporting."""
import os
import time
import json
import pandas as pd


def make_epoch_reporter(pl, out_dir, max_epochs, ref_median=None, patience=6):
    """Lightning callback: live per-epoch report on stdout + training_history.csv."""
    Callback = pl.Callback

    class EpochReporter(Callback):
        def __init__(self):
            super().__init__()
            self.out_dir = out_dir
            self.max_epochs = int(max_epochs)
            self.patience = int(patience)
            self.ref_median = float(ref_median) if ref_median else 100.0
            self.history = []
            self._hook = None
            self._pred = None
            self._apply = None
            self._ae = 0.0
            self._se = 0.0
            self._n = 0
            self._t_epoch = time.time()
            self._best_path = ""
            self._best_epoch = None
            self._warned = False

        # ---- capture the forward pass during validation (costs nothing extra) ----
        def on_validation_start(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            self._ae = self._se = 0.0
            self._n = 0
            self._pred = None
            self._apply = None
            if self._hook is None:
                def _capture(module, inputs, output):
                    self._pred = _prediction_tensor(output)
                try:
                    self._hook = pl_module.register_forward_hook(_capture)
                except Exception:
                    self._hook = None

        def on_validation_end(self, trainer, pl_module):
            if self._hook is not None:
                try:
                    self._hook.remove()
                except Exception:
                    pass
                self._hook = None
            self._pred = None

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                    batch_idx, dataloader_idx=0):
            """Accumulate squared/absolute errors -> epoch val MAE + RMSE in NPR."""
            try:
                if trainer.sanity_checking or self._pred is None:
                    return
                if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                    return
                x, y = batch
                if isinstance(y, (tuple, list)):     # pf returns (target, weight)
                    y = y[0]
                if y is None or not hasattr(y, "detach") or not hasattr(x, "get"):
                    return
                scale = x.get("target_scale")
                if scale is None:
                    return
                p = self._pred.detach().float().cpu().numpy()
                t = y.detach().float().cpu().numpy()
                if p.ndim == t.ndim + 1:             # (B,H,Q) -> median quantile
                    q = p.shape[-1]
                    p = p[..., 3] if q >= 7 else p[..., q // 2]
                if p.ndim != t.ndim or p.shape[0] != t.shape[0]:
                    return
                # one decision for BOTH sides -> same space guaranteed
                if self._apply is None:
                    self._apply = abs(float(np.median(t))) < 10 and self.ref_median > 20
                e = (_to_price(p, scale, self.ref_median, self._apply)
                     - _to_price(t, scale, self.ref_median, self._apply)).ravel()
                e = e[np.isfinite(e)]
                if e.size == 0:
                    return
                self._ae += float(np.abs(e).sum())
                self._se += float((e ** 2).sum())
                self._n += int(e.size)
            except Exception:
                if not self._warned:
                    self._warned = True
                    print("  (note: per-epoch MAE/RMSE unavailable - loss-only report)",
                          flush=True)

        def on_train_epoch_start(self, trainer, pl_module):
            self._t_epoch = time.time()

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            cm = trainer.callback_metrics
            epoch = int(trainer.current_epoch) + 1
            row = {
                "epoch": epoch,
                "train_loss": _cm_first(cm, "train_loss", "train_epoch_loss", "training_loss"),
                "val_loss": _cm_first(cm, "val_loss", "validation_loss"),
                "val_MAE": (self._ae / self._n) if self._n else None,
                "val_RMSE": (self._se / self._n) ** 0.5 if self._n else None,
                "lr": _cm_lr(cm),
                "seconds": round(time.time() - self._t_epoch, 1),
            }
            ckpt = getattr(trainer, "checkpoint_callback", None)
            path = getattr(ckpt, "best_model_path", "") or ""
            row["is_best"] = bool(path) and path != self._best_path
            if row["is_best"]:
                self._best_path = path
                self._best_epoch = epoch
            row["best_epoch"] = self._best_epoch
            row["stale_epochs"] = epoch - (self._best_epoch or epoch)
            self.history.append(row)
            print(_epoch_line(row, self.max_epochs, self.patience), flush=True)
            try:
                self._write()
            except Exception:
                pass

        # ---- persistence (written after EVERY epoch so the report survives a crash) ----
        def _write(self):
            if not self.history:
                return
            pd.DataFrame(self.history).to_csv(
                os.path.join(self.out_dir, "training_history.csv"), index=False)
            with open(os.path.join(self.out_dir, "training_history.json"), "w") as fh:
                json.dump(self.history, fh, indent=2)

        def save(self):
            if not self.history:
                print("(no epoch history recorded)")
                return self.history
            self._write()
            print(f"saved {os.path.join(self.out_dir, 'training_history.csv')} "
                  f"({len(self.history)} epochs)")
            return self.history

        def summary(self):
            rows = [r for r in self.history if r.get("val_loss") is not None]
            if not rows:
                return ""
            best = min(rows, key=lambda r: r["val_loss"])
            last = self.history[-1]
            tail = ""
            if last["stale_epochs"] >= self.patience:
                tail = f" | early stopping after {self.patience} stale epochs"
            return (f"epochs run: {last['epoch']}/{self.max_epochs} | "
                    f"best epoch: {best['epoch']} (val_loss={best['val_loss']:.4f}) | "
                    f"val MAE={_num(best['val_MAE'], 1)} RMSE={_num(best['val_RMSE'], 1)} NPR"
                    f"{tail}")

    return EpochReporter()


# Import helpers from utils (avoid circular import)
from utils import _num, _cm_first, _cm_lr, _epoch_line, _prediction_tensor, _to_price
import numpy as np