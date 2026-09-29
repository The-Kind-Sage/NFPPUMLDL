"""Logger setup - TensorBoardLogger for experiment tracking (recognized by Lightning to suppress tips)."""
def get_logger(out_dir):
    """Create TensorBoardLogger for Lightning (suppresses litlogger tips)."""
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir=out_dir, name="training_logs")
    except ImportError:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            return TensorBoardLogger(save_dir=out_dir, name="training_logs")
        except ImportError:
            # Fallback
            try:
                from lightning.pytorch.loggers import CSVLogger
                return CSVLogger(save_dir=out_dir, name="training_logs")
            except ImportError:
                from pytorch_lightning.loggers import CSVLogger
                return CSVLogger(save_dir=out_dir, name="training_logs")