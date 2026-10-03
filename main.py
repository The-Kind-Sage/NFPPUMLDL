"""CLI dispatcher for Nepal Food Price Forecasting with TFT."""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Nepal Food Price Forecasting with TFT")
    ap.add_argument("--mode", default="train", choices=["prep", "smoke", "train", "eval"],
                    help="Run mode: prep (EDA only), smoke (2-epoch test), train (full), "
                         "eval (re-score newest checkpoint, no training)")
    args = ap.parse_args()

    if args.mode == "prep":
        from scripts.prep import run_prep
        run_prep()
    elif args.mode == "smoke":
        from scripts.smoke import run_smoke
        run_smoke()
    elif args.mode == "train":
        from scripts.train import run_train
        run_train()
    elif args.mode == "eval":
        from scripts.eval import run_eval
        run_eval()


if __name__ == "__main__":
    main()