"""Data loading and EDA reporting."""
import pandas as pd
from pathlib import Path
from config import CSV_CANDIDATES


def find_csv():
    here = Path(__file__).resolve().parent.parent
    for c in CSV_CANDIDATES:
        for p in (Path(c), here / c):
            if p.exists():
                return str(p)
    raise FileNotFoundError(f"CSV not found. Tried: {CSV_CANDIDATES}")


def load_and_report(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print("=" * 70)
    print("DATASET ANALYSIS - WFP Nepal food prices")
    print("=" * 70)
    print(f"file            : {csv_path}")
    print(f"shape           : {df.shape}")
    print(f"date range      : {df['date'].min().date()} -> {df['date'].max().date()}")
    print("frequency       : MONTHLY (observed on the 15th)")
    print(f"markets         : {df['market'].nunique()}  | commodities: {df['commodity'].nunique()} "
          f"| units: {sorted(df['unit'].unique().tolist())}")
    print(f"total series (market x commodity x unit): {df.groupby(['market','commodity','unit']).ngroups}")
    print(f"price (NPR)     : min {df['price'].min():.1f} | median {df['price'].median():.1f} "
          f"| mean {df['price'].mean():.1f} | max {df['price'].max():.1f}")
    print(f"missing values  : {int(df.isna().sum().sum())}")
    print("\nrows per year (note: dense from 2019, sparse before):")
    print(df["date"].dt.year.value_counts().sort_index().to_string())
    print("\nTop commodities by rows:")
    print(df["commodity"].value_counts().head(10).to_string())
    return df