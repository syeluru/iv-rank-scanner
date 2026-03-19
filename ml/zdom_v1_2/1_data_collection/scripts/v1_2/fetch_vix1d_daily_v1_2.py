"""
Fetch/build VIX1D daily OHLC for v1_2 feature engineering.

Source: Bootstrapped from v1_1 ThetaData-derived vix1d_daily.parquet.
  yfinance ^VIX1D ticker is unreliable (delisted/intermittent as of March 2026).
  The v1_1 source was built from ThetaData index/history/ohlc (VIX1D, 1m)
  aggregated to daily bars.

VIX1D was launched by CBOE on 2023-04-24. No history exists before that date.
This is a hard constraint — features that depend on VIX1D will have nulls
for dates before 2023-04-24 plus any rolling lookback warmup period.

Output: data/v1_2/vix1d_daily_v1_2.parquet
  One row per trading day, keyed by date.
  Coverage: 2023-04-24 through 2026-03-13

Columns: date, vix1d_open, vix1d_high, vix1d_low, vix1d_close

Join lag: T-1 (daily close not known until after 4pm).
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
DATA_V1_2 = BASE / "data" / "v1_2"

CLIP = pd.Timestamp("2026-03-13")


def main():
    DATA_V1_2.mkdir(parents=True, exist_ok=True)

    print("Building vix1d_daily_v1_2 from v1_1 ThetaData source...")
    print("  Note: yfinance ^VIX1D is unreliable. Using ThetaData-derived v1_1 data.")
    print("  Note: VIX1D launched 2023-04-24. No earlier data exists.")

    vix1d = pd.read_parquet(RAW_V1_1 / "vix1d_daily.parquet")
    vix1d["date"] = pd.to_datetime(vix1d["date"])
    vix1d = vix1d[vix1d["vix1d_close"] > 0]
    vix1d = vix1d[vix1d["date"] <= CLIP]
    vix1d = vix1d.sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / "vix1d_daily_v1_2.parquet"
    vix1d.to_parquet(out, index=False)

    print(f"\nvix1d_daily_v1_2.parquet")
    print(f"  Shape: {vix1d.shape}")
    print(f"  Range: {vix1d['date'].min().date()} to {vix1d['date'].max().date()}")
    print(f"  Nulls: {vix1d.isna().sum().sum()}")
    print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
