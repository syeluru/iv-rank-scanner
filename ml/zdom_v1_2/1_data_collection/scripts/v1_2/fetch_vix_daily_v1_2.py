"""
Fetch long-history VIX daily OHLC for v1_2 feature engineering.

Source: yfinance (^VIX — CBOE Volatility Index)

Provides extended daily history for long-lookback VIX features:
  - vix_ratio_sma_10day, vix_ratio_sma_20day
  - vix_vxv_ratio, vix_vxv_zscore_21d
  - any future 50/100/200 day VIX features

Output: data/v1_2/vix_daily_v1_2.parquet
  One row per trading day, keyed by date.
  Coverage: ~2021-01-01 through 2026-03-13

Columns: date, vix_open, vix_high, vix_low, vix_close

Join lag: T-1 (daily close not known until after 4pm).
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_V1_2 = BASE / "data" / "v1_2"

FETCH_START = "2021-01-01"
CLIP = pd.Timestamp("2026-03-13")


def main():
    DATA_V1_2.mkdir(parents=True, exist_ok=True)

    print("Fetching ^VIX daily from yfinance...")
    raw = yf.download("^VIX", start=FETCH_START, auto_adjust=True, progress=False)

    if raw.empty:
        print("ERROR: No data returned")
        return

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]

    rename = {}
    for col in raw.columns:
        cl = col.lower().strip()
        if cl in ("date", "price"):
            rename[col] = "date"
        elif cl == "open":
            rename[col] = "vix_open"
        elif cl == "high":
            rename[col] = "vix_high"
        elif cl == "low":
            rename[col] = "vix_low"
        elif cl == "close":
            rename[col] = "vix_close"

    raw = raw.rename(columns=rename)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    keep = ["date", "vix_open", "vix_high", "vix_low", "vix_close"]
    raw = raw[keep].copy()
    raw = raw[raw["date"].dt.dayofweek < 5]
    raw = raw[raw["vix_close"] > 0]
    raw = raw[raw["date"] <= CLIP]
    raw = raw.sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / "vix_daily_v1_2.parquet"
    raw.to_parquet(out, index=False)

    print(f"\nvix_daily_v1_2.parquet")
    print(f"  Shape: {raw.shape}")
    print(f"  Range: {raw['date'].min().date()} to {raw['date'].max().date()}")
    print(f"  Nulls: {raw.isna().sum().sum()}")
    print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
