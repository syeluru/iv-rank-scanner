"""
Fetch long-history SPX daily OHLC for v1_2 feature engineering.

Source: yfinance (^GSPC — S&P 500 index)

This provides the extended daily history needed for long-lookback features:
  - SMA 50/100/200
  - RSI 14
  - MACD
  - Bollinger Bands
  - Yang-Zhang HV (5/10/20/30 day)
  - ATR, efficiency ratio, choppiness

yfinance ^GSPC gives the actual SPX index values (not SPY ETF).

Output: data/v1_2/spx_daily_v1_2.parquet
  One row per trading day, keyed by date.
  Coverage: ~2021-01-01 through 2026-03-13
  (starts well before 2022-06-06 to allow 200+ day lookback warmup)

Columns: date, spx_open, spx_high, spx_low, spx_close, spx_volume

Join lag: T-1 (daily close not known until after 4pm).

Note: The v1_2 raw/spx_1min.parquet starts at 2023-03-13 and is NOT suitable
for long-lookback daily features. This table fills that gap.
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

    print(f"Fetching ^GSPC (SPX) daily from yfinance...")
    print(f"  Start: {FETCH_START}")
    print(f"  Clip: {CLIP.date()}")

    raw = yf.download("^GSPC", start=FETCH_START, auto_adjust=True, progress=False)

    if raw.empty:
        print("ERROR: No data returned from yfinance")
        return

    raw = raw.reset_index()

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]

    # Standardize column names
    rename = {}
    for col in raw.columns:
        cl = col.lower().strip()
        if cl in ("date", "price"):
            rename[col] = "date"
        elif cl == "open":
            rename[col] = "spx_open"
        elif cl == "high":
            rename[col] = "spx_high"
        elif cl == "low":
            rename[col] = "spx_low"
        elif cl == "close":
            rename[col] = "spx_close"
        elif cl == "volume":
            rename[col] = "spx_volume"

    raw = raw.rename(columns=rename)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    # Keep only needed columns
    keep = ["date", "spx_open", "spx_high", "spx_low", "spx_close"]
    if "spx_volume" in raw.columns:
        keep.append("spx_volume")
    raw = raw[keep].copy()

    # Drop weekends / zero rows
    raw = raw[raw["date"].dt.dayofweek < 5]
    raw = raw[raw["spx_close"] > 0]

    # Clip
    raw = raw[raw["date"] <= CLIP]
    raw = raw.sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / "spx_daily_v1_2.parquet"
    raw.to_parquet(out, index=False)

    print(f"\nspx_daily_v1_2.parquet")
    print(f"  Shape: {raw.shape}")
    print(f"  Range: {raw['date'].min().date()} to {raw['date'].max().date()}")
    print(f"  Trading days: {raw['date'].nunique()}")
    print(f"  Nulls: {raw.isna().sum().sum()}")
    print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
