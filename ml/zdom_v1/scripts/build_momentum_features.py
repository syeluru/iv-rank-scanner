"""
Build intraday momentum features from 1-minute SPX data.

Captures the first 30 minutes of trading — a well-documented signal for
0DTE options. Opening momentum predicts continuation vs. reversal.

Inputs:
  data/spx_merged.parquet  — 1-min SPX data with VIX

Output:
  data/momentum_features.parquet — One row per trading day

Features:
  first_30m_return_pct   — (price at 10:00 - open) / open * 100
  first_30m_range_pct    — (high - low of first 30m) / open * 100
  first_30m_direction    — 1 if positive return, -1 if negative, 0 if flat
  first_30m_vol_ratio    — volume in first 30m / total daily volume
  first_15m_return_pct   — (price at 9:45 - open) / open * 100
  first_60m_return_pct   — (price at 10:30 - open) / open * 100
  open_drive_strength    — first_30m_return / first_30m_range (directional efficiency)
  prev_close_gap_reversal — sign(gap) != sign(first_30m_return)  (gap fill signal)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def compute_momentum_for_day(day_df):
    """Compute intraday momentum features for a single day.

    Args:
        day_df: DataFrame of 1-min bars for one trading day, sorted by datetime.

    Returns:
        dict of momentum features
    """
    feats = {}

    if len(day_df) < 30:
        return {k: np.nan for k in [
            "first_30m_return_pct", "first_30m_range_pct", "first_30m_direction",
            "first_30m_vol_ratio", "first_15m_return_pct", "first_60m_return_pct",
            "open_drive_strength", "prev_close_gap_reversal",
        ]}

    day_open = day_df.iloc[0]["open"]
    if pd.isna(day_open) or day_open <= 0:
        return {k: np.nan for k in [
            "first_30m_return_pct", "first_30m_range_pct", "first_30m_direction",
            "first_30m_vol_ratio", "first_15m_return_pct", "first_60m_return_pct",
            "open_drive_strength", "prev_close_gap_reversal",
        ]}

    total_volume = day_df["volume"].sum() if "volume" in day_df.columns else 0

    # First 15 minutes (9:30 - 9:44)
    first_15m = day_df.iloc[:15]
    price_15m = first_15m.iloc[-1]["close"] if len(first_15m) >= 15 else np.nan
    feats["first_15m_return_pct"] = (
        (price_15m - day_open) / day_open * 100
        if not pd.isna(price_15m) else np.nan
    )

    # First 30 minutes (9:30 - 9:59)
    first_30m = day_df.iloc[:30]
    price_30m = first_30m.iloc[-1]["close"]
    high_30m = first_30m["high"].max()
    low_30m = first_30m["low"].min()
    vol_30m = first_30m["volume"].sum() if "volume" in first_30m.columns else 0

    feats["first_30m_return_pct"] = (price_30m - day_open) / day_open * 100
    feats["first_30m_range_pct"] = (high_30m - low_30m) / day_open * 100

    ret_30m = feats["first_30m_return_pct"]
    if abs(ret_30m) < 0.001:
        feats["first_30m_direction"] = 0
    else:
        feats["first_30m_direction"] = 1 if ret_30m > 0 else -1

    feats["first_30m_vol_ratio"] = (
        vol_30m / total_volume if total_volume > 0 else np.nan
    )

    # First 60 minutes (9:30 - 10:29)
    first_60m = day_df.iloc[:60]
    price_60m = first_60m.iloc[-1]["close"] if len(first_60m) >= 60 else np.nan
    feats["first_60m_return_pct"] = (
        (price_60m - day_open) / day_open * 100
        if not pd.isna(price_60m) else np.nan
    )

    # Open drive strength: directional efficiency of first 30m
    range_30m = feats["first_30m_range_pct"]
    feats["open_drive_strength"] = (
        ret_30m / range_30m if range_30m > 0.001 else 0.0
    )

    # Gap reversal signal (needs previous day close — set downstream)
    feats["prev_close_gap_reversal"] = np.nan  # placeholder

    return feats


def main():
    print("Loading spx_merged data...")
    df = pd.read_parquet(DATA_DIR / "spx_merged.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime")
    print(f"  Shape: {df.shape}")

    # Get previous day close for gap reversal signal
    daily_close = df.groupby("date")["close"].last().reset_index()
    daily_close = daily_close.rename(columns={"close": "prev_day_close"})
    daily_close["prev_day_close"] = daily_close["prev_day_close"].shift(1)
    daily_close_map = daily_close.set_index("date")["prev_day_close"].to_dict()

    trading_days = sorted(df["date"].unique())
    print(f"  Trading days: {len(trading_days)}")

    rows = []
    for i, date in enumerate(trading_days):
        if (i + 1) % 50 == 0:
            print(f"  Processing day {i+1}/{len(trading_days)}...")

        day_df = df[df["date"] == date].sort_values("datetime").reset_index(drop=True)
        feats = compute_momentum_for_day(day_df)
        feats["date"] = date

        # Gap reversal: did first 30m reverse the overnight gap?
        prev_close = daily_close_map.get(date, np.nan)
        day_open = day_df.iloc[0]["open"] if len(day_df) > 0 else np.nan
        if not pd.isna(prev_close) and not pd.isna(day_open) and prev_close > 0:
            gap_pct = (day_open - prev_close) / prev_close * 100
            ret_30m = feats.get("first_30m_return_pct", np.nan)
            if not pd.isna(ret_30m) and abs(gap_pct) > 0.05:
                feats["prev_close_gap_reversal"] = (
                    1.0 if np.sign(gap_pct) != np.sign(ret_30m) else 0.0
                )

        rows.append(feats)

    result = pd.DataFrame(rows)
    result = result.sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])

    print(f"\n{'='*60}")
    print(f"Momentum features: {result.shape[0]} rows x {result.shape[1]} columns")
    print(f"Date range: {result['date'].min().date()} -> {result['date'].max().date()}")

    null_pcts = result.isna().mean() * 100
    print("\nNull rates:")
    for col in result.columns:
        if col != "date" and null_pcts[col] > 0:
            print(f"  {col:35s} {null_pcts[col]:.1f}%")

    print(f"\nSample stats:")
    for col in ["first_30m_return_pct", "first_30m_range_pct", "open_drive_strength"]:
        if col in result.columns:
            print(f"  {col}: mean={result[col].mean():.4f}, std={result[col].std():.4f}")

    outfile = DATA_DIR / "momentum_features.parquet"
    result.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
