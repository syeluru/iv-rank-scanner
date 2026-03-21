"""
Build IV surface features from ThetaData options data.

Extracts per-trading-day:
  From intraday greeks (0DTE, 10 AM snapshot):
    atm_iv, skew_25d, skew_10d, iv_smile_curve, etc.
  From term structure (multiple DTEs, Black-Scholes inversion):
    ts_iv_7dte, ts_iv_14dte, term structure slopes
  Rolling features:
    5-day and 21-day rolling means, stds, z-scores

Inputs:
  data/spxw_0dte_intraday_greeks.parquet
  data/spxw_term_structure.parquet
  data/spx_daily.parquet

Output:
  data/iv_surface_features.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time as dt_time

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Import IV surface module from src/
sys.path.insert(0, str(PROJECT_DIR))
from src.iv_surface import (
    extract_snapshot_iv_features, extract_term_structure_iv,
    add_rolling_features, compute_term_structure_slopes,
)

# Snapshot time: 10:00 AM ET (after opening noise settles, before lunch lull)
SNAPSHOT_HOUR = 10
SNAPSHOT_MINUTE = 0
SNAPSHOT_WINDOW_MINUTES = 15


def get_snapshot(day_data):
    """Extract the 10:00 AM snapshot from one day of intraday greeks.

    Tries exact time, then window, then hour 10, then all data as fallback.
    Aggregates to per-strike median if multiple timestamps.
    """
    target = dt_time(SNAPSHOT_HOUR, SNAPSHOT_MINUTE)

    snapshot = day_data[day_data["timestamp"].dt.time == target]
    if snapshot.empty:
        window_start = dt_time(SNAPSHOT_HOUR, SNAPSHOT_MINUTE)
        window_end = dt_time(SNAPSHOT_HOUR, SNAPSHOT_MINUTE + SNAPSHOT_WINDOW_MINUTES)
        snapshot = day_data[
            (day_data["timestamp"].dt.time >= window_start)
            & (day_data["timestamp"].dt.time <= window_end)
        ]
        if snapshot.empty:
            snapshot = day_data[day_data["timestamp"].dt.hour == SNAPSHOT_HOUR]

    if snapshot.empty:
        snapshot = day_data

    if snapshot["timestamp"].nunique() > 1:
        snapshot = (
            snapshot.groupby(["strike", "right"])
            .agg({
                "implied_vol": "median",
                "delta": "median",
                "iv_error": "min",
                "underlying_price": "median",
                "mid": "median",
            })
            .reset_index()
        )

    return snapshot


def main():
    print("=" * 60)
    print("Building IV surface features")
    print("=" * 60)

    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    spx["date"] = pd.to_datetime(spx["date"])
    spx_map = spx.set_index("date")[["spx_open", "spx_close"]].to_dict("index")

    ts_file = DATA_DIR / "spxw_term_structure.parquet"
    has_ts = ts_file.exists()
    if has_ts:
        ts = pd.read_parquet(ts_file)
        ts["date"] = pd.to_datetime(ts["date"])
        print(f"  Term structure: {ts.shape[0]:,} rows, {ts['date'].nunique()} days")
    else:
        ts = pd.DataFrame()
        print("  Term structure: not found")

    print("  Loading intraday greeks (this may take a moment)...")
    greeks_cols = ["date", "strike", "right", "timestamp", "underlying_price",
                   "delta", "implied_vol", "iv_error", "mid"]
    greeks = pd.read_parquet(
        DATA_DIR / "spxw_0dte_intraday_greeks.parquet",
        columns=greeks_cols,
    )
    greeks["date"] = pd.to_datetime(greeks["date"])
    greeks["timestamp"] = pd.to_datetime(greeks["timestamp"])
    print(f"  Intraday greeks: {greeks.shape[0]:,} rows, {greeks['date'].nunique()} days")

    trading_days = sorted(greeks["date"].unique())
    print(f"\nProcessing {len(trading_days)} trading days...")

    rows = []
    for i, date in enumerate(trading_days):
        if (i + 1) % 50 == 0 or (i + 1) == len(trading_days):
            print(f"  [{i+1}/{len(trading_days)}] {pd.Timestamp(date).date()}")

        row = {"date": date}

        # Intraday IV features (0DTE) — uses src/iv_surface module
        day_greeks = greeks[greeks["date"] == date]
        snapshot = get_snapshot(day_greeks)
        intraday_feats = extract_snapshot_iv_features(snapshot)
        row.update(intraday_feats)

        # Term structure IV features — uses src/iv_surface module
        spx_info = spx_map.get(date, spx_map.get(pd.Timestamp(date), {}))
        spx_close = spx_info.get("spx_close", np.nan) if spx_info else np.nan

        ts_day = ts[ts["date"] == date] if has_ts else pd.DataFrame()
        ts_feats = extract_term_structure_iv(ts_day, spx_close)
        row.update(ts_feats)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Term structure slopes — uses src/iv_surface module
    df = compute_term_structure_slopes(df)

    # Rolling features — uses src/iv_surface module
    print("\nComputing rolling features...")
    for col in ["atm_iv", "skew_25d"]:
        add_rolling_features(df, col, windows=(5, 21))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"IV surface features: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    feature_cols = [c for c in df.columns if c != "date"]
    print(f"\nFeatures ({len(feature_cols)}):")
    for col in feature_cols:
        valid_pct = (1 - df[col].isna().mean()) * 100
        if valid_pct > 0:
            vals = df[col].dropna()
            print(f"  {col:30s} valid={valid_pct:5.1f}%  "
                  f"mean={vals.mean():8.4f}  std={vals.std():8.4f}")
        else:
            print(f"  {col:30s} valid={valid_pct:5.1f}%")

    outfile = DATA_DIR / "iv_surface_features.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
