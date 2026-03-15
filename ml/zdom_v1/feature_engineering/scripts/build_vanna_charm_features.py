"""
Build per-day Vanna/Charm exposure features from ThetaData EOD greeks.

Inputs:
  data/spxw_0dte_greeks.parquet  — EOD greeks per strike (delta, gamma, iv, theta, vega)
  data/spxw_0dte_oi.parquet      — Open interest per strike
  data/spx_daily.parquet         — SPX daily OHLC (for spot price)

Output:
  data/vanna_charm_features.parquet — One row per trading day

Features:
  net_vanna_exposure      — aggregate dealer vanna across all strikes
  net_vanna_normalized    — net_vanna / spot (scale-independent)
  vanna_imbalance         — call_vanna / (call_vanna + |put_vanna|)
  net_charm_exposure      — aggregate dealer charm across all strikes
  net_charm_normalized    — net_charm / spot (scale-independent)
  charm_imbalance         — call_charm / (call_charm + |put_charm|)
  vanna_charm_ratio       — |net_vanna| / (|net_vanna| + |net_charm|)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

sys.path.insert(0, str(PROJECT_DIR))
from src.vanna_charm import compute_vanna_charm_features


def main():
    print("Loading data...")

    greeks = pd.read_parquet(DATA_DIR / "spxw_0dte_greeks.parquet")
    greeks["date"] = pd.to_datetime(greeks["date"])
    print(f"  EOD greeks: {greeks.shape}")

    oi_file = DATA_DIR / "spxw_0dte_oi.parquet"
    if oi_file.exists():
        oi = pd.read_parquet(oi_file)
        oi["date"] = pd.to_datetime(oi["date"])
        print(f"  OI data: {oi.shape}")
    else:
        # Try getting OI from EOD file
        eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
        eod["date"] = pd.to_datetime(eod["date"])
        oi_data = DATA_DIR / "spxw_0dte_oi.parquet"
        if oi_data.exists():
            oi = pd.read_parquet(oi_data)
        else:
            print("ERROR: No OI data found. Cannot compute Vanna/Charm.")
            return

    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    spx["date"] = pd.to_datetime(spx["date"])
    price_map = spx.set_index("date")["spx_open"].to_dict()

    trading_days = sorted(greeks["date"].unique())
    print(f"  Trading days: {len(trading_days)}")

    rows = []
    for i, date in enumerate(trading_days):
        if (i + 1) % 50 == 0:
            print(f"  Processing day {i+1}/{len(trading_days)}...")

        spot = price_map.get(date, np.nan)
        if pd.isna(spot):
            continue

        greeks_day = greeks[greeks["date"] == date]
        oi_day = oi[oi["date"] == date] if "date" in oi.columns else pd.DataFrame()

        # For 0DTE, T ~ fraction of a day. Use 1/365 as conservative estimate
        # (EOD snapshot, so actual remaining time < 1 day)
        feats = compute_vanna_charm_features(
            greeks_day, oi_day, spot, dte_years=0.5/365, risk_free_rate=0.05
        )
        feats["date"] = date
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n{'='*60}")
    print(f"Vanna/Charm features: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    null_pcts = df.isna().mean() * 100
    print("\nNull rates:")
    for col in df.columns:
        if col != "date" and null_pcts[col] > 0:
            print(f"  {col:35s} {null_pcts[col]:.1f}%")

    outfile = DATA_DIR / "vanna_charm_features.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
