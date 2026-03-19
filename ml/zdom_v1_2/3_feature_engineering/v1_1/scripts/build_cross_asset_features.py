"""
Build cross-asset momentum features from ETF data.

Captures relative strength and momentum across asset classes
to detect risk-on/risk-off regimes.

Inputs:  yfinance data (^GSPC, IWM, QQQ, EEM, TLT)
Output:  data/cross_asset_features.parquet (one row per trading day)

Features (~20):
  SPX momentum:
    spx_ret_5d, spx_ret_20d, spx_ret_60d

  SPX/IWM relative strength:
    spx_iwm_ratio, spx_iwm_ratio_chg_5d, spx_iwm_ratio_chg_20d

  QQQ/SPX ratio:
    qqq_spx_ratio, qqq_spx_ratio_chg_5d, qqq_spx_ratio_chg_20d

  IWM/SPX ratio:
    iwm_spx_ratio, iwm_spx_ratio_chg_5d, iwm_spx_ratio_chg_20d

  EEM/SPX ratio:
    eem_spx_ratio, eem_spx_ratio_chg_5d, eem_spx_ratio_chg_20d

  TLT direction:
    tlt_ret_5d, tlt_ret_20d

  TLT/SPX ratio:
    tlt_spx_ratio_chg_5d, tlt_spx_ratio_chg_20d

  Bond-equity correlation:
    tlt_spx_corr_20d

Usage:
  python scripts/build_cross_asset_features.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TICKERS = ["^GSPC", "IWM", "QQQ", "EEM", "TLT"]
START_DATE = "2018-01-01"


def fetch_etf_data():
    """Fetch daily close prices for all ETFs from yfinance."""
    print(f"Fetching ETF data from {START_DATE} to today...")
    print(f"  Tickers: {', '.join(TICKERS)}")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
    )

    # Extract close prices
    closes = data["Close"].copy()
    closes.columns = [c.lower() for c in closes.columns]

    # Rename ^GSPC column to spx
    if "^gspc" in closes.columns:
        closes = closes.rename(columns={"^gspc": "spx"})

    # Forward-fill missing data
    closes = closes.ffill()
    closes = closes.dropna()

    closes = closes.reset_index()
    closes = closes.rename(columns={"Date": "date"})
    closes["date"] = pd.to_datetime(closes["date"])

    print(f"  Fetched {len(closes)} trading days")
    print(f"  Date range: {closes['date'].min().date()} -> {closes['date'].max().date()}")

    return closes


def compute_ratio_features(df, numerator, denominator, prefix):
    """Compute ratio and its 5d/20d changes.

    Returns dict of column_name -> Series.
    """
    ratio = df[numerator] / df[denominator]
    return {
        f"{prefix}_ratio": ratio,
        f"{prefix}_ratio_chg_5d": ratio.pct_change(5) * 100,
        f"{prefix}_ratio_chg_20d": ratio.pct_change(20) * 100,
    }


def build_features(closes):
    """Compute all cross-asset features from close prices."""
    print("\nComputing features...")
    df = closes.copy()
    feats = pd.DataFrame({"date": df["date"]})

    # --- SPX momentum ---
    feats["spx_ret_5d"] = df["spx"].pct_change(5) * 100
    feats["spx_ret_20d"] = df["spx"].pct_change(20) * 100
    feats["spx_ret_60d"] = df["spx"].pct_change(60) * 100
    print("  SPX momentum: 3 features")

    # --- SPX/IWM relative strength ---
    spx_iwm = compute_ratio_features(df, "spx", "iwm", "spx_iwm")
    for col, vals in spx_iwm.items():
        feats[col] = vals.values
    print("  SPX/IWM relative strength: 3 features")

    # --- QQQ/SPX ratio ---
    qqq_spx = compute_ratio_features(df, "qqq", "spx", "qqq_spx")
    for col, vals in qqq_spx.items():
        feats[col] = vals.values
    print("  QQQ/SPX ratio: 3 features")

    # --- IWM/SPX ratio ---
    iwm_spx = compute_ratio_features(df, "iwm", "spx", "iwm_spx")
    for col, vals in iwm_spx.items():
        feats[col] = vals.values
    print("  IWM/SPX ratio: 3 features")

    # --- EEM/SPX ratio ---
    eem_spx = compute_ratio_features(df, "eem", "spx", "eem_spx")
    for col, vals in eem_spx.items():
        feats[col] = vals.values
    print("  EEM/SPX ratio: 3 features")

    # --- TLT direction ---
    feats["tlt_ret_5d"] = df["tlt"].pct_change(5).values * 100
    feats["tlt_ret_20d"] = df["tlt"].pct_change(20).values * 100
    print("  TLT direction: 2 features")

    # --- TLT/SPX ratio ---
    tlt_spx_ratio = df["tlt"] / df["spx"]
    feats["tlt_spx_ratio_chg_5d"] = tlt_spx_ratio.pct_change(5).values * 100
    feats["tlt_spx_ratio_chg_20d"] = tlt_spx_ratio.pct_change(20).values * 100
    print("  TLT/SPX ratio: 2 features")

    # --- Bond-equity correlation proxy ---
    spx_daily_ret = df["spx"].pct_change()
    tlt_daily_ret = df["tlt"].pct_change()
    feats["tlt_spx_corr_20d"] = (
        spx_daily_ret.rolling(20).corr(tlt_daily_ret).values
    )
    print("  Bond-equity correlation: 1 feature")

    feature_cols = [c for c in feats.columns if c != "date"]
    print(f"\n  Total features: {len(feature_cols)}")

    return feats


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    closes = fetch_etf_data()
    feats = build_features(closes)

    # Drop rows where all features are NaN (initial lookback period)
    feature_cols = [c for c in feats.columns if c != "date"]
    feats = feats.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Cross-asset features: {feats.shape[0]} rows x {feats.shape[1]} columns")
    print(f"Date range: {feats['date'].min().date()} -> {feats['date'].max().date()}")

    null_pcts = feats[feature_cols].isna().mean() * 100
    print("\nNull rates:")
    for col in feature_cols:
        if null_pcts[col] > 0:
            print(f"  {col:35s} {null_pcts[col]:.1f}%")

    print("\nSample stats:")
    for col in ["spx_ret_5d", "spx_ret_20d", "tlt_spx_corr_20d", "spx_iwm_ratio"]:
        if col in feats.columns:
            vals = feats[col].dropna()
            print(f"  {col:35s} mean={vals.mean():.4f}  std={vals.std():.4f}")

    outfile = DATA_DIR / "cross_asset_features.parquet"
    feats.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")
    print(f"  File size: {outfile.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
