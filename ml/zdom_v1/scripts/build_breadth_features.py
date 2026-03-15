"""
Build market breadth & internals features.

Fetches ETF data via yfinance and computes breadth proxies.
Does NOT require Theta Terminal.

Inputs:  yfinance data (RSP, SPY, ^VIX, ^VIX3M, XLF, XLK, XLE, IWM)
Output:  data/breadth_features.parquet (one row per trading day)

Features produced:
  - vix_vxv_ratio          : ^VIX / ^VIX3M — <1 = contango (calm), >1 = backwardation (stress)
  - vix_vxv_zscore_21d     : z-score of vix_vxv_ratio over 21-day window
  - rsp_spy_ratio          : RSP / SPY price ratio (breadth proxy)
  - rsp_spy_ratio_5d_chg   : 5-day change in RSP/SPY ratio
  - rsp_spy_ratio_20d_chg  : 20-day change in RSP/SPY ratio
  - iwm_spy_ret_spread_20d : IWM 20d return minus SPY 20d return (% above 200 SMA proxy)
  - sector_divergence_flag : 1 if 2+ sectors diverge >1% from SPY daily return
  - sector_divergence_count: number of sectors with |return - SPY return| > 1%
  - breadth_thrust          : RSP 5d return minus SPY 5d return
  - xlf_spy_ret_spread_1d  : XLF daily return minus SPY daily return
  - xlk_spy_ret_spread_1d  : XLK daily return minus SPY daily return
  - xle_spy_ret_spread_1d  : XLE daily return minus SPY daily return
  - vix_term_structure_slope: (^VIX3M - ^VIX) — positive = contango, negative = backwardation
  - rsp_spy_ratio_zscore_21d: z-score of RSP/SPY ratio over 21 days
  - breadth_thrust_cumulative_5d: cumulative breadth_thrust over 5 days

Usage:
  python scripts/build_breadth_features.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TICKERS = ["RSP", "SPY", "^VIX", "^VIX3M", "XLF", "XLK", "XLE", "IWM"]


def fetch_etf_data(start="2018-01-01"):
    """Fetch daily close data for all breadth tickers via yfinance."""
    print(f"Fetching ETF data from {start} to today...")
    print(f"  Tickers: {', '.join(TICKERS)}")

    data = yf.download(
        TICKERS,
        start=start,
        auto_adjust=True,
        progress=True,
    )

    # yf.download returns MultiIndex columns: (Price, Ticker)
    # Extract Close prices for each ticker
    close = data["Close"]

    # Flatten to a simple DataFrame with ticker names as columns
    df = pd.DataFrame(index=close.index)
    for ticker in TICKERS:
        col_name = ticker.replace("^", "").lower()
        if ticker in close.columns:
            df[col_name] = close[ticker]
        else:
            print(f"  WARNING: {ticker} not found in downloaded data")
            df[col_name] = np.nan

    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill missing data (holidays, gaps)
    for col in df.columns:
        if col != "date":
            df[col] = df[col].ffill()

    print(f"  Fetched {len(df)} trading days ({df['date'].min().date()} -> {df['date'].max().date()})")
    null_counts = df.isna().sum()
    if null_counts.sum() > 0:
        print("  Remaining nulls after ffill:")
        for col in null_counts[null_counts > 0].index:
            print(f"    {col}: {null_counts[col]}")

    return df


def compute_breadth_features(df):
    """Compute all breadth/internals features from raw ETF closes."""
    print("\nComputing breadth features...")
    out = pd.DataFrame()
    out["date"] = df["date"]

    # --- VIX / VXV term structure ---
    out["vix_vxv_ratio"] = df["vix"] / df["vix3m"]
    rolling_mean = out["vix_vxv_ratio"].rolling(21, min_periods=10).mean()
    rolling_std = out["vix_vxv_ratio"].rolling(21, min_periods=10).std()
    out["vix_vxv_zscore_21d"] = (out["vix_vxv_ratio"] - rolling_mean) / rolling_std
    out["vix_term_structure_slope"] = df["vix3m"] - df["vix"]

    # --- RSP / SPY breadth proxy ---
    out["rsp_spy_ratio"] = df["rsp"] / df["spy"]
    out["rsp_spy_ratio_5d_chg"] = out["rsp_spy_ratio"].pct_change(5) * 100
    out["rsp_spy_ratio_20d_chg"] = out["rsp_spy_ratio"].pct_change(20) * 100

    rsp_spy_mean = out["rsp_spy_ratio"].rolling(21, min_periods=10).mean()
    rsp_spy_std = out["rsp_spy_ratio"].rolling(21, min_periods=10).std()
    out["rsp_spy_ratio_zscore_21d"] = (out["rsp_spy_ratio"] - rsp_spy_mean) / rsp_spy_std

    # --- % above 200 SMA proxy: IWM vs SPY 20d returns ---
    iwm_ret_20d = df["iwm"].pct_change(20) * 100
    spy_ret_20d = df["spy"].pct_change(20) * 100
    out["iwm_spy_ret_spread_20d"] = iwm_ret_20d - spy_ret_20d

    # --- Sector divergence ---
    spy_ret_1d = df["spy"].pct_change() * 100
    xlf_ret_1d = df["xlf"].pct_change() * 100
    xlk_ret_1d = df["xlk"].pct_change() * 100
    xle_ret_1d = df["xle"].pct_change() * 100

    out["xlf_spy_ret_spread_1d"] = xlf_ret_1d - spy_ret_1d
    out["xlk_spy_ret_spread_1d"] = xlk_ret_1d - spy_ret_1d
    out["xle_spy_ret_spread_1d"] = xle_ret_1d - spy_ret_1d

    # Count sectors diverging > 1% from SPY
    sector_spreads = pd.DataFrame({
        "xlf": (xlf_ret_1d - spy_ret_1d).abs(),
        "xlk": (xlk_ret_1d - spy_ret_1d).abs(),
        "xle": (xle_ret_1d - spy_ret_1d).abs(),
    })
    out["sector_divergence_count"] = (sector_spreads > 1.0).sum(axis=1)
    out["sector_divergence_flag"] = (out["sector_divergence_count"] >= 2).astype(int)

    # --- Breadth thrust ---
    rsp_ret_5d = df["rsp"].pct_change(5) * 100
    spy_ret_5d = df["spy"].pct_change(5) * 100
    out["breadth_thrust"] = rsp_ret_5d - spy_ret_5d
    out["breadth_thrust_cumulative_5d"] = out["breadth_thrust"].rolling(5, min_periods=1).sum()

    # Drop rows where we can't compute features (first ~20 rows)
    out = out.dropna(subset=["vix_vxv_ratio"]).reset_index(drop=True)

    print(f"  Computed {len(out.columns) - 1} features over {len(out)} trading days")
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_etf_data(start="2018-01-01")
    result = compute_breadth_features(df)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Market Breadth Features Summary")
    print(f"{'='*60}")
    print(f"Date range: {result['date'].min().date()} -> {result['date'].max().date()}")
    print(f"Shape: {result.shape[0]} rows x {result.shape[1]} columns")

    feature_cols = [c for c in result.columns if c != "date"]
    print(f"\nFeatures ({len(feature_cols)}):")
    for col in feature_cols:
        vals = result[col].dropna()
        if len(vals) > 0:
            print(f"  {col:35s} mean={vals.mean():8.4f}  std={vals.std():8.4f}  null={result[col].isna().mean()*100:.1f}%")
        else:
            print(f"  {col:35s} ALL NULL")

    outfile = DATA_DIR / "breadth_features.parquet"
    result.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")
    print(f"  File size: {outfile.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
