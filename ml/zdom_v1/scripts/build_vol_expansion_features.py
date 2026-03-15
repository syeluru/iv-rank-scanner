"""
Build expanded volatility surface features.

Adds VVIX (vol-of-vol), realized vs implied vol spreads,
and post-FOMC vol crush signals.

Inputs:  yfinance (^VVIX), data/vix_daily.parquet, data/spx_daily.parquet,
         data/fomc_dates.parquet (optional)
Output:  data/vol_expansion_features.parquet (one row per trading day)

Features produced:
  - vvix_close             : raw VVIX close
  - vvix_ma20              : 20-day moving average of VVIX
  - vvix_zscore_21d        : z-score of VVIX over 21-day window
  - vvix_vs_vix_ratio      : VVIX / VIX — measures how jumpy implied vol is relative to level
  - iv_crush_post_fomc     : 1 if within 2 trading days after FOMC, else 0
  - rv_iv_spread_10d       : 10-day realized vol minus ATM IV (positive = vol underpriced)
  - rv_iv_spread_20d       : 20-day version
  - vol_regime_persistence : rolling 20d autocorrelation of VIX daily changes

Usage:
  python scripts/build_vol_expansion_features.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

START_DATE = "2018-01-01"


def fetch_vvix():
    """Fetch VVIX (vol of vol) from yfinance."""
    print("Fetching ^VVIX from yfinance...")
    vvix = yf.download("^VVIX", start=START_DATE, auto_adjust=True, progress=False)
    if vvix.empty:
        print("  WARNING: No VVIX data returned from yfinance")
        return pd.DataFrame(columns=["date", "vvix_close"])

    vvix = vvix.reset_index()

    # Handle MultiIndex columns from yfinance
    if isinstance(vvix.columns, pd.MultiIndex):
        vvix.columns = [col[0] if isinstance(col, tuple) else col for col in vvix.columns]

    vvix = vvix.rename(columns={"Date": "date", "Close": "vvix_close"})
    vvix["date"] = pd.to_datetime(vvix["date"]).dt.tz_localize(None)
    vvix = vvix[["date", "vvix_close"]].dropna()
    print(f"  VVIX: {len(vvix)} rows ({vvix['date'].min().date()} → {vvix['date'].max().date()})")
    return vvix


def load_vix():
    """Load VIX daily data from parquet."""
    path = DATA_DIR / "vix_daily.parquet"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame(columns=["date", "vix_close"])

    vix = pd.read_parquet(path)
    vix["date"] = pd.to_datetime(vix["date"])
    print(f"  VIX daily: {len(vix)} rows")
    return vix


def load_spx():
    """Load SPX daily data from parquet."""
    path = DATA_DIR / "spx_daily.parquet"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame(columns=["date", "spx_close"])

    spx = pd.read_parquet(path)
    spx["date"] = pd.to_datetime(spx["date"])
    print(f"  SPX daily: {len(spx)} rows")
    return spx


def load_fomc_dates():
    """Load FOMC meeting dates if available."""
    path = DATA_DIR / "fomc_dates.parquet"
    if not path.exists():
        print("  FOMC dates: not found, iv_crush_post_fomc will be NaN")
        return None

    fomc = pd.read_parquet(path)
    # Expect a column with FOMC dates — try common names
    date_col = None
    for col in ["date", "fomc_date", "meeting_date"]:
        if col in fomc.columns:
            date_col = col
            break
    if date_col is None:
        date_col = fomc.columns[0]

    fomc_dates = pd.to_datetime(fomc[date_col]).sort_values().reset_index(drop=True)
    print(f"  FOMC dates: {len(fomc_dates)} meetings loaded")
    return fomc_dates


def compute_days_since_fomc(dates, fomc_dates):
    """For each trading date, compute business days since the most recent FOMC meeting."""
    if fomc_dates is None:
        return pd.Series(np.nan, index=dates.index)

    fomc_arr = fomc_dates.values
    result = []
    for d in dates:
        past = fomc_arr[fomc_arr <= np.datetime64(d)]
        if len(past) == 0:
            result.append(np.nan)
        else:
            last_fomc = pd.Timestamp(past[-1])
            bdays = np.busday_count(last_fomc.date(), d.date())
            result.append(bdays)
    return pd.Series(result, index=dates.index)


def compute_realized_vol(log_returns, window):
    """Compute annualized realized vol over a rolling window."""
    return log_returns.rolling(window).std() * np.sqrt(252)


def rolling_autocorrelation(series, window):
    """Compute rolling autocorrelation (lag-1) over a window."""
    result = series.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan,
        raw=False
    )
    return result


def main():
    print("Loading data...")
    vvix = fetch_vvix()
    vix = load_vix()
    spx = load_spx()
    fomc_dates = load_fomc_dates()

    # Start with VVIX as the base frame
    if vvix.empty:
        print("\nERROR: Cannot build features without VVIX data. Exiting.")
        return

    df = vvix.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ── VVIX features ──
    print("\nComputing VVIX features...")
    df["vvix_ma20"] = df["vvix_close"].rolling(20, min_periods=10).mean()

    vvix_roll_mean = df["vvix_close"].rolling(21, min_periods=10).mean()
    vvix_roll_std = df["vvix_close"].rolling(21, min_periods=10).std()
    df["vvix_zscore_21d"] = (df["vvix_close"] - vvix_roll_mean) / vvix_roll_std

    # ── Merge VIX ──
    if not vix.empty:
        df = df.merge(vix[["date", "vix_close"]], on="date", how="left")
        df["vvix_vs_vix_ratio"] = df["vvix_close"] / df["vix_close"]
    else:
        df["vix_close"] = np.nan
        df["vvix_vs_vix_ratio"] = np.nan

    # ── FOMC vol crush signal ──
    print("Computing FOMC vol crush signal...")
    days_since = compute_days_since_fomc(df["date"], fomc_dates)
    df["iv_crush_post_fomc"] = (days_since <= 2).astype(float)
    df.loc[days_since.isna(), "iv_crush_post_fomc"] = np.nan

    # ── Realized vs Implied vol spreads ──
    print("Computing realized vs implied vol spreads...")
    if not spx.empty:
        spx_sorted = spx.sort_values("date").reset_index(drop=True)
        spx_sorted["spx_log_return"] = np.log(
            spx_sorted["spx_close"] / spx_sorted["spx_close"].shift(1)
        )

        spx_sorted["rv_10d"] = compute_realized_vol(spx_sorted["spx_log_return"], 10)
        spx_sorted["rv_20d"] = compute_realized_vol(spx_sorted["spx_log_return"], 20)

        df = df.merge(
            spx_sorted[["date", "rv_10d", "rv_20d"]],
            on="date", how="left"
        )

        # Use VIX as ATM IV proxy (VIX ≈ 30-day ATM implied vol, annualized, in %)
        # Convert VIX from percentage to decimal for comparison with realized vol
        vix_decimal = df["vix_close"] / 100.0
        df["rv_iv_spread_10d"] = df["rv_10d"] - vix_decimal
        df["rv_iv_spread_20d"] = df["rv_20d"] - vix_decimal
        df = df.drop(columns=["rv_10d", "rv_20d"])
    else:
        df["rv_iv_spread_10d"] = np.nan
        df["rv_iv_spread_20d"] = np.nan

    # ── Vol regime persistence ──
    print("Computing vol regime persistence...")
    if "vix_close" in df.columns and df["vix_close"].notna().sum() > 20:
        vix_changes = df["vix_close"].diff()
        df["vol_regime_persistence"] = rolling_autocorrelation(vix_changes, 20)
    else:
        df["vol_regime_persistence"] = np.nan

    # ── Select and order output columns ──
    out_cols = [
        "date",
        "vvix_close",
        "vvix_ma20",
        "vvix_zscore_21d",
        "vvix_vs_vix_ratio",
        "iv_crush_post_fomc",
        "rv_iv_spread_10d",
        "rv_iv_spread_20d",
        "vol_regime_persistence",
    ]
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].copy()
    out = out.sort_values("date").reset_index(drop=True)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Vol expansion features: {out.shape[0]} rows x {out.shape[1]} columns")
    print(f"Date range: {out['date'].min().date()} → {out['date'].max().date()}")

    null_pcts = out.isna().mean() * 100
    print("\nNull rates:")
    for col in out.columns:
        if col == "date":
            continue
        print(f"  {col:30s} {null_pcts[col]:.1f}%")

    # ── Save ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile = DATA_DIR / "vol_expansion_features.parquet"
    out.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
