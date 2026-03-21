"""
Fetch macro regime features from FRED API and yfinance.

Pulls and caches:
  - Fed Funds Rate (DFF — daily effective rate)
  - 2Y Treasury yield (DGS2)
  - 10Y Treasury yield (DGS10)
  - 2Y/10Y spread (computed)
  - WTI Crude Oil (DCOILWTICO)
  - Gold (GOLDAMGBD228NLBM — London AM fix in USD)
  - DXY (via yfinance ^DXY or DTWEXBGS from FRED)
  - MOVE Index (ICE BofA MOVE Index via FRED: BAMLMOVE)
  - IG credit spread (BAMLC0A0CM — option-adjusted spread)
  - HY credit spread (BAMLH0A0HYM2 — option-adjusted spread)

Saves to: data/macro_regime.parquet

Setup:
  Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
  Add to .env: FRED_API_KEY=your_key_here

Usage:
  python3 scripts/fetch_macro_features.py
  python3 scripts/fetch_macro_features.py --start 2020-01-01
  python3 scripts/fetch_macro_features.py --no-cache  # force full refresh
"""

import argparse
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# Load .env
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env.development"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "macro_regime.parquet"

FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Default lookback if no existing data
DEFAULT_START = "2018-01-01"

# ── FRED series definitions ────────────────────────────────────────────────────
FRED_SERIES = {
    # Fed policy
    "fed_funds_rate":     "DFF",           # Daily effective fed funds rate
    # Treasury yields
    "yield_2y":           "DGS2",          # 2-year constant maturity
    "yield_10y":          "DGS10",         # 10-year constant maturity
    # Oil
    "wti_crude":          "DCOILWTICO",    # WTI crude oil spot price (daily)
    # Gold
    "gold_fix_am":        "GOLDAMGBD228NLBM",  # London Gold fix AM (USD/troy oz)
    # Dollar
    "dxy_broad":          "DTWEXBGS",      # Broad Real Dollar Index (weekly, interpolated)
    # Volatility / Credit
    "move_index":         "BAMLMOVE",      # ICE BofA MOVE Index (bond vol)
    "ig_spread":          "BAMLC0A0CM",    # IG corp OAS (bps)
    "hy_spread":          "BAMLH0A0HYM2",  # HY corp OAS (bps)
}

# Series with different frequency (will be forward-filled to daily)
LOWER_FREQ_SERIES = {"dxy_broad"}


# ── FRED fetch ─────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str, end: str,
                      api_key: str, retries: int = 3) -> pd.Series:
    """
    Fetch a FRED series and return a daily pd.Series indexed by date.
    Missing observations are returned as NaN.
    """
    params = {
        "series_id":    series_id,
        "observation_start": start,
        "observation_end":   end,
        "file_type":    "json",
        "frequency":    "d",     # daily aggregation (FRED will auto-aggregate)
        "aggregation_method": "avg",
    }
    if api_key:
        params["api_key"] = api_key

    for attempt in range(retries):
        try:
            resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "observations" not in data:
                print(f"  [warn] {series_id}: no observations in response")
                return pd.Series(dtype=float)

            rows = [
                (obs["date"], float(obs["value"]))
                for obs in data["observations"]
                if obs["value"] not in (".", "")
            ]
            if not rows:
                return pd.Series(dtype=float)

            dates, values = zip(*rows)
            s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
            return s

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 400:
                # Bad series or invalid param — don't retry
                print(f"  [warn] {series_id}: HTTP 400 (bad request) — skipping")
                return pd.Series(dtype=float)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] {series_id}: HTTP error after {retries} tries: {e}")
                return pd.Series(dtype=float)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] {series_id}: {e}")
                return pd.Series(dtype=float)


# ── DXY fallback via yfinance ──────────────────────────────────────────────────

def fetch_dxy_yfinance(start: str, end: str) -> pd.Series:
    """Fetch DXY from yfinance as fallback."""
    try:
        import yfinance as yf
        df = yf.download("DX-Y.NYB", start=start, end=end,
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"].squeeze()
        close.index = pd.to_datetime(close.index).normalize()
        close.name = "dxy"
        return close.astype(float)
    except Exception as e:
        print(f"  [warn] DXY yfinance fallback failed: {e}")
        return pd.Series(dtype=float)


# ── Derived features ───────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed macro features to the DataFrame."""

    # Yield curve spread (2Y/10Y)
    if "yield_2y" in df.columns and "yield_10y" in df.columns:
        df["yield_spread_2y10y"] = df["yield_10y"] - df["yield_2y"]
        df["yield_curve_inverted"] = (df["yield_spread_2y10y"] < 0).astype(int)

    # Yield changes (1W, 1M, 3M)
    for col in ["yield_2y", "yield_10y", "yield_spread_2y10y"]:
        if col in df.columns:
            df[f"{col}_chg_5d"]  = df[col].diff(5)
            df[f"{col}_chg_21d"] = df[col].diff(21)
            df[f"{col}_chg_63d"] = df[col].diff(63)

    # Fed policy direction (hiking=1, cutting=-1, neutral=0)
    if "fed_funds_rate" in df.columns:
        df["fed_funds_chg_21d"] = df["fed_funds_rate"].diff(21)
        df["fed_policy_direction"] = np.where(
            df["fed_funds_chg_21d"] > 0.05,  1,   # hiking
            np.where(df["fed_funds_chg_21d"] < -0.05, -1,  # cutting
                     0)                                      # neutral
        )

    # DXY trend
    dxy_col = "dxy" if "dxy" in df.columns else "dxy_broad"
    if dxy_col in df.columns:
        df["dxy_ma20"]     = df[dxy_col].rolling(20).mean()
        df["dxy_trend"]    = df[dxy_col] / df["dxy_ma20"] - 1  # % above/below MA
        df["dxy_chg_5d"]   = df[dxy_col].diff(5)
        df["dxy_chg_20d"]  = df[dxy_col].diff(20)

    # Oil trend
    if "wti_crude" in df.columns:
        df["oil_ma20"]    = df["wti_crude"].rolling(20).mean()
        df["oil_trend"]   = df["wti_crude"] / df["oil_ma20"] - 1
        df["oil_chg_5d"]  = df["wti_crude"].diff(5)
        df["oil_chg_20d"] = df["wti_crude"].diff(20)

    # Gold trend + Gold/SPY ratio (added later when SPX available)
    if "gold_fix_am" in df.columns:
        df["gold_ma20"]    = df["gold_fix_am"].rolling(20).mean()
        df["gold_trend"]   = df["gold_fix_am"] / df["gold_ma20"] - 1
        df["gold_chg_5d"]  = df["gold_fix_am"].diff(5)
        df["gold_chg_20d"] = df["gold_fix_am"].diff(20)

    # Credit stress indicator (HY - IG spread differential)
    if "hy_spread" in df.columns and "ig_spread" in df.columns:
        df["credit_stress"] = df["hy_spread"] - df["ig_spread"]
        df["credit_stress_chg_20d"] = df["credit_stress"].diff(20)

    # MOVE index trend (bond volatility leading indicator)
    if "move_index" in df.columns:
        df["move_ma20"]    = df["move_index"].rolling(20).mean()
        df["move_trend"]   = df["move_index"] / df["move_ma20"] - 1
        df["move_chg_5d"]  = df["move_index"].diff(5)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch FRED macro features")
    parser.add_argument("--start",    default=None,
                        help="Start date (YYYY-MM-DD, default: resume from last record)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force full refresh from DEFAULT_START")
    args = parser.parse_args()

    # ── Determine date range ──
    end_date = str(date.today())

    if args.no_cache or not OUTPUT_FILE.exists():
        start_date = args.start or DEFAULT_START
        existing_df = None
        print(f"Full fetch from {start_date} → {end_date}")
    else:
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        last_date  = existing_df["date"].max()
        start_date = args.start or str((last_date - timedelta(days=5)).date())
        print(f"Incremental fetch from {start_date} → {end_date}")
        print(f"  Existing: {len(existing_df)} rows through {last_date.date()}")

    if not FRED_API_KEY:
        print("\n[warn] FRED_API_KEY not set — requests may fail.")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Add to .env.development: FRED_API_KEY=your_key_here\n")

    # ── Build date index (business days) ──
    date_idx = pd.bdate_range(start=start_date, end=end_date)

    # ── Fetch all FRED series ──
    series_dict = {}
    for col_name, series_id in FRED_SERIES.items():
        print(f"  Fetching {series_id:25s} ({col_name})...")
        s = fetch_fred_series(series_id, start_date, end_date, FRED_API_KEY)
        if not s.empty:
            s.index = pd.to_datetime(s.index).normalize()
            series_dict[col_name] = s
            print(f"    Got {len(s)} observations")
        else:
            print(f"    [empty]")

    # ── DXY fallback via yfinance ──
    if "dxy_broad" not in series_dict or series_dict.get("dxy_broad", pd.Series()).empty:
        print(f"  Fetching DXY via yfinance (FRED fallback)...")
        dxy_s = fetch_dxy_yfinance(start_date, end_date)
        if not dxy_s.empty:
            series_dict["dxy"] = dxy_s
            print(f"    Got {len(dxy_s)} observations")

    if not series_dict:
        print("\n[error] No data fetched — check FRED_API_KEY and connectivity.")
        sys.exit(1)

    # ── Align to business day index ──
    df_new = pd.DataFrame(index=date_idx)
    df_new.index.name = "date"

    for col_name, s in series_dict.items():
        s = s[~s.index.duplicated(keep="last")]
        df_new[col_name] = s.reindex(df_new.index)
        # Forward-fill lower-frequency series (DTWEXBGS is weekly)
        if col_name in LOWER_FREQ_SERIES or col_name == "dxy_broad":
            df_new[col_name] = df_new[col_name].fillna(method="ffill", limit=7)

    # Forward-fill all other series up to 5 business days (handles weekends/holidays)
    df_new = df_new.fillna(method="ffill", limit=5)

    # ── Add derived features ──
    df_new = add_derived_features(df_new)

    df_new = df_new.reset_index()
    df_new["date"] = pd.to_datetime(df_new["date"])

    # ── Merge with existing ──
    if existing_df is not None and not args.no_cache:
        # Remove overlap from existing
        existing_df = existing_df[existing_df["date"] < df_new["date"].min()]
        # Align columns
        all_cols = sorted(set(existing_df.columns) | set(df_new.columns))
        for c in all_cols:
            if c not in existing_df.columns:
                existing_df[c] = np.nan
            if c not in df_new.columns:
                df_new[c] = np.nan
        df_final = pd.concat([existing_df[all_cols], df_new[all_cols]], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # ── Save ──
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_final)} rows to {OUTPUT_FILE}")
    print(f"Date range: {df_final['date'].min().date()} → {df_final['date'].max().date()}")
    print(f"Features: {len(df_final.columns) - 1}")

    # Print snapshot of latest values
    print(f"\nLatest macro snapshot:")
    latest = df_final.tail(1).iloc[0]
    display_cols = [
        ("fed_funds_rate",       "Fed Funds Rate",  "%.2f%%"),
        ("yield_2y",             "2Y Yield",        "%.2f%%"),
        ("yield_10y",            "10Y Yield",       "%.2f%%"),
        ("yield_spread_2y10y",   "2/10Y Spread",    "%.2f%%"),
        ("yield_curve_inverted", "Inverted",        "%d"),
        ("wti_crude",            "WTI Crude",       "$%.2f"),
        ("gold_fix_am",          "Gold",            "$%.2f"),
        ("ig_spread",            "IG Spread",       "%.0f bps"),
        ("hy_spread",            "HY Spread",       "%.0f bps"),
        ("move_index",           "MOVE Index",      "%.1f"),
    ]
    for col, label, fmt in display_cols:
        if col in latest and not pd.isna(latest[col]):
            print(f"  {label:20s}: {fmt % latest[col]}")


if __name__ == "__main__":
    main()
