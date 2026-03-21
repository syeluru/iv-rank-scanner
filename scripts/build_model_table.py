"""
Build the final model table: one row per trading day with all features + target.

Inputs:
  data/spy_features.parquet      — 95 SPY technical indicators (per-minute, daily cols broadcast)
  data/options_features.parquet  — options-derived features (per day)
  data/target.parquet            — IC simulation outcomes (per day)

Output:
  data/model_table.parquet       — one row per trading day, all features + target

Also drops:
  - Rows where target is NaN (no valid IC simulation)
  - Lookahead-prone columns (e.g. d_close, d_high, d_low — known only at EOD)
  - Raw price levels (use % differences instead to avoid non-stationarity)
  - Duplicate/redundant columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Columns to drop from spy_features (lookahead or non-stationary raw levels)
SPY_DROP = [
    # Raw price levels (non-stationary)
    "open", "high", "low", "close", "volume",
    "d_open", "d_high", "d_low", "d_close", "d_volume",
    "prev_close",
    # Moving average levels (use *_vs_sma_* % differences instead)
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_5", "ema_20",
    "bb_upper", "bb_lower",
    "atr_14",
    # MACD levels (keep hist, it's a difference)
    "macd_line", "macd_signal",
    # Intraday (minute-level, not useful for daily model)
    "datetime", "time_sin", "time_cos", "minutes_since_open", "minutes_to_close",
    # VIX levels (keep vs_sma and ratio features)
    "vix_sma_5", "vix_sma_10", "vix_sma_20",
    "vol_sma_20",
]

# Columns from options_features to exclude from model features
OPTIONS_META = ["spx_open", "atm_strike"]  # keep for reference but not as model features


def main():
    print("Loading feature tables...")

    # ── SPY features (1-min table → collapse to one row per day) ──
    spy = pd.read_parquet(DATA_DIR / "spy_features.parquet")
    spy["date"] = pd.to_datetime(spy["date"])

    print(f"  spy_features: {spy.shape}")

    # All daily features are already broadcast — take the first row per day
    daily_cols = [c for c in spy.columns if c not in
                  ["date", "datetime", "open", "high", "low", "close", "volume",
                   "time_sin", "time_cos", "minutes_since_open", "minutes_to_close"]]

    # Get the last row per day (EOD) for daily cols — some like rsi need the final value
    # Actually for daily features they're identical across all minute rows, so first is fine
    spy_daily = spy.groupby("date")[daily_cols].first().reset_index()
    print(f"  spy_daily (collapsed): {spy_daily.shape}")

    # Drop unwanted columns
    spy_drop_actual = [c for c in SPY_DROP if c in spy_daily.columns]
    spy_daily = spy_daily.drop(columns=spy_drop_actual, errors="ignore")

    # ── Options features ──
    opts = pd.read_parquet(DATA_DIR / "options_features.parquet")
    opts["date"] = pd.to_datetime(opts["date"])
    print(f"  options_features: {opts.shape}")

    # ── Target ──
    target = pd.read_parquet(DATA_DIR / "target.parquet")
    target["date"] = pd.to_datetime(target["date"])
    print(f"  target: {target.shape}")

    # Keep only the target + metadata from target table
    target_cols = ["date", "spx_open", "spx_close", "atm_strike",
                   "short_call_strike", "short_put_strike",
                   "ic_credit", "ic_pnl_per_share", "ic_pnl_pct",
                   "ic_profitable", "ic_tp25_hit"]
    target = target[[c for c in target_cols if c in target.columns]]

    # ── Join ──
    print("\nJoining tables...")
    df = spy_daily.merge(opts, on="date", how="inner", suffixes=("", "_opts"))
    df = df.merge(target, on="date", how="inner")
    print(f"  After join: {df.shape}")

    # ── Drop rows with no target ──
    before = len(df)
    df = df.dropna(subset=["ic_profitable"])
    print(f"  After dropping missing target: {len(df)} rows (dropped {before - len(df)})")

    # ── Drop duplicate columns (from join suffixes) ──
    dup_cols = [c for c in df.columns if c.endswith("_opts")]
    df = df.drop(columns=dup_cols, errors="ignore")

    # ── Sort chronologically ──
    df = df.sort_values("date").reset_index(drop=True)

    # ── Summary ──
    feature_cols = [c for c in df.columns if c not in
                    ["date", "spx_open", "spx_close", "atm_strike",
                     "short_call_strike", "short_put_strike",
                     "ic_credit", "ic_pnl_per_share", "ic_pnl_pct",
                     "ic_profitable", "ic_tp25_hit"]]

    print(f"\n{'='*60}")
    print(f"Model table: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nTarget distribution (ic_profitable):")
    vc = df["ic_profitable"].value_counts()
    print(f"  Win  (1): {vc.get(1.0, 0):4d} ({vc.get(1.0, 0)/len(df):.1%})")
    print(f"  Loss (0): {vc.get(0.0, 0):4d} ({vc.get(0.0, 0)/len(df):.1%})")

    # High null-rate features
    null_rates = df[feature_cols].isna().mean()
    high_null = null_rates[null_rates > 0.20]
    if len(high_null) > 0:
        print(f"\nHigh null-rate features (>20%):")
        for col, rate in high_null.items():
            print(f"  {col:35s} {rate:.1%}")

    outfile = DATA_DIR / "model_table.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.2f} MB)")

    # Save feature list for reference
    with open(PROJECT_DIR / "feature_list.txt", "w") as f:
        f.write(f"# Features used in model ({len(feature_cols)} total)\n")
        f.write(f"# Generated from model_table.parquet\n\n")
        for col in sorted(feature_cols):
            f.write(f"{col}\n")
    print(f"Feature list saved to feature_list.txt")


if __name__ == "__main__":
    main()
