"""
Build microstructure features from existing pipeline data.

Captures prior IC outcomes, gap fill rates, and intraday auction
metrics — all computed from existing parquet files with no
external API calls.

Inputs:  data/target.parquet, data/spx_features.parquet,
         data/momentum_features.parquet
Output:  data/microstructure_features.parquet (one row per trading day)

Features:
  prev_ic_outcome        — Previous day's IC profitable flag (1/0)
  prev_ic_pnl_pct        — Previous day's IC PnL percentage
  consecutive_wins       — Rolling count of consecutive winning IC days
  consecutive_losses     — Rolling count of consecutive losing IC days
  gap_fill_rate_5d       — 5-day rolling % of gaps that reversed in first 30m
  gap_fill_rate_20d      — 20-day rolling version
  intraday_range_vs_atr  — First-30m range as % of 14d ATR
  overnight_range_pct    — (open - prev_close) / prev_close * 100
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def compute_consecutive_streaks(profitable_series):
    """Compute consecutive win and loss streaks.

    Args:
        profitable_series: Series of 1/0/NaN IC profitable flags.

    Returns:
        Tuple of (consecutive_wins, consecutive_losses) Series.
    """
    wins = []
    losses = []
    win_count = 0
    loss_count = 0

    for val in profitable_series:
        if pd.isna(val):
            wins.append(np.nan)
            losses.append(np.nan)
            continue
        if val == 1.0:
            win_count += 1
            loss_count = 0
        else:
            loss_count += 1
            win_count = 0
        wins.append(win_count)
        losses.append(loss_count)

    return pd.Series(wins, index=profitable_series.index), pd.Series(losses, index=profitable_series.index)


def main():
    # ── Load target data ────────────────────────────────────────────
    print("Loading target data...")
    target_path = DATA_DIR / "target.parquet"
    if not target_path.exists():
        print(f"  ERROR: {target_path} not found")
        return
    target = pd.read_parquet(target_path)
    target["date"] = pd.to_datetime(target["date"])
    target = target.sort_values("date").reset_index(drop=True)
    print(f"  Shape: {target.shape}, date range: {target['date'].min().date()} -> {target['date'].max().date()}")

    # ── Load spx_features (daily rows) ──────────────────────────────
    print("Loading spx_features data...")
    spy_path = DATA_DIR / "spx_features.parquet"
    if not spy_path.exists():
        print(f"  ERROR: {spy_path} not found")
        return
    spy = pd.read_parquet(spy_path)
    spy["date"] = pd.to_datetime(spy["date"])

    # spx_features has intraday rows — aggregate to daily
    # Get first-30m range per day and daily ATR
    spy["datetime"] = pd.to_datetime(spy["datetime"])
    spy = spy.sort_values("datetime")

    daily_atr = spy.groupby("date")["atr_14"].last().reset_index()
    daily_atr.columns = ["date", "atr_14"]

    # Compute first-30m range per day
    print("  Computing first-30m range per day...")
    daily_30m_range = []
    for date, grp in spy.groupby("date"):
        grp = grp.sort_values("datetime")
        first_30m = grp.iloc[:30]
        if len(first_30m) >= 10:
            range_val = first_30m["high"].max() - first_30m["low"].min()
        else:
            range_val = np.nan
        daily_30m_range.append({"date": date, "first_30m_range": range_val})
    daily_30m_range = pd.DataFrame(daily_30m_range)

    # Get daily open and prev close for overnight range
    daily_ohlc = spy.groupby("date").agg(
        d_open=("open", "first"),
        d_close=("close", "last"),
    ).reset_index()
    daily_ohlc["prev_close"] = daily_ohlc["d_close"].shift(1)

    # ── Load momentum features (for gap fill signals) ───────────────
    print("Loading momentum features...")
    mom_path = DATA_DIR / "momentum_features.parquet"
    if not mom_path.exists():
        print(f"  WARNING: {mom_path} not found, gap fill features will be NaN")
        mom = pd.DataFrame(columns=["date", "prev_close_gap_reversal"])
    else:
        mom = pd.read_parquet(mom_path)
    mom["date"] = pd.to_datetime(mom["date"])
    mom = mom.sort_values("date").reset_index(drop=True)
    print(f"  Shape: {mom.shape}")

    # ── Build features ──────────────────────────────────────────────
    print("\nBuilding microstructure features...")

    # Start from target dates as the canonical date index
    result = target[["date"]].copy()

    # 1. prev_ic_outcome: shift(1) on ic_profitable — prior day's result
    result["prev_ic_outcome"] = target["ic_profitable"].shift(1)

    # 2. prev_ic_pnl_pct: shift(1) on ic_pnl_pct
    result["prev_ic_pnl_pct"] = target["ic_pnl_pct"].shift(1)

    # 3-4. Consecutive wins / losses (based on shifted series — prior streaks)
    cons_wins, cons_losses = compute_consecutive_streaks(target["ic_profitable"])
    # Shift by 1 to avoid lookahead — streak count as of prior day
    result["consecutive_wins"] = cons_wins.shift(1)
    result["consecutive_losses"] = cons_losses.shift(1)

    # 5-6. Gap fill rates from momentum features
    if "prev_close_gap_reversal" in mom.columns:
        gap_fill = mom[["date", "prev_close_gap_reversal"]].copy()
        result = result.merge(gap_fill, on="date", how="left")
        result["gap_fill_rate_5d"] = result["prev_close_gap_reversal"].rolling(5, min_periods=1).mean()
        result["gap_fill_rate_20d"] = result["prev_close_gap_reversal"].rolling(20, min_periods=5).mean()
        result = result.drop(columns=["prev_close_gap_reversal"])
    else:
        result["gap_fill_rate_5d"] = np.nan
        result["gap_fill_rate_20d"] = np.nan

    # 7. intraday_range_vs_atr: first-30m range / 14d ATR
    range_atr = daily_30m_range.merge(daily_atr, on="date", how="outer")
    range_atr["intraday_range_vs_atr"] = np.where(
        range_atr["atr_14"] > 0,
        range_atr["first_30m_range"] / range_atr["atr_14"] * 100,
        np.nan,
    )
    result = result.merge(range_atr[["date", "intraday_range_vs_atr"]], on="date", how="left")

    # 8. overnight_range_pct: (open - prev_close) / prev_close * 100
    overnight = daily_ohlc[["date", "d_open", "prev_close"]].copy()
    overnight["overnight_range_pct"] = np.where(
        overnight["prev_close"] > 0,
        (overnight["d_open"] - overnight["prev_close"]) / overnight["prev_close"] * 100,
        np.nan,
    )
    result = result.merge(overnight[["date", "overnight_range_pct"]], on="date", how="left")

    # ── Finalize ────────────────────────────────────────────────────
    result = result.sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])

    feature_cols = [c for c in result.columns if c != "date"]

    print(f"\n{'='*60}")
    print(f"Microstructure features: {result.shape[0]} rows x {result.shape[1]} columns")
    print(f"Date range: {result['date'].min().date()} -> {result['date'].max().date()}")

    null_pcts = result[feature_cols].isna().mean() * 100
    print("\nNull rates:")
    for col in feature_cols:
        if null_pcts[col] > 0:
            print(f"  {col:30s} {null_pcts[col]:.1f}%")

    print("\nSample stats:")
    for col in feature_cols:
        valid = result[col].dropna()
        if len(valid) > 0:
            print(f"  {col:30s} mean={valid.mean():8.4f}  std={valid.std():8.4f}  min={valid.min():8.4f}  max={valid.max():8.4f}")

    outfile = DATA_DIR / "microstructure_features.parquet"
    result.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
