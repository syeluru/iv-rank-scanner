#!/usr/bin/env python3
"""
Build 15-delta and 20-delta training CSVs by joining 10-delta features
with 15/20-delta targets from the backtest sweep cache.

All 26 v5 features are delta-independent (SPX price action + time), so we
reuse them from the existing 10-delta training CSV and swap in delta-specific
target columns (hit_25pct, hit_50pct, entry_credit, pnl_at_close, strikes).

Usage:
    python scripts/build_delta_training_csv.py
    python scripts/build_delta_training_csv.py --delta 0.15
    python scripts/build_delta_training_csv.py --delta 0.20
    python scripts/build_delta_training_csv.py --delta 0.15 --delta 0.20
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
SWEEP_CACHE_DIR = TRAINING_DATA_DIR / "cache" / "backtest_sweep"

# Columns from sweep cache that replace 10-delta values
TARGET_COLS = [
    "hit_25pct", "hit_50pct", "entry_credit", "pnl_at_close",
    "short_put_strike", "short_call_strike", "underlying_price",
]


def load_sweep_targets(delta: float) -> pd.DataFrame:
    """Load all backtest sweep JSON files for a given delta into a DataFrame."""
    suffix = f"_d{delta:.2f}.json"
    files = sorted(SWEEP_CACHE_DIR.glob(f"*{suffix}"))
    logger.info(f"  Found {len(files)} cache files for delta={delta}")

    rows = []
    for f in files:
        with open(f) as fh:
            entries = json.load(fh)
        rows.extend(entries)

    df = pd.DataFrame(rows)
    logger.info(f"  Loaded {len(df)} rows from sweep cache")
    return df


def build_delta_csv(base_df: pd.DataFrame, delta: float, output_path: Path):
    """Join 10-delta features with delta-specific targets."""
    logger.info(f"\nBuilding {delta}-delta training CSV...")

    sweep_df = load_sweep_targets(delta)
    if sweep_df.empty:
        logger.error(f"  No sweep data for delta={delta}")
        return

    # Ensure join keys are strings
    base_df["trade_date"] = base_df["trade_date"].astype(str)
    base_df["entry_time"] = base_df["entry_time"].astype(str)
    sweep_df["trade_date"] = sweep_df["trade_date"].astype(str)
    sweep_df["entry_time"] = sweep_df["entry_time"].astype(str)

    # Drop target columns from base (will be replaced)
    # Also drop time_to_*_min and max_adverse_pct, win_50pct_or_close
    drop_cols = TARGET_COLS + [
        "time_to_25pct_min", "time_to_50pct_min",
        "max_adverse_pct", "win_50pct_or_close",
    ]
    feature_df = base_df.drop(columns=[c for c in drop_cols if c in base_df.columns])

    # Keep only target columns from sweep
    sweep_targets = sweep_df[["trade_date", "entry_time"] + [c for c in TARGET_COLS if c in sweep_df.columns]]

    # Inner join on (trade_date, entry_time)
    merged = feature_df.merge(sweep_targets, on=["trade_date", "entry_time"], how="inner")

    logger.info(f"  Base rows: {len(base_df)}")
    logger.info(f"  Sweep rows: {len(sweep_df)}")
    logger.info(f"  Merged rows: {len(merged)} (inner join)")

    # Log base rates
    if "hit_25pct" in merged.columns:
        rate_25 = merged["hit_25pct"].mean()
        logger.info(f"  hit_25pct base rate: {rate_25:.1%}")
    if "hit_50pct" in merged.columns:
        rate_50 = merged["hit_50pct"].mean()
        logger.info(f"  hit_50pct base rate: {rate_50:.1%}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  Columns: {len(merged.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Build delta-specific training CSVs")
    parser.add_argument(
        "--delta", type=float, action="append", default=None,
        help="Delta level(s) to build (default: 0.15 and 0.20)",
    )
    parser.add_argument(
        "--base-csv", type=Path,
        default=TRAINING_DATA_DIR / "training_dataset_multi_w25.csv",
        help="Path to 10-delta base training CSV",
    )
    args = parser.parse_args()

    deltas = args.delta or [0.15, 0.20]

    logger.info("Loading base 10-delta training CSV...")
    base_df = pd.read_csv(args.base_csv)
    logger.info(f"  Loaded {len(base_df)} rows, {len(base_df.columns)} columns")

    for delta in deltas:
        suffix = f"d{int(delta*100)}"
        output_path = TRAINING_DATA_DIR / f"training_dataset_multi_w25_{suffix}.csv"
        build_delta_csv(base_df, delta, output_path)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
