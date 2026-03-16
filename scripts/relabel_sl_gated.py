#!/usr/bin/env python3
"""
Re-label training dataset with stop-loss-gated TP targets.

Reads existing training CSV, re-fetches leg timeseries from LOCAL CACHE,
and computes new labels where TP is only counted if it occurs BEFORE stop loss.

Two SL levels: 200% (debit = 2x credit) and 300% (debit = 3x credit).

New columns added:
  - hit_25pct_sl200, hit_50pct_sl200  (TP gated by 200% SL)
  - hit_25pct_sl300, hit_50pct_sl300  (TP gated by 300% SL)
  - time_to_sl200_min, time_to_sl300_min  (when SL was hit)
  - sl200_before_tp25, sl200_before_tp50  (debug: was SL hit before TP?)
  - sl300_before_tp25, sl300_before_tp50

Usage:
    python scripts/relabel_sl_gated.py
    python scripts/relabel_sl_gated.py --input ml/training_data/training_dataset_multi_w25.csv
"""

import argparse
import json
import sys
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "ml" / "training_data" / "cache"
LEGS_CACHE = CACHE_DIR / "legs"


def load_leg_timeseries(trade_date: date, strike: float, right: str,
                        start_time: dtime, end_time: dtime) -> list:
    """Load leg timeseries from local cache (no API calls)."""
    s = f"{start_time.hour:02d}{start_time.minute:02d}"
    e = f"{end_time.hour:02d}{end_time.minute:02d}"
    cache_file = LEGS_CACHE / f"{trade_date.isoformat()}_{right}_{strike}_{s}_{e}.json"

    if not cache_file.exists():
        return []

    with open(cache_file) as f:
        return json.load(f)


def compute_sl_gated_labels(trade_date: date, entry_time: dtime,
                            credit: float, short_put: float, short_call: float,
                            long_put: float, long_call: float) -> dict:
    """Compute TP labels gated by stop loss at 200% and 300%."""
    exit_time = dtime(15, 0)

    defaults = {
        "hit_25pct_sl200": None, "hit_50pct_sl200": None,
        "hit_25pct_sl300": None, "hit_50pct_sl300": None,
        "time_to_sl200_min": None, "time_to_sl300_min": None,
        "sl200_before_tp25": None, "sl200_before_tp50": None,
        "sl300_before_tp25": None, "sl300_before_tp50": None,
        "pnl_at_close_sl200": None, "pnl_at_close_sl300": None,
    }

    legs = [
        (short_put, "put", -1),
        (long_put, "put", 1),
        (short_call, "call", -1),
        (long_call, "call", 1),
    ]

    # Load all leg timeseries from cache
    leg_data = {}
    for strike, right, direction in legs:
        data = load_leg_timeseries(trade_date, strike, right, entry_time, exit_time)
        if not data:
            return defaults
        leg_data[(strike, right, direction)] = data

    min_len = min(len(d) for d in leg_data.values())
    if min_len < 2:
        return defaults

    # Compute debit at each timestamp
    debits = []
    for i in range(min_len):
        debit = 0.0
        for (strike, right, direction), data in leg_data.items():
            pt = data[i]
            if direction == -1:  # Short: buy back at ask
                debit += pt.get("ask", 0) or 0
            else:  # Long: sell at bid
                debit -= pt.get("bid", 0) or 0
        debits.append(debit)

    debits = np.array(debits)

    # TP thresholds (debit must drop TO this level = profit taken)
    target_25_debit = credit * 0.75
    target_50_debit = credit * 0.50

    # SL thresholds (debit rises TO this level = stopped out)
    sl_200_debit = credit * 2.0   # 200% of credit = 100% loss
    sl_300_debit = credit * 3.0   # 300% of credit = 200% loss

    # Find first occurrence indices
    tp25_indices = np.where(debits <= target_25_debit)[0]
    tp50_indices = np.where(debits <= target_50_debit)[0]
    sl200_indices = np.where(debits >= sl_200_debit)[0]
    sl300_indices = np.where(debits >= sl_300_debit)[0]

    first_tp25 = int(tp25_indices[0]) if len(tp25_indices) > 0 else None
    first_tp50 = int(tp50_indices[0]) if len(tp50_indices) > 0 else None
    first_sl200 = int(sl200_indices[0]) if len(sl200_indices) > 0 else None
    first_sl300 = int(sl300_indices[0]) if len(sl300_indices) > 0 else None

    # --- SL200 gated labels ---
    if first_sl200 is not None:
        # SL200 was hit — only count TP if it happened BEFORE SL
        hit_25_sl200 = 1 if (first_tp25 is not None and first_tp25 < first_sl200) else 0
        hit_50_sl200 = 1 if (first_tp50 is not None and first_tp50 < first_sl200) else 0
        sl200_before_tp25 = 1 if (first_tp25 is None or first_sl200 < first_tp25) else 0
        sl200_before_tp50 = 1 if (first_tp50 is None or first_sl200 < first_tp50) else 0
        # PnL = stop loss hit, so loss = -(sl_200_debit - credit)
        pnl_sl200 = float(credit - sl_200_debit)  # negative
    else:
        # SL200 never hit — TP labels are same as original
        hit_25_sl200 = 1 if first_tp25 is not None else 0
        hit_50_sl200 = 1 if first_tp50 is not None else 0
        sl200_before_tp25 = 0
        sl200_before_tp50 = 0
        pnl_sl200 = float(credit - debits[-1])  # held to close

    # --- SL300 gated labels ---
    if first_sl300 is not None:
        hit_25_sl300 = 1 if (first_tp25 is not None and first_tp25 < first_sl300) else 0
        hit_50_sl300 = 1 if (first_tp50 is not None and first_tp50 < first_sl300) else 0
        sl300_before_tp25 = 1 if (first_tp25 is None or first_sl300 < first_tp25) else 0
        sl300_before_tp50 = 1 if (first_tp50 is None or first_sl300 < first_tp50) else 0
        pnl_sl300 = float(credit - sl_300_debit)
    else:
        hit_25_sl300 = 1 if first_tp25 is not None else 0
        hit_50_sl300 = 1 if first_tp50 is not None else 0
        sl300_before_tp25 = 0
        sl300_before_tp50 = 0
        pnl_sl300 = float(credit - debits[-1])

    return {
        "hit_25pct_sl200": hit_25_sl200,
        "hit_50pct_sl200": hit_50_sl200,
        "hit_25pct_sl300": hit_25_sl300,
        "hit_50pct_sl300": hit_50_sl300,
        "time_to_sl200_min": float(first_sl200 * 5) if first_sl200 is not None else None,
        "time_to_sl300_min": float(first_sl300 * 5) if first_sl300 is not None else None,
        "sl200_before_tp25": sl200_before_tp25,
        "sl200_before_tp50": sl200_before_tp50,
        "sl300_before_tp25": sl300_before_tp25,
        "sl300_before_tp50": sl300_before_tp50,
        "pnl_at_close_sl200": pnl_sl200,
        "pnl_at_close_sl300": pnl_sl300,
    }


def infer_long_strikes(row: pd.Series, wing_width: int = 25) -> tuple:
    """Infer long strikes from short strikes and wing width."""
    sp = row["short_put_strike"]
    sc = row["short_call_strike"]
    return (sp - wing_width, sc + wing_width)


def main():
    parser = argparse.ArgumentParser(description="Re-label with SL-gated TP targets")
    parser.add_argument("--input", default="ml/training_data/training_dataset_multi_w25.csv",
                        help="Input CSV path (relative to project root)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: adds _slgated suffix)")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        output_path = input_path.with_name(input_path.stem + "_slgated.csv")

    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    # Filter to rows that have valid label data
    valid_mask = df["entry_credit"].notna() & df["short_put_strike"].notna()
    logger.info(f"Rows with valid labels: {valid_mask.sum()}")

    new_cols = []
    skipped = 0
    processed = 0

    for idx, row in df.iterrows():
        if not valid_mask.iloc[idx]:
            new_cols.append({k: None for k in [
                "hit_25pct_sl200", "hit_50pct_sl200",
                "hit_25pct_sl300", "hit_50pct_sl300",
                "time_to_sl200_min", "time_to_sl300_min",
                "sl200_before_tp25", "sl200_before_tp50",
                "sl300_before_tp25", "sl300_before_tp50",
                "pnl_at_close_sl200", "pnl_at_close_sl300",
            ]})
            skipped += 1
            continue

        trade_date = date.fromisoformat(row["trade_date"])
        entry_hour = int(row["entry_hour"])
        entry_min = int(row.get("entry_minute", 0)) if "entry_minute" in row else 0
        entry_time = dtime(entry_hour, entry_min)

        wing_width = int(row.get("wing_width", 25))
        long_put, long_call = infer_long_strikes(row, wing_width)

        result = compute_sl_gated_labels(
            trade_date=trade_date,
            entry_time=entry_time,
            credit=float(row["entry_credit"]),
            short_put=float(row["short_put_strike"]),
            short_call=float(row["short_call_strike"]),
            long_put=long_put,
            long_call=long_call,
        )

        new_cols.append(result)
        processed += 1

        if processed % 100 == 0:
            logger.info(f"Processed {processed} rows...")

    new_df = pd.DataFrame(new_cols)
    result_df = pd.concat([df, new_df], axis=1)

    # --- Summary Statistics ---
    valid = result_df[result_df["hit_25pct_sl200"].notna()]

    logger.info("\n" + "=" * 70)
    logger.info("LABEL COMPARISON: Original vs SL-Gated")
    logger.info("=" * 70)

    if len(valid) > 0:
        logger.info(f"\nTotal valid rows: {len(valid)}")
        logger.info(f"Skipped (no labels): {skipped}")

        logger.info(f"\n{'Metric':<35} {'Original':>10} {'SL200':>10} {'SL300':>10}")
        logger.info("-" * 70)

        orig_25 = valid["hit_25pct"].mean()
        sl200_25 = valid["hit_25pct_sl200"].mean()
        sl300_25 = valid["hit_25pct_sl300"].mean()
        logger.info(f"{'Hit 25% TP rate':<35} {orig_25:>9.1%} {sl200_25:>9.1%} {sl300_25:>9.1%}")

        orig_50 = valid["hit_50pct"].mean()
        sl200_50 = valid["hit_50pct_sl200"].mean()
        sl300_50 = valid["hit_50pct_sl300"].mean()
        logger.info(f"{'Hit 50% TP rate':<35} {orig_50:>9.1%} {sl200_50:>9.1%} {sl300_50:>9.1%}")

        logger.info(f"\n{'Contamination (SL hit before TP):'}")
        logger.info(f"  SL200 hit before TP25: {valid['sl200_before_tp25'].mean():>8.1%}")
        logger.info(f"  SL200 hit before TP50: {valid['sl200_before_tp50'].mean():>8.1%}")
        logger.info(f"  SL300 hit before TP25: {valid['sl300_before_tp25'].mean():>8.1%}")
        logger.info(f"  SL300 hit before TP50: {valid['sl300_before_tp50'].mean():>8.1%}")

        # How many labels actually flip?
        if "hit_25pct" in valid.columns:
            flip_25_200 = ((valid["hit_25pct"] == 1) & (valid["hit_25pct_sl200"] == 0)).sum()
            flip_50_200 = ((valid["hit_50pct"] == 1) & (valid["hit_50pct_sl200"] == 0)).sum()
            flip_25_300 = ((valid["hit_25pct"] == 1) & (valid["hit_25pct_sl300"] == 0)).sum()
            flip_50_300 = ((valid["hit_50pct"] == 1) & (valid["hit_50pct_sl300"] == 0)).sum()

            logger.info(f"\n{'Labels that FLIP (1→0):'}")
            logger.info(f"  TP25 flips at SL200: {flip_25_200:>5d} ({flip_25_200/len(valid):.1%})")
            logger.info(f"  TP50 flips at SL200: {flip_50_200:>5d} ({flip_50_200/len(valid):.1%})")
            logger.info(f"  TP25 flips at SL300: {flip_25_300:>5d} ({flip_25_300/len(valid):.1%})")
            logger.info(f"  TP50 flips at SL300: {flip_50_300:>5d} ({flip_50_300/len(valid):.1%})")

        sl200_hit_rate = valid["time_to_sl200_min"].notna().mean()
        sl300_hit_rate = valid["time_to_sl300_min"].notna().mean()
        logger.info(f"\n  SL200 ever hit: {sl200_hit_rate:.1%}")
        logger.info(f"  SL300 ever hit: {sl300_hit_rate:.1%}")

    result_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
