#!/usr/bin/env python3
"""
Analyze long IC take-profit hit rates and P&L across entry times and deltas.

For each trading day from 2023-06-01 to 2026-03-10, fetches intraday option
chain snapshots at 5-min intervals via ThetaData bulk API, constructs long
iron condors at 10/15/20 delta, and tracks the IC value path to compute:

1. TP hit rates at 10%, 15%, 20%, 25%, 50%
2. Average P&L if held to 3:00 PM close
3. Average P&L if exited at TP (with 3:00 PM fallback if not hit)
4. Average P&L if exited at TP (with expiration/hold fallback if not hit)

Outputs 3 tables (one per delta) as CSV and prints formatted summary.

Usage:
    python -m scripts.analyze_long_ic_tp
    python -m scripts.analyze_long_ic_tp --start-date 2024-01-01 --end-date 2024-12-31
    python -m scripts.analyze_long_ic_tp --resume
"""

import argparse
import json
import logging
import os
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse backfill infrastructure
from scripts.backfill_feature_store import (
    THETA_HOST,
    THETA_SEMAPHORE,
    WING_WIDTH,
    IC_DELTAS,
    ensure_thetadata_running,
    _theta_get,
    _parse_response,
    _format_date,
    _format_time,
    _timestamp_to_slot_key,
    _fetch_greeks_one_right,
    fetch_chain_greeks_bulk,
    _find_delta_strike,
    _mid_price,
)

logger = logging.getLogger("analyze_long_ic_tp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TP_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.50]  # take-profit thresholds
ENTRY_START = time(9, 35)
ENTRY_END = time(14, 0)
ENTRY_INTERVAL_MIN = 5  # every 5 minutes
CLOSE_TIME = time(15, 0)  # 3:00 PM ET exit

# Progress tracking
PROGRESS_FILE = "ml/features/v8/data/long_ic_tp_progress.json"
OUTPUT_DIR = "output/long_ic_analysis"


def generate_entry_slots() -> list[time]:
    """Generate entry time slots from 9:35 to 14:00 every 5 minutes."""
    slots = []
    cur_min = ENTRY_START.hour * 60 + ENTRY_START.minute
    end_min = ENTRY_END.hour * 60 + ENTRY_END.minute
    while cur_min <= end_min:
        h, m = divmod(cur_min, 60)
        slots.append(time(h, m))
        cur_min += ENTRY_INTERVAL_MIN
    return slots


def generate_monitor_slots(entry_slot: time) -> list[str]:
    """Generate all 5-min slot keys from entry_slot to CLOSE_TIME."""
    slots = []
    cur_min = entry_slot.hour * 60 + entry_slot.minute + ENTRY_INTERVAL_MIN
    end_min = CLOSE_TIME.hour * 60 + CLOSE_TIME.minute
    while cur_min <= end_min:
        h, m = divmod(cur_min, 60)
        slots.append(f"{h:02d}:{m:02d}")
        cur_min += ENTRY_INTERVAL_MIN
    return slots


def compute_ic_value(chain: pd.DataFrame, strikes: dict) -> Optional[float]:
    """
    Compute IC net value (short IC perspective) from chain at given strikes.
    Returns (short_put + short_call) - (long_put + long_call) mid prices.
    Returns None if any leg has missing data.
    """
    sp = _mid_price(chain, strikes["short_put"], "P")
    sc = _mid_price(chain, strikes["short_call"], "C")
    lp = _mid_price(chain, strikes["long_put"], "P")
    lc = _mid_price(chain, strikes["long_call"], "C")

    if any(np.isnan(v) for v in [sp, sc, lp, lc]):
        return None

    return (sp + sc) - (lp + lc)


def compute_expiration_value(spx_close: float, strikes: dict) -> float:
    """
    Compute IC intrinsic value at expiration given SPX close price.
    For long IC: pay debit at entry, receive intrinsic payoff at expiry.
    Returns net IC value from short IC perspective (intrinsic).
    """
    sp_intrinsic = max(strikes["short_put"] - spx_close, 0)
    sc_intrinsic = max(spx_close - strikes["short_call"], 0)
    lp_intrinsic = max(strikes["long_put"] - spx_close, 0)
    lc_intrinsic = max(spx_close - strikes["long_call"], 0)

    return (sp_intrinsic + sc_intrinsic) - (lp_intrinsic + lc_intrinsic)


def analyze_day(
    session,
    d: date,
    entry_slots: list[time],
    spx_close_price: float,
) -> list[dict]:
    """
    Analyze one trading day for all entry times and deltas.

    Fetches bulk Greeks (2 API calls), then for each entry slot and delta:
    - Constructs the IC
    - Tracks 5-min value path through close
    - Computes TP hits and P&L scenarios

    Returns list of result dicts, one per (entry_time, delta) combo.
    """
    import requests as req

    # Fetch bulk Greeks for the full day (2 API calls: calls + puts)
    greeks_by_slot = fetch_chain_greeks_bulk(session, d)
    if not greeks_by_slot:
        logger.warning(f"  {d}: no bulk Greeks data — skipping")
        return []

    close_key = CLOSE_TIME.strftime("%H:%M")
    results = []

    for entry_slot in entry_slots:
        entry_key = entry_slot.strftime("%H:%M")
        entry_chain = greeks_by_slot.get(entry_key, pd.DataFrame())
        if entry_chain.empty:
            continue

        close_chain = greeks_by_slot.get(close_key, pd.DataFrame())

        for target_delta in IC_DELTAS:
            prefix = f"d{int(target_delta * 100)}"

            # Find IC strikes from entry chain
            short_put = _find_delta_strike(entry_chain, target_delta, "P")
            short_call = _find_delta_strike(entry_chain, target_delta, "C")
            if short_put is None or short_call is None:
                continue

            strikes = {
                "short_put": short_put,
                "short_call": short_call,
                "long_put": short_put - WING_WIDTH,
                "long_call": short_call + WING_WIDTH,
            }

            # Entry IC value = credit from short IC perspective
            entry_value = compute_ic_value(entry_chain, strikes)
            if entry_value is None or entry_value <= 0:
                continue

            # For LONG IC: we PAY the credit (debit = entry_value)
            debit = entry_value

            # Track IC value at every subsequent 5-min slot
            monitor_slots = generate_monitor_slots(entry_slot)

            # Find first TP hit for each level
            tp_hit_slot = {tp: None for tp in TP_LEVELS}
            tp_hit_value = {tp: None for tp in TP_LEVELS}

            for slot_key in monitor_slots:
                chain_at_slot = greeks_by_slot.get(slot_key, pd.DataFrame())
                if chain_at_slot.empty:
                    continue

                ic_value = compute_ic_value(chain_at_slot, strikes)
                if ic_value is None:
                    continue

                # Long IC P&L = debit - current_value (profit when IC value drops)
                # Actually: long IC buyer pays debit, position value = debit - ic_value
                # TP% means long IC has gained TP% of debit
                long_pnl_pct = (debit - ic_value) / debit

                for tp in TP_LEVELS:
                    if tp_hit_slot[tp] is None and long_pnl_pct >= tp:
                        tp_hit_slot[tp] = slot_key
                        tp_hit_value[tp] = ic_value

            # P&L at 3:00 PM close
            close_value = compute_ic_value(close_chain, strikes) if not close_chain.empty else None
            if close_value is not None:
                pnl_close = debit - close_value  # long IC P&L
                pnl_close_pct = pnl_close / debit
            else:
                pnl_close = np.nan
                pnl_close_pct = np.nan

            # P&L at expiration (intrinsic value)
            if spx_close_price > 0:
                exp_value = compute_expiration_value(spx_close_price, strikes)
                pnl_exp = debit - exp_value
                pnl_exp_pct = pnl_exp / debit
            else:
                pnl_exp = np.nan
                pnl_exp_pct = np.nan

            # Build result row
            row = {
                "date": d.isoformat(),
                "entry_time": entry_key,
                "delta": prefix,
                "debit": debit,
                "short_put": short_put,
                "short_call": short_call,
                "pnl_close": pnl_close,
                "pnl_close_pct": pnl_close_pct,
                "pnl_exp": pnl_exp,
                "pnl_exp_pct": pnl_exp_pct,
            }

            # For each TP level: hit?, P&L if exit at TP (or fallback)
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                hit = tp_hit_slot[tp] is not None
                row[f"{tp_key}_hit"] = 1.0 if hit else 0.0

                if hit:
                    # Exit at TP
                    tp_pnl = debit - tp_hit_value[tp]
                    row[f"{tp_key}_pnl_3pm"] = tp_pnl  # same as TP exit (already exited)
                    row[f"{tp_key}_pnl_exp"] = tp_pnl
                else:
                    # Fallback: hold to 3pm or expiration
                    row[f"{tp_key}_pnl_3pm"] = pnl_close  # didn't hit TP, sell at 3pm
                    row[f"{tp_key}_pnl_exp"] = pnl_exp  # didn't hit TP, hold to exp

            results.append(row)

    return results


def get_trading_days(start: date, end: date) -> list[date]:
    """Generate list of weekday dates."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def load_progress() -> set:
    """Load set of already-processed dates."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return set(data.get("completed_dates", []))
    return set()


def save_progress(completed: set):
    """Save progress to disk."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed_dates": sorted(completed)}, f)


def get_spx_close_prices(start: date, end: date) -> dict:
    """Fetch SPX daily close prices from yfinance for expiration P&L."""
    try:
        import yfinance as yf
        df = yf.download(
            "^GSPC",
            start=start,
            end=end + timedelta(days=3),
            progress=False,
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result = {}
        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, 'date') else idx
            result[d] = float(row["Close"])
        return result
    except Exception as e:
        logger.warning(f"yfinance SPX close fetch failed: {e}")
        return {}


def generate_tables(all_results: pd.DataFrame):
    """Generate and save the 3 analysis tables (one per delta)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for delta in IC_DELTAS:
        prefix = f"d{int(delta * 100)}"
        df = all_results[all_results["delta"] == prefix].copy()

        if df.empty:
            logger.warning(f"No data for {prefix}")
            continue

        # Group by entry_time
        entry_times = sorted(df["entry_time"].unique())

        table_rows = []
        for et in entry_times:
            subset = df[df["entry_time"] == et]
            n = len(subset)

            row = {"entry_time": et, "n_trades": n}

            # Average debit (cost to enter long IC)
            row["avg_debit"] = subset["debit"].mean()

            # TP hit rates
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                row[f"{tp_key}_hit_rate"] = subset[f"{tp_key}_hit"].mean()

            # Average P&L held to close
            row["avg_pnl_close"] = subset["pnl_close"].mean()
            row["avg_pnl_close_pct"] = subset["pnl_close_pct"].mean()

            # Average P&L held to expiration
            row["avg_pnl_exp"] = subset["pnl_exp"].mean()
            row["avg_pnl_exp_pct"] = subset["pnl_exp_pct"].mean()

            # Average P&L with TP exit + 3pm fallback
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                row[f"avg_{tp_key}_pnl_3pm"] = subset[f"{tp_key}_pnl_3pm"].mean()
                pnl_pct = subset[f"{tp_key}_pnl_3pm"] / subset["debit"]
                row[f"avg_{tp_key}_pnl_pct_3pm"] = pnl_pct.mean()

            # Average P&L with TP exit + expiration fallback
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                row[f"avg_{tp_key}_pnl_exp"] = subset[f"{tp_key}_pnl_exp"].mean()
                pnl_pct = subset[f"{tp_key}_pnl_exp"] / subset["debit"]
                row[f"avg_{tp_key}_pnl_pct_exp"] = pnl_pct.mean()

            # Win rate (held to close)
            row["win_rate_close"] = (subset["pnl_close"] > 0).mean()
            row["win_rate_exp"] = (subset["pnl_exp"] > 0).mean()

            table_rows.append(row)

        table_df = pd.DataFrame(table_rows)

        # Save to CSV
        csv_path = os.path.join(OUTPUT_DIR, f"long_ic_{prefix}_tp_analysis.csv")
        table_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")

        # Print formatted table
        print(f"\n{'='*120}")
        print(f"LONG IC — {int(delta*100)}-DELTA — TP Hit Rates & P&L Analysis")
        print(f"{'='*120}")
        print(f"{'Entry':>6} | {'N':>4} | {'Avg$':>6} | "
              f"{'TP10%':>5} {'TP15%':>5} {'TP20%':>5} {'TP25%':>5} {'TP50%':>5} | "
              f"{'PnL@3pm':>8} {'PnL@Exp':>8} | "
              f"{'WR@3pm':>6} {'WR@Exp':>6}")
        print("-" * 120)

        for _, r in table_df.iterrows():
            print(f"{r['entry_time']:>6} | "
                  f"{r['n_trades']:4.0f} | "
                  f"${r['avg_debit']:5.2f} | "
                  f"{r['tp10_hit_rate']:5.1%} "
                  f"{r['tp15_hit_rate']:5.1%} "
                  f"{r['tp20_hit_rate']:5.1%} "
                  f"{r['tp25_hit_rate']:5.1%} "
                  f"{r['tp50_hit_rate']:5.1%} | "
                  f"{r['avg_pnl_close_pct']:+7.1%} "
                  f"{r['avg_pnl_exp_pct']:+7.1%} | "
                  f"{r['win_rate_close']:5.1%} "
                  f"{r['win_rate_exp']:5.1%}")

        # Print TP exit P&L tables
        print(f"\n  P&L with TP Exit + 3:00 PM Fallback:")
        print(f"  {'Entry':>6} | ", end="")
        for tp in TP_LEVELS:
            print(f"  TP{int(tp*100)}%  ", end="")
        print()
        print("  " + "-" * 60)
        for _, r in table_df.iterrows():
            print(f"  {r['entry_time']:>6} | ", end="")
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                pct = r[f"avg_{tp_key}_pnl_pct_3pm"]
                print(f" {pct:+6.1%}  ", end="")
            print()

        print(f"\n  P&L with TP Exit + Hold to Expiration Fallback:")
        print(f"  {'Entry':>6} | ", end="")
        for tp in TP_LEVELS:
            print(f"  TP{int(tp*100)}%  ", end="")
        print()
        print("  " + "-" * 60)
        for _, r in table_df.iterrows():
            print(f"  {r['entry_time']:>6} | ", end="")
            for tp in TP_LEVELS:
                tp_key = f"tp{int(tp * 100)}"
                pct = r[f"avg_{tp_key}_pnl_pct_exp"]
                print(f" {pct:+6.1%}  ", end="")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze long IC take-profit hit rates and P&L"
    )
    parser.add_argument("--start-date", default="2023-06-01")
    parser.add_argument("--end-date", default="2026-03-10")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last completed date")
    parser.add_argument("--tables-only", action="store_true",
                       help="Skip data collection, just regenerate tables from saved data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/long_ic_tp_analysis.log"),
        ],
    )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    raw_path = os.path.join(OUTPUT_DIR, "long_ic_tp_raw.parquet")

    if args.tables_only:
        if not os.path.exists(raw_path):
            logger.error(f"No raw data found at {raw_path}")
            sys.exit(1)
        all_results = pd.read_parquet(raw_path)
        logger.info(f"Loaded {len(all_results)} results from {raw_path}")
        generate_tables(all_results)
        return

    # Ensure ThetaData is running
    if not ensure_thetadata_running():
        logger.error("ThetaData Terminal not available")
        sys.exit(1)

    # Fetch SPX daily closes for expiration P&L
    logger.info("Fetching SPX daily close prices...")
    spx_closes = get_spx_close_prices(start, end)
    logger.info(f"Got {len(spx_closes)} daily SPX closes")

    # Generate trading days and entry slots
    trading_days = get_trading_days(start, end)
    entry_slots = generate_entry_slots()
    logger.info(f"Processing {len(trading_days)} trading days, "
                f"{len(entry_slots)} entry slots per day")

    # Resume support
    completed = load_progress() if args.resume else set()
    if completed:
        logger.info(f"Resuming: {len(completed)} days already done")

    # Load existing results if resuming
    all_rows = []
    if args.resume and os.path.exists(raw_path):
        existing = pd.read_parquet(raw_path)
        all_rows = existing.to_dict("records")
        logger.info(f"Loaded {len(all_rows)} existing results")

    import requests
    session = requests.Session()

    days_to_process = [d for d in trading_days if d.isoformat() not in completed]
    total = len(days_to_process)
    logger.info(f"Days to process: {total}")

    t0 = _time.time()
    for i, d in enumerate(days_to_process):
        day_t0 = _time.time()

        spx_close = spx_closes.get(d, 0.0)
        day_results = analyze_day(session, d, entry_slots, spx_close)
        all_rows.extend(day_results)

        completed.add(d.isoformat())
        elapsed = _time.time() - day_t0

        # Log progress
        if (i + 1) % 10 == 0 or i == 0:
            total_elapsed = _time.time() - t0
            rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
            remaining = (total - i - 1) / rate / 3600 if rate > 0 else 0
            logger.info(
                f"  [{i+1}/{total}] {d}: {len(day_results)} results, "
                f"{elapsed:.1f}s/day, ETA: {remaining:.1f}h"
            )

        # Save progress every 25 days
        if (i + 1) % 25 == 0:
            save_progress(completed)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pd.DataFrame(all_rows).to_parquet(raw_path, index=False)
            logger.info(f"  Checkpoint saved: {len(all_rows)} results")

    # Final save
    save_progress(completed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = pd.DataFrame(all_rows)
    all_results.to_parquet(raw_path, index=False)
    logger.info(f"Saved {len(all_results)} results to {raw_path}")

    # Generate tables
    generate_tables(all_results)

    total_time = _time.time() - t0
    logger.info(f"Done in {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()
