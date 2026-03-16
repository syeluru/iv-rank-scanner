#!/usr/bin/env python3
"""
Hedged IC Backtest with Intraday Stop Loss / Take Profit

Walks each trade minute-by-minute from entry to 15:59, checking 6 SL/TP combos.
Runs only for top 50 configs from Phase 2.

SL/TP Combos:
  SL levels: 200%, 300% of net credit
  TP levels: 10%, 25%, 50% of net credit (after round-trip fees)

Usage:
    python -m scripts.backtest_hedged_ic_sltp
    python -m scripts.backtest_hedged_ic_sltp --resume
"""

import argparse
import json
import logging
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_hedged_ic import (
    load_day,
    build_lookup_fast,
    find_short_strikes,
    get_spx_close_fast,
    generate_entry_times,
    PHASE2_START,
    PHASE2_END,
    PHASE2_INTERVAL,
    COST_PER_LEG,
)

logger = logging.getLogger("hedged_ic_sltp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output/hedged_ic_backtest")
GREEKS_PATH = Path("data/spxw_0dte_intraday_greeks.parquet")

OPEN_FEES = COST_PER_LEG * 8 / 100   # $0.052/sh
CLOSE_FEES = COST_PER_LEG * 8 / 100  # $0.052/sh
ROUND_TRIP_FEES = OPEN_FEES + CLOSE_FEES  # $0.104/sh

# SL/TP combos to test
SL_MULTIPLIERS = [2.0, 3.0]
TP_PCTS = [0.10, 0.25, 0.50]

COMBO_NAMES = []
for sl in SL_MULTIPLIERS:
    for tp in TP_PCTS:
        COMBO_NAMES.append("sl{:.0f}_tp{:.0f}".format(sl * 100, tp * 100))
# → ['sl200_tp10', 'sl200_tp25', 'sl200_tp50', 'sl300_tp10', 'sl300_tp25', 'sl300_tp50']


def load_top50_configs():
    """Load top 50 config labels and parse into permutation dicts."""
    top50_df = pd.read_csv(OUTPUT_DIR / "top50_labels.csv")
    perms = []
    for _, row in top50_df.iterrows():
        perms.append({
            "delta": row["delta"],
            "si_width": int(row["si_width"]),
            "gap": int(row["gap"]),
            "li_width": int(row["li_width"]),
            "label": row["label"],
        })
    return perms


def compute_debit_to_close(lookup, minute_key, strikes):
    """
    Compute current debit-to-close for the 8-leg position.

    debit_to_close = current_si_value - current_li_value
    where si_value = (short_put + short_call) - (long_put + long_call) at current mids
    and   li_value = (long_put + long_call) - (short_put + short_call) at current mids
    """
    legs = {
        "si_short_put": (strikes["si_short_put"], "P"),
        "si_long_put": (strikes["si_long_put"], "P"),
        "si_short_call": (strikes["si_short_call"], "C"),
        "si_long_call": (strikes["si_long_call"], "C"),
        "li_long_put": (strikes["li_long_put"], "P"),
        "li_short_put": (strikes["li_short_put"], "P"),
        "li_long_call": (strikes["li_long_call"], "C"),
        "li_short_call": (strikes["li_short_call"], "C"),
    }

    mids = {}
    for name, (strike, right) in legs.items():
        data = lookup.get((strike, right, minute_key))
        if data is None or data["mid"] <= 0:
            return None
        mids[name] = data["mid"]

    si_value = (mids["si_short_put"] + mids["si_short_call"]) - \
               (mids["si_long_put"] + mids["si_long_call"])
    li_value = (mids["li_long_put"] + mids["li_long_call"]) - \
               (mids["li_short_put"] + mids["li_short_call"])

    return si_value - li_value


def walk_trade(lookup, entry_time, strikes, net_credit, spx_close, minute_keys_after_entry):
    """
    Walk a single trade minute-by-minute, checking all 6 SL/TP combos.

    Returns dict with results for each combo + baseline (hold to expiry).
    """
    # Precompute thresholds for all combos
    thresholds = {}
    for sl_mult in SL_MULTIPLIERS:
        for tp_pct in TP_PCTS:
            combo = "sl{:.0f}_tp{:.0f}".format(sl_mult * 100, tp_pct * 100)
            thresholds[combo] = {
                "sl_thresh": net_credit * sl_mult,
                "tp_thresh": net_credit * (1 - tp_pct) - ROUND_TRIP_FEES,
                "sl_mult": sl_mult,
                "tp_pct": tp_pct,
            }

    # Track which combos are still active (haven't triggered yet)
    active = {c: True for c in COMBO_NAMES}
    results = {}

    # Walk forward minute by minute
    for mk in minute_keys_after_entry:
        dtc = compute_debit_to_close(lookup, mk, strikes)
        if dtc is None:
            continue  # missing data this minute, skip

        for combo in COMBO_NAMES:
            if not active[combo]:
                continue

            th = thresholds[combo]

            # Check SL first (conservative — if both trigger same minute, SL wins)
            if dtc >= th["sl_thresh"]:
                pnl_per_share = net_credit - dtc - ROUND_TRIP_FEES
                results[combo] = {
                    "exit_type": "SL",
                    "exit_time": mk,
                    "exit_debit": dtc,
                    "pnl_per_share": pnl_per_share,
                    "pnl_dollar": pnl_per_share * 100,
                }
                active[combo] = False

            elif dtc <= th["tp_thresh"]:
                pnl_per_share = net_credit - dtc - ROUND_TRIP_FEES
                results[combo] = {
                    "exit_type": "TP",
                    "exit_time": mk,
                    "exit_debit": dtc,
                    "pnl_per_share": pnl_per_share,
                    "pnl_dollar": pnl_per_share * 100,
                }
                active[combo] = False

        # Early exit if all combos triggered
        if not any(active.values()):
            break

    # For any combo that didn't trigger: hold to expiry
    spx = spx_close
    si_intrinsic = (
        -max(strikes["si_short_put"] - spx, 0)
        + max(strikes["si_long_put"] - spx, 0)
        - max(spx - strikes["si_short_call"], 0)
        + max(spx - strikes["si_long_call"], 0)
    )
    li_intrinsic = (
        + max(strikes["li_long_put"] - spx, 0)
        - max(strikes["li_short_put"] - spx, 0)
        + max(spx - strikes["li_long_call"], 0)
        - max(spx - strikes["li_short_call"], 0)
    )
    expiry_pnl_per_share = si_intrinsic + li_intrinsic + net_credit - OPEN_FEES

    for combo in COMBO_NAMES:
        if active[combo]:
            results[combo] = {
                "exit_type": "EXPIRY",
                "exit_time": "15:59",
                "exit_debit": 0,
                "pnl_per_share": expiry_pnl_per_share,
                "pnl_dollar": expiry_pnl_per_share * 100,
            }

    # Baseline (no SL/TP)
    results["baseline"] = {
        "exit_type": "EXPIRY",
        "exit_time": "15:59",
        "exit_debit": 0,
        "pnl_per_share": expiry_pnl_per_share,
        "pnl_dollar": expiry_pnl_per_share * 100,
    }

    return results


def run_sltp_backtest(resume=False):
    """Main orchestration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "sltp_backtest.log"),
            logging.StreamHandler(),
        ],
    )

    # Load configs
    perms = load_top50_configs()
    logger.info("Loaded {} permutations".format(len(perms)))

    # Entry times
    entry_times = generate_entry_times(PHASE2_START, PHASE2_END, PHASE2_INTERVAL)
    logger.info("Entry times: {}".format(len(entry_times)))

    # All minute keys from 10:00 to 15:59
    all_minutes = []
    for h in range(10, 16):
        end_m = 60 if h < 15 else 60
        for m in range(0, end_m):
            all_minutes.append("{:02d}:{:02d}".format(h, m))

    # Get dates
    pf = pd.read_parquet(GREEKS_PATH, columns=["date"])
    all_dates = sorted(pf["date"].unique())
    all_dates = [pd.Timestamp(d).date() for d in all_dates]
    logger.info("Total dates: {}".format(len(all_dates)))

    # Progress / resume
    progress_file = OUTPUT_DIR / "progress_sltp.json"
    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed_dates": []}
    completed = set(progress.get("completed_dates", []))
    remaining = [d for d in all_dates if str(d) not in completed]
    logger.info("Remaining: {} (skipping {} completed)".format(len(remaining), len(completed)))

    # Results file
    results_file = OUTPUT_DIR / "results_sltp.csv"
    write_header = not results_file.exists() or not resume

    total_results = 0
    t0 = _time.time()

    for i, d in enumerate(remaining):
        t_day = _time.time()

        day_df = load_day(d)
        if day_df.empty:
            logger.warning("[{}/{}] {}: no data".format(i + 1, len(remaining), d))
            completed.add(str(d))
            progress["completed_dates"] = sorted(completed)
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
            continue

        lookup = build_lookup_fast(day_df)
        spx_close = get_spx_close_fast(day_df)
        if spx_close is None:
            completed.add(str(d))
            progress["completed_dates"] = sorted(completed)
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
            continue

        day_results = []

        for entry_time in entry_times:
            entry_chain = day_df[day_df["minute_key"] == entry_time]
            if entry_chain.empty:
                continue

            spx_entry = float(entry_chain["underlying_price"].iloc[0])

            # Minutes after entry for the walk
            entry_idx = all_minutes.index(entry_time) if entry_time in all_minutes else -1
            if entry_idx < 0:
                continue
            minutes_after = all_minutes[entry_idx + 1:]

            # Cache short strikes by delta
            unique_deltas = set(p["delta"] for p in perms)
            delta_strikes = {}
            for delta in unique_deltas:
                s = find_short_strikes(entry_chain, delta)
                if s is not None:
                    delta_strikes[delta] = s

            for perm in perms:
                delta = perm["delta"]
                if delta not in delta_strikes:
                    continue

                short_put = delta_strikes[delta]["put"]
                short_call = delta_strikes[delta]["call"]
                si_w = perm["si_width"]
                gap = perm["gap"]
                li_w = perm["li_width"]

                # Compute strikes
                strikes = {
                    "si_short_put": short_put,
                    "si_long_put": short_put - si_w,
                    "si_short_call": short_call,
                    "si_long_call": short_call + si_w,
                    "li_long_put": short_put - si_w - gap,
                    "li_short_put": short_put - si_w - gap - li_w,
                    "li_long_call": short_call + si_w + gap,
                    "li_short_call": short_call + si_w + gap + li_w,
                }

                # Get entry prices for all 8 legs
                entry_mids = {}
                skip = False
                for name, strike in strikes.items():
                    right = "P" if "put" in name else "C"
                    data = lookup.get((strike, right, entry_time))
                    if data is None or data["mid"] <= 0:
                        skip = True
                        break
                    entry_mids[name] = data["mid"]
                if skip:
                    continue

                si_credit = (entry_mids["si_short_put"] + entry_mids["si_short_call"]) - \
                            (entry_mids["si_long_put"] + entry_mids["si_long_call"])
                li_debit = (entry_mids["li_long_put"] + entry_mids["li_long_call"]) - \
                           (entry_mids["li_short_put"] + entry_mids["li_short_call"])
                net_credit = si_credit - li_debit

                if net_credit - OPEN_FEES <= 0:
                    continue

                # Walk the trade
                combo_results = walk_trade(
                    lookup, entry_time, strikes, net_credit,
                    spx_close, minutes_after,
                )

                # Build result row
                base = {
                    "date": str(d),
                    "entry_time": entry_time,
                    "label": perm["label"],
                    "delta": delta,
                    "si_width": si_w,
                    "gap": gap,
                    "li_width": li_w,
                    "spx_entry": spx_entry,
                    "spx_close": spx_close,
                    "net_credit": net_credit,
                }

                for combo_name in COMBO_NAMES + ["baseline"]:
                    cr = combo_results[combo_name]
                    base[combo_name + "_exit"] = cr["exit_type"]
                    base[combo_name + "_time"] = cr["exit_time"]
                    base[combo_name + "_pnl"] = cr["pnl_dollar"]

                day_results.append(base)

        # Write results
        if day_results:
            df_out = pd.DataFrame(day_results)
            df_out.to_csv(
                results_file,
                mode="a" if not write_header else "w",
                header=write_header,
                index=False,
            )
            write_header = False
            total_results += len(day_results)

        # Checkpoint
        completed.add(str(d))
        progress["completed_dates"] = sorted(completed)
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        elapsed = _time.time() - t_day
        total_elapsed = _time.time() - t0
        rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
        eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

        logger.info(
            "[{}/{}] {}: {} results in {:.1f}s "
            "(total: {:,}, ETA: {:.0f}min)".format(
                i + 1, len(remaining), d,
                len(day_results), elapsed,
                total_results, eta / 60,
            )
        )

    total_time = _time.time() - t0
    logger.info("SLTP backtest complete: {:,} results in {:.1f} min".format(
        total_results, total_time / 60))

    # Generate summary
    if results_file.exists():
        generate_sltp_summary(results_file)


def generate_sltp_summary(results_path):
    """Summarize results across all SL/TP combos."""
    logger.info("Generating SLTP summary...")
    df = pd.read_csv(results_path)

    all_combos = COMBO_NAMES + ["baseline"]
    rows = []

    for label in df["label"].unique():
        sub = df[df["label"] == label]
        for combo in all_combos:
            pnl_col = combo + "_pnl"
            exit_col = combo + "_exit"

            pnl = sub[pnl_col]
            exits = sub[exit_col]

            tp_count = (exits == "TP").sum()
            sl_count = (exits == "SL").sum()
            exp_count = (exits == "EXPIRY").sum()
            total = len(sub)

            rows.append({
                "label": label,
                "combo": combo,
                "num_trades": total,
                "avg_pnl": pnl.mean(),
                "median_pnl": pnl.median(),
                "total_pnl": pnl.sum(),
                "win_rate": (pnl > 0).mean(),
                "std_pnl": pnl.std(),
                "tp_rate": tp_count / total if total > 0 else 0,
                "sl_rate": sl_count / total if total > 0 else 0,
                "expiry_rate": exp_count / total if total > 0 else 0,
                "p5_pnl": pnl.quantile(0.05),
                "p95_pnl": pnl.quantile(0.95),
            })

    summary = pd.DataFrame(rows)
    summary["sharpe"] = np.where(
        summary["std_pnl"] > 0,
        (summary["avg_pnl"] / summary["std_pnl"]) * np.sqrt(252),
        0,
    )
    summary = summary.sort_values(["label", "combo"])
    summary.to_csv(OUTPUT_DIR / "summary_sltp.csv", index=False)
    logger.info("Summary saved to summary_sltp.csv")

    # Print comparison for top 10 configs
    top10 = df["label"].value_counts().head(10).index.tolist()
    # Actually use sorted by baseline avg_pnl
    baseline_avg = summary[summary["combo"] == "baseline"].sort_values("avg_pnl", ascending=False)
    top10 = baseline_avg.head(10)["label"].tolist()

    print("\n" + "=" * 120)
    print("TOP 10 CONFIGS — SL/TP COMPARISON")
    print("=" * 120)
    for label in top10:
        print("\n  {}:".format(label))
        label_summary = summary[summary["label"] == label]
        print("    {:15s}  {:>8s}  {:>5s}  {:>9s}  {:>6s}  {:>6s}  {:>6s}  {:>6s}  {:>9s}  {:>8s}".format(
            "Combo", "Avg P&L", "Win%", "Total P&L", "Sharpe", "TP%", "SL%", "Exp%", "P5", "P95"))
        print("    " + "-" * 95)
        for _, r in label_summary.iterrows():
            print("    {:15s}  ${:+7.1f}  {:.1%}  ${:+9,.0f}  {:6.2f}  {:.1%}  {:.1%}  {:.1%}  ${:+8.0f}  ${:+7.0f}".format(
                r["combo"], r["avg_pnl"], r["win_rate"], r["total_pnl"],
                r["sharpe"], r["tp_rate"], r["sl_rate"], r["expiry_rate"],
                r["p5_pnl"], r["p95_pnl"]))


def main():
    parser = argparse.ArgumentParser(description="Hedged IC SL/TP Backtest")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    run_sltp_backtest(resume=args.resume)


if __name__ == "__main__":
    main()
