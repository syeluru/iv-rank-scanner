#!/usr/bin/env python3
"""
Find days where a short iron condor hits the stop loss (debit >= 2x credit)
intraday but then RECOVERS by 3pm to be profitable or near breakeven.

Scans chain files to find ~10-delta strikes at 10:30, then loads leg files
to track the IC mid debit bar-by-bar.
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

CHAINS_DIR = Path("/Users/saiyeluru/Documents/Projects/iv-rank-scanner/ml/training_data/cache/chains")
LEGS_DIR = Path("/Users/saiyeluru/Documents/Projects/iv-rank-scanner/ml/training_data/cache/legs")
WING_WIDTH = 25.0
ENTRY_TIME = "1030"
EXIT_TIME = "1500"
SL_MULT = 2.0  # debit >= 2x credit = stop loss

# Priority candidates (known reversal days)
PRIORITY_DATES = [
    "2024-08-05", "2024-04-19", "2024-07-11", "2024-09-06", "2024-10-31",
    "2025-01-13", "2024-03-28", "2024-01-25", "2024-11-01", "2025-02-21",
    "2025-03-04", "2025-03-10",
]


@dataclass
class ICResult:
    date: str
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    entry_credit: float
    max_debit: float
    max_debit_time: str
    final_debit: float
    final_time: str
    sl_ratio: float  # max_debit / entry_credit
    recovery_ratio: float  # (max_debit - final_debit) / (max_debit - entry_credit)
    bars: list  # list of (time, sp_mid, lp_mid, sc_mid, lc_mid, ic_debit)


def find_closest_strike(chain, target_delta):
    """Find the strike with delta closest to target_delta."""
    best = None
    best_diff = float('inf')
    for opt in chain:
        d = opt.get('delta', 0)
        mid = (opt.get('bid', 0) + opt.get('ask', 0)) / 2
        if mid <= 0.01:  # skip zero-value options
            continue
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = opt
    return best


def load_chain(date, time, opt_type):
    """Load a chain file. Returns list of dicts or None."""
    path = CHAINS_DIR / f"{date}_{time}_{opt_type}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_leg(date, opt_type, strike, entry_time, exit_time):
    """Load a leg file. Returns list of {timestamp, bid, ask} or None."""
    path = LEGS_DIR / f"{date}_{opt_type}_{strike}_{entry_time}_{exit_time}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_entry_time(date):
    """Find the best available entry time chain for a date (prefer 1030, then nearby)."""
    for t in ["1030", "1025", "1035", "1020", "1040", "1015", "1045", "1000", "1050"]:
        if (CHAINS_DIR / f"{date}_{t}_put.json").exists() and \
           (CHAINS_DIR / f"{date}_{t}_call.json").exists():
            return t
    return None


def analyze_date(date, entry_time=None, verbose=False):
    """Analyze a single date for SL breach + recovery. Returns ICResult or None."""
    if entry_time is None:
        entry_time = find_entry_time(date)
    if entry_time is None:
        if verbose:
            print(f"  {date}: No chain files found")
        return None

    # Load chains
    put_chain = load_chain(date, entry_time, "put")
    call_chain = load_chain(date, entry_time, "call")
    if not put_chain or not call_chain:
        if verbose:
            print(f"  {date}: Chain load failed")
        return None

    # Find ~10-delta strikes
    short_put_opt = find_closest_strike(put_chain, -0.10)
    short_call_opt = find_closest_strike(call_chain, 0.10)
    if not short_put_opt or not short_call_opt:
        if verbose:
            print(f"  {date}: Could not find 10-delta strikes")
        return None

    sp_strike = short_put_opt['strike']
    sc_strike = short_call_opt['strike']
    lp_strike = sp_strike - WING_WIDTH
    lc_strike = sc_strike + WING_WIDTH

    if verbose:
        print(f"  {date}: SP={sp_strike} LP={lp_strike} SC={sc_strike} LC={lc_strike} (entry={entry_time})")

    # Load all 4 leg files
    sp_leg = load_leg(date, "put", sp_strike, entry_time, EXIT_TIME)
    lp_leg = load_leg(date, "put", lp_strike, entry_time, EXIT_TIME)
    sc_leg = load_leg(date, "call", sc_strike, entry_time, EXIT_TIME)
    lc_leg = load_leg(date, "call", lc_strike, entry_time, EXIT_TIME)

    missing = []
    if not sp_leg: missing.append(f"put_{sp_strike}")
    if not lp_leg: missing.append(f"put_{lp_strike}")
    if not sc_leg: missing.append(f"call_{sc_strike}")
    if not lc_leg: missing.append(f"call_{lc_strike}")

    if missing:
        if verbose:
            print(f"    Missing legs: {', '.join(missing)}")
        # Try nearby strikes for puts and calls (+-5)
        return None

    # Build time-aligned bars
    def to_mid_map(leg_data):
        m = {}
        for bar in leg_data:
            ts = bar['timestamp']
            time_part = ts.split('T')[1][:5].replace(':', '')  # "HH:MM" -> "HHMM"
            mid = (bar['bid'] + bar['ask']) / 2
            m[time_part] = mid
        return m

    sp_mids = to_mid_map(sp_leg)
    lp_mids = to_mid_map(lp_leg)
    sc_mids = to_mid_map(sc_leg)
    lc_mids = to_mid_map(lc_leg)

    # Get common timestamps
    common_times = sorted(set(sp_mids.keys()) & set(lp_mids.keys()) &
                          set(sc_mids.keys()) & set(lc_mids.keys()))

    if len(common_times) < 3:
        if verbose:
            print(f"    Only {len(common_times)} common bars")
        return None

    # Entry credit = first bar's IC mid
    t0 = common_times[0]
    entry_credit = sp_mids[t0] + sc_mids[t0] - lp_mids[t0] - lc_mids[t0]

    if entry_credit <= 0.10:
        if verbose:
            print(f"    Entry credit too low: {entry_credit:.2f}")
        return None

    # Walk through bars
    bars = []
    max_debit = -999
    max_debit_time = ""
    for t in common_times:
        ic_debit = sp_mids[t] + sc_mids[t] - lp_mids[t] - lc_mids[t]
        bars.append((t, sp_mids[t], lp_mids[t], sc_mids[t], lc_mids[t], ic_debit))
        if ic_debit > max_debit:
            max_debit = ic_debit
            max_debit_time = t

    final_debit = bars[-1][5]
    final_time = bars[-1][0]

    sl_ratio = max_debit / entry_credit if entry_credit > 0 else 0
    if max_debit > entry_credit:
        recovery_ratio = (max_debit - final_debit) / (max_debit - entry_credit)
    else:
        recovery_ratio = 0

    return ICResult(
        date=date,
        short_put=sp_strike,
        long_put=lp_strike,
        short_call=sc_strike,
        long_call=lc_strike,
        entry_credit=entry_credit,
        max_debit=max_debit,
        max_debit_time=max_debit_time,
        final_debit=final_debit,
        final_time=final_time,
        sl_ratio=sl_ratio,
        recovery_ratio=recovery_ratio,
        bars=bars,
    )


def print_bar_walk(result):
    """Print full bar-by-bar IC walk."""
    print(f"\n{'='*90}")
    print(f"DATE: {result.date}")
    print(f"IC: Short Put {result.short_put} / Long Put {result.long_put} / "
          f"Short Call {result.short_call} / Long Call {result.long_call}")
    print(f"Entry Credit: ${result.entry_credit:.2f}  |  "
          f"SL Level (2x): ${result.entry_credit * SL_MULT:.2f}")
    print(f"Max Debit: ${result.max_debit:.2f} at {result.max_debit_time}  "
          f"({result.sl_ratio:.2f}x credit)")
    print(f"Final Debit: ${result.final_debit:.2f} at {result.final_time}  "
          f"(P&L: ${result.entry_credit - result.final_debit:+.2f})")
    print(f"Recovery: {result.recovery_ratio:.1%} of the way back from max debit to entry")
    print(f"{'='*90}")
    print(f"{'Time':>6}  {'SP Mid':>8}  {'LP Mid':>8}  {'SC Mid':>8}  {'LC Mid':>8}  "
          f"{'IC Debit':>9}  {'P&L':>8}  {'xCredit':>8}  {'Note':>10}")
    print("-" * 90)

    for t, sp, lp, sc, lc, debit in result.bars:
        pnl = result.entry_credit - debit
        ratio = debit / result.entry_credit if result.entry_credit > 0 else 0
        note = ""
        if ratio >= SL_MULT:
            note = "** SL **"
        elif ratio >= 1.5:
            note = "! warn"
        elif pnl > 0:
            note = "profit"

        t_fmt = f"{t[:2]}:{t[2:]}"
        print(f"{t_fmt:>6}  {sp:>8.2f}  {lp:>8.2f}  {sc:>8.2f}  {lc:>8.2f}  "
              f"${debit:>8.2f}  ${pnl:>7.2f}  {ratio:>7.2f}x  {note:>10}")


def scan_all_dates():
    """Scan all available dates in the chains directory."""
    dates = set()
    for f in CHAINS_DIR.iterdir():
        if f.name.endswith("_put.json"):
            parts = f.stem.split("_")
            dates.add(parts[0])
    return sorted(dates)


def main():
    print("=" * 70)
    print("SCANNING FOR STOP-LOSS BREACH + RECOVERY DAYS")
    print("=" * 70)

    # Phase 1: Check priority candidates
    print("\n--- Phase 1: Priority candidate dates ---")
    results = []
    for date in PRIORITY_DATES:
        r = analyze_date(date, verbose=True)
        if r:
            results.append(r)
            print(f"    => Credit=${r.entry_credit:.2f}  MaxDebit=${r.max_debit:.2f} "
                  f"({r.sl_ratio:.2f}x) at {r.max_debit_time}  "
                  f"FinalDebit=${r.final_debit:.2f}  "
                  f"FinalP&L=${r.entry_credit - r.final_debit:+.2f}  "
                  f"Recovery={r.recovery_ratio:.0%}")

    # Phase 2: Scan ALL dates if we haven't found a great example
    sl_breaches = [r for r in results if r.sl_ratio >= SL_MULT]
    recoveries = [r for r in sl_breaches if r.recovery_ratio > 0.5]

    if not recoveries:
        print(f"\n--- Phase 2: Scanning ALL available dates ({len(scan_all_dates())} total) ---")
        all_dates = scan_all_dates()
        priority_set = set(PRIORITY_DATES)
        for date in all_dates:
            if date in priority_set:
                continue
            r = analyze_date(date, verbose=False)
            if r and r.sl_ratio >= 1.5:  # broader filter first
                results.append(r)
                if r.sl_ratio >= SL_MULT:
                    print(f"  {date}: Credit=${r.entry_credit:.2f}  "
                          f"MaxDebit=${r.max_debit:.2f} ({r.sl_ratio:.2f}x) at {r.max_debit_time}  "
                          f"FinalDebit=${r.final_debit:.2f}  "
                          f"P&L=${r.entry_credit - r.final_debit:+.2f}  "
                          f"Recovery={r.recovery_ratio:.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL SL BREACH DAYS (debit >= 2x credit)")
    print("=" * 70)

    sl_breaches = sorted([r for r in results if r.sl_ratio >= SL_MULT],
                         key=lambda r: r.recovery_ratio, reverse=True)

    if not sl_breaches:
        print("No days found with debit >= 2x credit.")
        # Show near-misses
        near = sorted([r for r in results if r.sl_ratio >= 1.5],
                      key=lambda r: r.recovery_ratio, reverse=True)[:10]
        if near:
            print("\nNear-miss days (debit >= 1.5x credit):")
            for r in near:
                print(f"  {r.date}: {r.sl_ratio:.2f}x at {r.max_debit_time}  "
                      f"Final={r.final_debit:.2f}  Recovery={r.recovery_ratio:.0%}")
        return

    for r in sl_breaches:
        pnl = r.entry_credit - r.final_debit
        print(f"  {r.date}: Max {r.sl_ratio:.2f}x at {r.max_debit_time} | "
              f"Final P&L: ${pnl:+.2f} | Recovery: {r.recovery_ratio:.0%}")

    # Print full bar walk for the best recovery
    best = sl_breaches[0]
    print(f"\n*** BEST RECOVERY: {best.date} ***")
    print_bar_walk(best)

    # Also show the top 3 if available
    if len(sl_breaches) > 1:
        print(f"\n--- Also showing next best recoveries ---")
        for r in sl_breaches[1:3]:
            print_bar_walk(r)


if __name__ == "__main__":
    main()
