#!/usr/bin/env python3
"""
Analyze path-dependent TP/SL scenarios for 10-delta short iron condors.
Finds Type A (all TPs hit), Type B (SL before TP), Type C (expires flat) days.
"""

import json
import os
import sys
from pathlib import Path

CHAINS_DIR = Path("/Users/saiyeluru/Documents/Projects/iv-rank-scanner/ml/training_data/cache/chains")
LEGS_DIR = Path("/Users/saiyeluru/Documents/Projects/iv-rank-scanner/ml/training_data/cache/legs")

WING_WIDTH = 25.0
ENTRY_TIME = "1030"
SL_MULT = 2.0  # SL at 2x credit
TP10_PCT = 0.90
TP25_PCT = 0.75
TP50_PCT = 0.50

CANDIDATE_DATES = [
    # Volatile event days
    ("2024-09-18", "FOMC Sep"),
    ("2024-08-05", "JPY carry unwind"),
    ("2024-11-06", "Election"),
    ("2024-12-18", "FOMC Dec"),
    ("2025-01-15", "CPI Jan"),
    ("2024-04-15", "Iran/Israel"),
    # Flat / low-vol days
    ("2024-06-03", "Flat day"),
    ("2024-07-08", "Flat day"),
    ("2024-05-20", "Flat day"),
    # Additional candidates for Type C (choppy, range-bound)
    ("2024-03-25", "Pre-Q1 end"),
    ("2024-02-26", "Mon flat"),
    ("2024-10-07", "Mon flat"),
    ("2024-10-14", "Columbus Day"),
    ("2024-08-19", "Flat Mon"),
    ("2024-07-15", "Flat Mon"),
    ("2024-01-22", "Jan flat"),
    ("2024-05-06", "May flat"),
    ("2024-11-25", "Post-Thanksgiving"),
    ("2024-09-09", "Sep flat"),
    ("2025-02-10", "Feb flat"),
    ("2025-03-03", "Mar flat"),
    ("2024-06-17", "Jun flat"),
    ("2024-04-22", "Apr flat"),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_10_delta_strikes(date, entry_time):
    """From chain files, find ~10-delta short put and short call strikes."""
    put_file = CHAINS_DIR / f"{date}_{entry_time}_put.json"
    call_file = CHAINS_DIR / f"{date}_{entry_time}_call.json"

    if not put_file.exists() or not call_file.exists():
        return None, None, None

    puts = load_json(put_file)
    calls = load_json(call_file)

    # Find put closest to -0.10 delta (short put)
    best_put = None
    best_put_dist = float('inf')
    for p in puts:
        d = p.get('delta', 0)
        if d is None or d == 0:
            continue
        dist = abs(abs(d) - 0.10)
        if dist < best_put_dist:
            best_put_dist = dist
            best_put = p

    # Find call closest to +0.10 delta (short call)
    best_call = None
    best_call_dist = float('inf')
    for c in calls:
        d = c.get('delta', 0)
        if d is None or d == 0:
            continue
        dist = abs(d - 0.10)
        if dist < best_call_dist:
            best_call_dist = dist
            best_call = c

    if not best_put or not best_call:
        return None, None, None

    underlying = best_put.get('underlying_price', 0)
    return best_put['strike'], best_call['strike'], underlying


def find_leg_file(date, leg_type, strike, entry_time):
    """Find the leg file. leg_type is 'call' or 'put'."""
    # Try exact entry time first, then nearby times
    for time_offset in range(0, 30, 5):  # try 0, 5, 10, ... minutes later
        h = int(entry_time[:2])
        m = int(entry_time[2:]) + time_offset
        if m >= 60:
            h += 1
            m -= 60
        t = f"{h:02d}{m:02d}"
        fname = f"{date}_{leg_type}_{strike}_{t}_1500.json"
        fpath = LEGS_DIR / fname
        if fpath.exists():
            return fpath, t
    return None, None


def compute_ic_walk(date, short_put, short_call, entry_time):
    """Load 4 legs and walk forward computing IC debit at each bar."""
    long_put = short_put - WING_WIDTH
    long_call = short_call + WING_WIDTH

    legs_needed = [
        ('put', short_put, 'short_put'),
        ('call', short_call, 'short_call'),
        ('put', long_put, 'long_put'),
        ('call', long_call, 'long_call'),
    ]

    leg_data = {}
    actual_entry_time = None

    for leg_type, strike, name in legs_needed:
        fpath, found_time = find_leg_file(date, leg_type, strike, entry_time)
        if fpath is None:
            return None, f"Missing leg: {leg_type} {strike}"
        if actual_entry_time is None:
            actual_entry_time = found_time
        leg_data[name] = load_json(fpath)

    # Build time-indexed mid prices for each leg
    def mid(snap):
        b = snap.get('bid', 0) or 0
        a = snap.get('ask', 0) or 0
        if b == 0 and a == 0:
            return 0
        if b == 0:
            return a
        if a == 0:
            return b
        return (b + a) / 2

    # Index by timestamp
    leg_by_time = {}
    for name, snaps in leg_data.items():
        for snap in snaps:
            ts = snap['timestamp']
            if ts not in leg_by_time:
                leg_by_time[ts] = {}
            leg_by_time[ts][name] = mid(snap)

    # Walk forward - only bars where all 4 legs present
    bars = []
    for ts in sorted(leg_by_time.keys()):
        d = leg_by_time[ts]
        if len(d) == 4:
            # Short IC credit = short_put + short_call - long_put - long_call
            # At entry, this is the credit received
            # At any point, the closing debit = same formula (what you'd pay to close)
            ic_mid = d['short_put'] + d['short_call'] - d['long_put'] - d['long_call']
            bars.append((ts, ic_mid, d))

    if not bars:
        return None, "No bars with all 4 legs"

    return bars, actual_entry_time


def analyze_day(date, label):
    """Full analysis of one day."""
    short_put, short_call, underlying = find_10_delta_strikes(date, ENTRY_TIME)
    if short_put is None:
        return None, f"Could not find 10-delta strikes"

    long_put = short_put - WING_WIDTH
    long_call = short_call + WING_WIDTH

    result = compute_ic_walk(date, short_put, short_call, ENTRY_TIME)
    if result[0] is None:
        return None, result[1]

    bars, actual_entry = result

    # Entry credit is the first bar's IC mid
    entry_credit = bars[0][1]

    if entry_credit <= 0:
        return None, f"Non-positive entry credit: {entry_credit:.2f}"

    # Thresholds
    sl_debit = entry_credit * SL_MULT
    tp10_debit = entry_credit * TP10_PCT
    tp25_debit = entry_credit * TP25_PCT
    tp50_debit = entry_credit * TP50_PCT

    # Walk forward
    first_sl = None
    first_tp10 = None
    first_tp25 = None
    first_tp50 = None

    bar_details = []

    for ts, ic_mid, legs in bars:
        time_str = ts.split('T')[1][:5] if 'T' in ts else ts[-5:]
        pnl_pct = (entry_credit - ic_mid) / entry_credit * 100

        bar_details.append({
            'time': time_str,
            'ic_mid': ic_mid,
            'pnl_pct': pnl_pct,
            'sp': legs['short_put'],
            'sc': legs['short_call'],
            'lp': legs['long_put'],
            'lc': legs['long_call'],
        })

        # Check SL: debit >= 2x credit means loss
        # When IC mid goes UP, that means options are more expensive to close = loss
        if first_sl is None and ic_mid >= sl_debit:
            first_sl = time_str

        # Check TPs: debit drops below threshold = profit
        # When IC mid goes DOWN, options cheaper to close = profit
        if first_tp10 is None and ic_mid <= tp10_debit:
            first_tp10 = time_str
        if first_tp25 is None and ic_mid <= tp25_debit:
            first_tp25 = time_str
        if first_tp50 is None and ic_mid <= tp50_debit:
            first_tp50 = time_str

    close_mid = bars[-1][1]
    close_pnl_pct = (entry_credit - close_mid) / entry_credit * 100

    # Classify
    all_tps_hit = first_tp10 and first_tp25 and first_tp50
    sl_hit = first_sl is not None

    if all_tps_hit and not sl_hit:
        day_type = "A"  # All TPs, no SL
    elif all_tps_hit and sl_hit:
        # Check order: did all TPs come before SL?
        if first_tp50 < first_sl:
            day_type = "A"  # TPs all hit first
        else:
            day_type = "B"  # SL hit before at least some TP
    elif sl_hit:
        day_type = "B"  # SL hit, not all TPs
    elif first_tp50:
        day_type = "A"  # At least TP50 hit, no SL (partial winner)
    else:
        day_type = "C"  # Neither SL nor TP50

    # Refine Type B: SL hit before at least one TP level
    if sl_hit and (not first_tp10 or first_sl < first_tp10 or
                   not first_tp25 or first_sl < first_tp25 or
                   not first_tp50 or first_sl < first_tp50):
        if sl_hit:
            day_type = "B"

    info = {
        'date': date,
        'label': label,
        'type': day_type,
        'underlying': underlying,
        'short_put': short_put,
        'short_call': short_call,
        'long_put': long_put,
        'long_call': long_call,
        'entry_credit': entry_credit,
        'close_mid': close_mid,
        'close_pnl_pct': close_pnl_pct,
        'sl_debit': sl_debit,
        'tp10_debit': tp10_debit,
        'tp25_debit': tp25_debit,
        'tp50_debit': tp50_debit,
        'first_sl': first_sl,
        'first_tp10': first_tp10,
        'first_tp25': first_tp25,
        'first_tp50': first_tp50,
        'bars': bar_details,
        'actual_entry': actual_entry,
    }

    return info, None


def print_summary(info):
    """Print one-line summary for a day."""
    i = info
    sl_str = i['first_sl'] or '---'
    tp10_str = i['first_tp10'] or '---'
    tp25_str = i['first_tp25'] or '---'
    tp50_str = i['first_tp50'] or '---'

    print(f"  {i['date']} ({i['label']:20s}) | Type {i['type']} | "
          f"SPX={i['underlying']:.0f} | IC: {i['short_put']:.0f}p/{i['short_call']:.0f}c | "
          f"Credit=${i['entry_credit']:.2f} | "
          f"TP10={tp10_str} TP25={tp25_str} TP50={tp50_str} SL={sl_str} | "
          f"3pm P&L={i['close_pnl_pct']:+.1f}%")


def print_bar_walk(info):
    """Print detailed bar-by-bar walk for a day."""
    i = info
    print(f"\n{'='*100}")
    print(f"  TYPE {i['type']} EXAMPLE: {i['date']} ({i['label']})")
    print(f"{'='*100}")
    print(f"  SPX={i['underlying']:.2f} | Short Put={i['short_put']:.0f} | Short Call={i['short_call']:.0f} | "
          f"Long Put={i['long_put']:.0f} | Long Call={i['long_call']:.0f}")
    print(f"  Entry Credit: ${i['entry_credit']:.2f}")
    print(f"  Thresholds -> SL: ${i['sl_debit']:.2f} (2x) | "
          f"TP10: ${i['tp10_debit']:.2f} | TP25: ${i['tp25_debit']:.2f} | TP50: ${i['tp50_debit']:.2f}")
    print(f"  Triggers -> SL={i['first_sl'] or '---'} | "
          f"TP10={i['first_tp10'] or '---'} | TP25={i['first_tp25'] or '---'} | TP50={i['first_tp50'] or '---'}")
    print(f"  3pm Close: ${i['close_mid']:.2f} (P&L={i['close_pnl_pct']:+.1f}%)")
    print()
    print(f"  {'Time':>5s}  {'IC Mid':>8s}  {'P&L%':>7s}  {'ShortP':>7s}  {'ShortC':>7s}  {'LongP':>7s}  {'LongC':>7s}  Events")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")

    for bar in i['bars']:
        events = []
        t = bar['time']
        if t == i['first_sl']:
            events.append("** SL HIT **")
        if t == i['first_tp10']:
            events.append("TP10")
        if t == i['first_tp25']:
            events.append("TP25")
        if t == i['first_tp50']:
            events.append("TP50")

        evt_str = ' '.join(events)
        print(f"  {t:>5s}  ${bar['ic_mid']:>7.2f}  {bar['pnl_pct']:>+6.1f}%  "
              f"${bar['sp']:>6.2f}  ${bar['sc']:>6.2f}  ${bar['lp']:>6.2f}  ${bar['lc']:>6.2f}  {evt_str}")


def scan_for_type_c():
    """Scan many dates to find Type C days."""
    import os
    chains_dir = CHAINS_DIR
    # Get all dates with 1030 chain files
    all_dates = set()
    for f in os.listdir(chains_dir):
        if f.endswith("_1030_put.json"):
            d = f.replace("_1030_put.json", "")
            all_dates.add(d)

    type_c_dates = []
    for date in sorted(all_dates):
        info, err = analyze_day(date, "scan")
        if err:
            continue
        if info['type'] == 'C':
            type_c_dates.append(info)
            if len(type_c_dates) >= 5:
                break
    return type_c_dates


def main():
    print("=" * 100)
    print("  10-DELTA SHORT IRON CONDOR: PATH-DEPENDENT TP/SL SCENARIO ANALYSIS")
    print("  Entry at 10:30 ET | $25 wings | SL=2x credit | TP10/25/50")
    print("=" * 100)
    print()

    results = []
    errors = []

    for date, label in CANDIDATE_DATES:
        info, err = analyze_day(date, label)
        if err:
            errors.append((date, label, err))
        else:
            results.append(info)

    # If no Type C found, scan for one
    type_c = [r for r in results if r['type'] == 'C']
    if not type_c:
        print("  No Type C in candidate dates. Scanning full dataset...")
        type_c_scan = scan_for_type_c()
        for tc in type_c_scan:
            tc['label'] = 'scan-found'
            results.append(tc)
        print(f"  Found {len(type_c_scan)} Type C days from scan.\n")

    # Print summary table
    print("SUMMARY")
    print("-" * 100)

    type_a = [r for r in results if r['type'] == 'A']
    type_b = [r for r in results if r['type'] == 'B']
    type_c = [r for r in results if r['type'] == 'C']

    if type_a:
        print(f"\n  Type A (All TPs hit, no SL / TPs before SL):")
        for r in type_a:
            print_summary(r)

    if type_b:
        print(f"\n  Type B (SL hit before at least one TP level):")
        for r in type_b:
            print_summary(r)

    if type_c:
        print(f"\n  Type C (Neither SL nor TP50 hit, expires at 3pm):")
        for r in type_c:
            print_summary(r)

    if errors:
        print(f"\n  ERRORS (could not analyze):")
        for date, label, err in errors:
            print(f"    {date} ({label}): {err}")

    # Print detailed walks for one representative of each type found
    printed_types = set()
    for r in results:
        if r['type'] not in printed_types:
            print_bar_walk(r)
            printed_types.add(r['type'])

    print(f"\n{'='*100}")
    print(f"  Found: {len(type_a)} Type A, {len(type_b)} Type B, {len(type_c)} Type C out of {len(results)} analyzable days")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
