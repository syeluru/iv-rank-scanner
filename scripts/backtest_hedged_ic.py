#!/usr/bin/env python3
"""
Hedged Iron Condor Backtest Engine (v2 — corrected 8-leg logic)

Tests 350 permutations of hedged IC strategies across 3 years of 0DTE data.

Strategy: Short IC (credit) + Long IC (debit) wrapping outside for tail protection.
Always 8 legs. li_width must be > si_width for hedge to provide tail recovery.

Dimensions (350 permutations after li_width > si_width filter):
  - Short IC short strike delta: 5, 10, 15, 20, 25
  - Short IC width (short→wing): $10, $15, $20, $25
  - Gap (SI wing → LI inner leg): -$5, $0, $5, $10, $15, $20, $25
  - Long IC width (inner→outer): $15, $20, $25, $30

Usage:
    # Phase 1: noon-only entry across all days
    python -m scripts.backtest_hedged_ic

    # Phase 2: 5-minute entry intervals
    python -m scripts.backtest_hedged_ic --all-entries

    # Resume from checkpoint
    python -m scripts.backtest_hedged_ic --resume

    # Date range
    python -m scripts.backtest_hedged_ic --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import json
import logging
import time as _time
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("hedged_ic_backtest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
GREEKS_PATH = DATA_DIR / "spxw_0dte_intraday_greeks.parquet"
OUTPUT_DIR = Path("output/hedged_ic_backtest")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Permutation dimensions
SHORT_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25]
SI_WIDTHS = [10, 15, 20, 25]
GAPS = [-5, 0, 5, 10, 15, 20, 25]
LI_WIDTHS = [15, 20, 25, 30]

# Cost per leg per contract (commissions + fees)
COST_PER_LEG = 0.65
FEES_8_LEG = COST_PER_LEG * 8  # $5.20 — always 8 legs

# Phase 1 entry time
NOON_ENTRY = "12:00"

# Phase 2 entry times: every 5 min from 10:00 to 15:00
PHASE2_START = "10:00"
PHASE2_END = "15:00"
PHASE2_INTERVAL = 5  # minutes


# ---------------------------------------------------------------------------
# Data Loading (reuse pattern from compute_v8_labels.py)
# ---------------------------------------------------------------------------

def load_day(d: date) -> pd.DataFrame:
    """Load and clean one day of greeks data."""
    target = pd.Timestamp(d)
    df = pd.read_parquet(
        GREEKS_PATH,
        filters=[("date", "=", target)],
    )

    if df.empty:
        return df

    # Filter bad data
    df = df[
        (df["implied_vol"] > 0) &
        (df["implied_vol"] < 5.0) &
        (df["underlying_price"] > 0) &
        ((df["bid"] > 0) | (df["ask"] > 0))
    ].copy()

    # Normalize right
    df["right"] = df["right"].str.upper().str[0]

    # Compute mid price
    df["mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2.0,
        np.maximum(df["bid"], df["ask"]),
    )

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute_key"] = df["timestamp"].dt.strftime("%H:%M")

    return df


def build_lookup(day_df: pd.DataFrame) -> dict:
    """
    Build (strike, right, minute_key) -> {mid, bid, ask, delta} lookup.
    Much faster than repeated DataFrame filtering.
    """
    lookup = {}
    for _, row in day_df.iterrows():
        key = (row["strike"], row["right"], row["minute_key"])
        lookup[key] = {
            "mid": row["mid"],
            "bid": row.get("bid", 0) or 0,
            "ask": row.get("ask", 0) or 0,
            "delta": row.get("delta", 0) or 0,
        }
    return lookup


def build_lookup_fast(day_df: pd.DataFrame) -> dict:
    """
    Vectorized lookup builder — much faster than iterrows.
    Returns (strike, right, minute_key) -> {mid, bid, ask, delta}.
    """
    strikes = day_df["strike"].values
    rights = day_df["right"].values
    minute_keys = day_df["minute_key"].values
    mids = day_df["mid"].values
    bids = day_df["bid"].values if "bid" in day_df.columns else np.zeros(len(day_df))
    asks = day_df["ask"].values if "ask" in day_df.columns else np.zeros(len(day_df))
    deltas = day_df["delta"].values if "delta" in day_df.columns else np.zeros(len(day_df))

    lookup = {}
    for i in range(len(day_df)):
        key = (strikes[i], rights[i], minute_keys[i])
        lookup[key] = {
            "mid": mids[i],
            "bid": bids[i] if bids[i] == bids[i] else 0,  # NaN check
            "ask": asks[i] if asks[i] == asks[i] else 0,
            "delta": deltas[i] if deltas[i] == deltas[i] else 0,
        }
    return lookup


# ---------------------------------------------------------------------------
# Strike Selection
# ---------------------------------------------------------------------------

def find_short_strikes(chain: pd.DataFrame, target_delta: float) -> dict | None:
    """
    Find put and call strikes closest to target |delta| from entry chain.
    Returns {"put": strike, "call": strike, "spx": underlying_price} or None.
    """
    puts = chain[chain["right"] == "P"]
    calls = chain[chain["right"] == "C"]

    if puts.empty or calls.empty:
        return None

    # Find put closest to target delta (puts have negative delta, use abs)
    put_diffs = (puts["delta"].abs() - target_delta).abs()
    put_idx = put_diffs.idxmin()
    put_strike = float(puts.loc[put_idx, "strike"])

    # Find call closest to target delta
    call_diffs = (calls["delta"].abs() - target_delta).abs()
    call_idx = call_diffs.idxmin()
    call_strike = float(calls.loc[call_idx, "strike"])

    spx = float(chain["underlying_price"].iloc[0])

    return {"put": put_strike, "call": call_strike, "spx": spx}


# ---------------------------------------------------------------------------
# Hedged IC Construction
# ---------------------------------------------------------------------------

def build_hedged_ic(
    lookup: dict,
    minute_key: str,
    short_put: float,
    short_call: float,
    si_width: int,
    gap: int,
    li_width: int,
) -> dict | None:
    """
    Construct 8-leg hedged IC and compute entry prices.

    Returns dict with all strikes, prices, credits/debits, or None if missing data.

    Strike layout:
        PUT SIDE:  LI_short_put ... LI_long_put [GAP] SI_long_put ... SI_short_put
        CALL SIDE: SI_short_call ... SI_long_call [GAP] LI_long_call ... LI_short_call
    """
    # Compute all 8 strikes
    si_short_put = short_put
    si_long_put = short_put - si_width
    li_long_put = short_put - si_width - gap
    li_short_put = short_put - si_width - gap - li_width

    si_short_call = short_call
    si_long_call = short_call + si_width
    li_long_call = short_call + si_width + gap
    li_short_call = short_call + si_width + gap + li_width

    # Always look up all 8 leg prices — even when gap=0, we BUY the shared
    # strike twice (once for SI, once for LI). No legs cancel.
    legs = {
        "si_short_put": (si_short_put, "P"),    # sell
        "si_long_put": (si_long_put, "P"),       # buy
        "si_short_call": (si_short_call, "C"),   # sell
        "si_long_call": (si_long_call, "C"),     # buy
        "li_long_put": (li_long_put, "P"),       # buy
        "li_short_put": (li_short_put, "P"),     # sell
        "li_long_call": (li_long_call, "C"),     # buy
        "li_short_call": (li_short_call, "C"),   # sell
    }

    prices = {}
    for leg_name, (strike, right) in legs.items():
        data = lookup.get((strike, right, minute_key))
        if data is None or data["mid"] <= 0:
            return None
        prices[leg_name] = data["mid"]

    # SI credit (sell short strikes, buy long strikes)
    si_credit = (prices["si_short_put"] + prices["si_short_call"]) - \
                (prices["si_long_put"] + prices["si_long_call"])

    # LI debit (buy long strikes, sell short strikes) — ALWAYS full 4-leg
    li_debit = (prices["li_long_put"] + prices["li_long_call"]) - \
               (prices["li_short_put"] + prices["li_short_call"])

    net_credit = si_credit - li_debit
    fees_per_share = FEES_8_LEG / 100  # always 8 legs
    net_credit_after_fees = net_credit - fees_per_share

    if net_credit_after_fees <= 0:
        return None

    return {
        # Strikes
        "si_short_put": si_short_put,
        "si_long_put": si_long_put,
        "si_short_call": si_short_call,
        "si_long_call": si_long_call,
        "li_long_put": li_long_put,
        "li_short_put": li_short_put,
        "li_long_call": li_long_call,
        "li_short_call": li_short_call,
        # Prices
        "si_credit": si_credit,
        "li_debit": li_debit,
        "net_credit": net_credit,
        "fees_per_share": fees_per_share,
        "net_credit_after_fees": net_credit_after_fees,
    }


# ---------------------------------------------------------------------------
# P&L Computation
# ---------------------------------------------------------------------------

def compute_expiry_pnl(ic: dict, spx_close: float) -> dict:
    """
    Compute actual P&L at expiration given closing SPX price.

    For each leg, intrinsic value at expiry:
      Put:  max(strike - spx, 0)
      Call: max(spx - strike, 0)

    Short IC: we sold short strikes, bought long strikes
    Long IC: we bought inner (long) strikes, sold outer (short) strikes
    """
    spx = spx_close

    # Short IC intrinsic P&L (from our perspective as seller)
    si_short_put_val = -max(ic["si_short_put"] - spx, 0)   # sold put, we pay
    si_long_put_val = max(ic["si_long_put"] - spx, 0)      # bought put, we receive
    si_short_call_val = -max(spx - ic["si_short_call"], 0)  # sold call, we pay
    si_long_call_val = max(spx - ic["si_long_call"], 0)     # bought call, we receive

    si_intrinsic = si_short_put_val + si_long_put_val + si_short_call_val + si_long_call_val
    si_pnl = si_intrinsic + ic["si_credit"]

    # Long IC intrinsic P&L (we buy inner longs, sell outer shorts) — always 4 legs
    li_long_put_val = max(ic["li_long_put"] - spx, 0)       # bought put
    li_short_put_val = -max(ic["li_short_put"] - spx, 0)    # sold put
    li_long_call_val = max(spx - ic["li_long_call"], 0)     # bought call
    li_short_call_val = -max(spx - ic["li_short_call"], 0)  # sold call
    li_intrinsic = li_long_put_val + li_short_put_val + li_long_call_val + li_short_call_val

    li_pnl = li_intrinsic - ic["li_debit"]

    total_pnl = si_pnl + li_pnl  # = si_intrinsic + li_intrinsic + net_credit
    total_pnl_after_fees = total_pnl - ic["fees_per_share"]

    # Max possible loss for this structure
    # Put side max loss: between SI inner legs and LI inner legs (the gap zone)
    # Call side max loss: similar gap zone
    # Overall max loss = -si_width - gap + net_credit (worst case in gap zone)

    return {
        "si_pnl": si_pnl,
        "li_pnl": li_pnl,
        "total_pnl": total_pnl,
        "total_pnl_after_fees": total_pnl_after_fees,
        "total_pnl_dollar": total_pnl_after_fees * 100,  # per contract
    }


# ---------------------------------------------------------------------------
# Get SPX close price
# ---------------------------------------------------------------------------

def get_spx_close(lookup: dict, day_df: pd.DataFrame) -> float | None:
    """Get SPX price at end of day (try 15:59, 16:00, then last available)."""
    # Try common closing times
    for mk in ["15:59", "16:00", "15:58"]:
        # Find any entry at this minute for underlying price
        for key, val in lookup.items():
            if key[2] == mk:
                # Get underlying_price from the dataframe for this minute
                subset = day_df[day_df["minute_key"] == mk]
                if not subset.empty:
                    return float(subset["underlying_price"].iloc[0])

    # Fallback: last available minute
    last_mk = day_df["minute_key"].max()
    subset = day_df[day_df["minute_key"] == last_mk]
    if not subset.empty:
        return float(subset["underlying_price"].iloc[0])

    return None


def get_spx_close_fast(day_df: pd.DataFrame) -> float | None:
    """Get SPX closing price efficiently."""
    for mk in ["15:59", "16:00", "15:58"]:
        subset = day_df[day_df["minute_key"] == mk]
        if not subset.empty:
            return float(subset["underlying_price"].iloc[0])

    # Fallback: last minute
    last_mk = day_df["minute_key"].max()
    subset = day_df[day_df["minute_key"] == last_mk]
    if not subset.empty:
        return float(subset["underlying_price"].iloc[0])
    return None


# ---------------------------------------------------------------------------
# Permutation Generator
# ---------------------------------------------------------------------------

def generate_permutations(
    deltas: list[float] | None = None,
    si_widths: list[int] | None = None,
    gaps: list[int] | None = None,
    li_widths: list[int] | None = None,
) -> list[dict]:
    """Generate permutation configs, optionally filtered."""
    d_list = deltas or SHORT_DELTAS
    s_list = si_widths or SI_WIDTHS
    g_list = gaps or GAPS
    l_list = li_widths or LI_WIDTHS
    perms = []
    for delta, si_w, gap, li_w in product(d_list, s_list, g_list, l_list):
        # Constraint: li_width must be strictly greater than si_width
        if li_w <= si_w:
            continue
        # Constraint: for negative gaps, gap must be > -si_width
        # (LI_long must stay below SI_short)
        if gap < 0 and gap <= -si_w:
            continue
        perms.append({
            "delta": delta,
            "si_width": si_w,
            "gap": gap,
            "li_width": li_w,
            "label": f"d{int(delta*100)}_si{si_w}_g{gap}_li{li_w}",
        })
    return perms


def generate_entry_times(start: str, end: str, interval: int) -> list[str]:
    """Generate entry time strings at given interval."""
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    times = []
    cur = s_h * 60 + s_m
    end_min = e_h * 60 + e_m
    while cur <= end_min:
        h, m = divmod(cur, 60)
        times.append(f"{h:02d}:{m:02d}")
        cur += interval
    return times


# ---------------------------------------------------------------------------
# Single Day Processing
# ---------------------------------------------------------------------------

def run_single_day(
    d: date,
    day_df: pd.DataFrame,
    entry_times: list[str],
    permutations: list[dict],
) -> list[dict]:
    """
    Process one day across all entry times and permutations.
    Returns list of result dicts.
    """
    if day_df.empty:
        return []

    # Build fast lookup
    lookup = build_lookup_fast(day_df)

    # Get SPX close
    spx_close = get_spx_close_fast(day_df)
    if spx_close is None:
        return []

    results = []

    # Cache: delta -> {put, call, spx} for each entry time
    # Short strikes only depend on delta + entry time
    for entry_time in entry_times:
        entry_chain = day_df[day_df["minute_key"] == entry_time]
        if entry_chain.empty:
            continue

        spx_entry = float(entry_chain["underlying_price"].iloc[0])

        # Find short strikes for each delta level (cache across permutations)
        unique_deltas = set(p["delta"] for p in permutations)
        delta_strikes = {}
        for delta in unique_deltas:
            strikes = find_short_strikes(entry_chain, delta)
            if strikes is not None:
                delta_strikes[delta] = strikes

        # Now test each permutation
        for perm in permutations:
            delta = perm["delta"]
            if delta not in delta_strikes:
                continue

            short_put = delta_strikes[delta]["put"]
            short_call = delta_strikes[delta]["call"]

            # Build the hedged IC
            ic = build_hedged_ic(
                lookup, entry_time,
                short_put, short_call,
                perm["si_width"], perm["gap"], perm["li_width"],
            )
            if ic is None:
                continue

            # Compute P&L at expiry
            pnl = compute_expiry_pnl(ic, spx_close)

            results.append({
                "date": str(d),
                "entry_time": entry_time,
                "spx_entry": spx_entry,
                "spx_close": spx_close,
                "spx_move": spx_close - spx_entry,
                "spx_move_pct": (spx_close - spx_entry) / spx_entry * 100,
                # Permutation
                "delta": delta,
                "si_width": perm["si_width"],
                "gap": perm["gap"],
                "li_width": perm["li_width"],
                "label": perm["label"],
                # Strikes
                "si_short_put": ic["si_short_put"],
                "si_long_put": ic["si_long_put"],
                "si_short_call": ic["si_short_call"],
                "si_long_call": ic["si_long_call"],
                "li_short_put": ic["li_short_put"],
                "li_long_put": ic["li_long_put"],
                "li_short_call": ic["li_short_call"],
                "li_long_call": ic["li_long_call"],
                # Prices
                "si_credit": ic["si_credit"],
                "li_debit": ic["li_debit"],
                "net_credit": ic["net_credit"],
                "fees_per_share": ic["fees_per_share"],
                "net_credit_after_fees": ic["net_credit_after_fees"],
                # P&L
                "si_pnl": pnl["si_pnl"],
                "li_pnl": pnl["li_pnl"],
                "total_pnl": pnl["total_pnl"],
                "total_pnl_after_fees": pnl["total_pnl_after_fees"],
                "pnl_dollar": pnl["total_pnl_dollar"],
                # Derived
                "pnl_pct": pnl["total_pnl_after_fees"] / ic["net_credit_after_fees"] * 100
                    if ic["net_credit_after_fees"] > 0 else 0,
                "win": 1 if pnl["total_pnl_after_fees"] > 0 else 0,
            })

    return results


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load checkpoint progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_dates": [], "phase": None}


def save_progress(progress: dict):
    """Save checkpoint progress."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------------

def generate_summary(results_path: Path) -> pd.DataFrame:
    """Generate summary statistics grouped by permutation label."""
    df = pd.read_csv(results_path)

    summary = df.groupby("label").agg(
        delta=("delta", "first"),
        si_width=("si_width", "first"),
        gap=("gap", "first"),
        li_width=("li_width", "first"),
        num_trades=("pnl_dollar", "count"),
        avg_pnl_dollar=("pnl_dollar", "mean"),
        median_pnl_dollar=("pnl_dollar", "median"),
        total_pnl_dollar=("pnl_dollar", "sum"),
        win_rate=("win", "mean"),
        avg_net_credit=("net_credit_after_fees", "mean"),
        avg_pnl_pct=("pnl_pct", "mean"),
        std_pnl_dollar=("pnl_dollar", "std"),
        min_pnl_dollar=("pnl_dollar", "min"),
        max_pnl_dollar=("pnl_dollar", "max"),
        p5_pnl=("pnl_dollar", lambda x: x.quantile(0.05)),
        p95_pnl=("pnl_dollar", lambda x: x.quantile(0.95)),
    ).reset_index()

    # Sharpe-like ratio (annualized, ~252 trading days)
    summary["sharpe"] = np.where(
        summary["std_pnl_dollar"] > 0,
        (summary["avg_pnl_dollar"] / summary["std_pnl_dollar"]) * np.sqrt(252),
        0,
    )

    # Profit factor
    wins = df[df["pnl_dollar"] > 0].groupby("label")["pnl_dollar"].sum()
    losses = df[df["pnl_dollar"] < 0].groupby("label")["pnl_dollar"].sum().abs()
    pf = (wins / losses).replace([np.inf, -np.inf], np.nan).fillna(0)
    summary["profit_factor"] = summary["label"].map(pf).fillna(0)

    # Sort by total P&L
    summary = summary.sort_values("total_pnl_dollar", ascending=False)

    return summary


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def get_all_dates() -> list[date]:
    """Get all unique dates from the greeks parquet."""
    pf = pd.read_parquet(GREEKS_PATH, columns=["date"])
    dates = sorted(pf["date"].unique())
    return [pd.Timestamp(d).date() for d in dates]


def run_backtest(
    start_date: date | None = None,
    end_date: date | None = None,
    all_entries: bool = False,
    resume: bool = False,
    tag: str | None = None,
    filter_deltas: list[float] | None = None,
    filter_si_widths: list[int] | None = None,
    filter_gaps: list[int] | None = None,
    filter_li_widths: list[int] | None = None,
):
    """Main backtest orchestration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = OUTPUT_DIR / "backtest.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    # Get dates
    all_dates = get_all_dates()
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    logger.info(f"Total dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Entry times
    if all_entries:
        entry_times = generate_entry_times(PHASE2_START, PHASE2_END, PHASE2_INTERVAL)
        phase = "phase2"
    else:
        entry_times = [NOON_ENTRY]
        phase = "phase1"

    logger.info(f"Phase: {phase}, Entry times: {len(entry_times)}")

    # Permutations (optionally filtered)
    permutations = generate_permutations(
        deltas=filter_deltas,
        si_widths=filter_si_widths,
        gaps=filter_gaps,
        li_widths=filter_li_widths,
    )
    logger.info(f"Permutations: {len(permutations)}")

    # File suffix: use tag if provided, otherwise phase
    suffix = tag or phase
    progress_file = OUTPUT_DIR / f"progress_{suffix}.json"

    # Resume support
    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed_dates": [], "phase": phase, "tag": suffix}
    completed = set(progress.get("completed_dates", []))

    remaining_dates = [d for d in all_dates if str(d) not in completed]
    logger.info(f"Remaining dates: {len(remaining_dates)} (skipping {len(completed)} completed)")

    # Group permutations into families by (delta, si_width, li_width)
    # Each family varies only by gap — share same data load
    from collections import defaultdict
    families = defaultdict(list)
    for perm in permutations:
        fam_key = f"d{int(perm['delta']*100)}_si{perm['si_width']}_li{perm['li_width']}"
        families[fam_key].append(perm)

    # Results file — one combined file, plus per-family summaries
    results_file = OUTPUT_DIR / f"results_{suffix}.csv"
    write_header = not results_file.exists() or not resume

    total_results = 0
    t0 = _time.time()
    family_results_accum = defaultdict(list)  # fam_key -> list of result dicts

    for i, d in enumerate(remaining_dates):
        t_day = _time.time()

        # Load day data (once — this is the bottleneck)
        day_df = load_day(d)
        if day_df.empty:
            logger.warning(f"[{i+1}/{len(remaining_dates)}] {d}: no data, skipping")
            completed.add(str(d))
            progress["completed_dates"] = sorted(completed)
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
            continue

        # Run all permutations for this day
        results = run_single_day(d, day_df, entry_times, permutations)

        # Write results
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(
                results_file,
                mode="a" if not write_header else "w",
                header=write_header,
                index=False,
            )
            write_header = False  # Only write header once
            total_results += len(results)

            # Accumulate per-family for incremental summaries
            for r in results:
                fam_key = f"d{int(r['delta']*100)}_si{r['si_width']}_li{r['li_width']}"
                family_results_accum[fam_key].append(r)

        # Update checkpoint
        completed.add(str(d))
        progress["completed_dates"] = sorted(completed)
        progress["phase"] = phase
        progress["tag"] = suffix
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        elapsed = _time.time() - t_day
        total_elapsed = _time.time() - t0
        rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
        eta = (len(remaining_dates) - i - 1) / rate if rate > 0 else 0

        logger.info(
            f"[{i+1}/{len(remaining_dates)}] {d}: "
            f"{len(results)} results in {elapsed:.1f}s "
            f"(total: {total_results:,}, ETA: {eta/60:.0f}min)"
        )

        # Write incremental per-family summaries every 100 days
        if (i + 1) % 100 == 0 or (i + 1) == len(remaining_dates):
            for fam_key, fam_results in family_results_accum.items():
                if len(fam_results) > 0:
                    fam_df = pd.DataFrame(fam_results)
                    fam_file = OUTPUT_DIR / f"family_{fam_key}.csv"
                    fam_df.to_csv(fam_file, index=False)
            logger.info(f"  Wrote {len(family_results_accum)} family files (incremental)")

    # Generate summary
    logger.info("Generating summary...")
    if results_file.exists():
        summary = generate_summary(results_file)
        summary_file = OUTPUT_DIR / f"summary_{suffix}.csv"
        summary.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")

        # Top 10 by total P&L
        best_file = OUTPUT_DIR / f"best_combos_{suffix}.csv"
        best = summary.head(min(10, len(summary)))
        best.to_csv(best_file, index=False)
        logger.info(f"Best combos saved to {best_file}")

        # Print top 10
        print("\n" + "=" * 80)
        print("TOP 10 PERMUTATIONS BY TOTAL P&L")
        print("=" * 80)
        for _, row in best.iterrows():
            print(
                f"  {row['label']:25s}  "
                f"trades={row['num_trades']:5.0f}  "
                f"win={row['win_rate']:.1%}  "
                f"avg=${row['avg_pnl_dollar']:+7.1f}  "
                f"total=${row['total_pnl_dollar']:+10.0f}  "
                f"sharpe={row['sharpe']:.2f}  "
                f"PF={row['profit_factor']:.2f}"
            )

        # Print bottom 5
        if len(summary) > 5:
            print("\nBOTTOM 5 PERMUTATIONS:")
            for _, row in summary.tail(5).iterrows():
                print(
                    f"  {row['label']:25s}  "
                    f"trades={row['num_trades']:5.0f}  "
                    f"win={row['win_rate']:.1%}  "
                    f"avg=${row['avg_pnl_dollar']:+7.1f}  "
                    f"total=${row['total_pnl_dollar']:+10.0f}"
                )

        # Print per-family summaries
        if len(families) > 1:
            print("\n" + "=" * 80)
            print("PER-FAMILY SUMMARY")
            print("=" * 80)
            for fam_key in sorted(families.keys()):
                fam_labels = [p["label"] for p in families[fam_key]]
                fam_summary = summary[summary["label"].isin(fam_labels)]
                if not fam_summary.empty:
                    best_row = fam_summary.iloc[0]
                    print(
                        f"  {fam_key:25s}  "
                        f"best={best_row['label']:25s}  "
                        f"win={best_row['win_rate']:.1%}  "
                        f"avg=${best_row['avg_pnl_dollar']:+7.1f}  "
                        f"total=${best_row['total_pnl_dollar']:+10.0f}"
                    )

    total_time = _time.time() - t0
    logger.info(f"Backtest complete: {total_results:,} results in {total_time/60:.1f} min")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]

def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]

def main():
    parser = argparse.ArgumentParser(description="Hedged IC Backtest")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all-entries", action="store_true",
                        help="Phase 2: test all 5-min entry intervals (default: noon only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--tag", type=str,
                        help="Tag for output files (e.g., 'd20_si20_li25')")
    parser.add_argument("--delta", type=str,
                        help="Filter short deltas (comma-sep, e.g., '0.20' or '0.10,0.15,0.20')")
    parser.add_argument("--si-width", type=str,
                        help="Filter SI widths (comma-sep, e.g., '20' or '15,20,25')")
    parser.add_argument("--gap", type=str,
                        help="Filter gaps (comma-sep, e.g., '0,5' or '0,5,10')")
    parser.add_argument("--li-width", type=str,
                        help="Filter LI widths (comma-sep, e.g., '25' or '10,15,20,25')")
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None

    f_deltas = parse_float_list(args.delta) if args.delta else None
    f_si = parse_int_list(args.si_width) if args.si_width else None
    f_gap = parse_int_list(args.gap) if args.gap else None
    f_li = parse_int_list(args.li_width) if args.li_width else None

    run_backtest(
        start_date=start,
        end_date=end,
        all_entries=args.all_entries,
        resume=args.resume,
        tag=args.tag,
        filter_deltas=f_deltas,
        filter_si_widths=f_si,
        filter_gaps=f_gap,
        filter_li_widths=f_li,
    )


if __name__ == "__main__":
    main()
