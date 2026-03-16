#!/usr/bin/env python3
"""
V8 Label Computation — 1-minute walk-forward for Short + Long IC labels.

Computes labels across:
  - 2 sides (short, long) with SL logic: 18 labels × 3 fills = 108 columns
  - 1 side (long_nosl) without SL: 12 labels × 3 fills = 36 columns
  - Total: 144 label columns + 6 entry prices + 6 metadata = 156 columns

Walk-forward runs from entry to 4pm ET (expiration).

Uses local 1-min greeks parquet (data/spxw_0dte_intraday_greeks.parquet).

Usage:
    python -m scripts.compute_v8_labels
    python -m scripts.compute_v8_labels --start-date 2024-01-01 --end-date 2024-01-31
    python -m scripts.compute_v8_labels --resume
"""

import argparse
import json
import logging
import math
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("v8_labels")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("ml/features/v8/data")
PROGRESS_FILE = OUTPUT_DIR / "v3_labels_progress.json"

GREEKS_PATH = DATA_DIR / "spxw_0dte_intraday_greeks.parquet"

WING_WIDTH = 25.0
TARGET_DELTA = 0.10
CLOSE_TIME = time(16, 0)  # 4pm ET (expiration)

# Short IC thresholds (multipliers of credit)
SHORT_TP_LEVELS = [0.90, 0.75, 0.50]  # IC value must be <= this fraction of credit
SHORT_SL_MULT = 2.00                   # IC value >= 2x credit = 100% loss

# Long IC thresholds (multipliers of debit)
LONG_TP_LEVELS = [1.10, 1.25, 1.50]   # IC value must be >= this fraction of debit
LONG_SL_MULT = 0.50                    # IC value <= 0.5x debit = 50% loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_down_05(val: float) -> float:
    """Round down to nearest $0.05."""
    return math.floor(val * 20) / 20


def round_up_05(val: float) -> float:
    """Round up to nearest $0.05."""
    return math.ceil(val * 20) / 20


def find_delta_strike(chain: pd.DataFrame, target_delta: float, right: str) -> float | None:
    """Find strike closest to target |delta| for given right ('C' or 'P')."""
    subset = chain[chain["right"] == right]
    if subset.empty:
        return None
    diffs = (subset["delta"].abs() - abs(target_delta)).abs()
    best_idx = diffs.idxmin()
    return float(subset.loc[best_idx, "strike"])


def get_leg_prices(chain: pd.DataFrame, strike: float, right: str) -> dict | None:
    """Get bid/ask/mid for a specific (strike, right) from chain."""
    rows = chain[(chain["strike"] == strike) & (chain["right"] == right)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    if bid <= 0 and ask <= 0:
        return None
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
    return {"bid": bid, "ask": ask, "mid": mid}


# ---------------------------------------------------------------------------
# IC Price Computation
# ---------------------------------------------------------------------------

def compute_ic_strikes(chain: pd.DataFrame) -> dict | None:
    """
    Find 10-delta IC strikes from the entry chain.
    Returns dict with sp, sc, lp, lc strikes or None if can't find.
    """
    sp = find_delta_strike(chain, TARGET_DELTA, "P")
    sc = find_delta_strike(chain, TARGET_DELTA, "C")
    if sp is None or sc is None:
        return None
    return {
        "sp": sp,                    # short put
        "sc": sc,                    # short call
        "lp": sp - WING_WIDTH,      # long put
        "lc": sc + WING_WIDTH,      # long call
    }


def compute_entry_prices(chain: pd.DataFrame, strikes: dict) -> dict | None:
    """
    Compute entry prices under all 3 fill scenarios.
    Returns dict with 6 values (3 short credits, 3 long debits) or None.
    """
    sp = get_leg_prices(chain, strikes["sp"], "P")
    sc = get_leg_prices(chain, strikes["sc"], "C")
    lp = get_leg_prices(chain, strikes["lp"], "P")
    lc = get_leg_prices(chain, strikes["lc"], "C")

    if any(x is None for x in [sp, sc, lp, lc]):
        return None

    # Mid fill: all legs at mid
    mid_raw = (sp["mid"] + sc["mid"]) - (lp["mid"] + lc["mid"])

    # Worst fill for short: sell at bid, buy at ask
    worst_short_raw = (sp["bid"] + sc["bid"]) - (lp["ask"] + lc["ask"])

    # Worst fill for long: buy at ask, sell at bid
    worst_long_raw = (sp["ask"] + sc["ask"]) - (lp["bid"] + lc["bid"])

    # Apply rounding
    short_credit_mid = round_down_05(mid_raw)
    short_credit_mid10 = round_down_05(mid_raw - 0.10)
    short_credit_worst = round_down_05(worst_short_raw)

    long_debit_mid = round_up_05(mid_raw)
    long_debit_mid10 = round_up_05(mid_raw + 0.10)
    long_debit_worst = round_up_05(worst_long_raw)

    # Validate: all must be positive
    if short_credit_mid <= 0 or long_debit_mid <= 0:
        return None

    return {
        "short_credit_mid": short_credit_mid,
        "short_credit_mid10": short_credit_mid10 if short_credit_mid10 > 0 else np.nan,
        "short_credit_worst": short_credit_worst if short_credit_worst > 0 else np.nan,
        "long_debit_mid": long_debit_mid,
        "long_debit_mid10": long_debit_mid10,
        "long_debit_worst": long_debit_worst,
    }


def compute_ic_mid_value(tick_data: dict) -> float | None:
    """
    Compute IC mid-price from pre-looked-up leg prices at a given tick.
    tick_data has keys 'sp', 'sc', 'lp', 'lc' each with 'mid' price.
    """
    if any(tick_data[k] is None for k in ["sp", "sc", "lp", "lc"]):
        return None
    return (tick_data["sp"] + tick_data["sc"]) - (tick_data["lp"] + tick_data["lc"])


# ---------------------------------------------------------------------------
# Walk-Forward Label Engine
# ---------------------------------------------------------------------------

def compute_labels_for_entry(
    entry_prices: dict,
    ic_values: list[float],      # IC mid at each minute from entry onward
    minutes_from_entry: list[int],  # corresponding minutes elapsed
    ic_3pm: float | None,        # IC mid at exactly 3pm
    ic_4pm: float | None,        # IC mid at exactly 4pm (expiration)
) -> dict:
    """
    Compute all labels for one (date, entry_time) combination.

    ic_values: list of IC mid-prices at each 1-min tick from entry to 4pm
    minutes_from_entry: parallel list of elapsed minutes
    ic_3pm: IC value at 3pm
    ic_4pm: IC value at 4pm (expiration)
    """
    labels = {}

    # Process each fill scenario
    for fill in ["mid", "mid10", "worst"]:
        short_credit = entry_prices.get(f"short_credit_{fill}", np.nan)
        long_debit = entry_prices.get(f"long_debit_{fill}", np.nan)

        # Store entry prices
        labels[f"short_{fill}_credit"] = short_credit
        labels[f"long_{fill}_debit"] = long_debit

        # Short IC labels (with SL)
        short_labels = _compute_short_labels(short_credit, ic_values, minutes_from_entry, ic_3pm, ic_4pm, fill)
        labels.update(short_labels)

        # Long IC labels (with SL)
        long_labels = _compute_long_labels(long_debit, ic_values, minutes_from_entry, ic_3pm, ic_4pm, fill)
        labels.update(long_labels)

        # Long IC labels (no SL — max loss = debit paid)
        long_nosl_labels = _compute_long_nosl_labels(long_debit, ic_values, minutes_from_entry, ic_4pm, fill)
        labels.update(long_nosl_labels)

    return labels


def _compute_short_labels(
    credit: float,
    ic_values: list[float],
    minutes: list[int],
    ic_3pm: float | None,
    ic_4pm: float | None,
    fill: str,
) -> dict:
    """Compute all 18 short IC labels for one fill scenario."""
    prefix = f"short_{fill}"
    labels = {}

    if np.isnan(credit) or credit <= 0:
        for suffix in _all_short_suffixes():
            labels[f"{prefix}_{suffix}"] = np.nan
        return labels

    # Thresholds
    tp_levels = {10: credit * 0.90, 25: credit * 0.75, 50: credit * 0.50}
    sl_level = credit * SHORT_SL_MULT

    # Walk forward — track events
    tp_hit = {10: False, 25: False, 50: False}
    tp_hit_time = {10: np.nan, 25: np.nan, 50: np.nan}
    tp_hit_price = {10: np.nan, 25: np.nan, 50: np.nan}
    sl_hit = False
    sl_hit_time = np.nan
    sl_hit_price = np.nan
    first_profitable_time = np.nan
    max_loss_pnl = 0.0
    max_loss_time = 0

    for ic_val, elapsed in zip(ic_values, minutes):
        if ic_val is None or np.isnan(ic_val):
            continue

        pnl_pct = (credit - ic_val) / credit

        if pnl_pct < max_loss_pnl:
            max_loss_pnl = pnl_pct
            max_loss_time = elapsed

        if np.isnan(first_profitable_time) and ic_val < credit:
            first_profitable_time = elapsed

        if not sl_hit and ic_val >= sl_level:
            sl_hit = True
            sl_hit_time = elapsed
            sl_hit_price = ic_val

        for tp_pct, tp_lvl in tp_levels.items():
            if not tp_hit[tp_pct] and not sl_hit and ic_val <= tp_lvl:
                tp_hit[tp_pct] = True
                tp_hit_time[tp_pct] = elapsed
                tp_hit_price[tp_pct] = ic_val

    # --- Binary TP labels ---
    labels[f"{prefix}_tp10"] = 1.0 if tp_hit[10] else 0.0
    labels[f"{prefix}_tp25"] = 1.0 if tp_hit[25] else 0.0
    labels[f"{prefix}_tp50"] = 1.0 if tp_hit[50] else 0.0

    # --- Binary status labels ---
    any_exit = sl_hit or any(tp_hit.values())
    if any_exit or ic_3pm is None:
        labels[f"{prefix}_3pm_profitable_if_open"] = np.nan
    else:
        labels[f"{prefix}_3pm_profitable_if_open"] = 1.0 if ic_3pm < credit else 0.0

    if ic_3pm is not None:
        labels[f"{prefix}_3pm_profitable_raw"] = 1.0 if ic_3pm < credit else 0.0
    else:
        labels[f"{prefix}_3pm_profitable_raw"] = np.nan

    if ic_4pm is not None:
        labels[f"{prefix}_4pm_profitable_raw"] = 1.0 if ic_4pm < credit else 0.0
    else:
        labels[f"{prefix}_4pm_profitable_raw"] = np.nan

    # --- Realized P&L labels ---
    def _short_pnl(exit_price):
        return (credit - exit_price) / credit

    labels[f"{prefix}_pnl_tp10_sl_3pm"] = _realized_pnl_short(
        credit, tp_hit[10], tp_hit_price[10], tp_hit_time[10],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    labels[f"{prefix}_pnl_tp25_sl_3pm"] = _realized_pnl_short(
        credit, tp_hit[25], tp_hit_price[25], tp_hit_time[25],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    labels[f"{prefix}_pnl_tp50_sl_3pm"] = _realized_pnl_short(
        credit, tp_hit[50], tp_hit_price[50], tp_hit_time[50],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    if sl_hit:
        labels[f"{prefix}_pnl_sl_3pm"] = _short_pnl(sl_hit_price)
    elif ic_3pm is not None:
        labels[f"{prefix}_pnl_sl_3pm"] = _short_pnl(ic_3pm)
    else:
        labels[f"{prefix}_pnl_sl_3pm"] = np.nan

    labels[f"{prefix}_pnl_3pm_raw"] = _short_pnl(ic_3pm) if ic_3pm is not None else np.nan
    labels[f"{prefix}_pnl_4pm_raw"] = _short_pnl(ic_4pm) if ic_4pm is not None else np.nan

    # --- Time-to-event labels ---
    labels[f"{prefix}_time_to_tp10"] = tp_hit_time[10] if tp_hit[10] else np.nan
    labels[f"{prefix}_time_to_tp25"] = tp_hit_time[25] if tp_hit[25] else np.nan
    labels[f"{prefix}_time_to_tp50"] = tp_hit_time[50] if tp_hit[50] else np.nan
    labels[f"{prefix}_time_to_sl"] = sl_hit_time if sl_hit else np.nan
    labels[f"{prefix}_time_to_profitable"] = first_profitable_time
    labels[f"{prefix}_time_to_max_loss"] = float(max_loss_time)

    return labels


def _realized_pnl_short(
    credit, tp_hit, tp_price, tp_time,
    sl_hit, sl_price, sl_time,
    ic_3pm,
) -> float:
    """
    Compute realized P&L for short IC given TP and SL events.
    First event wins: TP vs SL vs 3pm fallback.
    """
    # Determine which happened first
    tp_happened = tp_hit and not np.isnan(tp_time)
    sl_happened = sl_hit and not np.isnan(sl_time)

    if tp_happened and sl_happened:
        # Both hit — which was first?
        if tp_time <= sl_time:
            return (credit - tp_price) / credit
        else:
            return (credit - sl_price) / credit
    elif tp_happened:
        return (credit - tp_price) / credit
    elif sl_happened:
        return (credit - sl_price) / credit
    elif ic_3pm is not None:
        return (credit - ic_3pm) / credit
    else:
        return np.nan


def _compute_long_labels(
    debit: float,
    ic_values: list[float],
    minutes: list[int],
    ic_3pm: float | None,
    ic_4pm: float | None,
    fill: str,
) -> dict:
    """Compute all 18 long IC labels for one fill scenario (with SL)."""
    prefix = f"long_{fill}"
    labels = {}

    if np.isnan(debit) or debit <= 0:
        for suffix in _all_long_suffixes():
            labels[f"{prefix}_{suffix}"] = np.nan
        return labels

    # Thresholds (long profits when IC value INCREASES)
    tp_levels = {10: debit * 1.10, 25: debit * 1.25, 50: debit * 1.50}
    sl_level = debit * LONG_SL_MULT  # IC drops to half debit

    # Walk forward
    tp_hit = {10: False, 25: False, 50: False}
    tp_hit_time = {10: np.nan, 25: np.nan, 50: np.nan}
    tp_hit_price = {10: np.nan, 25: np.nan, 50: np.nan}
    sl_hit = False
    sl_hit_time = np.nan
    sl_hit_price = np.nan
    first_profitable_time = np.nan
    max_loss_pnl = 0.0
    max_loss_time = 0

    for ic_val, elapsed in zip(ic_values, minutes):
        if ic_val is None or np.isnan(ic_val):
            continue

        pnl_pct = (ic_val - debit) / debit

        if pnl_pct < max_loss_pnl:
            max_loss_pnl = pnl_pct
            max_loss_time = elapsed

        if np.isnan(first_profitable_time) and ic_val > debit:
            first_profitable_time = elapsed

        if not sl_hit and ic_val <= sl_level:
            sl_hit = True
            sl_hit_time = elapsed
            sl_hit_price = ic_val

        for tp_pct, tp_lvl in tp_levels.items():
            if not tp_hit[tp_pct] and not sl_hit and ic_val >= tp_lvl:
                tp_hit[tp_pct] = True
                tp_hit_time[tp_pct] = elapsed
                tp_hit_price[tp_pct] = ic_val

    # --- Binary TP labels ---
    labels[f"{prefix}_tp10"] = 1.0 if tp_hit[10] else 0.0
    labels[f"{prefix}_tp25"] = 1.0 if tp_hit[25] else 0.0
    labels[f"{prefix}_tp50"] = 1.0 if tp_hit[50] else 0.0

    # --- Binary status labels ---
    any_exit = sl_hit or any(tp_hit.values())
    if any_exit or ic_3pm is None:
        labels[f"{prefix}_3pm_profitable_if_open"] = np.nan
    else:
        labels[f"{prefix}_3pm_profitable_if_open"] = 1.0 if ic_3pm > debit else 0.0

    if ic_3pm is not None:
        labels[f"{prefix}_3pm_profitable_raw"] = 1.0 if ic_3pm > debit else 0.0
    else:
        labels[f"{prefix}_3pm_profitable_raw"] = np.nan

    if ic_4pm is not None:
        labels[f"{prefix}_4pm_profitable_raw"] = 1.0 if ic_4pm > debit else 0.0
    else:
        labels[f"{prefix}_4pm_profitable_raw"] = np.nan

    # --- Realized P&L labels ---
    def _long_pnl(exit_price):
        return (exit_price - debit) / debit

    labels[f"{prefix}_pnl_tp10_sl_3pm"] = _realized_pnl_long(
        debit, tp_hit[10], tp_hit_price[10], tp_hit_time[10],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    labels[f"{prefix}_pnl_tp25_sl_3pm"] = _realized_pnl_long(
        debit, tp_hit[25], tp_hit_price[25], tp_hit_time[25],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    labels[f"{prefix}_pnl_tp50_sl_3pm"] = _realized_pnl_long(
        debit, tp_hit[50], tp_hit_price[50], tp_hit_time[50],
        sl_hit, sl_hit_price, sl_hit_time, ic_3pm,
    )
    if sl_hit:
        labels[f"{prefix}_pnl_sl_3pm"] = _long_pnl(sl_hit_price)
    elif ic_3pm is not None:
        labels[f"{prefix}_pnl_sl_3pm"] = _long_pnl(ic_3pm)
    else:
        labels[f"{prefix}_pnl_sl_3pm"] = np.nan

    labels[f"{prefix}_pnl_3pm_raw"] = _long_pnl(ic_3pm) if ic_3pm is not None else np.nan
    labels[f"{prefix}_pnl_4pm_raw"] = _long_pnl(ic_4pm) if ic_4pm is not None else np.nan

    # --- Time-to-event labels ---
    labels[f"{prefix}_time_to_tp10"] = tp_hit_time[10] if tp_hit[10] else np.nan
    labels[f"{prefix}_time_to_tp25"] = tp_hit_time[25] if tp_hit[25] else np.nan
    labels[f"{prefix}_time_to_tp50"] = tp_hit_time[50] if tp_hit[50] else np.nan
    labels[f"{prefix}_time_to_sl"] = sl_hit_time if sl_hit else np.nan
    labels[f"{prefix}_time_to_profitable"] = first_profitable_time
    labels[f"{prefix}_time_to_max_loss"] = float(max_loss_time)

    return labels


def _realized_pnl_long(
    debit, tp_hit, tp_price, tp_time,
    sl_hit, sl_price, sl_time,
    ic_3pm,
) -> float:
    """Realized P&L for long IC. First event wins."""
    tp_happened = tp_hit and not np.isnan(tp_time)
    sl_happened = sl_hit and not np.isnan(sl_time)

    if tp_happened and sl_happened:
        if tp_time <= sl_time:
            return (tp_price - debit) / debit
        else:
            return (sl_price - debit) / debit
    elif tp_happened:
        return (tp_price - debit) / debit
    elif sl_happened:
        return (sl_price - debit) / debit
    elif ic_3pm is not None:
        return (ic_3pm - debit) / debit
    else:
        return np.nan


def _compute_long_nosl_labels(
    debit: float,
    ic_values: list[float],
    minutes: list[int],
    ic_4pm: float | None,
    fill: str,
) -> dict:
    """
    Compute long IC labels WITHOUT stop loss (12 per fill scenario).

    Rationale: max loss on a long IC is the debit paid, so SL is unnecessary.
    TPs are unconditional — "did the IC ever reach X?" regardless of prior drawdown.
    Fallback is always 4pm (expiration).
    """
    prefix = f"long_nosl_{fill}"
    labels = {}

    if np.isnan(debit) or debit <= 0:
        for suffix in _all_long_nosl_suffixes():
            labels[f"{prefix}_{suffix}"] = np.nan
        return labels

    tp_levels = {10: debit * 1.10, 25: debit * 1.25, 50: debit * 1.50}

    # Walk forward — no SL, TPs are unconditional
    tp_hit = {10: False, 25: False, 50: False}
    tp_hit_time = {10: np.nan, 25: np.nan, 50: np.nan}
    tp_hit_price = {10: np.nan, 25: np.nan, 50: np.nan}
    first_profitable_time = np.nan

    for ic_val, elapsed in zip(ic_values, minutes):
        if ic_val is None or np.isnan(ic_val):
            continue

        if np.isnan(first_profitable_time) and ic_val > debit:
            first_profitable_time = elapsed

        # Check TPs unconditionally (no SL gate)
        for tp_pct, tp_lvl in tp_levels.items():
            if not tp_hit[tp_pct] and ic_val >= tp_lvl:
                tp_hit[tp_pct] = True
                tp_hit_time[tp_pct] = elapsed
                tp_hit_price[tp_pct] = ic_val

    def _long_pnl(exit_price):
        return (exit_price - debit) / debit

    # --- Binary TP labels (unconditional) ---
    labels[f"{prefix}_tp10"] = 1.0 if tp_hit[10] else 0.0
    labels[f"{prefix}_tp25"] = 1.0 if tp_hit[25] else 0.0
    labels[f"{prefix}_tp50"] = 1.0 if tp_hit[50] else 0.0

    # --- Binary 4pm profitable ---
    if ic_4pm is not None:
        labels[f"{prefix}_4pm_profitable_raw"] = 1.0 if ic_4pm > debit else 0.0
    else:
        labels[f"{prefix}_4pm_profitable_raw"] = np.nan

    # --- Realized P&L: exit at TP if hit, else hold to 4pm ---
    for tp_pct in [10, 25, 50]:
        if tp_hit[tp_pct]:
            labels[f"{prefix}_pnl_tp{tp_pct}_4pm"] = _long_pnl(tp_hit_price[tp_pct])
        elif ic_4pm is not None:
            labels[f"{prefix}_pnl_tp{tp_pct}_4pm"] = _long_pnl(ic_4pm)
        else:
            labels[f"{prefix}_pnl_tp{tp_pct}_4pm"] = np.nan

    # 4pm raw (pure hold to expiration, no exits)
    labels[f"{prefix}_pnl_4pm_raw"] = _long_pnl(ic_4pm) if ic_4pm is not None else np.nan

    # --- Time-to-event labels ---
    labels[f"{prefix}_time_to_tp10"] = tp_hit_time[10] if tp_hit[10] else np.nan
    labels[f"{prefix}_time_to_tp25"] = tp_hit_time[25] if tp_hit[25] else np.nan
    labels[f"{prefix}_time_to_tp50"] = tp_hit_time[50] if tp_hit[50] else np.nan
    labels[f"{prefix}_time_to_profitable"] = first_profitable_time

    return labels


def _all_short_suffixes():
    return [
        "tp10", "tp25", "tp50",
        "3pm_profitable_if_open", "3pm_profitable_raw", "4pm_profitable_raw",
        "pnl_tp10_sl_3pm", "pnl_tp25_sl_3pm", "pnl_tp50_sl_3pm",
        "pnl_sl_3pm", "pnl_3pm_raw", "pnl_4pm_raw",
        "time_to_tp10", "time_to_tp25", "time_to_tp50",
        "time_to_sl", "time_to_profitable", "time_to_max_loss",
    ]


def _all_long_suffixes():
    return _all_short_suffixes()  # same structure


def _all_long_nosl_suffixes():
    return [
        "tp10", "tp25", "tp50",
        "4pm_profitable_raw",
        "pnl_tp10_4pm", "pnl_tp25_4pm", "pnl_tp50_4pm", "pnl_4pm_raw",
        "time_to_tp10", "time_to_tp25", "time_to_tp50",
        "time_to_profitable",
    ]


# ---------------------------------------------------------------------------
# Data Loading — 1-min resolution
# ---------------------------------------------------------------------------

def load_greeks_for_date_1min(d: date) -> pd.DataFrame:
    """
    Load all 1-min greeks for a single date.
    Returns DataFrame with columns: timestamp, strike, right, bid, ask, mid, delta, ...
    """
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

    # Normalize right: 'call'/'put' -> 'C'/'P'
    df["right"] = df["right"].str.upper().str[0]

    # Compute mid price
    df["mid"] = df.apply(
        lambda r: (r["bid"] + r["ask"]) / 2.0 if r["bid"] > 0 and r["ask"] > 0
        else max(r["bid"], r["ask"]), axis=1
    )

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute_key"] = df["timestamp"].dt.strftime("%H:%M")

    return df


def precompute_leg_mid_by_minute(
    day_greeks: pd.DataFrame,
    strikes: dict,
) -> dict[str, dict[str, float]]:
    """
    Pre-index mid prices for 4 IC legs at every minute.
    Returns dict[minute_key -> {"sp": mid, "sc": mid, "lp": mid, "lc": mid}].
    Much faster than per-minute DataFrame filtering.
    """
    legs = [
        ("sp", strikes["sp"], "P"),
        ("sc", strikes["sc"], "C"),
        ("lp", strikes["lp"], "P"),
        ("lc", strikes["lc"], "C"),
    ]

    # Build a lookup: (strike, right, minute_key) -> mid
    # Filter to only the 4 strikes we care about
    strike_set = {strikes["sp"], strikes["sc"], strikes["lp"], strikes["lc"]}
    subset = day_greeks[day_greeks["strike"].isin(strike_set)]

    # Vectorized: create tuple index and map to mid
    lookup = dict(zip(
        zip(subset["strike"], subset["right"], subset["minute_key"]),
        subset["mid"],
    ))

    # Build per-minute IC leg prices
    all_minutes = sorted(day_greeks["minute_key"].unique())
    result = {}
    for mk in all_minutes:
        leg_mids = {}
        complete = True
        for leg_name, strike, right in legs:
            val = lookup.get((strike, right, mk))
            if val is None:
                complete = False
                break
            leg_mids[leg_name] = val
        result[mk] = leg_mids if complete else None

    return result


def build_minute_ic_timeline(
    leg_prices_by_minute: dict[str, dict | None],
    entry_minute: str,
) -> tuple[list[float], list[int], float | None, float | None]:
    """
    Build timeline of IC mid-values from entry_minute to 16:00 (4pm).
    Returns (ic_values, minutes_elapsed, ic_3pm, ic_4pm).
    """
    all_minutes = sorted(leg_prices_by_minute.keys())
    timeline_minutes = [m for m in all_minutes if entry_minute <= m <= "16:00"]

    if not timeline_minutes:
        return [], [], None, None

    entry_h, entry_m = map(int, entry_minute.split(":"))
    entry_total_min = entry_h * 60 + entry_m

    ic_values = []
    minutes_elapsed = []

    for mk in timeline_minutes:
        legs = leg_prices_by_minute[mk]
        if legs is None:
            ic_values.append(np.nan)
        else:
            ic_val = (legs["sp"] + legs["sc"]) - (legs["lp"] + legs["lc"])
            ic_values.append(ic_val)

        h, m = map(int, mk.split(":"))
        elapsed = (h * 60 + m) - entry_total_min
        minutes_elapsed.append(elapsed)

    def _find_ic_at(target_key):
        if target_key in timeline_minutes:
            idx = timeline_minutes.index(target_key)
            if not np.isnan(ic_values[idx]):
                return ic_values[idx]
        return None

    # IC at 3pm — fallback to last valid value before 3pm
    ic_3pm = _find_ic_at("15:00")
    if ic_3pm is None:
        pre_3pm = [v for v, mk in zip(ic_values, timeline_minutes) if mk <= "15:00" and not np.isnan(v)]
        if pre_3pm:
            ic_3pm = pre_3pm[-1]

    # IC at 4pm — fallback to last valid value
    ic_4pm = _find_ic_at("16:00")
    if ic_4pm is None and ic_values:
        for v in reversed(ic_values):
            if not np.isnan(v):
                ic_4pm = v
                break

    return ic_values, minutes_elapsed, ic_3pm, ic_4pm


# ---------------------------------------------------------------------------
# Process one day
# ---------------------------------------------------------------------------

def generate_entry_slots(entry_start="09:35", entry_end="14:00", interval_min=1):
    """Generate entry time slots."""
    start_h, start_m = map(int, entry_start.split(":"))
    end_h, end_m = map(int, entry_end.split(":"))
    slots = []
    cur_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    while cur_min <= end_min:
        h, m = divmod(cur_min, 60)
        slots.append(time(h, m))
        cur_min += interval_min
    return slots


def process_day(d: date, slots: list[time], day_greeks: pd.DataFrame) -> list[dict]:
    """Process all entry slots for one day."""
    rows = []

    # Cache: strikes -> precomputed leg prices (avoids recomputing for same strikes)
    leg_cache = {}

    for slot in slots:
        slot_key = slot.strftime("%H:%M")

        # Get entry chain (all rows at entry minute)
        entry_chain = day_greeks[day_greeks["minute_key"] == slot_key]
        if entry_chain.empty:
            continue

        # Find IC strikes from entry chain
        strikes = compute_ic_strikes(entry_chain)
        if strikes is None:
            continue

        # Compute entry prices (3 fill scenarios)
        entry_prices = compute_entry_prices(entry_chain, strikes)
        if entry_prices is None:
            continue

        # Pre-compute leg prices for these strikes (cached across slots)
        cache_key = (strikes["sp"], strikes["sc"], strikes["lp"], strikes["lc"])
        if cache_key not in leg_cache:
            leg_cache[cache_key] = precompute_leg_mid_by_minute(day_greeks, strikes)

        # Build 1-min IC value timeline from entry to 4pm
        ic_values, minutes_elapsed, ic_3pm, ic_4pm = build_minute_ic_timeline(
            leg_cache[cache_key], slot_key,
        )

        if not ic_values:
            continue

        # Compute all labels
        try:
            labels = compute_labels_for_entry(
                entry_prices, ic_values, minutes_elapsed, ic_3pm, ic_4pm,
            )
        except Exception as e:
            logger.warning(f"  {d} {slot_key}: label error: {e}")
            continue

        # Add row ID + strike info
        row = {
            "date": d.isoformat(),
            "entry_time": slot_key,
            "sp_strike": strikes["sp"],
            "sc_strike": strikes["sc"],
            "lp_strike": strikes["lp"],
            "lc_strike": strikes["lc"],
        }
        row.update(labels)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("completed_dates", []))
    return set()


def save_progress(completed: set):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed_dates": sorted(completed)}, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute v8 IC labels with 1-min walk-forward")
    parser.add_argument("--start-date", default="2023-02-27")
    parser.add_argument("--end-date", default="2026-03-11")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/v8_labels_v3.log"),
        ],
    )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    # Get available dates from greeks parquet
    logger.info("Scanning available dates...")
    dates_col = pd.read_parquet(GREEKS_PATH, columns=["date"])
    dates_col["date"] = pd.to_datetime(dates_col["date"]).dt.date
    all_dates = sorted(dates_col["date"].unique())
    available_dates = [d for d in all_dates if start <= d <= end]
    logger.info(f"Date range: {start} to {end}, {len(available_dates)} days available")

    slots = generate_entry_slots()
    logger.info(f"Entry slots: {len(slots)} ({slots[0].strftime('%H:%M')} to {slots[-1].strftime('%H:%M')})")

    # Resume
    completed = load_progress() if args.resume else set()
    days_to_process = [d for d in available_dates if d.isoformat() not in completed]
    logger.info(f"Days to process: {len(days_to_process)} (already done: {len(completed)})")

    # Load existing if resuming
    output_path = OUTPUT_DIR / "labels_v3.parquet"
    all_rows = []
    if args.resume and output_path.exists():
        all_rows = pd.read_parquet(output_path).to_dict("records")
        logger.info(f"Loaded {len(all_rows)} existing rows")

    t0 = _time.time()

    for i, d in enumerate(days_to_process):
        day_t0 = _time.time()

        # Load 1-min greeks for this day
        day_greeks = load_greeks_for_date_1min(d)
        if day_greeks.empty:
            completed.add(d.isoformat())
            continue

        # Process all entry slots
        day_rows = process_day(d, slots, day_greeks)
        all_rows.extend(day_rows)
        completed.add(d.isoformat())

        elapsed = _time.time() - day_t0

        if (i + 1) % 10 == 0 or i == 0:
            total_elapsed = _time.time() - t0
            rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
            remaining = (len(days_to_process) - i - 1) / rate / 3600 if rate > 0 else 0
            logger.info(
                f"  [{i+1}/{len(days_to_process)}] {d}: {len(day_rows)} rows, "
                f"{elapsed:.1f}s, ETA: {remaining:.1f}h"
            )

        # Checkpoint every 25 days
        if (i + 1) % 25 == 0:
            save_progress(completed)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_rows).to_parquet(output_path, index=False)
            logger.info(f"  Checkpoint: {len(all_rows)} rows saved")

    # Final save
    save_progress(completed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)

    total_time = _time.time() - t0
    logger.info(f"Done: {len(df)} rows, {len(df.columns)} columns in {total_time/3600:.1f}h")

    # Quick stats
    if not df.empty:
        for fill in ["mid", "mid10", "worst"]:
            tp10_col = f"short_{fill}_tp10"
            if tp10_col in df.columns:
                tp10_rate = df[tp10_col].mean()
                tp25_rate = df[f"short_{fill}_tp25"].mean()
                tp50_rate = df[f"short_{fill}_tp50"].mean()
                avg_3pm = df[f"short_{fill}_pnl_3pm_raw"].mean()
                avg_4pm = df[f"short_{fill}_pnl_4pm_raw"].mean()
                logger.info(
                    f"  {fill}: Short TP10={tp10_rate:.1%} TP25={tp25_rate:.1%} "
                    f"TP50={tp50_rate:.1%} avg_pnl_3pm={avg_3pm:.3f} avg_pnl_4pm={avg_4pm:.3f}"
                )
            long_tp10_col = f"long_{fill}_tp10"
            if long_tp10_col in df.columns:
                ltp10 = df[long_tp10_col].mean()
                ltp25 = df[f"long_{fill}_tp25"].mean()
                lavg_3pm = df[f"long_{fill}_pnl_3pm_raw"].mean()
                lavg_4pm = df[f"long_{fill}_pnl_4pm_raw"].mean()
                logger.info(
                    f"  {fill}: Long (SL) TP10={ltp10:.1%} TP25={ltp25:.1%} "
                    f"avg_pnl_3pm={lavg_3pm:.3f} avg_pnl_4pm={lavg_4pm:.3f}"
                )
            nosl_tp10_col = f"long_nosl_{fill}_tp10"
            if nosl_tp10_col in df.columns:
                ntp10 = df[nosl_tp10_col].mean()
                ntp25 = df[f"long_nosl_{fill}_tp25"].mean()
                ntp50 = df[f"long_nosl_{fill}_tp50"].mean()
                navg = df[f"long_nosl_{fill}_pnl_4pm_raw"].mean()
                ntp25_pnl = df[f"long_nosl_{fill}_pnl_tp25_4pm"].mean()
                logger.info(
                    f"  {fill}: Long (noSL) TP10={ntp10:.1%} TP25={ntp25:.1%} "
                    f"TP50={ntp50:.1%} avg_pnl_4pm={navg:.3f} avg_pnl_tp25_or_4pm={ntp25_pnl:.3f}"
                )


if __name__ == "__main__":
    main()
