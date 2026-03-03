#!/usr/bin/env python3
"""
Jade Lizard / Big Lizard 0DTE-to-14DTE Backtest

Tests jade lizard strategy performance across:
- 4 DTE targets: 0 (0DTE), 3, 7, 14
- 4 IV rank thresholds: 30, 40, 50, 60
- ~3 years of data: March 2023 to February 2026

Jade Lizard = Short OTM put + Short OTM call + Long further OTM call (3 legs).
Safety condition: total_credit >= call_spread_width → eliminates upside risk.

Phase 1 (this script):
  Fixed deltas (12δ put, 10δ call), fixed 20pt call spread width.
  Sweeps DTE × IV rank threshold to find optimal regime.

Metrics per combination:
  Win rate, expectancy, Sharpe, max DD, profit factor, safety rate,
  VIX regime breakdown (< 15, 15-25, > 25), exit reason distribution.

Usage:
    python scripts/backtest_jade_lizard.py                        # Full run
    python scripts/backtest_jade_lizard.py --max-days 50          # Quick test
    python scripts/backtest_jade_lizard.py --dtes 0 3             # Subset DTEs
    python scripts/backtest_jade_lizard.py --iv-mins 40 50        # Subset IV thresholds
    python scripts/backtest_jade_lizard.py --workers 2            # Fewer workers
"""

import argparse
import json
import os
import subprocess
import sys
import time as time_module
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, time as dtime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from loguru import logger

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_training_dataset import (
    download_daily_data,
    fetch_leg_timeseries,
    fetch_option_chain,
    get_trading_days,
    theta_available,
    theta_request,
    CACHE_DIR,
    TRAINING_DATA_DIR,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
JL_CACHE_DIR = CACHE_DIR / "backtest_jade_lizard"
OUTPUT_DIR = PROJECT_ROOT / "output" / "jade_lizard"

# ── ThetaData recovery ───────────────────────────────────────────────────────
THETA_JAR = os.path.expanduser("~/Downloads/ThetaTerminalv3.jar")
THETA_CREDS = "/tmp/theta_creds.txt"
THETA_HEALTH_URL = "http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPXW"

# ── Progress tracking ────────────────────────────────────────────────────────
_lock = Lock()
_progress = {"done": 0, "errors": 0, "cached": 0, "total": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JLConfig:
    """Phase 1 configuration for a single backtest run."""
    dte: int = 0
    iv_min: float = 30.0
    put_delta: float = 0.12
    call_short_delta: float = 0.10
    spread_width: float = 20.0
    entry_time: dtime = dtime(12, 0)
    tp_pct: float = 0.50
    delta_stop: float = 0.40
    time_stop_dte: int = 1
    sma_filter: bool = False

    @property
    def label(self) -> str:
        return f"dte{self.dte}_iv{int(self.iv_min)}"


PHASE1_DTES = [0, 3, 7, 14]
PHASE1_IV_MINS = [30, 40, 50, 60]


# ═══════════════════════════════════════════════════════════════════════════════
# THETADATA RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _theta_is_alive() -> bool:
    try:
        r = urllib.request.urlopen(THETA_HEALTH_URL, timeout=5)
        return r.status == 200
    except Exception:
        return False


def restart_thetadata() -> bool:
    """Kill and restart ThetaData Terminal. Returns True if it comes back up."""
    logger.warning("  Restarting ThetaData Terminal...")

    for pattern in ["ThetaTerminal", "202602131.jar"]:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern],
                           capture_output=True, timeout=5)
        except Exception:
            pass
    time_module.sleep(3)

    if not os.path.exists(THETA_JAR):
        logger.error(f"  ThetaData jar not found: {THETA_JAR}")
        return False

    if not os.path.exists(THETA_CREDS):
        try:
            from config.settings import settings
            with open(THETA_CREDS, "w") as f:
                f.write(f"{settings.THETADATA_USERNAME}\n{settings.THETADATA_PASSWORD}\n")
        except Exception as e:
            logger.error(f"  Failed to write creds: {e}")
            return False

    try:
        subprocess.Popen(
            ["java", "-jar", THETA_JAR, "--creds-file", THETA_CREDS],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(THETA_JAR),
        )
    except Exception as e:
        logger.error(f"  Failed to launch ThetaData: {e}")
        return False

    for i in range(18):
        time_module.sleep(5)
        if _theta_is_alive():
            logger.info(f"  ThetaData back up after {(i+1)*5}s")
            return True
        logger.info(f"  Waiting for ThetaData... ({(i+1)*5}s/90s)")

    logger.error("  ThetaData failed to restart within 90s")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# IV RANK & FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iv_rank(vix_closes: np.ndarray, idx: int, lookback: int = 252) -> float:
    """Standard IV rank: (current - 252d_min) / (252d_max - 252d_min) × 100.

    Uses VIX as proxy for SPX IV environment.
    Args:
        vix_closes: Array of VIX close values (full history).
        idx: Index of the current day in vix_closes.
        lookback: Lookback window (default 252 trading days = 1 year).
    Returns:
        IV rank as percentage (0-100), or -1 if insufficient data.
    """
    if idx < lookback:
        return -1.0

    window = vix_closes[idx - lookback:idx]
    current = vix_closes[idx]
    lo, hi = window.min(), window.max()

    if hi == lo:
        return 50.0

    return float((current - lo) / (hi - lo) * 100)


def check_sma_filter(spx_closes: np.ndarray, idx: int, period: int = 20) -> bool:
    """True if SPX close > 20-day SMA."""
    if idx < period:
        return True  # skip filter if insufficient data
    sma = spx_closes[idx - period:idx].mean()
    return float(spx_closes[idx]) > sma


# ═══════════════════════════════════════════════════════════════════════════════
# STRIKE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_jade_lizard_strikes(puts: list, calls: list, config: JLConfig) -> dict | None:
    """Find jade lizard strikes and compute entry credit.

    Jade Lizard = Short OTM put + Short OTM call + Long further OTM call.

    Args:
        puts: Put option chain (list of dicts with strike, delta, bid, ask).
        calls: Call option chain.
        config: JLConfig with put_delta, call_short_delta, spread_width.

    Returns:
        Dict with strikes, credit, safety info — or None if can't construct.
    """
    if not puts or not calls:
        return None

    # Get underlying price
    underlying_prices = [
        p["underlying_price"] for p in puts
        if p.get("underlying_price") and p["underlying_price"] > 0
    ]
    if not underlying_prices:
        underlying_prices = [
            c["underlying_price"] for c in calls
            if c.get("underlying_price") and c["underlying_price"] > 0
        ]
    if not underlying_prices:
        return None
    underlying = underlying_prices[0]

    # ── Short put: closest to target |delta| ──────────────────────────────
    valid_puts = [
        p for p in puts
        if (p.get("delta") or 0) < 0
        and (p.get("bid") or 0) > 0
        and p.get("strike", 0) < underlying
    ]
    if not valid_puts:
        return None

    short_put = min(valid_puts, key=lambda x: abs(abs(x["delta"]) - config.put_delta))

    # ── Short call: closest to target delta ───────────────────────────────
    valid_calls = [
        c for c in calls
        if (c.get("delta") or 0) > 0
        and (c.get("bid") or 0) > 0
        and c.get("strike", 0) > underlying
    ]
    if not valid_calls:
        return None

    short_call = min(valid_calls, key=lambda x: abs(x["delta"] - config.call_short_delta))

    # ── Long call: short_call_strike + spread_width ───────────────────────
    lc_target = short_call["strike"] + config.spread_width
    valid_longs = [
        c for c in calls
        if c.get("strike", 0) > short_call["strike"]
        and (c.get("ask") or 0) > 0
    ]
    if not valid_longs:
        return None

    long_call = min(valid_longs, key=lambda x: abs(x["strike"] - lc_target))

    # Reject if no strike within $10 of target
    if abs(long_call["strike"] - lc_target) > 10:
        return None

    actual_spread_width = long_call["strike"] - short_call["strike"]

    # ── Entry credit (conservative: sell@bid, buy@ask) ────────────────────
    put_credit = short_put.get("bid") or 0
    call_credit = (short_call.get("bid") or 0) - (long_call.get("ask") or 0)
    total_credit = put_credit + call_credit

    if total_credit <= 0:
        return None

    # ── Safety condition ──────────────────────────────────────────────────
    # If total_credit >= call_spread_width → no upside risk
    safety_met = total_credit >= actual_spread_width
    max_upside_risk = max(0.0, actual_spread_width - total_credit)
    max_downside_risk = short_put["strike"] - total_credit  # put is naked

    return {
        "short_put_strike": short_put["strike"],
        "short_call_strike": short_call["strike"],
        "long_call_strike": long_call["strike"],
        "call_spread_width": actual_spread_width,
        "entry_credit": total_credit,
        "put_credit": put_credit,
        "call_spread_credit": call_credit,
        "underlying_price": underlying,
        "safety_met": safety_met,
        "max_upside_risk": max_upside_risk,
        "max_downside_risk": max_downside_risk,
        "short_put_delta": short_put.get("delta", 0),
        "short_call_delta": short_call.get("delta", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SETTLEMENT P&L
# ═══════════════════════════════════════════════════════════════════════════════

def compute_settlement_pnl(
    spx_close: float,
    sp_strike: float,
    sc_strike: float,
    lc_strike: float,
    credit: float,
) -> float:
    """Compute jade lizard P&L at expiration from intrinsic values.

    Short put: max(0, sp_strike - spx_close)
    Short call: max(0, spx_close - sc_strike)
    Long call: max(0, spx_close - lc_strike)
    P&L = credit - short_put_intrinsic - short_call_intrinsic + long_call_intrinsic
    """
    sp_intrinsic = max(0.0, sp_strike - spx_close)
    sc_intrinsic = max(0.0, spx_close - sc_strike)
    lc_intrinsic = max(0.0, spx_close - lc_strike)
    return credit - sp_intrinsic - sc_intrinsic + lc_intrinsic


# ═══════════════════════════════════════════════════════════════════════════════
# 0DTE TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_0dte_trade(
    trade_date: date,
    jl: dict,
    config: JLConfig,
) -> dict | None:
    """Simulate a 0DTE jade lizard trade using cached 5-min leg timeseries.

    Entry at config.entry_time, monitor every 5 min until 3:00 PM.
    Exit on: TP hit, delta stop, or settlement at close.

    Returns trade result dict or None if data insufficient.
    """
    entry_time = config.entry_time
    exit_time = dtime(15, 0)
    credit = jl["entry_credit"]

    # 3 legs: short put, short call, long call
    legs = [
        (jl["short_put_strike"], "put", -1),
        (jl["short_call_strike"], "call", -1),
        (jl["long_call_strike"], "call", 1),
    ]

    leg_data = {}
    for strike, right, direction in legs:
        data = fetch_leg_timeseries(trade_date, strike, right, entry_time, exit_time)
        if not data:
            return None
        leg_data[(strike, right, direction)] = data

    min_len = min(len(d) for d in leg_data.values())
    if min_len < 2:
        return None

    # ── Build debit-to-close curve ────────────────────────────────────────
    debits = np.zeros(min_len)
    for i in range(min_len):
        for (strike, right, direction), data in leg_data.items():
            pt = data[i]
            if direction == -1:   # Short: buy back at ask
                debits[i] += pt.get("ask", 0) or 0
            else:                 # Long: sell at bid
                debits[i] -= pt.get("bid", 0) or 0

    pnls = credit - debits

    # ── Scan for exit conditions ──────────────────────────────────────────
    tp_target = credit * (1.0 - config.tp_pct)  # 50% TP → debit <= 50% of credit
    exit_reason = "expiration"
    exit_idx = min_len - 1
    exit_pnl = float(pnls[-1])

    for i in range(min_len):
        # TP check
        if debits[i] <= tp_target and debits[i] >= 0:
            exit_reason = "tp"
            exit_idx = i
            exit_pnl = float(pnls[i])
            break

    # If no TP, check delta stop via chain snapshots
    # (delta stop requires re-reading chain data — use debit proxy instead)
    # For 0DTE, if debit exceeds credit significantly → similar to delta stop
    if exit_reason == "expiration":
        # Delta stop proxy: if debit exceeds 2× credit (shorts going deep ITM)
        for i in range(min_len):
            if debits[i] >= credit * 2.0:
                exit_reason = "delta_stop"
                exit_idx = i
                exit_pnl = float(pnls[i])
                break

    max_debit = float(np.max(debits[:exit_idx + 1]))
    max_adverse_pct = ((max_debit / credit) - 1) * 100 if credit > 0 else 0.0
    pnl_pct = (exit_pnl / credit) * 100 if credit > 0 else 0.0

    return {
        "pnl": exit_pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "exit_idx": exit_idx,
        "max_adverse_pct": max_adverse_pct,
        "max_debit": max_debit,
        "n_snapshots": min_len,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-DTE TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_chain_for_expiry(
    check_date: date,
    expiration: date,
    check_time: dtime,
    right: str,
) -> list:
    """Fetch option chain for a specific expiration date on a given check date.

    Uses a separate cache directory for multi-DTE chains.
    """
    time_str = f"{check_time.hour:02d}{check_time.minute:02d}"
    cache_dir = CACHE_DIR / "chains_dte"
    cache_file = cache_dir / f"{check_date.isoformat()}_{expiration.isoformat()}_{time_str}_{right}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    end_min = check_time.minute + 1 if check_time.minute < 59 else 59
    end_time = dtime(check_time.hour, end_min)

    data = theta_request("/v3/option/history/greeks/first_order", {
        "symbol": "SPXW",
        "expiration": expiration.strftime("%Y%m%d"),
        "right": right,
        "date": check_date.strftime("%Y%m%d"),
        "interval": "1m",
        "start_time": check_time.strftime("%H:%M:%S"),
        "end_time": end_time.strftime("%H:%M:%S"),
        "strike": "*",
        "format": "json",
    })

    if not data:
        return []

    response = data.get("response", data if isinstance(data, list) else [])

    flattened = []
    for row in response:
        if not isinstance(row, dict):
            continue
        contract = row.get("contract", {})
        data_points = row.get("data", [])
        if not data_points:
            continue
        dp = data_points[0]
        flattened.append({
            "strike": contract.get("strike", 0),
            "right": contract.get("right", ""),
            "bid": dp.get("bid", 0),
            "ask": dp.get("ask", 0),
            "delta": dp.get("delta", 0),
            "implied_vol": dp.get("implied_vol", 0),
            "underlying_price": dp.get("underlying_price", 0),
        })

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(flattened, f)

    return flattened


def simulate_multi_dte_trade(
    entry_date: date,
    expiration: date,
    jl: dict,
    config: JLConfig,
    trading_days: list[date],
) -> dict | None:
    """Simulate a multi-DTE jade lizard trade.

    Checks at entry_time each subsequent trading day.
    Exits on: TP hit, delta stop, time stop (DTE <= time_stop_dte), or expiration.
    """
    credit = jl["entry_credit"]
    sp_strike = jl["short_put_strike"]
    sc_strike = jl["short_call_strike"]
    lc_strike = jl["long_call_strike"]

    # Get check dates: entry_date+1 through expiration
    entry_idx = None
    exp_idx = None
    for i, d in enumerate(trading_days):
        if d == entry_date:
            entry_idx = i
        if d == expiration:
            exp_idx = i

    if entry_idx is None or exp_idx is None or exp_idx <= entry_idx:
        return None

    check_dates = trading_days[entry_idx + 1:exp_idx + 1]
    if not check_dates:
        return None

    exit_reason = "expiration"
    exit_pnl = None
    max_adverse_pct = 0.0
    max_debit = 0.0

    for check_date in check_dates:
        remaining_dte = (expiration - check_date).days

        # Fetch chains for the expiration on this check date
        puts = _fetch_chain_for_expiry(check_date, expiration, config.entry_time, "put")
        calls = _fetch_chain_for_expiry(check_date, expiration, config.entry_time, "call")

        if not puts or not calls:
            continue

        # Find our strikes in the chain and compute debit-to-close
        sp_opt = next((p for p in puts if p["strike"] == sp_strike), None)
        sc_opt = next((c for c in calls if c["strike"] == sc_strike), None)
        lc_opt = next((c for c in calls if c["strike"] == lc_strike), None)

        if not sp_opt or not sc_opt or not lc_opt:
            continue

        # Debit to close: buy back shorts at ask, sell long at bid
        debit = (sp_opt.get("ask") or 0) + (sc_opt.get("ask") or 0) - (lc_opt.get("bid") or 0)
        pnl = credit - debit
        pnl_pct = (pnl / credit) * 100 if credit > 0 else 0.0

        if debit > max_debit:
            max_debit = debit
            max_adverse_pct = ((debit / credit) - 1) * 100 if credit > 0 else 0.0

        # TP check
        tp_target = credit * (1.0 - config.tp_pct)
        if debit <= tp_target and debit >= 0:
            exit_reason = "tp"
            exit_pnl = pnl
            break

        # Delta stop: check if shorts have gone deep ITM
        sp_delta = abs(sp_opt.get("delta") or 0)
        sc_delta = sc_opt.get("delta") or 0
        if sp_delta >= config.delta_stop or sc_delta >= config.delta_stop:
            exit_reason = "delta_stop"
            exit_pnl = pnl
            break

        # Time stop
        if remaining_dte <= config.time_stop_dte:
            exit_reason = "time_stop"
            exit_pnl = pnl
            break

    # Settlement at expiration if not already exited
    if exit_pnl is None:
        # Get SPX close on expiration day from chain underlying price
        calls_exp = _fetch_chain_for_expiry(expiration, expiration, dtime(15, 0), "call")
        spx_close = None
        if calls_exp:
            prices = [c["underlying_price"] for c in calls_exp if c.get("underlying_price", 0) > 0]
            if prices:
                spx_close = prices[0]

        if spx_close is None:
            return None

        exit_pnl = compute_settlement_pnl(spx_close, sp_strike, sc_strike, lc_strike, credit)

    pnl_pct = (exit_pnl / credit) * 100 if credit > 0 else 0.0

    return {
        "pnl": exit_pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "max_adverse_pct": max_adverse_pct,
        "max_debit": max_debit,
        "n_check_days": len(check_dates),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DAY PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_json(row: dict) -> dict:
    """Convert numpy types for JSON serialization."""
    clean = {}
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            clean[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        else:
            clean[k] = v
    return clean


def process_day(
    trade_date: date,
    config: JLConfig,
    vix_closes: np.ndarray,
    vix_dates: np.ndarray,
    spx_closes: np.ndarray,
    spx_dates: np.ndarray,
    trading_days: list[date],
) -> dict | None:
    """Process one trading day for a given JL config.

    Returns trade result dict or None if filtered/skipped.
    """
    # ── Cache check ───────────────────────────────────────────────────────
    cache_file = JL_CACHE_DIR / f"{trade_date.isoformat()}_{config.label}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        # Return None for skipped trades (cached as {"skipped": true})
        return cached if not cached.get("skipped") else None

    # ── IV rank filter ────────────────────────────────────────────────────
    vix_idx = np.searchsorted(vix_dates, trade_date)
    if vix_idx >= len(vix_dates) or vix_dates[vix_idx] != trade_date:
        # Try previous day (VIX data uses prior day close)
        vix_idx -= 1
    if vix_idx < 0 or vix_idx >= len(vix_closes):
        return None

    iv_rank = compute_iv_rank(vix_closes, vix_idx)
    if iv_rank < 0:
        return None  # insufficient history

    vix_level = float(vix_closes[vix_idx])

    if iv_rank < config.iv_min:
        # Cache the skip for future runs
        JL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"skipped": True, "reason": "iv_rank", "iv_rank": iv_rank}, f)
        return None

    # ── SMA filter (optional) ─────────────────────────────────────────────
    if config.sma_filter:
        spx_idx = np.searchsorted(spx_dates, trade_date)
        if spx_idx > 0:
            spx_idx -= 1  # use prior day close
        if not check_sma_filter(spx_closes, spx_idx):
            JL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"skipped": True, "reason": "sma_filter"}, f)
            return None

    # ── Fetch option chain at entry time ──────────────────────────────────
    if config.dte == 0:
        # 0DTE: expiration = trade_date, use standard chain cache
        puts = fetch_option_chain(trade_date, config.entry_time, "put")
        calls = fetch_option_chain(trade_date, config.entry_time, "call")
    else:
        # Multi-DTE: find expiration N trading days ahead
        td_idx = None
        for i, d in enumerate(trading_days):
            if d == trade_date:
                td_idx = i
                break
        if td_idx is None:
            return None

        exp_idx = td_idx + config.dte
        if exp_idx >= len(trading_days):
            return None
        expiration = trading_days[exp_idx]

        puts = _fetch_chain_for_expiry(trade_date, expiration, config.entry_time, "put")
        calls = _fetch_chain_for_expiry(trade_date, expiration, config.entry_time, "call")

    if not puts or not calls:
        return None

    # ── Find jade lizard strikes ──────────────────────────────────────────
    jl = find_jade_lizard_strikes(puts, calls, config)
    if jl is None:
        return None

    # ── Simulate trade ────────────────────────────────────────────────────
    if config.dte == 0:
        result = simulate_0dte_trade(trade_date, jl, config)
    else:
        expiration = trading_days[td_idx + config.dte]
        result = simulate_multi_dte_trade(trade_date, expiration, jl, config, trading_days)

    if result is None:
        return None

    # ── Assemble trade record ─────────────────────────────────────────────
    vix_regime = "low" if vix_level < 15 else ("mid" if vix_level <= 25 else "high")

    record = {
        "trade_date": trade_date.isoformat(),
        "dte": config.dte,
        "iv_min": config.iv_min,
        "iv_rank": round(iv_rank, 1),
        "vix_level": round(vix_level, 2),
        "vix_regime": vix_regime,
        "underlying_price": jl["underlying_price"],
        "short_put_strike": jl["short_put_strike"],
        "short_call_strike": jl["short_call_strike"],
        "long_call_strike": jl["long_call_strike"],
        "call_spread_width": jl["call_spread_width"],
        "entry_credit": jl["entry_credit"],
        "put_credit": jl["put_credit"],
        "call_spread_credit": jl["call_spread_credit"],
        "safety_met": jl["safety_met"],
        "max_upside_risk": jl["max_upside_risk"],
        "short_put_delta": jl["short_put_delta"],
        "short_call_delta": jl["short_call_delta"],
        "pnl": result["pnl"],
        "pnl_pct": result["pnl_pct"],
        "exit_reason": result["exit_reason"],
        "max_adverse_pct": result.get("max_adverse_pct", 0),
    }

    # Cache result
    JL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(_sanitize_json(record), f)

    return record


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (dte, iv_min) and compute comprehensive summary statistics."""
    rows = []
    for (dte, iv_min), grp in df.groupby(["dte", "iv_min"]):
        n = len(grp)
        if n == 0:
            continue

        pnls = grp["pnl"].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        # Core metrics
        win_rate = len(wins) / n * 100
        expectancy = pnls.mean()
        std_pnl = pnls.std() if n > 1 else 0.0

        # Sharpe: annualize based on DTE
        if dte == 0:
            trades_per_year = 252
        else:
            trades_per_year = 252 / max(dte, 1)
        sharpe = (expectancy / std_pnl * np.sqrt(trades_per_year)) if std_pnl > 0 else 0.0

        # Max drawdown (cumulative P&L)
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # Profit factor
        sum_wins = wins.sum() if len(wins) > 0 else 0.0
        sum_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else float("inf")

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        # Safety rate
        safety_rate = grp["safety_met"].mean() * 100

        # Exit reason distribution
        exit_counts = grp["exit_reason"].value_counts(normalize=True) * 100
        tp_pct = exit_counts.get("tp", 0.0)
        delta_stop_pct = exit_counts.get("delta_stop", 0.0)
        time_stop_pct = exit_counts.get("time_stop", 0.0)
        expiration_pct = exit_counts.get("expiration", 0.0)

        # VIX regime breakdown
        regime_stats = {}
        for regime in ["low", "mid", "high"]:
            rg = grp[grp["vix_regime"] == regime]
            if len(rg) > 0:
                regime_stats[f"n_{regime}"] = len(rg)
                regime_stats[f"wr_{regime}"] = (rg["pnl"] > 0).mean() * 100
                regime_stats[f"exp_{regime}"] = rg["pnl"].mean()
            else:
                regime_stats[f"n_{regime}"] = 0
                regime_stats[f"wr_{regime}"] = 0.0
                regime_stats[f"exp_{regime}"] = 0.0

        row = {
            "dte": int(dte),
            "iv_min": int(iv_min),
            "n_trades": n,
            "win_rate": round(win_rate, 1),
            "expectancy": round(expectancy, 2),
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_credit": round(grp["entry_credit"].mean(), 2),
            "safety_rate": round(safety_rate, 1),
            "avg_upside_risk": round(grp["max_upside_risk"].mean(), 2),
            "tp_pct": round(tp_pct, 1),
            "delta_stop_pct": round(delta_stop_pct, 1),
            "time_stop_pct": round(time_stop_pct, 1),
            "expiration_pct": round(expiration_pct, 1),
            "total_pnl": round(pnls.sum(), 2),
            **{k: round(v, 1) if isinstance(v, float) else v for k, v in regime_stats.items()},
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def display_results(summary: pd.DataFrame, n_days: int, date_range: str):
    """Display backtest results as formatted console tables."""
    if summary.empty:
        logger.warning("No results to display.")
        return

    # ── Main results table (ranked by Sharpe) ─────────────────────────────
    print()
    print("═" * 140)
    print("  JADE LIZARD BACKTEST — Phase 1 Results")
    print(f"  {n_days} trading days | {date_range}")
    print(f"  12δ put, 10δ/20pt call spread | 50% TP, 2× delta stop")
    print("═" * 140)
    print()
    print(f"  {'OVERALL METRICS (ranked by Sharpe)':^130}")
    print()

    hdr1 = (
        f"{'DTE':>4s}  {'IVR':>5s}  {'Trades':>6s}  {'Win%':>6s}  {'Expect':>8s}  "
        f"{'Sharpe':>7s}  {'MaxDD':>8s}  {'PF':>6s}  "
        f"{'AvgWin':>8s}  {'AvgLoss':>8s}  {'Credit':>7s}  "
        f"{'Safety':>7s}  {'Total':>10s}"
    )
    hdr2 = (
        f"{'':>4s}  {'Min':>5s}  {'':>6s}  {'':>6s}  {'$/trade':>8s}  "
        f"{'':>7s}  {'$':>8s}  {'':>6s}  "
        f"{'$':>8s}  {'$':>8s}  {'$':>7s}  "
        f"{'Rate%':>7s}  {'P&L $':>10s}"
    )
    sep = (
        "─" * 4 + "  " + "─" * 5 + "  " + "─" * 6 + "  " + "─" * 6 + "  " + "─" * 8
        + "  " + "─" * 7 + "  " + "─" * 8 + "  " + "─" * 6
        + "  " + "─" * 8 + "  " + "─" * 8 + "  " + "─" * 7
        + "  " + "─" * 7 + "  " + "─" * 10
    )

    print(hdr1)
    print(hdr2)
    print(sep)

    ranked = summary.sort_values("sharpe", ascending=False)
    for _, row in ranked.iterrows():
        pf_str = f"{row['profit_factor']:5.2f}" if row['profit_factor'] < 100 else "  inf"
        print(
            f"{int(row['dte']):4d}  {int(row['iv_min']):5d}  {int(row['n_trades']):6d}  "
            f"{row['win_rate']:5.1f}%  {row['expectancy']:+7.2f}  "
            f"{row['sharpe']:+6.3f}  {row['max_dd']:8.2f}  {pf_str}  "
            f"{row['avg_win']:+7.2f}  {row['avg_loss']:+7.2f}  "
            f"${row['avg_credit']:5.2f}  "
            f"{row['safety_rate']:5.1f}%  "
            f"{row['total_pnl']:+9.2f}"
        )

    print(sep)
    print()

    # ── Exit reason distribution ──────────────────────────────────────────
    print(f"  {'EXIT REASON DISTRIBUTION':^100}")
    print()
    hdr_exit = (
        f"{'DTE':>4s}  {'IVR':>5s}  {'TP%':>7s}  {'DeltaStop%':>10s}  "
        f"{'TimeStop%':>10s}  {'Expiry%':>8s}"
    )
    sep_exit = "─" * 4 + "  " + "─" * 5 + "  " + "─" * 7 + "  " + "─" * 10 + "  " + "─" * 10 + "  " + "─" * 8
    print(hdr_exit)
    print(sep_exit)

    for _, row in ranked.iterrows():
        print(
            f"{int(row['dte']):4d}  {int(row['iv_min']):5d}  "
            f"{row['tp_pct']:5.1f}%  {row['delta_stop_pct']:8.1f}%  "
            f"{row['time_stop_pct']:8.1f}%  {row['expiration_pct']:6.1f}%"
        )
    print(sep_exit)
    print()

    # ── VIX regime breakdown ──────────────────────────────────────────────
    print(f"  {'VIX REGIME BREAKDOWN':^120}")
    print()
    hdr_vix = (
        f"{'DTE':>4s}  {'IVR':>5s}  "
        f"{'':^28s}  {'':^28s}  {'':^28s}"
    )
    hdr_vix_sub = (
        f"{'':>4s}  {'Min':>5s}  "
        f"{'VIX < 15':^28s}  {'VIX 15-25':^28s}  {'VIX > 25':^28s}"
    )
    hdr_vix2 = (
        f"{'':>4s}  {'':>5s}  "
        f"{'N':>5s}  {'Win%':>6s}  {'Exp$':>8s}  {'':>5s}  "
        f"{'N':>5s}  {'Win%':>6s}  {'Exp$':>8s}  {'':>5s}  "
        f"{'N':>5s}  {'Win%':>6s}  {'Exp$':>8s}"
    )
    sep_vix = "─" * 4 + "  " + "─" * 5 + ("  " + "─" * 28) * 3
    print(hdr_vix_sub)
    print(hdr_vix2)
    print(sep_vix)

    for _, row in ranked.iterrows():
        print(
            f"{int(row['dte']):4d}  {int(row['iv_min']):5d}  "
            f"{int(row.get('n_low', 0)):5d}  {row.get('wr_low', 0):5.1f}%  {row.get('exp_low', 0):+7.2f}  {'':>5s}  "
            f"{int(row.get('n_mid', 0)):5d}  {row.get('wr_mid', 0):5.1f}%  {row.get('exp_mid', 0):+7.2f}  {'':>5s}  "
            f"{int(row.get('n_high', 0)):5d}  {row.get('wr_high', 0):5.1f}%  {row.get('exp_high', 0):+7.2f}"
        )
    print(sep_vix)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Jade Lizard Strategy Backtest")
    parser.add_argument("--start", default="2023-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-27", help="End date")
    parser.add_argument(
        "--dtes", nargs="+", type=int, default=PHASE1_DTES,
        help=f"DTE values to test (default: {PHASE1_DTES})",
    )
    parser.add_argument(
        "--iv-mins", nargs="+", type=float, default=PHASE1_IV_MINS,
        help=f"IV rank minimums to test (default: {PHASE1_IV_MINS})",
    )
    parser.add_argument("--max-days", type=int, default=0, help="Limit days for quick testing")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSV output")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    dtes = args.dtes
    iv_mins = args.iv_mins

    # Build config matrix
    configs = []
    for dte in dtes:
        for iv_min in iv_mins:
            configs.append(JLConfig(dte=dte, iv_min=iv_min))

    logger.info("═" * 72)
    logger.info("Jade Lizard Strategy Backtest — Phase 1")
    logger.info("═" * 72)
    logger.info(f"  Date range:    {start_date} to {end_date}")
    logger.info(f"  DTEs:          {dtes}")
    logger.info(f"  IV minimums:   {iv_mins}")
    logger.info(f"  Combinations:  {len(configs)}")
    logger.info(f"  Put delta:     12δ  |  Call delta: 10δ  |  Spread: $20")
    logger.info(f"  TP: 50%  |  Delta stop: 40δ  |  Time stop: DTE≤1")
    logger.info(f"  Workers:       {args.workers}")

    # Check ThetaData (needed for multi-DTE and uncached 0DTE)
    has_theta = theta_available()
    if has_theta:
        logger.info("  ThetaData:     Connected")
    else:
        logger.warning("  ThetaData:     NOT connected (will use cached data only)")
        if any(d > 0 for d in dtes):
            logger.warning("  Multi-DTE configs require ThetaData — will skip uncached days")

    # Get trading days
    trading_days = get_trading_days(start_date, end_date)
    if args.max_days > 0:
        trading_days = trading_days[:args.max_days]
    logger.info(f"  Trading days:  {len(trading_days)}")

    if not trading_days:
        logger.error("No trading days in range!")
        sys.exit(1)

    date_range = f"{trading_days[0]} to {trading_days[-1]}"

    # ── Download daily data ───────────────────────────────────────────────
    logger.info("\nStep 1: Loading daily data (VIX for IV rank)...")
    daily_data = download_daily_data(
        start=(start_date - timedelta(days=400)).isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
    )

    vix_df = daily_data.get("VIX")
    spx_df = daily_data.get("SPX")
    if vix_df is None or vix_df.empty:
        logger.error("Failed to load VIX data!")
        sys.exit(1)
    if spx_df is None or spx_df.empty:
        logger.error("Failed to load SPX data!")
        sys.exit(1)

    vix_closes = vix_df["Close"].values.astype(float)
    vix_dates = np.array([d.date() if hasattr(d, 'date') else d for d in vix_df.index])
    spx_closes = spx_df["Close"].values.astype(float)
    spx_dates = np.array([d.date() if hasattr(d, 'date') else d for d in spx_df.index])

    # ── Count cached results ──────────────────────────────────────────────
    JL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total_tasks = len(trading_days) * len(configs)
    cached = sum(
        1 for td in trading_days
        for cfg in configs
        if (JL_CACHE_DIR / f"{td.isoformat()}_{cfg.label}.json").exists()
    )
    logger.info(f"\nStep 2: Processing {total_tasks} day×config combinations...")
    logger.info(f"  Cached:        {cached}")
    logger.info(f"  To process:    {total_tasks - cached}")

    # ── Process all day×config combinations ───────────────────────────────
    t_start = time_module.time()
    _progress["done"] = 0
    _progress["errors"] = 0
    _progress["cached"] = cached

    all_results = []

    # Load cached results upfront
    for td in trading_days:
        for cfg in configs:
            cache_file = JL_CACHE_DIR / f"{td.isoformat()}_{cfg.label}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)
                if not data.get("skipped"):
                    all_results.append(data)

    # Build uncached tasks
    uncached_tasks = [
        (td, cfg) for td in trading_days for cfg in configs
        if not (JL_CACHE_DIR / f"{td.isoformat()}_{cfg.label}.json").exists()
    ]

    if uncached_tasks:
        logger.info(f"  Processing {len(uncached_tasks)} uncached tasks...")

        # Separate 0DTE vs multi-DTE tasks
        tasks_0dte = [(td, cfg) for td, cfg in uncached_tasks if cfg.dte == 0]
        tasks_multi = [(td, cfg) for td, cfg in uncached_tasks if cfg.dte > 0]

        # Process 0DTE first (fully cacheable, no ThetaData needed for cached chains)
        if tasks_0dte:
            logger.info(f"  Processing {len(tasks_0dte)} 0DTE tasks...")

            def _process_0dte(args_tuple):
                td, cfg = args_tuple
                try:
                    result = process_day(
                        td, cfg, vix_closes, vix_dates,
                        spx_closes, spx_dates, trading_days,
                    )
                    with _lock:
                        _progress["done"] += 1
                        done = _progress["done"]
                        if done % 100 == 0 or done == 1:
                            elapsed = time_module.time() - t_start
                            rate = done / elapsed if elapsed > 0 else 0
                            left = len(uncached_tasks) - done
                            eta = left / rate if rate > 0 else 0
                            logger.info(
                                f"  [{done + _progress['cached']:5d}/{total_tasks}] "
                                f"({_progress['errors']} errors, ETA {eta/60:.1f}m)"
                            )
                    return result
                except Exception as e:
                    with _lock:
                        _progress["errors"] += 1
                    logger.debug(f"  {td} {cfg.label}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_0dte, task): task for task in tasks_0dte}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)

        # Process multi-DTE tasks (may need ThetaData)
        if tasks_multi:
            if not has_theta and not _theta_is_alive():
                logger.warning(f"  Skipping {len(tasks_multi)} multi-DTE tasks (no ThetaData)")
            else:
                logger.info(f"  Processing {len(tasks_multi)} multi-DTE tasks...")

                BATCH_SIZE = 200
                restart_count = 0

                for batch_start in range(0, len(tasks_multi), BATCH_SIZE):
                    batch = tasks_multi[batch_start:batch_start + BATCH_SIZE]

                    # Health check before each batch
                    if not _theta_is_alive():
                        logger.warning("  ThetaData down — restarting...")
                        if restart_thetadata():
                            restart_count += 1
                        else:
                            logger.error("  ThetaData restart failed — stopping multi-DTE")
                            break

                    batch_errors = 0
                    batch_trades = 0

                    for td, cfg in batch:
                        try:
                            result = process_day(
                                td, cfg, vix_closes, vix_dates,
                                spx_closes, spx_dates, trading_days,
                            )
                            with _lock:
                                _progress["done"] += 1
                            if result is not None:
                                all_results.append(result)
                                batch_trades += 1
                        except Exception as e:
                            with _lock:
                                _progress["errors"] += 1
                                batch_errors += 1
                            logger.debug(f"  {td} {cfg.label}: {e}")

                    done = _progress["done"] + _progress["cached"]
                    logger.info(
                        f"  [{done:5d}/{total_tasks}] multi-DTE batch "
                        f"{batch_start // BATCH_SIZE + 1} done "
                        f"({batch_trades} trades, {batch_errors} errors, "
                        f"{restart_count} restarts)"
                    )

                    # Only restart if actual errors (not IV-filtered skips)
                    if batch_errors > 5:
                        if not _theta_is_alive():
                            if restart_thetadata():
                                restart_count += 1
                            else:
                                logger.error("  ThetaData restart failed — stopping")
                                break

    elapsed = time_module.time() - t_start
    logger.info(
        f"\n  Done: {len(all_results)} trades, {_progress['errors']} errors, "
        f"{_progress['cached']} pre-cached ({elapsed:.1f}s)"
    )

    if not all_results:
        logger.error("No trade data collected!")
        sys.exit(1)

    # ── Combine & aggregate ───────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    n_days = df_all["trade_date"].nunique()
    logger.info(f"\nStep 3: Aggregating {len(df_all)} trades across {n_days} days...")

    summary = aggregate_metrics(df_all)

    # ── Display ───────────────────────────────────────────────────────────
    display_results(summary, n_days, date_range)

    # ── Save ──────────────────────────────────────────────────────────────
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        trades_file = OUTPUT_DIR / "phase1_trades.csv"
        df_all.to_csv(trades_file, index=False)
        logger.info(f"  Trade details saved to {trades_file}")

        summary_file = OUTPUT_DIR / "phase1_results.csv"
        summary.to_csv(summary_file, index=False)
        logger.info(f"  Summary saved to {summary_file}")

    logger.info("\nBacktest complete.")


if __name__ == "__main__":
    main()
