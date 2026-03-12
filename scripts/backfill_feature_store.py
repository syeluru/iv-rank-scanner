#!/usr/bin/env python3
"""
Backfill v8 feature store from ThetaData historical data.

Streams SPX/VIX 1-min bars and 0DTE option chain snapshots day-by-day,
computes all v8 features at each 5-min entry slot, and writes to parquet.
Raw data never touches disk (unless --save-snapshots is set).

Usage:
    python -m scripts.backfill_feature_store --start-date 2023-06-01 --end-date 2026-03-10
    python -m scripts.backfill_feature_store --resume  # pick up where you left off
"""

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import time as _time
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from ml.features.v8.greeks_aggregate import compute_greeks_aggregate_features
from ml.features.v8.vol_surface import compute_vol_surface_features
from ml.features.v8.realized_vol import compute_realized_vol_features
from ml.features.v8.positioning import compute_positioning_features
from ml.features.v8.order_flow import compute_order_flow_features
from ml.features.v8.microstructure import compute_microstructure_features
from ml.features.v8.time_event import compute_time_event_features
from ml.features.v8.cross_asset import compute_cross_asset_features
from ml.features.v8.market_internals import compute_market_internals_features
from ml.features.v8.feature_store import _compute_interaction_features, _resample_to_5min
from ml.features.technical_indicators import compute_all_technical_indicators

logger = logging.getLogger("backfill_feature_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THETA_HOST = "http://127.0.0.1:25503"
THETA_TIMEOUT = 120  # seconds per request
WING_WIDTH = 25  # $25 wings for IC
CLOSE_TIME = time(15, 0)  # 3:00 PM ET — exit time for label computation
IC_DELTAS = [0.10, 0.15, 0.20]  # delta targets for IC labels
THETA_SEMAPHORE = Semaphore(6)  # PRO = 8 simultaneous, leave headroom


# ============================================================================
# ThetaData helpers
# ============================================================================

def _theta_get(session: requests.Session, path: str, params: dict) -> Optional[dict]:
    """GET from ThetaData v3 with semaphore and error handling."""
    url = f"{THETA_HOST}{path}"
    params["format"] = "json"
    with THETA_SEMAPHORE:
        try:
            resp = session.get(url, params=params, timeout=THETA_TIMEOUT)
            if resp.status_code != 200:
                logger.debug(f"ThetaData {resp.status_code}: {path} {params}")
                return None
            return resp.json()
        except Exception as e:
            logger.debug(f"ThetaData error: {path} — {e}")
            return None


def _parse_response(data: Optional[dict]) -> list:
    """Extract the response list from ThetaData v3 nested format."""
    if data is None:
        return []
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    if isinstance(data, list):
        return data
    return []


def _format_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _format_time(t: time) -> str:
    return t.strftime("%H:%M:%S")


# ============================================================================
# ThetaData Terminal launcher (reused from zero_dte_bot.py)
# ============================================================================

def ensure_thetadata_running() -> bool:
    """Check if ThetaData Terminal is reachable. If not, start it and wait."""
    HEALTH_URL = f"{THETA_HOST}/v3/option/list/expirations?symbol=SPXW"
    THETA_JAR = os.path.expanduser("~/Downloads/ThetaTerminalv3.jar")
    CREDS_FILE = "/tmp/theta_creds.txt"
    STARTUP_TIMEOUT = 90

    def _is_alive():
        try:
            req = urllib.request.urlopen(HEALTH_URL, timeout=5)
            return req.status == 200
        except Exception:
            return False

    if _is_alive():
        logger.info("ThetaData Terminal: running")
        return True

    # Kill stale processes
    logger.warning("ThetaData Terminal not responding. Killing stale processes...")
    try:
        subprocess.run(["pkill", "-9", "-f", "ThetaTerminal"], capture_output=True, timeout=5)
        subprocess.run(["pkill", "-9", "-f", "202602131.jar"], capture_output=True, timeout=5)
        _time.sleep(2)
    except Exception:
        pass

    if not os.path.exists(THETA_JAR):
        logger.error(f"ThetaData jar not found at {THETA_JAR}")
        return False

    # Ensure creds file
    if not os.path.exists(CREDS_FILE):
        try:
            with open(CREDS_FILE, "w") as f:
                f.write(f"{settings.THETADATA_USERNAME}\n{settings.THETADATA_PASSWORD}\n")
        except Exception as e:
            logger.error(f"Failed to write creds file: {e}")
            return False

    try:
        subprocess.Popen(
            ["java", "-jar", THETA_JAR, "--creds-file", CREDS_FILE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(THETA_JAR),
        )
        logger.info(f"ThetaData Terminal launched from {THETA_JAR}")
    except Exception as e:
        logger.error(f"Failed to start ThetaData Terminal: {e}")
        return False

    for i in range(STARTUP_TIMEOUT // 5):
        _time.sleep(5)
        if _is_alive():
            logger.info(f"ThetaData Terminal ready after {(i + 1) * 5}s")
            return True
        logger.info(f"Waiting for ThetaData... ({(i + 1) * 5}s/{STARTUP_TIMEOUT}s)")

    logger.error(f"ThetaData Terminal did not respond after {STARTUP_TIMEOUT}s")
    return False


# ============================================================================
# Cross-asset daily data (yfinance)
# ============================================================================

def fetch_cross_asset_daily(start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily data for SPX, VIX, VIX3M, VVIX, ^TNX, ^IRX via yfinance.
    Returns a DataFrame indexed by date with columns like spx_open, spx_close,
    vix_close, etc.  Missing tickers fill with NaN.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — cross-asset features will be NaN")
        return pd.DataFrame()

    tickers = {
        "^GSPC": "spx",
        "^VIX": "vix",
        "^VIX3M": "vix3m",
        "^VVIX": "vvix",
        "^TNX": "yield_10y",
        "^IRX": "yield_2y",
    }

    frames = {}
    for yf_sym, prefix in tickers.items():
        try:
            df = yf.download(
                yf_sym,
                start=start - timedelta(days=60),  # extra history for lookbacks
                end=end + timedelta(days=1),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                "Open": f"{prefix}_open",
                "High": f"{prefix}_high",
                "Low": f"{prefix}_low",
                "Close": f"{prefix}_close",
                "Volume": f"{prefix}_volume",
            })
            df.index = pd.to_datetime(df.index).date
            frames[prefix] = df
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {yf_sym}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1)
    combined = combined.sort_index()
    return combined


def build_cross_data_for_date(cross_daily: pd.DataFrame, d: date) -> dict:
    """Build the cross_data dict expected by compute_cross_asset_features."""
    result = {
        "vix": np.nan, "vix_prior": np.nan,
        "vix_history": [],
        "vvix": np.nan, "vix9d": np.nan, "vix3m": np.nan,
        "vix_future_1": np.nan, "vix_future_2": np.nan,
        "yield_10y": np.nan, "yield_10y_prior": np.nan,
        "yield_2y": np.nan,
        "hy_oas": np.nan, "hy_oas_prior": np.nan,
        "dxy": np.nan, "dxy_prior": np.nan,
        "gold_change_pct": np.nan, "oil_change_pct": np.nan,
        "spx_overnight_return": np.nan,
        "es_volume_ratio": np.nan,
    }
    if cross_daily.empty:
        return result

    # Get rows up to and including d
    mask = np.array([idx <= d for idx in cross_daily.index])
    available = cross_daily.loc[mask]
    if available.empty:
        return result

    today = available.iloc[-1] if available.index[-1] == d else None
    prior = available.iloc[-2] if len(available) >= 2 and today is not None else (
        available.iloc[-1] if today is None else None
    )

    def _safe(row, col):
        if row is None:
            return np.nan
        v = row.get(col, np.nan) if isinstance(row, dict) else getattr(row, col, np.nan) if hasattr(row, col) else np.nan
        try:
            v = float(v)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    result["vix"] = _safe(today, "vix_close")
    result["vix_prior"] = _safe(prior, "vix_close")
    result["vix3m"] = _safe(today, "vix3m_close")
    result["vvix"] = _safe(today, "vvix_close")
    result["yield_10y"] = _safe(today, "yield_10y_close")
    result["yield_10y_prior"] = _safe(prior, "yield_10y_close")
    result["yield_2y"] = _safe(today, "yield_2y_close")

    # VIX history (last 30 closes, most recent first)
    vix_col = "vix_close"
    if vix_col in available.columns:
        vix_hist = available[vix_col].dropna().values[-30:][::-1].tolist()
        result["vix_history"] = vix_hist

    # SPX overnight return
    spx_open = _safe(today, "spx_open")
    spx_prior_close = _safe(prior, "spx_close")
    if np.isfinite(spx_open) and np.isfinite(spx_prior_close) and spx_prior_close > 0:
        result["spx_overnight_return"] = (spx_open - spx_prior_close) / spx_prior_close

    return result


# ============================================================================
# ThetaData fetchers for a single day
# ============================================================================

def fetch_index_1m(session: requests.Session, symbol: str, d: date) -> pd.DataFrame:
    """Fetch 1-min OHLC bars for an index (SPX, VIX) for a single day."""
    data = _theta_get(session, "/v3/index/history/ohlc", {
        "symbol": symbol,
        "start_date": _format_date(d),
        "end_date": _format_date(d),
        "interval": "1m",
        "start_time": "09:30:00",
        "end_time": "16:00:00",
    })
    rows = _parse_response(data)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("timestamp", "time", "datetime"):
            col_map[c] = "timestamp"
        elif cl in ("o", "open"):
            col_map[c] = "open"
        elif cl in ("h", "high"):
            col_map[c] = "high"
        elif cl in ("l", "low"):
            col_map[c] = "low"
        elif cl in ("c", "close"):
            col_map[c] = "close"
        elif cl in ("v", "volume"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def fetch_chain_greeks(
    session: requests.Session, d: date, slot_time: time, end_time: time
) -> pd.DataFrame:
    """
    Fetch 0DTE option chain Greeks at a time slot.
    Returns DataFrame with [strike, right, bid, ask, delta, gamma, theta, vega,
    implied_vol, underlying_price].
    """
    all_rows = []
    for right in ("call", "put"):
        data = _theta_get(session, "/v3/option/history/greeks/first_order", {
            "symbol": "SPXW",
            "expiration": _format_date(d),
            "date": _format_date(d),
            "interval": "5m",
            "start_time": _format_time(slot_time),
            "end_time": _format_time(end_time),
            "strike": "*",
            "right": right,
        })
        response = _parse_response(data)
        for item in response:
            if not isinstance(item, dict):
                continue
            contract = item.get("contract", {})
            data_points = item.get("data", [])
            if not data_points:
                continue
            dp = data_points[0]  # use first data point closest to slot_time
            all_rows.append({
                "strike": float(contract.get("strike", 0)),
                "right": "C" if right == "call" else "P",
                "bid": float(dp.get("bid", 0) or 0),
                "ask": float(dp.get("ask", 0) or 0),
                "delta": float(dp.get("delta", 0) or 0),
                "gamma": float(dp.get("gamma", 0) or 0),
                "theta": float(dp.get("theta", 0) or 0),
                "vega": float(dp.get("vega", 0) or 0),
                "implied_vol": float(dp.get("implied_vol", 0) or 0),
                "underlying_price": float(dp.get("underlying_price", 0) or 0),
            })

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _timestamp_to_slot_key(ts_val) -> str:
    """Convert a ThetaData timestamp to HH:MM slot key.

    ThetaData v3 returns ISO strings like '2026-03-11T09:35:00.000'.
    These are in ET (exchange time), so we extract HH:MM directly.
    """
    try:
        if isinstance(ts_val, (int, float)):
            # Milliseconds since epoch — convert to ET
            dt = datetime.utcfromtimestamp(ts_val / 1000.0)
            return dt.strftime("%H:%M")
        elif isinstance(ts_val, str):
            # ISO format: '2026-03-11T09:35:00.000' — extract time directly
            if "T" in ts_val:
                time_part = ts_val.split("T")[1]  # '09:35:00.000'
                return time_part[:5]  # 'HH:MM'
            dt = datetime.fromisoformat(ts_val)
            return dt.strftime("%H:%M")
        return ""
    except Exception:
        return ""


def _fetch_greeks_one_right(
    session: requests.Session, d: date, right: str
) -> list[dict]:
    """Fetch bulk Greeks for one right (call or put). Thread-safe."""
    data = _theta_get(session, "/v3/option/history/greeks/first_order", {
        "symbol": "SPXW",
        "expiration": _format_date(d),
        "date": _format_date(d),
        "interval": "5m",
        "start_time": "09:30:00",
        "end_time": "15:10:00",
        "strike": "*",
        "right": right,
    })
    response = _parse_response(data)
    right_label = "C" if right == "call" else "P"
    rows = []
    for item in response:
        if not isinstance(item, dict):
            continue
        contract = item.get("contract", {})
        strike = float(contract.get("strike", 0))
        for dp in item.get("data", []):
            ts = dp.get("timestamp") or dp.get("time") or dp.get("datetime")
            slot_key = _timestamp_to_slot_key(ts)
            if not slot_key:
                continue
            rows.append((slot_key, {
                "strike": strike,
                "right": right_label,
                "bid": float(dp.get("bid", 0) or 0),
                "ask": float(dp.get("ask", 0) or 0),
                "delta": float(dp.get("delta", 0) or 0),
                "gamma": float(dp.get("gamma", 0) or 0),
                "theta": float(dp.get("theta", 0) or 0),
                "vega": float(dp.get("vega", 0) or 0),
                "implied_vol": float(dp.get("implied_vol", 0) or 0),
                "underlying_price": float(dp.get("underlying_price", 0) or 0),
            }))
    return rows


def fetch_chain_greeks_bulk(
    session: requests.Session, d: date
) -> dict[str, pd.DataFrame]:
    """
    Fetch ALL 0DTE Greeks for the full trading day in 2 parallel API calls.
    Returns dict keyed by HH:MM slot → DataFrame with same schema as fetch_chain_greeks.
    """
    slot_rows: dict[str, list[dict]] = {}

    # Fetch calls and puts in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_fetch_greeks_one_right, session, d, r): r
            for r in ("call", "put")
        }
        for fut in as_completed(futures):
            for slot_key, row in fut.result():
                slot_rows.setdefault(slot_key, []).append(row)

    result = {k: pd.DataFrame(v) for k, v in slot_rows.items()}
    logger.info(f"  Bulk Greeks: {len(result)} slots, "
                f"{sum(len(df) for df in result.values())} rows")
    return result


def fetch_chain_oi(session: requests.Session, d: date) -> pd.DataFrame:
    """Fetch open interest for 0DTE SPXW options."""
    all_rows = []
    for right in ("call", "put"):
        data = _theta_get(session, "/v3/option/history/open_interest", {
            "symbol": "SPXW",
            "expiration": _format_date(d),
            "date": _format_date(d),
            "right": right,
        })
        response = _parse_response(data)
        for item in response:
            if not isinstance(item, dict):
                continue
            contract = item.get("contract", {})
            data_points = item.get("data", [])
            if not data_points:
                continue
            dp = data_points[0]
            all_rows.append({
                "strike": float(contract.get("strike", 0)),
                "right": "C" if right == "call" else "P",
                "open_interest": int(dp.get("open_interest", 0) or 0),
            })
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def fetch_chain_volume(
    session: requests.Session, d: date, slot_time: time, end_time: time
) -> pd.DataFrame:
    """Fetch option volume for 0DTE SPXW at a time slot."""
    all_rows = []
    for right in ("call", "put"):
        data = _theta_get(session, "/v3/option/history/ohlc", {
            "symbol": "SPXW",
            "expiration": _format_date(d),
            "date": _format_date(d),
            "interval": "5m",
            "start_time": _format_time(slot_time),
            "end_time": _format_time(end_time),
            "strike": "*",
            "right": right,
        })
        response = _parse_response(data)
        for item in response:
            if not isinstance(item, dict):
                continue
            contract = item.get("contract", {})
            data_points = item.get("data", [])
            if not data_points:
                continue
            dp = data_points[0]
            vol = int(dp.get("volume", 0) or 0)
            all_rows.append({
                "strike": float(contract.get("strike", 0)),
                "right": "C" if right == "call" else "P",
                "volume": vol,
            })
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _fetch_volume_one_right(
    session: requests.Session, d: date, right: str
) -> list[dict]:
    """Fetch bulk volume for one right (call or put). Thread-safe."""
    data = _theta_get(session, "/v3/option/history/ohlc", {
        "symbol": "SPXW",
        "expiration": _format_date(d),
        "date": _format_date(d),
        "interval": "5m",
        "start_time": "09:30:00",
        "end_time": "15:10:00",
        "strike": "*",
        "right": right,
    })
    response = _parse_response(data)
    right_label = "C" if right == "call" else "P"
    rows = []
    for item in response:
        if not isinstance(item, dict):
            continue
        contract = item.get("contract", {})
        strike = float(contract.get("strike", 0))
        for dp in item.get("data", []):
            ts = dp.get("timestamp") or dp.get("time") or dp.get("datetime")
            slot_key = _timestamp_to_slot_key(ts)
            if not slot_key:
                continue
            rows.append((slot_key, {
                "strike": strike,
                "right": right_label,
                "volume": int(dp.get("volume", 0) or 0),
            }))
    return rows


def fetch_chain_volume_bulk(
    session: requests.Session, d: date
) -> dict[str, pd.DataFrame]:
    """
    Fetch ALL 0DTE option volume for the full day in 2 parallel API calls.
    Returns dict keyed by HH:MM slot → DataFrame with [strike, right, volume].
    """
    slot_rows: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_fetch_volume_one_right, session, d, r): r
            for r in ("call", "put")
        }
        for fut in as_completed(futures):
            for slot_key, row in fut.result():
                slot_rows.setdefault(slot_key, []).append(row)

    result = {k: pd.DataFrame(v) for k, v in slot_rows.items()}
    logger.info(f"  Bulk Volume: {len(result)} slots, "
                f"{sum(len(df) for df in result.values())} rows")
    return result


def fetch_trading_days(session: requests.Session, start: date, end: date) -> list[date]:
    """Get SPXW expiration dates as proxy for trading days."""
    data = _theta_get(session, "/v3/option/list/expirations", {"symbol": "SPXW"})
    rows = _parse_response(data)
    days = []
    for row in rows:
        exp_str = row.get("expiration") if isinstance(row, dict) else row
        if exp_str is None:
            continue
        try:
            if isinstance(exp_str, int):
                d = datetime.strptime(str(exp_str), "%Y%m%d").date()
            elif isinstance(exp_str, str) and "T" in exp_str:
                d = datetime.fromisoformat(exp_str.split(".")[0]).date()
            elif isinstance(exp_str, str):
                for fmt in ("%Y-%m-%d", "%Y%m%d"):
                    try:
                        d = datetime.strptime(exp_str, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    continue
            else:
                continue
            if start <= d <= end:
                days.append(d)
        except Exception:
            continue
    if days:
        return sorted(days)
    # Fallback: business days
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ============================================================================
# Label computation
# ============================================================================

def _find_delta_strike(chain: pd.DataFrame, target_delta: float, right: str) -> Optional[float]:
    """Find the strike closest to target |delta| for a given right ('C' or 'P')."""
    subset = chain[chain["right"] == right].copy()
    if subset.empty:
        return None
    subset["_abs_diff"] = (subset["delta"].abs() - abs(target_delta)).abs()
    best = subset.loc[subset["_abs_diff"].idxmin()]
    return float(best["strike"])


def _mid_price(chain: pd.DataFrame, strike: float, right: str) -> float:
    """Get mid price for a specific strike/right from chain."""
    row = chain[(chain["strike"] == strike) & (chain["right"] == right)]
    if row.empty:
        return np.nan
    r = row.iloc[0]
    bid = float(r.get("bid", 0) or 0)
    ask = float(r.get("ask", 0) or 0)
    if bid <= 0 and ask <= 0:
        return np.nan
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return (bid + ask) / 2.0


def compute_ic_labels(
    entry_chain: pd.DataFrame,
    close_chain: pd.DataFrame,
    target_delta: float,
) -> dict:
    """
    Compute labels for one IC at a given delta target.

    Returns dict with keys prefixed by delta (e.g., d10_pnl_pct, d10_hit_tp25, ...).
    """
    prefix = f"d{int(target_delta * 100)}"
    nan_labels = {
        f"{prefix}_credit": np.nan,
        f"{prefix}_pnl": np.nan,
        f"{prefix}_pnl_pct": np.nan,
        f"{prefix}_hit_tp25": np.nan,
        f"{prefix}_hit_tp50": np.nan,
        f"{prefix}_big_loss": np.nan,
        f"{prefix}_long_profitable": np.nan,
    }

    # Find short strikes at entry
    short_put_strike = _find_delta_strike(entry_chain, target_delta, "P")
    short_call_strike = _find_delta_strike(entry_chain, target_delta, "C")
    if short_put_strike is None or short_call_strike is None:
        return nan_labels

    # Wing strikes
    long_put_strike = short_put_strike - WING_WIDTH
    long_call_strike = short_call_strike + WING_WIDTH

    # Entry prices (mid)
    sp_mid = _mid_price(entry_chain, short_put_strike, "P")
    sc_mid = _mid_price(entry_chain, short_call_strike, "C")
    lp_mid = _mid_price(entry_chain, long_put_strike, "P")
    lc_mid = _mid_price(entry_chain, long_call_strike, "C")

    if any(np.isnan(v) for v in [sp_mid, sc_mid, lp_mid, lc_mid]):
        return nan_labels

    credit = (sp_mid + sc_mid) - (lp_mid + lc_mid)
    if credit <= 0:
        return nan_labels

    # Close prices
    sp_close = _mid_price(close_chain, short_put_strike, "P")
    sc_close = _mid_price(close_chain, short_call_strike, "C")
    lp_close = _mid_price(close_chain, long_put_strike, "P")
    lc_close = _mid_price(close_chain, long_call_strike, "C")

    if any(np.isnan(v) for v in [sp_close, sc_close, lp_close, lc_close]):
        return nan_labels

    value_at_close = (sp_close + sc_close) - (lp_close + lc_close)
    pnl = credit - value_at_close  # short IC P&L
    pnl_pct = pnl / credit

    return {
        f"{prefix}_credit": credit,
        f"{prefix}_pnl": pnl,
        f"{prefix}_pnl_pct": pnl_pct,
        f"{prefix}_hit_tp25": 1.0 if pnl_pct >= 0.25 else 0.0,
        f"{prefix}_hit_tp50": 1.0 if pnl_pct >= 0.50 else 0.0,
        f"{prefix}_big_loss": 1.0 if pnl_pct <= -2.0 else 0.0,
        f"{prefix}_long_profitable": 1.0 if pnl < 0 else 0.0,
    }


# ============================================================================
# Feature computation for a single entry slot
# ============================================================================

def compute_slot_features(
    d: date,
    slot_time: time,
    chain: pd.DataFrame,
    chain_oi: pd.DataFrame,
    spx_1m: pd.DataFrame,
    vix_1m: pd.DataFrame,
    cross_data: dict,
    prior_days_1m: list,
    chain_volume: pd.DataFrame = None,
    prior_chain: pd.DataFrame = None,
    prior_greeks: dict = None,
) -> dict:
    """Compute all v8 features for one entry slot. Returns flat dict."""
    import pytz
    eastern = pytz.timezone("US/Eastern")

    entry_dt = eastern.localize(datetime.combine(d, slot_time))
    spx_price = 0.0

    # Get SPX price from chain or 1m bars
    if not chain.empty and "underlying_price" in chain.columns:
        prices = chain["underlying_price"].dropna()
        if not prices.empty:
            spx_price = float(prices.iloc[0])

    if spx_price <= 0 and not spx_1m.empty:
        spx_price = float(spx_1m["close"].iloc[-1])

    # Slice 1m bars up to current slot time
    spx_up_to = _slice_up_to(spx_1m, slot_time)
    vix_up_to = _slice_up_to(vix_1m, slot_time) if not vix_1m.empty else pd.DataFrame()

    # Minutes to close
    slot_minutes = slot_time.hour * 60 + slot_time.minute
    close_minutes = 16 * 60  # 4:00 PM
    minutes_to_close = max(0, close_minutes - slot_minutes)

    # RV features need annualized value for vol_surface
    rv_result = compute_realized_vol_features(spx_up_to, prior_days_1m)
    rv_annualized = rv_result.get("rv_annualized", np.nan)

    features = {}

    # 1. Greeks aggregate (~30)
    features.update(compute_greeks_aggregate_features(
        chain, chain_oi, spx_price, minutes_to_close, prior_greeks,
    ))

    # 2. Vol surface (~30)
    features.update(compute_vol_surface_features(
        chain, spx_price, prior_chain, rv_annualized,
    ))

    # 3. Realized vol (~30)
    features.update(rv_result)

    # 4. Positioning (~20)
    features.update(compute_positioning_features(
        chain, chain_oi, spx_price, minutes_to_close,
    ))

    # 5. Order flow (~25)
    features.update(compute_order_flow_features(
        chain, chain_oi, spx_price, prior_oi=None, chain_volume=chain_volume,
    ))

    # 6. Microstructure (~25)
    features.update(compute_microstructure_features(chain, spx_price))

    # 7. Time/event (~25)
    features.update(compute_time_event_features(entry_dt))

    # 8. Cross-asset (~25)
    features.update(compute_cross_asset_features(cross_data))

    # 9. Market internals (~15)
    features.update(compute_market_internals_features(spx_up_to, vix_up_to))

    # 10. Technical indicators (11 features from existing v4 module)
    spx_5m = _resample_to_5min(spx_up_to) if not spx_up_to.empty else pd.DataFrame()
    tech_features = compute_all_technical_indicators(spx_5m)
    features.update({f"tech_{k}": v for k, v in tech_features.items()})

    # 11. Interaction & Regime features (~15)
    interaction_features = _compute_interaction_features(features)
    features.update(interaction_features)

    return features


def _slice_up_to(df: pd.DataFrame, t: time) -> pd.DataFrame:
    """Slice a 1-min DataFrame up to (inclusive) a given time."""
    if df.empty or "timestamp" not in df.columns:
        return df
    cutoff_str = t.strftime("%H:%M")
    mask = df["timestamp"].dt.strftime("%H:%M") <= cutoff_str
    return df[mask].copy()


# ============================================================================
# Generate entry time slots
# ============================================================================

def generate_slots(entry_start: str, entry_end: str, interval_min: int) -> list[time]:
    """Generate list of entry time slots."""
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


# ============================================================================
# Process a single day
# ============================================================================

def process_day(
    session: requests.Session,
    d: date,
    slots: list[time],
    cross_data: dict,
    prior_days_1m: list,
    save_snapshots: bool = False,
    snapshot_dir: str = "",
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Process one trading day: fetch data, compute features + labels for each slot.

    Uses bulk API fetching: 2 calls for Greeks + 2 calls for Volume cover all slots.
    Previously made 4 calls per slot × 54 slots = 216 calls; now just 4 + 4 day-level = 8 total.

    Returns (feature_rows, short_label_rows, long_label_rows).
    """
    feature_rows = []
    short_label_rows = []
    long_label_rows = []

    # Fetch ALL data for this day in parallel (8 API calls across 6 threads)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut_spx = ex.submit(fetch_index_1m, session, "SPX", d)
        fut_vix = ex.submit(fetch_index_1m, session, "VIX", d)
        fut_oi = ex.submit(fetch_chain_oi, session, d)
        fut_greeks = ex.submit(fetch_chain_greeks_bulk, session, d)
        fut_volume = ex.submit(fetch_chain_volume_bulk, session, d)

    spx_1m = fut_spx.result()
    if spx_1m.empty:
        logger.warning(f"  {d}: no SPX 1m data — skipping")
        return feature_rows, short_label_rows, long_label_rows

    vix_1m = fut_vix.result()
    chain_oi = fut_oi.result()
    greeks_by_slot = fut_greeks.result()
    volume_by_slot = fut_volume.result()

    if not greeks_by_slot:
        logger.warning(f"  {d}: no bulk Greeks data — skipping")
        return feature_rows, short_label_rows, long_label_rows

    # Get close chain from bulk data (for labels)
    close_key = CLOSE_TIME.strftime("%H:%M")
    close_chain = greeks_by_slot.get(close_key, pd.DataFrame())

    # Track prior chain/greeks for change features
    prior_chain = None
    prior_greeks = None

    for slot in slots:
        slot_key = slot.strftime("%H:%M")

        # Look up chain from bulk data
        chain = greeks_by_slot.get(slot_key, pd.DataFrame())
        if chain.empty:
            continue

        # Look up volume from bulk data
        chain_vol = volume_by_slot.get(slot_key, pd.DataFrame())

        # Compute features
        feats = compute_slot_features(
            d, slot, chain, chain_oi, spx_1m, vix_1m, cross_data,
            prior_days_1m, chain_vol, prior_chain, prior_greeks,
        )

        row_id = {"date": d.isoformat(), "entry_time": slot.strftime("%H:%M")}
        feature_rows.append({**row_id, **feats})

        # Compute labels for each delta
        short_labels = dict(row_id)
        long_labels = dict(row_id)
        if not close_chain.empty:
            for delta in IC_DELTAS:
                ic_labels = compute_ic_labels(chain, close_chain, delta)
                short_labels.update(ic_labels)
                # Long IC labels: invert profitability
                prefix = f"d{int(delta * 100)}"
                long_labels.update({
                    f"{prefix}_credit": ic_labels.get(f"{prefix}_credit", np.nan),
                    f"{prefix}_pnl": -ic_labels.get(f"{prefix}_pnl", np.nan),
                    f"{prefix}_pnl_pct": -ic_labels.get(f"{prefix}_pnl_pct", np.nan),
                    f"{prefix}_long_profitable": ic_labels.get(f"{prefix}_long_profitable", np.nan),
                })

        short_label_rows.append(short_labels)
        long_label_rows.append(long_labels)

        # Update priors for next slot
        prior_chain = chain
        # Store aggregate greeks as prior for change features
        prior_greeks = {
            k: v for k, v in feats.items()
            if k.startswith("gex_") or k.startswith("net_") or k.startswith("total_")
        }

    # Optionally save chain snapshots
    if save_snapshots and snapshot_dir and not close_chain.empty:
        _save_snapshot(d, close_chain, chain_oi, snapshot_dir)

    return feature_rows, short_label_rows, long_label_rows


def _save_snapshot(d: date, chain: pd.DataFrame, chain_oi: pd.DataFrame, snapshot_dir: str):
    """Save compressed daily chain snapshot."""
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot = {
        "date": d.isoformat(),
        "chain": chain.to_dict(orient="records"),
        "oi": chain_oi.to_dict(orient="records") if not chain_oi.empty else [],
    }
    path = os.path.join(snapshot_dir, f"{d.isoformat()}.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(snapshot, f)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backfill v8 feature store from ThetaData historical data"
    )
    parser.add_argument(
        "--start-date", type=str,
        default=(date.today() - timedelta(days=3 * 365)).isoformat(),
        help="Start date YYYY-MM-DD (default: 3 years ago)",
    )
    parser.add_argument(
        "--end-date", type=str,
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="End date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="ml/features/v8/data/",
        help="Output directory for parquet files",
    )
    parser.add_argument("--resume", action="store_true", help="Skip dates already in output")
    parser.add_argument(
        "--save-snapshots", action="store_true",
        help="Save compressed daily chain snapshots",
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="data_store/chain_snapshots/",
        help="Directory for chain snapshots",
    )
    parser.add_argument("--entry-start", type=str, default="09:35", help="First entry time (HH:MM)")
    parser.add_argument("--entry-end", type=str, default="14:00", help="Last entry time (HH:MM)")
    parser.add_argument("--entry-interval", type=int, default=5, help="Minutes between entry slots")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "training_features.parquet"
    short_labels_path = output_dir / "short_ic_labels.parquet"
    long_labels_path = output_dir / "long_ic_labels.parquet"

    slots = generate_slots(args.entry_start, args.entry_end, args.entry_interval)
    logger.info(f"Entry slots: {len(slots)} ({args.entry_start} to {args.entry_end} every {args.entry_interval}m)")

    # Determine dates already processed (for --resume)
    existing_dates = set()
    if args.resume and features_path.exists():
        try:
            existing_df = pd.read_parquet(features_path, columns=["date"])
            existing_dates = set(existing_df["date"].unique())
            logger.info(f"Resume mode: {len(existing_dates)} dates already processed")
        except Exception:
            pass

    # Ensure ThetaData is running
    if not ensure_thetadata_running():
        logger.error("Cannot start ThetaData Terminal — aborting")
        sys.exit(1)

    session = requests.Session()
    session.timeout = THETA_TIMEOUT

    # Fetch trading days
    logger.info(f"Fetching trading days from {start} to {end}...")
    trading_days = fetch_trading_days(session, start, end)
    logger.info(f"Found {len(trading_days)} trading days")

    if args.resume:
        trading_days = [d for d in trading_days if d.isoformat() not in existing_dates]
        logger.info(f"After resume filter: {len(trading_days)} days remaining")

    if not trading_days:
        logger.info("No days to process")
        return

    # Fetch cross-asset daily data upfront
    logger.info("Fetching cross-asset daily data (yfinance)...")
    cross_daily = fetch_cross_asset_daily(start, end)
    logger.info(f"Cross-asset data: {len(cross_daily)} rows, {len(cross_daily.columns)} columns")

    # Process days
    all_features = []
    all_short_labels = []
    all_long_labels = []
    prior_days_1m = []  # rolling buffer of prior day SPX 1m for multi-day RV features
    total_days = len(trading_days)
    errors = 0

    for i, d in enumerate(trading_days):
        try:
            cross_data = build_cross_data_for_date(cross_daily, d)

            feat_rows, short_rows, long_rows = process_day(
                session, d, slots, cross_data, prior_days_1m,
                save_snapshots=args.save_snapshots,
                snapshot_dir=args.snapshot_dir,
            )

            all_features.extend(feat_rows)
            all_short_labels.extend(short_rows)
            all_long_labels.extend(long_rows)

            n_feats = len(feat_rows[0]) - 2 if feat_rows else 0  # minus date, entry_time
            pct = (i + 1) / total_days * 100
            logger.info(
                f"{d}: {len(feat_rows)} slots, {n_feats} features each | "
                f"{i + 1}/{total_days} days ({pct:.1f}%)"
            )

            # Update prior_days_1m buffer (keep last 5 days)
            spx_1m = fetch_index_1m(session, "SPX", d)
            if not spx_1m.empty:
                prior_days_1m.insert(0, spx_1m)
                prior_days_1m = prior_days_1m[:5]

            # Small delay between days to avoid overwhelming ThetaData
            _time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Interrupted — saving partial results...")
            break
        except Exception as e:
            logger.warning(f"{d}: ERROR — {e}")
            errors += 1
            continue

    # Save results
    if not all_features:
        logger.warning("No features computed — nothing to save")
        return

    logger.info("Saving parquet files...")

    feat_df = pd.DataFrame(all_features)
    short_df = pd.DataFrame(all_short_labels)
    long_df = pd.DataFrame(all_long_labels)

    # If resuming, append to existing data
    if args.resume:
        for path, df_name in [
            (features_path, "feat_df"),
            (short_labels_path, "short_df"),
            (long_labels_path, "long_df"),
        ]:
            df = locals()[df_name]
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                    df = pd.concat([existing, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["date", "entry_time"], keep="last")
                    df = df.sort_values(["date", "entry_time"]).reset_index(drop=True)
                    locals()[df_name] = df
                except Exception as e:
                    logger.warning(f"Could not merge existing {path.name}: {e}")
        # Re-read locals after update
        feat_df = locals()["feat_df"]
        short_df = locals()["short_df"]
        long_df = locals()["long_df"]

    feat_df.to_parquet(features_path, index=False)
    short_df.to_parquet(short_labels_path, index=False)
    long_df.to_parquet(long_labels_path, index=False)

    # Summary
    feat_size = features_path.stat().st_size / (1024 * 1024)
    short_size = short_labels_path.stat().st_size / (1024 * 1024)
    long_size = long_labels_path.stat().st_size / (1024 * 1024)

    n_features = len(feat_df.columns) - 2  # minus date, entry_time
    n_dates = feat_df["date"].nunique()
    date_range = f"{feat_df['date'].min()} to {feat_df['date'].max()}"

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Rows:       {len(feat_df):,}")
    logger.info(f"  Features:   {n_features}")
    logger.info(f"  Days:       {n_dates}")
    logger.info(f"  Date range: {date_range}")
    logger.info(f"  Errors:     {errors}")
    logger.info(f"  Files:")
    logger.info(f"    {features_path}  ({feat_size:.1f} MB)")
    logger.info(f"    {short_labels_path}  ({short_size:.1f} MB)")
    logger.info(f"    {long_labels_path}  ({long_size:.1f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
