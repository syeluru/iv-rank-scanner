#!/usr/bin/env python3
"""
Build training dataset for 0DTE Iron Condor ML model.

Computes 37 features + labels for each trading day (Feb 2023 - Feb 2026).

Data sources:
    - yfinance: SPX (^GSPC), VIX (^VIX), VIX3M (^VIX3M) daily OHLCV
    - ThetaData: SPX 5-min intraday candles (/v3/index/history/ohlc)
    - ThetaData: SPXW option chain greeks (/v3/option/history/greeks/first_order)

Features (37):
    SPX Daily (8):  rv_close_5d/10d/20d, atr_14d_pct, gap_pct, prior_day_range/return, sma_20d_dist
    VIX (6):        vix_level, vix_change_1d/5d, vix_rank_30d, vix_sma_10d_dist, vix_term_slope
    Intraday (10):  intraday_rv, move_from_open, orb_range/contained, range_exhaustion,
                    high_low_range, price_vs_vwap, momentum_30min, trend_slope, volume_ratio
    Time (5):       time_sin/cos, dow_sin/cos, minutes_to_close
    Calendar (4):   is_fomc_day/week, is_friday, days_since_fomc
    Option (4):     atm_iv, iv_skew_10d, short_put/call_delta

Labels:  entry_credit, hit_25pct/50pct, pnl_at_close, max_adverse_pct, win_50pct_or_close

Usage:
    python scripts/build_training_dataset.py                          # Full run (noon entry, $25 wings)
    python scripts/build_training_dataset.py --entry-hour 13          # 1 PM entry
    python scripts/build_training_dataset.py --max-days 10            # Test with 10 days
    python scripts/build_training_dataset.py --features-only          # Skip label computation
    python scripts/build_training_dataset.py --no-theta               # yfinance features only
"""

import argparse
import json
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import requests
from loguru import logger

# Add project root to path for ml.features imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features.technical_indicators import compute_all_technical_indicators

# Thread-safe progress counters
_progress_lock = Lock()
_progress = {"done": 0, "errors": 0, "cached": 0}

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
CACHE_DIR = TRAINING_DATA_DIR / "cache"
DAILY_CACHE_DIR = CACHE_DIR / "daily_features"

# ── ThetaData ──────────────────────────────────────────────────────────────────
THETA_HOST = "http://127.0.0.1:25503"
THETA_TIMEOUT = 60
API_DELAY = 0.05  # seconds between ThetaData calls

# ── Strategy ───────────────────────────────────────────────────────────────────
TARGET_DELTA = 0.10

# ── FOMC Decision Dates (2023-2026) ───────────────────────────────────────────
FOMC_DATES = sorted([
    date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3), date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 17),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 16),
])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def download_daily_data(start: str, end: str) -> dict:
    """Download daily OHLCV from yfinance for SPX, VIX, VIX3M. Cached to parquet."""
    cache_file = CACHE_DIR / "yf_daily.parquet"

    if cache_file.exists():
        age_hours = (time_module.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info(f"Loading cached yfinance data ({age_hours:.1f}h old)")
            store = pd.read_parquet(cache_file)
            data = {}
            for name in store["_ticker"].unique():
                data[name] = store[store["_ticker"] == name].drop(columns="_ticker")
            return data

    import yfinance as yf

    tickers = {"SPX": "^GSPC", "VIX": "^VIX", "VIX3M": "^VIX3M"}
    data = {}

    for name, ticker in tickers.items():
        logger.info(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                logger.warning(f"  No data for {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df
            logger.info(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            logger.warning(f"  Failed to download {ticker}: {e}")

    if data:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frames = []
        for name, df in data.items():
            df_copy = df.copy()
            df_copy["_ticker"] = name
            frames.append(df_copy)
        pd.concat(frames).to_parquet(cache_file)

    return data


def theta_request(endpoint: str, params: dict) -> dict | None:
    """Make a ThetaData API request with rate limiting."""
    try:
        resp = requests.get(f"{THETA_HOST}{endpoint}", params=params, timeout=THETA_TIMEOUT)
        time_module.sleep(API_DELAY)
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        logger.debug(f"ThetaData request failed: {e}")
        return None


def theta_available() -> bool:
    """Check if ThetaData terminal is running."""
    try:
        resp = requests.get(
            f"{THETA_HOST}/v3/option/list/expirations",
            params={"symbol": "SPXW"},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_trading_days(start: date, end: date) -> list:
    """Get trading days from ThetaData SPXW expirations (cached)."""
    cache_file = CACHE_DIR / "trading_days.json"

    if cache_file.exists():
        with open(cache_file) as f:
            days = [date.fromisoformat(d) for d in json.load(f)]
        filtered = sorted(d for d in days if start <= d <= end)
        if filtered:
            return filtered

    data = theta_request("/v3/option/list/expirations", {"symbol": "SPXW", "format": "json"})

    if not data:
        logger.warning("ThetaData unavailable, using business day fallback")
        return [d.date() for d in pd.bdate_range(start, end)]

    response = data.get("response", data if isinstance(data, list) else [])
    trading_days = []
    for row in response:
        exp_str = row.get("expiration") if isinstance(row, dict) else row
        if exp_str:
            try:
                trading_days.append(date.fromisoformat(str(exp_str)[:10]))
            except (ValueError, TypeError):
                continue

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump([d.isoformat() for d in sorted(trading_days)], f)

    return sorted(d for d in trading_days if start <= d <= end)


def fetch_spx_intraday(trade_date: date) -> pd.DataFrame | None:
    """Fetch SPX 5-min candles for a day from ThetaData (cached)."""
    cache_file = CACHE_DIR / "spx_intraday" / f"{trade_date.isoformat()}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            rows = json.load(f)
        return pd.DataFrame(rows) if rows else None

    data = theta_request("/v3/index/history/ohlc", {
        "symbol": "SPX",
        "start_date": trade_date.strftime("%Y%m%d"),
        "end_date": trade_date.strftime("%Y%m%d"),
        "interval": "5m",
        "start_time": "09:30:00",
        "end_time": "16:00:00",
        "format": "json",
    })

    if not data:
        return None

    response = data.get("response", data if isinstance(data, list) else [])

    # Handle nested format (like options endpoint)
    if response and isinstance(response[0], dict) and "data" in response[0]:
        bars = []
        for entry in response:
            for dp in entry.get("data", []):
                bars.append(dp)
        response = bars

    if not response:
        return None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(response, f)

    return pd.DataFrame(response)


def fetch_option_chain(trade_date: date, entry_time: dtime, right: str) -> list:
    """Fetch SPXW option chain greeks at entry time (cached)."""
    time_str = f"{entry_time.hour:02d}{entry_time.minute:02d}"
    cache_file = CACHE_DIR / "chains" / f"{trade_date.isoformat()}_{time_str}_{right}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    end_min = entry_time.minute + 1 if entry_time.minute < 59 else 59
    end_time = dtime(entry_time.hour, end_min)

    data = theta_request("/v3/option/history/greeks/first_order", {
        "symbol": "SPXW",
        "expiration": trade_date.strftime("%Y%m%d"),
        "right": right,
        "date": trade_date.strftime("%Y%m%d"),
        "interval": "1m",
        "start_time": entry_time.strftime("%H:%M:%S"),
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
            "gamma": dp.get("gamma", 0),
            "theta": dp.get("theta", 0),
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(flattened, f)

    return flattened


def fetch_leg_timeseries(
    trade_date: date, strike: float, right: str,
    start_time: dtime, end_time: dtime,
) -> list:
    """Fetch 5-min bid/ask for a specific option leg (cached)."""
    s = f"{start_time.hour:02d}{start_time.minute:02d}"
    e = f"{end_time.hour:02d}{end_time.minute:02d}"
    cache_file = (
        CACHE_DIR / "legs"
        / f"{trade_date.isoformat()}_{right}_{strike}_{s}_{e}.json"
    )

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    data = theta_request("/v3/option/history/greeks/first_order", {
        "symbol": "SPXW",
        "expiration": trade_date.strftime("%Y%m%d"),
        "right": right,
        "date": trade_date.strftime("%Y%m%d"),
        "strike": f"{strike:.3f}",
        "interval": "5m",
        "start_time": start_time.strftime("%H:%M:%S"),
        "end_time": end_time.strftime("%H:%M:%S"),
        "format": "json",
    })

    if not data:
        return []

    response = data.get("response", data if isinstance(data, list) else [])
    points = []
    for row in response:
        if not isinstance(row, dict):
            continue
        for dp in row.get("data", []):
            points.append({
                "timestamp": dp.get("timestamp", ""),
                "bid": dp.get("bid", 0),
                "ask": dp.get("ask", 0),
            })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(points, f)

    return points


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spx_daily_features(daily_spx: pd.DataFrame, trade_date: date) -> dict:
    """8 daily SPX features using data strictly before trade_date."""
    defaults = dict.fromkeys([
        "rv_close_5d", "rv_close_10d", "rv_close_20d", "atr_14d_pct",
        "gap_pct", "prior_day_range_pct", "prior_day_return", "sma_20d_dist",
    ], 0.0)
    defaults["rv_close_5d"] = defaults["rv_close_10d"] = defaults["rv_close_20d"] = 15.0
    defaults["atr_14d_pct"] = 0.5

    hist = daily_spx[daily_spx.index.date < trade_date]
    if len(hist) < 25:
        return defaults

    closes = hist["Close"].values.astype(float)
    highs = hist["High"].values.astype(float)
    lows = hist["Low"].values.astype(float)
    n = len(closes)

    # Log returns
    log_ret = np.log(closes[1:] / closes[:-1])

    # Realized volatility (annualized %)
    rv_5d = float(np.std(log_ret[-5:]) * np.sqrt(252) * 100) if len(log_ret) >= 5 else 15.0
    rv_10d = float(np.std(log_ret[-10:]) * np.sqrt(252) * 100) if len(log_ret) >= 10 else 15.0
    rv_20d = float(np.std(log_ret[-20:]) * np.sqrt(252) * 100) if len(log_ret) >= 20 else 15.0

    # ATR (14-day)
    if n >= 16:
        h = highs[n - 14 : n]
        lo = lows[n - 14 : n]
        pc = closes[n - 15 : n - 1]
        tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
        atr_14 = float(np.mean(tr))
        atr_14d_pct = (atr_14 / closes[-1]) * 100
    else:
        atr_14d_pct = 0.5

    # Gap: today's open vs yesterday's close
    today_mask = daily_spx.index.date == trade_date
    if today_mask.any():
        today_open = float(daily_spx[today_mask]["Open"].values[0])
        gap_pct = ((today_open - closes[-1]) / closes[-1]) * 100
    else:
        gap_pct = 0.0

    # Prior day range and return
    prior_day_range_pct = ((highs[-1] - lows[-1]) / closes[-1]) * 100
    prior_day_return = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if n >= 2 else 0.0

    # SMA distance
    sma_20 = np.mean(closes[-20:])
    sma_20d_dist = ((closes[-1] - sma_20) / sma_20) * 100

    return {
        "rv_close_5d": rv_5d,
        "rv_close_10d": rv_10d,
        "rv_close_20d": rv_20d,
        "atr_14d_pct": float(atr_14d_pct),
        "gap_pct": float(gap_pct),
        "prior_day_range_pct": float(prior_day_range_pct),
        "prior_day_return": float(prior_day_return),
        "sma_20d_dist": float(sma_20d_dist),
    }


def compute_vix_features(
    daily_vix: pd.DataFrame,
    daily_vix3m: pd.DataFrame | None,
    trade_date: date,
) -> dict:
    """6 VIX features using data strictly before trade_date."""
    defaults = {
        "vix_level": 18.5, "vix_change_1d": 0.0, "vix_change_5d": 0.0,
        "vix_rank_30d": 50.0, "vix_sma_10d_dist": 0.0, "vix_term_slope": 0.0,
    }

    hist = daily_vix[daily_vix.index.date < trade_date]
    if len(hist) < 30:
        return defaults

    closes = hist["Close"].values.astype(float)
    vix_now = closes[-1]

    vix_change_1d = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 else 0.0
    vix_change_5d = ((closes[-1] - closes[-6]) / closes[-6]) * 100 if len(closes) >= 6 else 0.0

    last_30 = closes[-30:]
    vix_rank_30d = float(np.sum(last_30 < vix_now) / len(last_30) * 100)

    sma_10 = np.mean(closes[-10:])
    vix_sma_10d_dist = ((vix_now - sma_10) / sma_10) * 100

    vix_term_slope = 0.0
    if daily_vix3m is not None and not daily_vix3m.empty:
        hist3m = daily_vix3m[daily_vix3m.index.date < trade_date]
        if not hist3m.empty:
            vix3m = float(hist3m["Close"].values[-1])
            if vix_now > 0:
                vix_term_slope = (vix3m - vix_now) / vix_now

    return {
        "vix_level": float(vix_now),
        "vix_change_1d": float(vix_change_1d),
        "vix_change_5d": float(vix_change_5d),
        "vix_rank_30d": vix_rank_30d,
        "vix_sma_10d_dist": float(vix_sma_10d_dist),
        "vix_term_slope": float(vix_term_slope),
    }


def compute_intraday_features(
    intraday_df: pd.DataFrame | None,
    entry_time: dtime,
    atr_14d_pct: float,
) -> dict:
    """Intraday SPX features from 5-min candles up to entry_time.

    Returns basic intraday features + 11 technical indicators.
    Cross-features (gk_rv_vs_atm_iv, range_vs_vix_range, range_exhaustion_pct)
    are computed in process_day() after all component features are available.
    """
    ti_defaults = {
        "adx": np.nan, "efficiency_ratio": np.nan, "choppiness_index": np.nan,
        "rsi_dist_50": np.nan, "linreg_r2": np.nan, "atr_ratio": np.nan,
        "bbw_percentile": np.nan, "ttm_squeeze": np.nan, "bars_in_squeeze": np.nan,
        "garman_klass_rv": np.nan, "orb_failure": np.nan,
    }
    defaults = {
        "intraday_rv": 10.0, "move_from_open_pct": 0.0, "orb_range_pct": 0.0,
        "orb_contained": 1.0, "range_exhaustion": 0.5, "high_low_range_pct": 0.0,
        "momentum_30min": 0.0, "trend_slope_norm": 0.0,
        **ti_defaults,
    }

    if intraday_df is None or intraday_df.empty:
        return defaults

    df = intraday_df.copy()

    # Parse timestamps
    ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
    if ts_col is None:
        return defaults

    df["ts"] = pd.to_datetime(df[ts_col])
    base_date = df["ts"].iloc[0].normalize()
    entry_dt = base_date + pd.Timedelta(hours=entry_time.hour, minutes=entry_time.minute)
    open_dt = base_date + pd.Timedelta(hours=9, minutes=30)

    df = df[(df["ts"] >= open_dt) & (df["ts"] <= entry_dt)].copy()
    if len(df) < 3:
        return defaults

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    open_price = float(df.iloc[0]["open"])
    current_price = float(df.iloc[-1]["close"])
    session_high = float(df["high"].max())
    session_low = float(df["low"].min())

    if open_price <= 0:
        return defaults

    # Intraday realized vol (annualized from 5-min log returns)
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    intraday_rv = float(np.sqrt(np.sum(log_ret**2) * 252) * 100) if len(log_ret) >= 2 else 10.0

    move_from_open_pct = ((current_price - open_price) / open_price) * 100

    # Opening range (first 30 min = first 6 five-min bars)
    orb = df.head(6)
    orb_high = float(orb["high"].max())
    orb_low = float(orb["low"].min())
    orb_range_pct = ((orb_high - orb_low) / open_price) * 100
    orb_contained = 1.0 if orb_low <= current_price <= orb_high else 0.0

    # Range exhaustion
    high_low_range_pct = ((session_high - session_low) / open_price) * 100
    range_exhaustion = high_low_range_pct / atr_14d_pct if atr_14d_pct > 0 else 0.5

    # Momentum (last 30 min = last 6 bars)
    if len(df) >= 6:
        price_30ago = float(df.iloc[-6]["close"])
        momentum_30min = ((current_price - price_30ago) / price_30ago) * 100
    else:
        momentum_30min = 0.0

    # Trend slope (normalized)
    if len(df) >= 5:
        x = np.arange(len(df))
        slope = np.polyfit(x, df["close"].values.astype(float), 1)[0]
        trend_slope_norm = (slope / open_price) * 100 * len(df)
    else:
        trend_slope_norm = 0.0

    # ── Technical Indicators (11) ────────────────────────────────────────────
    tech = compute_all_technical_indicators(df[["open", "high", "low", "close"]], orb_bars=6)

    result = {
        "intraday_rv": intraday_rv,
        "move_from_open_pct": float(move_from_open_pct),
        "orb_range_pct": float(orb_range_pct),
        "orb_contained": orb_contained,
        "range_exhaustion": float(min(range_exhaustion, 3.0)),
        "high_low_range_pct": float(high_low_range_pct),
        "momentum_30min": float(momentum_30min),
        "trend_slope_norm": float(trend_slope_norm),
    }
    result.update(tech)
    return result


def compute_time_features(trade_date: date, entry_time: dtime) -> dict:
    """5 time/calendar features with cyclical encoding."""
    minutes_since_open = (entry_time.hour - 9) * 60 + (entry_time.minute - 30)
    total_minutes = 390  # 6.5 hours

    frac = minutes_since_open / total_minutes
    dow = trade_date.weekday()

    return {
        "time_sin": float(np.sin(2 * np.pi * frac)),
        "time_cos": float(np.cos(2 * np.pi * frac)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 5)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 5)),
        "minutes_to_close": float((16 - entry_time.hour) * 60 - entry_time.minute),
    }


def compute_calendar_features(trade_date: date) -> dict:
    """4 calendar/event features."""
    is_fomc_day = 1.0 if trade_date in FOMC_DATES else 0.0

    is_fomc_week = 0.0
    for fd in FOMC_DATES:
        if abs((trade_date - fd).days) <= 3:
            is_fomc_week = 1.0
            break

    past_fomc = [d for d in FOMC_DATES if d <= trade_date]
    days_since_fomc = float(min((trade_date - max(past_fomc)).days, 60)) if past_fomc else 30.0

    return {
        "is_fomc_day": is_fomc_day,
        "is_fomc_week": is_fomc_week,
        "is_friday": 1.0 if trade_date.weekday() == 4 else 0.0,
        "days_since_fomc": days_since_fomc,
    }


def compute_option_features(puts: list, calls: list) -> dict:
    """4 option-derived features from chain at entry time."""
    defaults = {
        "atm_iv": 15.0, "iv_skew_10d": 2.0,
        "short_put_delta": -0.10, "short_call_delta": 0.10,
    }

    if not puts or not calls:
        return defaults

    # Find ATM put (closest to -0.50 delta)
    valid_puts = [p for p in puts if p.get("delta") is not None and p["delta"] < 0]
    valid_calls = [c for c in calls if c.get("delta") is not None and c["delta"] > 0]

    if not valid_puts or not valid_calls:
        return defaults

    atm_put = min(valid_puts, key=lambda x: abs(x["delta"] + 0.50))
    iv_raw = atm_put.get("implied_vol") or 0
    atm_iv = iv_raw * 100 if 0 < iv_raw < 5 else iv_raw  # Handle decimal vs %
    if atm_iv <= 0:
        atm_iv = 15.0

    # 10-delta options
    put_10d = min(valid_puts, key=lambda x: abs(x["delta"] + TARGET_DELTA))
    call_10d = min(valid_calls, key=lambda x: abs(x["delta"] - TARGET_DELTA))

    p10_iv = (put_10d.get("implied_vol") or 0)
    c10_iv = (call_10d.get("implied_vol") or 0)
    if 0 < p10_iv < 5:
        p10_iv *= 100
    if 0 < c10_iv < 5:
        c10_iv *= 100
    iv_skew = p10_iv - c10_iv if (p10_iv > 0 and c10_iv > 0) else 2.0

    return {
        "atm_iv": float(atm_iv),
        "iv_skew_10d": float(iv_skew),
        "short_put_delta": float(put_10d.get("delta") or -0.10),
        "short_call_delta": float(call_10d.get("delta") or 0.10),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_ic_strikes(puts: list, calls: list, wing_width: int) -> dict | None:
    """Find 10-delta IC strikes and compute entry credit."""
    if not puts or not calls:
        return None

    valid_puts = [p for p in puts if (p.get("delta") or 0) < 0 and (p.get("bid") or 0) > 0]
    valid_calls = [c for c in calls if (c.get("delta") or 0) > 0 and (c.get("bid") or 0) > 0]

    if not valid_puts or not valid_calls:
        return None

    short_put = min(valid_puts, key=lambda x: abs(x["delta"] + TARGET_DELTA))
    short_call = min(valid_calls, key=lambda x: abs(x["delta"] - TARGET_DELTA))

    sp = short_put["strike"]
    sc = short_call["strike"]
    lp_target = sp - wing_width
    lc_target = sc + wing_width

    # Find long legs (buy at ask, need ask > 0)
    all_puts = [p for p in puts if p.get("ask", 0) is not None]
    all_calls = [c for c in calls if c.get("ask", 0) is not None]

    long_put = min(all_puts, key=lambda x: abs(x["strike"] - lp_target), default=None)
    long_call = min(all_calls, key=lambda x: abs(x["strike"] - lc_target), default=None)

    if not long_put or not long_call:
        return None
    if abs(long_put["strike"] - lp_target) > 5 or abs(long_call["strike"] - lc_target) > 5:
        return None

    # Conservative: sell at bid, buy at ask
    put_credit = (short_put.get("bid") or 0) - (long_put.get("ask") or 0)
    call_credit = (short_call.get("bid") or 0) - (long_call.get("ask") or 0)
    entry_credit = put_credit + call_credit

    if entry_credit <= 0:
        return None

    return {
        "short_put_strike": sp,
        "long_put_strike": long_put["strike"],
        "short_call_strike": sc,
        "long_call_strike": long_call["strike"],
        "entry_credit": entry_credit,
        "underlying_price": short_put.get("underlying_price", 0),
    }


def compute_labels(trade_date: date, ic: dict, entry_time: dtime) -> dict:
    """Compute IC profit/loss labels by tracking leg prices to 3 PM."""
    exit_time = dtime(15, 0)
    credit = ic["entry_credit"]

    label_defaults = {
        "entry_credit": credit,
        "underlying_price": ic["underlying_price"],
        "short_put_strike": ic["short_put_strike"],
        "short_call_strike": ic["short_call_strike"],
        "hit_25pct": 0, "hit_50pct": 0,
        "time_to_25pct_min": None, "time_to_50pct_min": None,
        "pnl_at_close": None, "max_adverse_pct": None,
        "win_50pct_or_close": None,
    }

    legs = [
        (ic["short_put_strike"], "put", -1),
        (ic["long_put_strike"], "put", 1),
        (ic["short_call_strike"], "call", -1),
        (ic["long_call_strike"], "call", 1),
    ]

    leg_data = {}
    for strike, right, direction in legs:
        data = fetch_leg_timeseries(trade_date, strike, right, entry_time, exit_time)
        if not data:
            return label_defaults
        leg_data[(strike, right, direction)] = data

    min_len = min(len(d) for d in leg_data.values())
    if min_len < 2:
        return label_defaults

    # Compute close debit at each timestamp
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
    pnls = credit - debits

    target_25_debit = credit * 0.75
    target_50_debit = credit * 0.50

    hit_25_idx = np.where(debits <= target_25_debit)[0]
    hit_50_idx = np.where(debits <= target_50_debit)[0]

    hit_25 = 1 if len(hit_25_idx) > 0 else 0
    hit_50 = 1 if len(hit_50_idx) > 0 else 0

    pnl_at_close = float(pnls[-1])
    max_adverse_debit = float(np.max(debits))
    max_adverse_pct = ((max_adverse_debit - credit) / credit) * 100 if credit > 0 else 0

    return {
        "entry_credit": float(credit),
        "underlying_price": float(ic["underlying_price"]),
        "short_put_strike": float(ic["short_put_strike"]),
        "short_call_strike": float(ic["short_call_strike"]),
        "hit_25pct": hit_25,
        "hit_50pct": hit_50,
        "time_to_25pct_min": float(hit_25_idx[0] * 5) if hit_25 else None,
        "time_to_50pct_min": float(hit_50_idx[0] * 5) if hit_50 else None,
        "pnl_at_close": pnl_at_close,
        "max_adverse_pct": float(max_adverse_pct),
        "win_50pct_or_close": 1 if (hit_50 or pnl_at_close > 0) else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_day(
    trade_date: date,
    entry_time: dtime,
    wing_width: int,
    daily_data: dict,
    features_only: bool = False,
    use_theta: bool = True,
) -> dict | None:
    """Process a single trading day: compute 37 features + labels."""
    suffix = f"_{entry_time.hour:02d}{entry_time.minute:02d}_w{wing_width}"
    cache_file = DAILY_CACHE_DIR / f"{trade_date.isoformat()}{suffix}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    row = {
        "trade_date": trade_date.isoformat(),
        "entry_hour": entry_time.hour,
        "wing_width": wing_width,
    }

    # ── SPX Daily Features (8) ──────────────────────────────────────────────
    spx_daily = daily_data.get("SPX")
    if spx_daily is None or spx_daily.empty:
        return None
    row.update(compute_spx_daily_features(spx_daily, trade_date))

    # ── VIX Features (6) ────────────────────────────────────────────────────
    vix_daily = daily_data.get("VIX")
    vix3m_daily = daily_data.get("VIX3M")
    if vix_daily is not None and not vix_daily.empty:
        row.update(compute_vix_features(vix_daily, vix3m_daily, trade_date))

    # ── SPX Intraday Features (10) ──────────────────────────────────────────
    intraday_df = fetch_spx_intraday(trade_date) if use_theta else None
    atr_pct = row.get("atr_14d_pct", 0.5)
    row.update(compute_intraday_features(intraday_df, entry_time, atr_pct))

    # Override gap_pct with intraday open if available
    if intraday_df is not None and not intraday_df.empty:
        first_open = pd.to_numeric(intraday_df.iloc[0].get("open", 0), errors="coerce")
        hist = spx_daily[spx_daily.index.date < trade_date]
        if first_open > 0 and not hist.empty:
            prev_close = float(hist["Close"].values[-1])
            if prev_close > 0:
                row["gap_pct"] = float(((first_open - prev_close) / prev_close) * 100)

    # ── Time/Calendar Features (5 + 4) ──────────────────────────────────────
    row.update(compute_time_features(trade_date, entry_time))
    row.update(compute_calendar_features(trade_date))

    # ── Option-Derived Features (4) ─────────────────────────────────────────
    puts, calls = [], []
    if use_theta:
        puts = fetch_option_chain(trade_date, entry_time, "put")
        calls = fetch_option_chain(trade_date, entry_time, "call")
    row.update(compute_option_features(puts, calls))

    # ── Derived Cross-Features (3) ───────────────────────────────────────────
    gk_rv = row.get("garman_klass_rv")
    atm_iv_val = row.get("atm_iv", 15.0)
    vix_val = row.get("vix_level", 18.5)
    atr_val = row.get("atr_14d_pct", 0.5)
    hl_range = row.get("high_low_range_pct", 0.0)

    if gk_rv is not None and not (isinstance(gk_rv, float) and np.isnan(gk_rv)) and atm_iv_val > 0:
        row["gk_rv_vs_atm_iv"] = gk_rv / atm_iv_val
    else:
        row["gk_rv_vs_atm_iv"] = None

    if vix_val > 0 and hl_range > 0:
        vix_daily_range = vix_val / np.sqrt(252)
        row["range_vs_vix_range"] = hl_range / vix_daily_range
    else:
        row["range_vs_vix_range"] = None

    if atr_val > 0:
        row["range_exhaustion_pct"] = hl_range / atr_val
    else:
        row["range_exhaustion_pct"] = None

    # ── Labels ──────────────────────────────────────────────────────────────
    if not features_only and use_theta:
        ic = find_ic_strikes(puts, calls, wing_width)
        if ic:
            row.update(compute_labels(trade_date, ic, entry_time))
        else:
            row["entry_credit"] = None
            row["win_50pct_or_close"] = None

    # Cache
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json_row = {}
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            json_row[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            json_row[k] = float(v)
        else:
            json_row[k] = v

    with open(cache_file, "w") as f:
        json.dump(json_row, f, indent=2)

    return json_row


def print_summary(df: pd.DataFrame, features_only: bool):
    """Print dataset summary statistics."""
    FEATURE_COLS = [
        "rv_close_5d", "rv_close_10d", "rv_close_20d", "atr_14d_pct",
        "gap_pct", "prior_day_range_pct", "prior_day_return", "sma_20d_dist",
        "vix_level", "vix_change_1d", "vix_change_5d", "vix_rank_30d",
        "vix_sma_10d_dist", "vix_term_slope",
        "intraday_rv", "move_from_open_pct", "orb_range_pct", "orb_contained",
        "range_exhaustion", "high_low_range_pct",
        "momentum_30min", "trend_slope_norm",
        "time_sin", "time_cos", "dow_sin", "dow_cos", "minutes_to_close",
        "is_fomc_day", "is_fomc_week", "is_friday", "days_since_fomc",
        "atm_iv", "iv_skew_10d", "short_put_delta", "short_call_delta",
        # Technical indicators (11)
        "adx", "efficiency_ratio", "choppiness_index", "rsi_dist_50",
        "linreg_r2", "atr_ratio", "bbw_percentile", "ttm_squeeze",
        "bars_in_squeeze", "garman_klass_rv", "orb_failure",
        # Derived cross-features (3)
        "gk_rv_vs_atm_iv", "range_vs_vix_range", "range_exhaustion_pct",
    ]

    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]

    logger.info("=" * 65)
    logger.info("DATASET SUMMARY")
    logger.info(f"  Total rows:     {len(df)}")
    logger.info(f"  Date range:     {df['trade_date'].min()} to {df['trade_date'].max()}")
    logger.info(f"  Features:       {len(present)} / {len(FEATURE_COLS)} expected")
    if missing:
        logger.info(f"  Missing:        {missing}")

    logger.info(f"\n  Feature stats:")
    for col in present:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                logger.info(
                    f"    {col:28s} mean={vals.mean():8.3f}  "
                    f"std={vals.std():7.3f}  "
                    f"min={vals.min():8.3f}  max={vals.max():8.3f}"
                )

    if not features_only and "win_50pct_or_close" in df.columns:
        valid = df[df["win_50pct_or_close"].notna()]
        if len(valid) > 0:
            logger.info(f"\n  LABEL STATS ({len(valid)} valid / {len(df)} total):")
            logger.info(f"    Win rate (50% TP or close):  {valid['win_50pct_or_close'].mean():.1%}")
            logger.info(f"    Hit 25% TP:                  {valid['hit_25pct'].mean():.1%}")
            logger.info(f"    Hit 50% TP:                  {valid['hit_50pct'].mean():.1%}")
            logger.info(f"    Avg entry credit:            ${valid['entry_credit'].mean():.2f}")
            if "pnl_at_close" in valid.columns:
                pnl = valid["pnl_at_close"].dropna()
                if len(pnl) > 0:
                    logger.info(f"    Avg P&L at 3PM close:        ${pnl.mean():.2f}")
                    logger.info(f"    Median P&L at 3PM close:     ${pnl.median():.2f}")


def generate_entry_times(start_str: str, end_str: str, interval_min: int) -> list:
    """Generate list of entry times from start to end at given interval."""
    sh, sm = map(int, start_str.split(':'))
    eh, em = map(int, end_str.split(':'))
    times = []
    cur = sh * 60 + sm
    end = eh * 60 + em
    while cur <= end:
        times.append(dtime(cur // 60, cur % 60))
        cur += interval_min
    return times


def main():
    parser = argparse.ArgumentParser(description="Build 0DTE IC training dataset")
    parser.add_argument("--start", default="2023-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--entry-hour", type=int, default=12, help="Entry hour ET (12 or 13)")
    parser.add_argument("--wing-width", type=int, default=25, help="Wing width in dollars")
    parser.add_argument("--features-only", action="store_true", help="Skip label computation")
    parser.add_argument("--no-theta", action="store_true", help="Skip ThetaData (yfinance only)")
    parser.add_argument("--max-days", type=int, default=0, help="Limit days (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--start-time", type=str, default=None,
                        help="Entry window start HH:MM (enables multi-entry mode)")
    parser.add_argument("--end-time", type=str, default=None,
                        help="Entry window end HH:MM")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between entry times (default: 5)")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    wing_width = args.wing_width
    use_theta = not args.no_theta

    # Multi-entry mode vs single-entry mode
    multi_entry = args.start_time is not None and args.end_time is not None
    if multi_entry:
        entry_times = generate_entry_times(args.start_time, args.end_time, args.interval)
        logger.info("=" * 65)
        logger.info("0DTE Iron Condor Training Dataset Builder (MULTI-ENTRY)")
        logger.info("=" * 65)
        logger.info(f"  Date range:   {start_date} to {end_date}")
        logger.info(f"  Entry window: {args.start_time} to {args.end_time} every {args.interval}min")
        logger.info(f"  Entry times:  {len(entry_times)} slots")
    else:
        entry_times = [dtime(args.entry_hour, 0)]
        logger.info("=" * 65)
        logger.info("0DTE Iron Condor Training Dataset Builder")
        logger.info("=" * 65)
        logger.info(f"  Date range:   {start_date} to {end_date}")
        logger.info(f"  Entry time:   {entry_times[0].hour}:{entry_times[0].minute:02d} ET")

    logger.info(f"  Wing width:   ${wing_width}")
    logger.info(f"  ThetaData:    {'enabled' if use_theta else 'DISABLED (yfinance only)'}")
    logger.info(f"  Labels:       {'enabled' if not args.features_only else 'DISABLED'}")
    logger.info(f"  Workers:      {args.workers}")

    # Create directories
    for d in [TRAINING_DATA_DIR, CACHE_DIR, DAILY_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Check ThetaData connection
    if use_theta:
        if theta_available():
            logger.info("  ThetaData:    Connected")
        else:
            logger.error("ThetaData Terminal not running! Start it or use --no-theta")
            sys.exit(1)

    # Step 1: Download daily data
    logger.info("\n" + "=" * 65)
    logger.info("Step 1: Downloading daily data from yfinance...")
    daily_data = download_daily_data(
        start=(start_date - timedelta(days=60)).isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
    )

    if "SPX" not in daily_data:
        logger.error("Failed to download SPX data. Cannot continue.")
        sys.exit(1)

    # Step 2: Get trading days
    logger.info("\n" + "=" * 65)
    logger.info("Step 2: Getting trading days...")
    trading_days = get_trading_days(start_date, end_date)
    logger.info(f"  Found {len(trading_days)} trading days")

    if args.max_days > 0:
        trading_days = trading_days[: args.max_days]
        logger.info(f"  Limited to {args.max_days} days")

    # Step 3: Process each day (parallel)
    n_workers = args.workers if use_theta else 1
    total_tasks = len(trading_days) * len(entry_times)
    logger.info("\n" + "=" * 65)
    logger.info(f"Step 3: Processing {len(trading_days)} days x {len(entry_times)} entry times = {total_tasks} tasks with {n_workers} workers...")

    # Count already-cached (date, entry_time) combos
    pre_cached = 0
    for td in trading_days:
        for et in entry_times:
            suffix = f"_{et.hour:02d}{et.minute:02d}_w{wing_width}"
            if (DAILY_CACHE_DIR / f"{td.isoformat()}{suffix}.json").exists():
                pre_cached += 1

    remaining = total_tasks - pre_cached
    logger.info(f"  {pre_cached} already cached, {remaining} to fetch")

    # Reset progress counters
    _progress["done"] = 0
    _progress["errors"] = 0
    _progress["cached"] = pre_cached
    t_start = time_module.time()

    def _process_one_entry(args_tuple):
        """Worker function for one (trading_day, entry_time) pair."""
        td, et = args_tuple
        try:
            row = process_day(td, et, wing_width, daily_data, args.features_only, use_theta)
            if row and multi_entry:
                row["entry_time"] = f"{et.hour:02d}:{et.minute:02d}"
            with _progress_lock:
                _progress["done"] += 1
                done = _progress["done"]
                if done % 100 == 0 or done == 1:
                    elapsed = time_module.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    left = total_tasks - done - _progress["cached"]
                    eta = left / rate if rate > 0 else 0
                    logger.info(
                        f"  [{done + _progress['cached']:5d}/{total_tasks}] "
                        f"{td} {et.hour:02d}:{et.minute:02d} ({_progress['errors']} errors, ETA {eta/60:.1f}m)"
                    )
            return row
        except Exception as e:
            with _progress_lock:
                _progress["errors"] += 1
            logger.error(f"  {td} {et.hour:02d}:{et.minute:02d}: {e}")
            return None

    # Build work items: all (date, entry_time) pairs
    work_items = [(td, et) for td in trading_days for et in entry_times]

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_entry, item): item for item in work_items}
        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)

    elapsed = time_module.time() - t_start
    errors = _progress["errors"]
    logger.info(
        f"\n  Done: {len(results)} rows, {errors} errors, "
        f"{_progress['cached']} pre-cached ({elapsed:.1f}s)"
    )

    # Step 4: Save
    logger.info("\n" + "=" * 65)
    logger.info("Step 4: Assembling dataset...")

    df = pd.DataFrame(results)
    sort_cols = ["trade_date", "entry_time"] if multi_entry else ["trade_date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if multi_entry:
        output_file = TRAINING_DATA_DIR / f"training_dataset_multi_w{wing_width}.csv"
    else:
        output_file = TRAINING_DATA_DIR / f"training_dataset_h{args.entry_hour}_w{wing_width}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"  Saved {len(df)} rows to {output_file}")

    print_summary(df, args.features_only)

    logger.info(f"\nDataset saved to: {output_file}")


if __name__ == "__main__":
    main()
