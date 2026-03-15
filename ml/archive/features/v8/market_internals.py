"""
Market internals features for v8 model (~15 features).

Computes VWAP, intraday momentum, range analysis, and SPX-VIX dynamics
from 1-minute SPX bars. Uses numpy/pandas only.
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def compute_market_internals_features(spx_1m: pd.DataFrame, vix_1m: pd.DataFrame = None) -> dict:
    """
    Compute ~15 market internals features from 1-min intraday data.

    Args:
        spx_1m: DataFrame with [timestamp, open, high, low, close, volume] - 1-min SPX bars today
        vix_1m: Optional DataFrame with [timestamp, open, high, low, close] - 1-min VIX bars today
    Returns:
        dict of feature_name -> float
    """
    features = {}

    if spx_1m is None or spx_1m.empty or len(spx_1m) < 3:
        return {k: np.nan for k in _all_feature_names()}

    spx = spx_1m.copy().reset_index(drop=True)
    o = spx['open'].values.astype(np.float64)
    h = spx['high'].values.astype(np.float64)
    lo = spx['low'].values.astype(np.float64)
    c = spx['close'].values.astype(np.float64)
    n = len(spx)

    # ---- VWAP (3) ----
    # Equal-weighted VWAP: cumulative average of typical price (H+L+C)/3
    typical = (h + lo + c) / 3.0
    cum_vwap = np.cumsum(typical) / np.arange(1, n + 1, dtype=np.float64)
    vwap_current = cum_vwap[-1]
    close_current = c[-1]

    # 1. vwap_distance_pct
    if vwap_current != 0:
        features['vwap_distance_pct'] = (close_current - vwap_current) / vwap_current * 100
    else:
        features['vwap_distance_pct'] = np.nan

    # 2. vwap_slope - linear regression slope of VWAP over last 30 bars, normalized by price
    lookback = min(30, n)
    vwap_window = cum_vwap[-lookback:]
    if lookback >= 3:
        x = np.arange(lookback, dtype=np.float64)
        x_mean = x.mean()
        y_mean = vwap_window.mean()
        slope = np.sum((x - x_mean) * (vwap_window - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-10)
        # Normalize by price (slope per bar as % of price)
        features['vwap_slope'] = slope / max(close_current, 1e-6) * 100
    else:
        features['vwap_slope'] = np.nan

    # 3. spx_in_value_area - 1.0 if within ±0.3% of VWAP
    if vwap_current != 0:
        pct_from_vwap = abs(close_current - vwap_current) / vwap_current * 100
        features['spx_in_value_area'] = 1.0 if pct_from_vwap <= 0.3 else 0.0
    else:
        features['spx_in_value_area'] = np.nan

    # ---- Build 5-min bars from 1-min data ----
    bars_5m = _resample_to_5min(spx)
    n5 = len(bars_5m)

    # ---- Intraday Momentum (6) ----

    # 4. intraday_rsi_14
    if n5 >= 15:
        features['intraday_rsi_14'] = _rsi(bars_5m['close'].values, 14)
    else:
        features['intraday_rsi_14'] = np.nan

    # 5. intraday_adx
    if n5 >= 16:
        features['intraday_adx'] = _adx(
            bars_5m['high'].values, bars_5m['low'].values, bars_5m['close'].values, 14
        )
    else:
        features['intraday_adx'] = np.nan

    # 6. intraday_efficiency_ratio
    if n5 >= 10:
        features['intraday_efficiency_ratio'] = _efficiency_ratio(bars_5m['close'].values, 10)
    else:
        features['intraday_efficiency_ratio'] = np.nan

    # 7. intraday_choppiness
    if n5 >= 14:
        features['intraday_choppiness'] = _choppiness(
            bars_5m['high'].values, bars_5m['low'].values, bars_5m['close'].values, 14
        )
    else:
        features['intraday_choppiness'] = np.nan

    # 8. return_autocorrelation - lag-1 autocorrelation of 5-min returns
    if n5 >= 10:
        rets = np.diff(bars_5m['close'].values) / bars_5m['close'].values[:-1]
        if len(rets) >= 5 and np.std(rets) > 1e-10:
            features['return_autocorrelation'] = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])
        else:
            features['return_autocorrelation'] = np.nan
    else:
        features['return_autocorrelation'] = np.nan

    # 9. drawdown_from_open_pct
    high_since_open = np.maximum.accumulate(h)
    hso = high_since_open[-1]
    if hso > 0:
        features['drawdown_from_open_pct'] = (close_current - hso) / hso * 100
    else:
        features['drawdown_from_open_pct'] = np.nan

    # ---- Range Analysis (4) ----
    # Opening range = first 30 min = first 6 bars (indices 0..5) for 5-min,
    # but we use 1-min data: first 30 bars (indices 0..29)
    or_bars = min(30, n)  # first 30 1-min bars
    or_high = np.max(h[:or_bars])
    or_low = np.min(lo[:or_bars])
    or_width_pts = or_high - or_low
    open_price = o[0]

    # 10. opening_range_width (as % of open)
    if open_price > 0:
        features['opening_range_width'] = or_width_pts / open_price * 100
    else:
        features['opening_range_width'] = np.nan

    # 11. opening_range_breakout_status
    if n > or_bars:
        if close_current > or_high:
            features['opening_range_breakout_status'] = 1.0
        elif close_current < or_low:
            features['opening_range_breakout_status'] = -1.0
        else:
            features['opening_range_breakout_status'] = 0.0
    else:
        features['opening_range_breakout_status'] = 0.0

    # 12. morning_trend_established - price at bar 60 (1 hour of 1-min) outside OR
    if n >= 60:
        price_1h = c[59]  # 60th 1-min bar (index 59)
        if price_1h > or_high or price_1h < or_low:
            features['morning_trend_established'] = 1.0
        else:
            features['morning_trend_established'] = 0.0
    else:
        features['morning_trend_established'] = np.nan

    # 13. range_expansion_rate
    session_range = np.max(h) - np.min(lo)
    if or_width_pts > 0:
        features['range_expansion_rate'] = session_range / or_width_pts
    else:
        features['range_expansion_rate'] = np.nan

    # ---- SPX-VIX Dynamics (2) ----

    has_vix = vix_1m is not None and not vix_1m.empty and len(vix_1m) >= 3

    # 14. spx_vix_correlation
    if has_vix:
        vix = vix_1m.copy().reset_index(drop=True)
        # Align lengths
        min_len = min(len(spx), len(vix))
        if min_len >= 31:
            spx_c = c[:min_len]
            vix_c = vix['close'].values[:min_len].astype(np.float64)
            spx_rets = np.diff(spx_c) / np.maximum(spx_c[:-1], 1e-10)
            vix_rets = np.diff(vix_c) / np.maximum(vix_c[:-1], 1e-10)
            # Use last 30 bars for rolling correlation
            sr = spx_rets[-30:]
            vr = vix_rets[-30:]
            if np.std(sr) > 1e-10 and np.std(vr) > 1e-10:
                features['spx_vix_correlation'] = float(np.corrcoef(sr, vr)[0, 1])
            else:
                features['spx_vix_correlation'] = np.nan
        else:
            features['spx_vix_correlation'] = np.nan
    else:
        features['spx_vix_correlation'] = np.nan

    # 15. vix_intraday_change_pct
    if has_vix:
        vix = vix_1m.copy().reset_index(drop=True)
        vix_open = vix['open'].iloc[0]
        vix_current = vix['close'].iloc[-1]
        if vix_open > 0:
            features['vix_intraday_change_pct'] = float((vix_current - vix_open) / vix_open * 100)
        else:
            features['vix_intraday_change_pct'] = np.nan
    else:
        features['vix_intraday_change_pct'] = np.nan

    return features


# ---- Helper functions ----

def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Sample every 5th bar from 1-min data to create 5-min OHLC bars."""
    n = len(df)
    if n < 5:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])

    o_arr = df['open'].values.astype(np.float64)
    h_arr = df['high'].values.astype(np.float64)
    lo_arr = df['low'].values.astype(np.float64)
    c_arr = df['close'].values.astype(np.float64)

    n_bars = n // 5
    rows = []
    for i in range(n_bars):
        s = i * 5
        e = s + 5
        rows.append({
            'open': o_arr[s],
            'high': np.max(h_arr[s:e]),
            'low': np.min(lo_arr[s:e]),
            'close': c_arr[e - 1],
        })

    return pd.DataFrame(rows)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI on close prices."""
    if len(close) < period + 1:
        return np.nan

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute True Range array (needs at least 2 bars)."""
    prev_close = np.roll(close, 1)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close[1:]),
            np.abs(low[1:] - prev_close[1:]),
        ),
    )
    return tr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index on OHLC data."""
    n = len(high)
    if n < period + 2:
        return np.nan

    period = min(period, n - 1)

    up_move = np.diff(high)
    down_move = -np.diff(low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)

    if len(tr) < period:
        return np.nan

    # Wilder's smoothing
    atr = np.mean(tr[:period])
    plus_di_sum = np.mean(plus_dm[:period])
    minus_di_sum = np.mean(minus_dm[:period])

    dx_vals = []
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
        plus_di_sum = (plus_di_sum * (period - 1) + plus_dm[i]) / period
        minus_di_sum = (minus_di_sum * (period - 1) + minus_dm[i]) / period

        if atr > 1e-10:
            plus_di = plus_di_sum / atr * 100
            minus_di = minus_di_sum / atr * 100
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx_vals.append(abs(plus_di - minus_di) / di_sum * 100)

    if not dx_vals:
        return np.nan

    # ADX = smoothed average of DX
    adx_val = np.mean(dx_vals[-period:]) if len(dx_vals) >= period else np.mean(dx_vals)
    return float(adx_val)


def _efficiency_ratio(close: np.ndarray, period: int = 10) -> float:
    """Kaufman efficiency ratio: net movement / total path."""
    if len(close) < period + 1:
        return np.nan

    net_change = abs(close[-1] - close[-period - 1])
    total_path = np.sum(np.abs(np.diff(close[-period - 1:])))

    if total_path < 1e-10:
        return np.nan

    return float(net_change / total_path)


def _choppiness(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Choppiness Index (0-100). Higher = choppier = good for IC."""
    n = len(high)
    if n < period + 1:
        return np.nan

    tr = _true_range(high, low, close)
    if len(tr) < period:
        return np.nan

    atr_sum = np.sum(tr[-period:])
    highest = np.max(high[-period:])
    lowest = np.min(low[-period:])
    hl_range = highest - lowest

    if hl_range < 1e-10 or atr_sum < 1e-10:
        return np.nan

    chop = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    return float(chop)


def _all_feature_names() -> list:
    """Return all feature names for nan-filling on bad input."""
    return [
        'vwap_distance_pct', 'vwap_slope', 'spx_in_value_area',
        'intraday_rsi_14', 'intraday_adx', 'intraday_efficiency_ratio',
        'intraday_choppiness', 'return_autocorrelation', 'drawdown_from_open_pct',
        'opening_range_width', 'opening_range_breakout_status',
        'morning_trend_established', 'range_expansion_rate',
        'spx_vix_correlation', 'vix_intraday_change_pct',
    ]
