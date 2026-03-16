"""
Price action & technical indicator features for v8 model (~38 features).

Computes MACD, oscillators, Bollinger Bands, VWAP, EMA ribbons, ATR,
z-scores, candlestick aggregates, multi-timeframe alignment, and
local extrema from 1-minute SPX bars. Pure numpy/pandas.
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def compute_price_action_features(spx_1m: pd.DataFrame, vix_1m: pd.DataFrame = None) -> dict:
    """
    Compute ~38 price action features from 1-min intraday data.

    Args:
        spx_1m: DataFrame with [timestamp, open, high, low, close] - 1-min SPX bars today
        vix_1m: Optional VIX 1-min bars (unused currently, reserved for future)
    Returns:
        dict of feature_name -> float
    """
    if spx_1m is None or spx_1m.empty or len(spx_1m) < 10:
        return {k: np.nan for k in _all_feature_names()}

    spx = spx_1m.copy().reset_index(drop=True)
    o = spx['open'].values.astype(np.float64)
    h = spx['high'].values.astype(np.float64)
    lo = spx['low'].values.astype(np.float64)
    c = spx['close'].values.astype(np.float64)
    n = len(spx)

    features = {}

    # ---- Build multi-timeframe bars ----
    bars_5m = _resample_ohlc(o, h, lo, c, 5)
    bars_15m = _resample_ohlc(o, h, lo, c, 15)

    # ---- MACD (5,13,9) on 1-min closes (3 features) ----
    if n >= 22:
        ema_fast = _ema(c, 5)
        ema_slow = _ema(c, 13)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, 9)
        histogram = macd_line - signal_line

        # 1. macd_histogram
        features['pa_macd_histogram'] = float(histogram[-1])

        # 2. macd_histogram_slope (last 5 bars)
        if len(histogram) >= 5:
            features['pa_macd_slope'] = float(histogram[-1] - histogram[-5])
        else:
            features['pa_macd_slope'] = np.nan

        # 3. macd_zero_cross_bars (bars since last zero cross)
        sign_changes = np.diff(np.sign(histogram))
        cross_idx = np.where(sign_changes != 0)[0]
        if len(cross_idx) > 0:
            features['pa_macd_zero_cross_bars'] = float(len(histogram) - 1 - cross_idx[-1])
        else:
            features['pa_macd_zero_cross_bars'] = float(len(histogram))
    else:
        features['pa_macd_histogram'] = np.nan
        features['pa_macd_slope'] = np.nan
        features['pa_macd_zero_cross_bars'] = np.nan

    # ---- Williams %R (14 bars on 5-min) ----
    n5 = len(bars_5m['close']) if bars_5m else 0
    if n5 >= 14:
        c5 = bars_5m['close']
        h5 = bars_5m['high']
        l5 = bars_5m['low']
        hh = np.max(h5[-14:])
        ll = np.min(l5[-14:])
        if hh - ll > 1e-10:
            features['pa_williams_r'] = float((hh - c5[-1]) / (hh - ll) * -100)
        else:
            features['pa_williams_r'] = np.nan
    else:
        features['pa_williams_r'] = np.nan

    # ---- Stochastic neutral zone (%K distance from 50) ----
    if n5 >= 14:
        c5 = bars_5m['close']
        h5 = bars_5m['high']
        l5 = bars_5m['low']
        hh = np.max(h5[-14:])
        ll = np.min(l5[-14:])
        if hh - ll > 1e-10:
            stoch_k = (c5[-1] - ll) / (hh - ll) * 100
            features['pa_stoch_neutral_dist'] = float(abs(stoch_k - 50))
        else:
            features['pa_stoch_neutral_dist'] = np.nan
    else:
        features['pa_stoch_neutral_dist'] = np.nan

    # ---- Intraday Momentum Index (IMI) ----
    if n >= 14:
        gains = np.where(c[-14:] > o[-14:], c[-14:] - o[-14:], 0.0)
        losses = np.where(c[-14:] < o[-14:], o[-14:] - c[-14:], 0.0)
        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)
        if sum_gains + sum_losses > 1e-10:
            features['pa_imi'] = float(sum_gains / (sum_gains + sum_losses) * 100)
        else:
            features['pa_imi'] = 50.0
    else:
        features['pa_imi'] = np.nan

    # ---- ROC at 5/15/30 bars, ROC divergence ----
    for period in [5, 15, 30]:
        key = f'pa_roc_{period}'
        if n > period and c[-period - 1] > 0:
            features[key] = float((c[-1] - c[-period - 1]) / c[-period - 1] * 100)
        else:
            features[key] = np.nan

    # ROC divergence: sign(ROC5) != sign(ROC30)
    roc5 = features.get('pa_roc_5', np.nan)
    roc30 = features.get('pa_roc_30', np.nan)
    if np.isfinite(roc5) and np.isfinite(roc30):
        features['pa_roc_divergence'] = 1.0 if (roc5 * roc30 < 0) else 0.0
    else:
        features['pa_roc_divergence'] = np.nan

    # ---- Bollinger Bands (20-bar on 5-min) ----
    if n5 >= 20:
        c5 = bars_5m['close']
        bb_mean = np.mean(c5[-20:])
        bb_std = np.std(c5[-20:], ddof=1)
        if bb_std > 1e-10:
            features['pa_bb_position'] = float((c5[-1] - bb_mean) / (2 * bb_std))
        else:
            features['pa_bb_position'] = np.nan

        # BB width change rate
        if n5 >= 25:
            bb_width_now = bb_std / max(bb_mean, 1e-10) * 100
            bb_mean_prev = np.mean(c5[-25:-5])
            bb_std_prev = np.std(c5[-25:-5], ddof=1)
            bb_width_prev = bb_std_prev / max(bb_mean_prev, 1e-10) * 100
            if bb_width_prev > 1e-10:
                features['pa_bb_width_change'] = float((bb_width_now - bb_width_prev) / bb_width_prev)
            else:
                features['pa_bb_width_change'] = np.nan
        else:
            features['pa_bb_width_change'] = np.nan
    else:
        features['pa_bb_position'] = np.nan
        features['pa_bb_width_change'] = np.nan

    # ---- VWAP z-score and touch count (30 bars) ----
    typical = (h + lo + c) / 3.0
    cum_vwap = np.cumsum(typical) / np.arange(1, n + 1, dtype=np.float64)
    vwap_current = cum_vwap[-1]

    lookback = min(30, n)
    vwap_deviations = c[-lookback:] - cum_vwap[-lookback:]
    vwap_std = np.std(vwap_deviations, ddof=1)
    if vwap_std > 1e-10:
        features['pa_vwap_zscore'] = float(vwap_deviations[-1] / vwap_std)
    else:
        features['pa_vwap_zscore'] = np.nan

    # VWAP touch count: close within 0.02% of VWAP in last 30 bars
    if vwap_current > 0:
        touch_threshold = vwap_current * 0.0002
        touches = np.sum(np.abs(c[-lookback:] - cum_vwap[-lookback:]) < touch_threshold)
        features['pa_vwap_touch_count'] = float(touches)
    else:
        features['pa_vwap_touch_count'] = np.nan

    # ---- EMA ribbon width (8/21) and MA alignment (8/21/50) ----
    if n >= 50:
        ema8 = _ema(c, 8)
        ema21 = _ema(c, 21)
        sma50 = _sma(c, 50)

        # EMA ribbon width (normalized by price)
        if c[-1] > 0:
            features['pa_ema_ribbon_width'] = float(abs(ema8[-1] - ema21[-1]) / c[-1] * 100)
        else:
            features['pa_ema_ribbon_width'] = np.nan

        # MA alignment score: +1 if 8>21>50 (bull), -1 if 8<21<50 (bear), 0 mixed
        if ema8[-1] > ema21[-1] > sma50[-1]:
            features['pa_ma_alignment'] = 1.0
        elif ema8[-1] < ema21[-1] < sma50[-1]:
            features['pa_ma_alignment'] = -1.0
        else:
            features['pa_ma_alignment'] = 0.0
    elif n >= 21:
        ema8 = _ema(c, 8)
        ema21 = _ema(c, 21)
        if c[-1] > 0:
            features['pa_ema_ribbon_width'] = float(abs(ema8[-1] - ema21[-1]) / c[-1] * 100)
        else:
            features['pa_ema_ribbon_width'] = np.nan
        features['pa_ma_alignment'] = np.nan
    else:
        features['pa_ema_ribbon_width'] = np.nan
        features['pa_ma_alignment'] = np.nan

    # ---- ATR acceleration, normalized ATR ----
    tr = _true_range(h, lo, c)
    if len(tr) >= 28:
        atr_14_now = np.mean(tr[-14:])
        atr_14_prev = np.mean(tr[-28:-14])
        if atr_14_prev > 1e-10:
            features['pa_atr_acceleration'] = float(atr_14_now / atr_14_prev - 1.0)
        else:
            features['pa_atr_acceleration'] = np.nan

        if c[-1] > 0:
            features['pa_normalized_atr'] = float(atr_14_now / c[-1] * 100)
        else:
            features['pa_normalized_atr'] = np.nan
    elif len(tr) >= 14:
        atr_14_now = np.mean(tr[-14:])
        features['pa_atr_acceleration'] = np.nan
        if c[-1] > 0:
            features['pa_normalized_atr'] = float(atr_14_now / c[-1] * 100)
        else:
            features['pa_normalized_atr'] = np.nan
    else:
        features['pa_atr_acceleration'] = np.nan
        features['pa_normalized_atr'] = np.nan

    # ---- Price z-scores (20/60 bar on 1-min closes) ----
    for window in [20, 60]:
        key = f'pa_price_zscore_{window}'
        if n >= window:
            mean_w = np.mean(c[-window:])
            std_w = np.std(c[-window:], ddof=1)
            if std_w > 1e-10:
                features[key] = float((c[-1] - mean_w) / std_w)
            else:
                features[key] = np.nan
        else:
            features[key] = np.nan

    # Return z-score (5-min return z-scored against session 1-min returns)
    if n >= 10:
        rets_session = np.diff(c) / np.maximum(c[:-1], 1e-10)
        ret_5m = (c[-1] - c[-6]) / max(c[-6], 1e-10)
        ret_std = np.std(rets_session)
        if ret_std > 1e-10:
            features['pa_return_zscore_5m'] = float(ret_5m / ret_std)
        else:
            features['pa_return_zscore_5m'] = np.nan
    else:
        features['pa_return_zscore_5m'] = np.nan

    # ---- Candlestick aggregates (30-bar window) ----
    lb = min(30, n)
    o_lb = o[-lb:]
    h_lb = h[-lb:]
    lo_lb = lo[-lb:]
    c_lb = c[-lb:]
    body = np.abs(c_lb - o_lb)
    full_range = h_lb - lo_lb

    # Doji count (body < 10% of range)
    doji_mask = body < 0.1 * np.maximum(full_range, 1e-10)
    features['pa_doji_count_30'] = float(np.sum(doji_mask))

    # Avg body ratio (body / range)
    valid_range = full_range > 1e-10
    if np.any(valid_range):
        features['pa_avg_body_ratio_30'] = float(np.mean(body[valid_range] / full_range[valid_range]))
    else:
        features['pa_avg_body_ratio_30'] = np.nan

    # Directional ratio (fraction of bullish candles)
    bullish = c_lb > o_lb
    features['pa_directional_ratio_30'] = float(np.mean(bullish))

    # Max body (normalized by price)
    if c[-1] > 0:
        features['pa_max_body_30'] = float(np.max(body) / c[-1] * 100)
    else:
        features['pa_max_body_30'] = np.nan

    # Wick ratio (upper wick + lower wick) / body
    upper_wick = h_lb - np.maximum(c_lb, o_lb)
    lower_wick = np.minimum(c_lb, o_lb) - lo_lb
    total_wick = upper_wick + lower_wick
    body_nonzero = body > 1e-10
    if np.any(body_nonzero):
        features['pa_wick_ratio_30'] = float(np.mean(total_wick[body_nonzero] / body[body_nonzero]))
    else:
        features['pa_wick_ratio_30'] = np.nan

    # ---- Multi-TF features ----
    # Trend alignment: sign of (close - SMA(20)) across 1m, 5m, 15m
    trend_scores = []
    # 1-min trend
    if n >= 20:
        sma20_1m = np.mean(c[-20:])
        trend_scores.append(1.0 if c[-1] > sma20_1m else -1.0)

    # 5-min trend
    if n5 >= 20:
        c5 = bars_5m['close']
        sma20_5m = np.mean(c5[-20:])
        trend_scores.append(1.0 if c5[-1] > sma20_5m else -1.0)
    elif n5 >= 5:
        c5 = bars_5m['close']
        sma_5m = np.mean(c5[-min(n5, 20):])
        trend_scores.append(1.0 if c5[-1] > sma_5m else -1.0)

    # 15-min trend
    n15 = len(bars_15m['close']) if bars_15m else 0
    if n15 >= 5:
        c15 = bars_15m['close']
        sma_15m = np.mean(c15[-min(n15, 20):])
        trend_scores.append(1.0 if c15[-1] > sma_15m else -1.0)

    if trend_scores:
        features['pa_trend_alignment'] = float(np.mean(trend_scores))
    else:
        features['pa_trend_alignment'] = np.nan

    # RSI spread: 1m RSI(14) - 15m RSI(14)
    rsi_1m = _rsi(c, 14) if n >= 15 else np.nan
    if n15 >= 15:
        rsi_15m = _rsi(bars_15m['close'], 14)
    else:
        rsi_15m = np.nan
    if np.isfinite(rsi_1m) and np.isfinite(rsi_15m):
        features['pa_rsi_spread_1m_15m'] = float(rsi_1m - rsi_15m)
    else:
        features['pa_rsi_spread_1m_15m'] = np.nan

    # ADX max across timeframes
    adx_vals = []
    if n5 >= 16:
        adx_5m = _adx(bars_5m['high'], bars_5m['low'], bars_5m['close'], 14)
        if np.isfinite(adx_5m):
            adx_vals.append(adx_5m)
    if n15 >= 16:
        adx_15m = _adx(bars_15m['high'], bars_15m['low'], bars_15m['close'], 14)
        if np.isfinite(adx_15m):
            adx_vals.append(adx_15m)
    features['pa_adx_max_tf'] = float(max(adx_vals)) if adx_vals else np.nan

    # ---- Local extrema (60-bar window) ----
    lb60 = min(60, n)
    c_lb60 = c[-lb60:]

    # Local extrema count using simple comparison with neighbors
    if lb60 >= 5:
        local_max = np.zeros(lb60, dtype=bool)
        local_min = np.zeros(lb60, dtype=bool)
        for i in range(2, lb60 - 2):
            if c_lb60[i] > c_lb60[i-1] and c_lb60[i] > c_lb60[i-2] and \
               c_lb60[i] > c_lb60[i+1] and c_lb60[i] > c_lb60[i+2]:
                local_max[i] = True
            if c_lb60[i] < c_lb60[i-1] and c_lb60[i] < c_lb60[i-2] and \
               c_lb60[i] < c_lb60[i+1] and c_lb60[i] < c_lb60[i+2]:
                local_min[i] = True

        features['pa_extrema_count_60'] = float(np.sum(local_max) + np.sum(local_min))

        # Higher highs / lower lows count
        max_prices = c_lb60[local_max]
        min_prices = c_lb60[local_min]

        if len(max_prices) >= 2:
            hh_count = np.sum(np.diff(max_prices) > 0)
            features['pa_higher_highs_60'] = float(hh_count)
        else:
            features['pa_higher_highs_60'] = 0.0

        if len(min_prices) >= 2:
            ll_count = np.sum(np.diff(min_prices) < 0)
            features['pa_lower_lows_60'] = float(ll_count)
        else:
            features['pa_lower_lows_60'] = 0.0
    else:
        features['pa_extrema_count_60'] = np.nan
        features['pa_higher_highs_60'] = np.nan
        features['pa_lower_lows_60'] = np.nan

    return features


# ---- Helper functions ----

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (returns last value only for efficiency)."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    cumsum = np.cumsum(data)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    result = np.full_like(data, np.nan)
    result[period - 1:] = cumsum[period - 1:] / period
    return result


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range array (n-1 elements)."""
    if len(close) < 2:
        return np.array([])
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ),
    )
    return tr


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """RSI on close prices."""
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


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index."""
    n = len(high)
    if n < period + 2:
        return np.nan
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close)
    if len(tr) < period:
        return np.nan
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
    adx_val = np.mean(dx_vals[-period:]) if len(dx_vals) >= period else np.mean(dx_vals)
    return float(adx_val)


def _resample_ohlc(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
                   period: int) -> dict:
    """Resample 1-min OHLC to N-min OHLC bars."""
    n = len(c)
    n_bars = n // period
    if n_bars < 1:
        return {'open': np.array([]), 'high': np.array([]), 'low': np.array([]),
                'close': np.array([])}
    trimmed = n_bars * period
    o_r = o[:trimmed].reshape(n_bars, period)
    h_r = h[:trimmed].reshape(n_bars, period)
    l_r = lo[:trimmed].reshape(n_bars, period)
    c_r = c[:trimmed].reshape(n_bars, period)
    return {
        'open': o_r[:, 0],
        'high': np.max(h_r, axis=1),
        'low': np.min(l_r, axis=1),
        'close': c_r[:, -1],
    }


def _all_feature_names() -> list:
    """Return all feature names for nan-filling."""
    return [
        'pa_macd_histogram', 'pa_macd_slope', 'pa_macd_zero_cross_bars',
        'pa_williams_r', 'pa_stoch_neutral_dist', 'pa_imi',
        'pa_roc_5', 'pa_roc_15', 'pa_roc_30', 'pa_roc_divergence',
        'pa_bb_position', 'pa_bb_width_change',
        'pa_vwap_zscore', 'pa_vwap_touch_count',
        'pa_ema_ribbon_width', 'pa_ma_alignment',
        'pa_atr_acceleration', 'pa_normalized_atr',
        'pa_price_zscore_20', 'pa_price_zscore_60', 'pa_return_zscore_5m',
        'pa_doji_count_30', 'pa_avg_body_ratio_30', 'pa_directional_ratio_30',
        'pa_max_body_30', 'pa_wick_ratio_30',
        'pa_trend_alignment', 'pa_rsi_spread_1m_15m', 'pa_adx_max_tf',
        'pa_extrema_count_60', 'pa_higher_highs_60', 'pa_lower_lows_60',
    ]
