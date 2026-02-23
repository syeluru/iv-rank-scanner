"""
Technical indicators for iron condor entry timing (v4 model).

All functions are pure numpy, operating on OHLC arrays from 5-minute SPX candles.
Each returns a scalar value. Adaptive periods handle sparse early-morning data.
Returns np.nan when insufficient data (< 3 bars minimum).

SPX is an index — volume is always 0, so all indicators are OHLC-only.
"""

import numpy as np
import pandas as pd


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


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """
    Average Directional Index (0-100). ADX < 20 = range-bound = good for ICs.

    Uses Wilder's smoothing. Adaptive period for early-morning sparse data.
    """
    n = len(high)
    if n < 4:
        return np.nan

    period = min(period, n - 1)

    # Directional movement
    up_move = np.diff(high)
    down_move = -np.diff(low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)

    if len(tr) < period:
        period = len(tr)
    if period < 2:
        return np.nan

    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr_smooth = tr[0]
    plus_dm_smooth = plus_dm[0]
    minus_dm_smooth = minus_dm[0]

    dx_values = []
    for i in range(1, len(tr)):
        atr_smooth = atr_smooth * (1 - alpha) + tr[i] * alpha
        plus_dm_smooth = plus_dm_smooth * (1 - alpha) + plus_dm[i] * alpha
        minus_dm_smooth = minus_dm_smooth * (1 - alpha) + minus_dm[i] * alpha

        if atr_smooth > 0:
            plus_di = 100.0 * plus_dm_smooth / atr_smooth
            minus_di = 100.0 * minus_dm_smooth / atr_smooth
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx_values.append(100.0 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < 2:
        return np.nan

    # Smooth DX to get ADX
    adx = dx_values[0]
    for dx in dx_values[1:]:
        adx = adx * (1 - alpha) + dx * alpha

    return float(np.clip(adx, 0, 100))


def compute_efficiency_ratio(close: np.ndarray, period: int = 14) -> float:
    """
    Kaufman Efficiency Ratio (0-1). ER < 0.3 = choppy = good for ICs.

    ER = |net move| / total path length.
    """
    n = len(close)
    if n < 3:
        return np.nan

    period = min(period, n - 1)
    recent = close[-period - 1:]

    net_move = abs(recent[-1] - recent[0])
    total_path = np.sum(np.abs(np.diff(recent)))

    if total_path == 0:
        return 0.0

    return float(np.clip(net_move / total_path, 0, 1))


def compute_choppiness_index(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> float:
    """
    Choppiness Index (0-100). CI > 61.8 = choppy/range-bound = good for ICs.

    CI = 100 * LOG10(SUM(ATR) / (highest_high - lowest_low)) / LOG10(period)
    """
    n = len(high)
    if n < 4:
        return np.nan

    period = min(period, n - 1)
    tr = _true_range(high, low, close)

    if len(tr) < period:
        period = len(tr)
    if period < 2:
        return np.nan

    atr_sum = np.sum(tr[-period:])
    highest = np.max(high[-period - 1:])
    lowest = np.min(low[-period - 1:])
    hl_range = highest - lowest

    if hl_range <= 0 or period <= 1:
        return np.nan

    ci = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    return float(np.clip(ci, 0, 100))


def compute_rsi_distance_from_50(close: np.ndarray, period: int = 14) -> float:
    """
    abs(RSI - 50). Near 0 = balanced momentum = good for ICs.
    """
    n = len(close)
    if n < 3:
        return np.nan

    period = min(period, n - 1)
    deltas = np.diff(close[-period - 1:])

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    if avg_loss == 0:
        rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return float(abs(rsi - 50.0))


def compute_linreg_r2(close: np.ndarray, period: int = 20) -> float:
    """
    R-squared of linear regression on close prices (0-1). Low R^2 = no trend = good.
    """
    n = len(close)
    if n < 3:
        return np.nan

    period = min(period, n)
    y = close[-period:]
    x = np.arange(period)

    # Manual R^2 (avoids scipy dependency)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0 or ss_yy == 0:
        return 0.0

    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    return float(np.clip(r ** 2, 0, 1))


def compute_atr_ratio(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    fast: int = 7, slow: int = 50,
) -> float:
    """
    ATR(fast) / ATR(slow). < 1.0 = volatility contracting = good for ICs.
    """
    n = len(high)
    if n < 4:
        return np.nan

    tr = _true_range(high, low, close)
    if len(tr) < 3:
        return np.nan

    fast_p = min(fast, len(tr))
    slow_p = min(slow, len(tr))

    atr_fast = np.mean(tr[-fast_p:])
    atr_slow = np.mean(tr[-slow_p:])

    if atr_slow == 0:
        return 1.0

    return float(atr_fast / atr_slow)


def compute_bbw_percentile(close: np.ndarray, period: int = 20) -> float:
    """
    Bollinger Band Width percentile rank vs session (0-100).
    Low = squeeze forming = good for ICs.
    """
    n = len(close)
    if n < 5:
        return np.nan

    period = min(period, n)

    # Compute rolling BBW for all available windows
    bbw_values = []
    for i in range(period, n + 1):
        window = close[i - period: i]
        sma = np.mean(window)
        std = np.std(window, ddof=1) if len(window) > 1 else 0
        if sma > 0:
            bbw_values.append(2 * 2 * std / sma)  # 2 std bands
        else:
            bbw_values.append(0)

    if not bbw_values:
        # Not enough data for a full window — use entire array
        sma = np.mean(close)
        std = np.std(close, ddof=1) if n > 1 else 0
        return 50.0

    current_bbw = bbw_values[-1]
    if len(bbw_values) < 2:
        return 50.0

    rank = np.sum(np.array(bbw_values) < current_bbw) / len(bbw_values) * 100
    return float(rank)


def compute_ttm_squeeze(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
) -> float:
    """
    TTM Squeeze: 1 if Bollinger Bands are inside Keltner Channels, 0 otherwise.
    Squeeze = compressed volatility = good for ICs.
    """
    n = len(close)
    if n < 5:
        return np.nan

    period = min(period, n)
    window_close = close[-period:]
    window_high = high[-period:]
    window_low = low[-period:]

    sma = np.mean(window_close)
    std = np.std(window_close, ddof=1) if period > 1 else 0

    # Bollinger Bands (2 std)
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std

    # Keltner Channels (1.5x ATR)
    tr = _true_range(high[-period - 1:], low[-period - 1:], close[-period - 1:])
    if len(tr) == 0:
        return np.nan
    atr = np.mean(tr[-min(period, len(tr)):])
    kc_upper = sma + 1.5 * atr
    kc_lower = sma - 1.5 * atr

    # Squeeze = BB inside KC
    return 1.0 if (bb_lower > kc_lower and bb_upper < kc_upper) else 0.0


def compute_bars_in_squeeze(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
) -> float:
    """
    Count consecutive bars currently in TTM squeeze. More = stronger compression.
    """
    n = len(close)
    if n < 5:
        return np.nan

    period = min(period, n)
    count = 0

    # Walk backwards, checking squeeze at each bar
    for end_idx in range(n, period, -1):
        h_win = high[end_idx - period: end_idx]
        l_win = low[end_idx - period: end_idx]
        c_win = close[end_idx - period: end_idx]

        sma = np.mean(c_win)
        std = np.std(c_win, ddof=1) if period > 1 else 0

        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std

        # Need period+1 for TR calc
        start = max(0, end_idx - period - 1)
        tr_h = high[start: end_idx]
        tr_l = low[start: end_idx]
        tr_c = close[start: end_idx]
        tr = _true_range(tr_h, tr_l, tr_c)
        if len(tr) == 0:
            break
        atr = np.mean(tr[-min(period, len(tr)):])

        kc_upper = sma + 1.5 * atr
        kc_lower = sma - 1.5 * atr

        if bb_lower > kc_lower and bb_upper < kc_upper:
            count += 1
        else:
            break

    return float(count)


def compute_garman_klass_rv(
    open_arr: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int = 20,
) -> float:
    """
    Garman-Klass realized volatility (annualized %). Uses OHLC for better RV estimation.

    GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    """
    n = len(close)
    if n < 3:
        return np.nan

    period = min(period, n)
    o = open_arr[-period:]
    h = high[-period:]
    l = low[-period:]
    c = close[-period:]

    # Avoid log of zero/negative
    mask = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    if mask.sum() < 2:
        return np.nan

    o, h, l, c = o[mask], h[mask], l[mask], c[mask]

    log_hl = np.log(h / l)
    log_co = np.log(c / o)

    gk_var = np.mean(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)

    if gk_var < 0:
        gk_var = 0.0

    # Annualize: 78 five-minute bars per day * 252 trading days
    bars_per_day = 78
    annualized_vol = np.sqrt(gk_var * bars_per_day * 252) * 100

    return float(annualized_vol)


def compute_orb_failure(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, orb_bars: int = 6
) -> float:
    """
    Opening Range Breakout failure (0/1).
    1 = price broke above/below ORB high/low but returned inside = failed breakout = range day signal.
    """
    n = len(close)
    if n < orb_bars + 2:
        return np.nan

    orb_high = np.max(high[:orb_bars])
    orb_low = np.min(low[:orb_bars])

    post_orb_high = high[orb_bars:]
    post_orb_low = low[orb_bars:]
    current_close = close[-1]

    # Did price break out then return?
    broke_high = np.any(post_orb_high > orb_high)
    broke_low = np.any(post_orb_low < orb_low)
    back_inside = orb_low <= current_close <= orb_high

    if (broke_high or broke_low) and back_inside:
        return 1.0
    return 0.0


def compute_all_technical_indicators(
    ohlc_df: pd.DataFrame, orb_bars: int = 6
) -> dict:
    """
    Compute all 11 technical indicators from 5-min OHLC candles.

    Args:
        ohlc_df: DataFrame with 'open', 'high', 'low', 'close' columns.
                 Should contain bars from 9:30 up to entry time.
        orb_bars: Number of bars for opening range (default 6 = 30 min).

    Returns:
        Dict of 11 indicator values. Values are np.nan when data insufficient.
    """
    if ohlc_df is None or ohlc_df.empty or len(ohlc_df) < 3:
        return {
            "adx": np.nan,
            "efficiency_ratio": np.nan,
            "choppiness_index": np.nan,
            "rsi_dist_50": np.nan,
            "linreg_r2": np.nan,
            "atr_ratio": np.nan,
            "bbw_percentile": np.nan,
            "ttm_squeeze": np.nan,
            "bars_in_squeeze": np.nan,
            "garman_klass_rv": np.nan,
            "orb_failure": np.nan,
        }

    o = ohlc_df["open"].values.astype(float)
    h = ohlc_df["high"].values.astype(float)
    l = ohlc_df["low"].values.astype(float)
    c = ohlc_df["close"].values.astype(float)

    return {
        "adx": compute_adx(h, l, c),
        "efficiency_ratio": compute_efficiency_ratio(c),
        "choppiness_index": compute_choppiness_index(h, l, c),
        "rsi_dist_50": compute_rsi_distance_from_50(c),
        "linreg_r2": compute_linreg_r2(c),
        "atr_ratio": compute_atr_ratio(h, l, c),
        "bbw_percentile": compute_bbw_percentile(c),
        "ttm_squeeze": compute_ttm_squeeze(h, l, c),
        "bars_in_squeeze": compute_bars_in_squeeze(h, l, c),
        "garman_klass_rv": compute_garman_klass_rv(o, h, l, c),
        "orb_failure": compute_orb_failure(h, l, c, orb_bars),
    }
