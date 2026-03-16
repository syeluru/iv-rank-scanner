"""
Trendline features for v8 model (~22 features).

Computes regression trendlines on swing highs/lows over 1-day and 5-day
windows, convergence/wedge detection, multi-TF agreement, and pattern
proxies from SPX 1-min bars. Pure numpy.
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def compute_trendline_features(
    spx_1m: pd.DataFrame,
    prior_days_1m: list = None,
    spx_price: float = 0.0,
) -> dict:
    """
    Compute ~22 trendline features.

    Args:
        spx_1m: Today's 1-min SPX bars
        prior_days_1m: List of prior day DataFrames (most recent first)
        spx_price: Current SPX price
    Returns:
        dict of feature_name -> float
    """
    if spx_1m is None or spx_1m.empty or len(spx_1m) < 10:
        return {k: np.nan for k in _all_feature_names()}

    if prior_days_1m is None:
        prior_days_1m = []

    features = {}
    price = spx_price if spx_price > 0 else float(spx_1m['close'].iloc[-1])

    # Compute ATR for normalization
    today_c = spx_1m['close'].values.astype(np.float64)
    today_h = spx_1m['high'].values.astype(np.float64)
    today_lo = spx_1m['low'].values.astype(np.float64)
    n = len(today_c)

    if n >= 15:
        tr = np.maximum(
            today_h[1:] - today_lo[1:],
            np.maximum(
                np.abs(today_h[1:] - today_c[:-1]),
                np.abs(today_lo[1:] - today_c[:-1]),
            ),
        )
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    else:
        atr = float(np.mean(today_h - today_lo)) if n > 0 else 1.0
    atr = max(atr, 0.01)

    # ---- Build multi-day 5-min closes ----
    all_5m_closes = []
    for i in range(min(5, len(prior_days_1m)) - 1, -1, -1):
        if prior_days_1m[i] is not None and not prior_days_1m[i].empty:
            dc = prior_days_1m[i]['close'].values.astype(np.float64)
            n_bars = len(dc) // 5
            if n_bars >= 1:
                c5 = dc[:n_bars * 5].reshape(n_bars, 5)[:, -1]
                all_5m_closes.extend(c5.tolist())

    n_bars_today = n // 5
    if n_bars_today >= 1:
        c5_today = today_c[:n_bars_today * 5].reshape(n_bars_today, 5)[:, -1]
        all_5m_closes.extend(c5_today.tolist())

    all_5m = np.array(all_5m_closes)

    # ---- 5-day trendlines on swing highs/lows (8 features) ----
    if len(all_5m) >= 20:
        swing_highs_idx, swing_highs_val = _find_swings(all_5m, mode='high')
        swing_lows_idx, swing_lows_val = _find_swings(all_5m, mode='low')

        # Resistance trendline (on swing highs)
        if len(swing_highs_idx) >= 3:
            slope_r, r2_r, intercept_r = _fit_trendline(swing_highs_idx, swing_highs_val)
            # Predict current level
            current_idx = len(all_5m) - 1
            tl_level = slope_r * current_idx + intercept_r
            features['tl_5d_resist_slope'] = float(slope_r / atr)  # normalized
            features['tl_5d_resist_r2'] = float(r2_r)
            features['tl_5d_resist_dist_pct'] = float((price - tl_level) / price * 100) if price > 0 else np.nan
            features['tl_5d_resist_touches'] = float(len(swing_highs_idx))
        else:
            features['tl_5d_resist_slope'] = np.nan
            features['tl_5d_resist_r2'] = np.nan
            features['tl_5d_resist_dist_pct'] = np.nan
            features['tl_5d_resist_touches'] = np.nan

        # Support trendline (on swing lows)
        if len(swing_lows_idx) >= 3:
            slope_s, r2_s, intercept_s = _fit_trendline(swing_lows_idx, swing_lows_val)
            current_idx = len(all_5m) - 1
            tl_level = slope_s * current_idx + intercept_s
            features['tl_5d_support_slope'] = float(slope_s / atr)
            features['tl_5d_support_r2'] = float(r2_s)
            features['tl_5d_support_dist_pct'] = float((price - tl_level) / price * 100) if price > 0 else np.nan
            features['tl_5d_support_touches'] = float(len(swing_lows_idx))
        else:
            features['tl_5d_support_slope'] = np.nan
            features['tl_5d_support_r2'] = np.nan
            features['tl_5d_support_dist_pct'] = np.nan
            features['tl_5d_support_touches'] = np.nan
    else:
        for k in ['tl_5d_resist_slope', 'tl_5d_resist_r2', 'tl_5d_resist_dist_pct',
                   'tl_5d_resist_touches', 'tl_5d_support_slope', 'tl_5d_support_r2',
                   'tl_5d_support_dist_pct', 'tl_5d_support_touches']:
            features[k] = np.nan

    # ---- Intraday trendlines (4 features) ----
    today_5m = today_c[:n_bars_today * 5].reshape(n_bars_today, 5)[:, -1] if n_bars_today >= 1 else np.array([])

    if len(today_5m) >= 10:
        intra_highs_idx, intra_highs_val = _find_swings(today_5m, mode='high')
        intra_lows_idx, intra_lows_val = _find_swings(today_5m, mode='low')

        if len(intra_highs_idx) >= 2:
            slope_r, _, _ = _fit_trendline(intra_highs_idx, intra_highs_val)
            features['tl_1d_resist_slope'] = float(slope_r / atr)
        else:
            features['tl_1d_resist_slope'] = np.nan

        if len(intra_lows_idx) >= 2:
            slope_s, _, _ = _fit_trendline(intra_lows_idx, intra_lows_val)
            features['tl_1d_support_slope'] = float(slope_s / atr)
        else:
            features['tl_1d_support_slope'] = np.nan

        # Intraday trendline distances
        if len(intra_highs_idx) >= 2:
            slope_r, _, intercept_r = _fit_trendline(intra_highs_idx, intra_highs_val)
            tl_r = slope_r * (len(today_5m) - 1) + intercept_r
            features['tl_1d_resist_dist_pct'] = float((price - tl_r) / price * 100) if price > 0 else np.nan
        else:
            features['tl_1d_resist_dist_pct'] = np.nan

        if len(intra_lows_idx) >= 2:
            slope_s, _, intercept_s = _fit_trendline(intra_lows_idx, intra_lows_val)
            tl_s = slope_s * (len(today_5m) - 1) + intercept_s
            features['tl_1d_support_dist_pct'] = float((price - tl_s) / price * 100) if price > 0 else np.nan
        else:
            features['tl_1d_support_dist_pct'] = np.nan
    else:
        features['tl_1d_resist_slope'] = np.nan
        features['tl_1d_support_slope'] = np.nan
        features['tl_1d_resist_dist_pct'] = np.nan
        features['tl_1d_support_dist_pct'] = np.nan

    # ---- Convergence / Wedge features (5 features) ----
    resist_slope = features.get('tl_5d_resist_slope', np.nan)
    support_slope = features.get('tl_5d_support_slope', np.nan)

    if np.isfinite(resist_slope) and np.isfinite(support_slope):
        # Convergence rate (negative = converging)
        features['tl_convergence_rate'] = float(resist_slope - support_slope)

        # Channel slope (average of both trendlines)
        features['tl_channel_slope'] = float((resist_slope + support_slope) / 2)

        # Angle difference
        features['tl_angle_diff'] = float(abs(resist_slope - support_slope))

        # Bars to apex (if converging)
        if resist_slope - support_slope < -1e-10:
            # Estimate using current channel width
            r_level = features.get('tl_5d_resist_dist_pct', 0)
            s_level = features.get('tl_5d_support_dist_pct', 0)
            if np.isfinite(r_level) and np.isfinite(s_level):
                channel_width = abs(r_level - s_level)
                convergence_rate = abs(resist_slope - support_slope)
                if convergence_rate > 1e-10:
                    features['tl_bars_to_apex'] = float(channel_width / convergence_rate)
                else:
                    features['tl_bars_to_apex'] = np.nan
            else:
                features['tl_bars_to_apex'] = np.nan
        else:
            features['tl_bars_to_apex'] = np.nan  # diverging

        # Multi-TF agreement (1d vs 5d slope direction)
        intra_resist_slope = features.get('tl_1d_resist_slope', np.nan)
        if np.isfinite(intra_resist_slope):
            features['tl_mtf_agreement'] = 1.0 if np.sign(resist_slope) == np.sign(intra_resist_slope) else 0.0
        else:
            features['tl_mtf_agreement'] = np.nan
    else:
        features['tl_convergence_rate'] = np.nan
        features['tl_channel_slope'] = np.nan
        features['tl_angle_diff'] = np.nan
        features['tl_bars_to_apex'] = np.nan
        features['tl_mtf_agreement'] = np.nan

    # ---- Pattern proxies (5 features) ----
    # Symmetry score: how symmetric is the channel
    if np.isfinite(resist_slope) and np.isfinite(support_slope):
        total = abs(resist_slope) + abs(support_slope)
        if total > 1e-10:
            features['tl_symmetry_score'] = float(1.0 - abs(abs(resist_slope) - abs(support_slope)) / total)
        else:
            features['tl_symmetry_score'] = np.nan
    else:
        features['tl_symmetry_score'] = np.nan

    # Triangle convergence (absolute convergence metric)
    features['tl_triangle_convergence'] = features.get('tl_convergence_rate', np.nan)

    # Average bounce magnitude from trendlines
    if len(all_5m) >= 20:
        # Use deviations from linear fit as bounce proxy
        x = np.arange(len(all_5m), dtype=np.float64)
        if len(x) >= 3:
            coeffs = np.polyfit(x, all_5m, 1)
            residuals = all_5m - np.polyval(coeffs, x)
            features['tl_avg_bounce_magnitude'] = float(np.std(residuals) / atr)
        else:
            features['tl_avg_bounce_magnitude'] = np.nan
    else:
        features['tl_avg_bounce_magnitude'] = np.nan

    # Touch recency decay (exponential decay of touch importance)
    if len(all_5m) >= 20:
        swing_highs_idx, _ = _find_swings(all_5m, mode='high')
        if len(swing_highs_idx) >= 2:
            total_bars = len(all_5m)
            decay_factor = 0.95
            recency_score = sum(
                decay_factor ** (total_bars - 1 - idx) for idx in swing_highs_idx
            )
            features['tl_touch_recency_decay'] = float(recency_score)
        else:
            features['tl_touch_recency_decay'] = np.nan
    else:
        features['tl_touch_recency_decay'] = np.nan

    return features


def _find_swings(closes: np.ndarray, mode: str = 'high', window: int = 3) -> tuple:
    """
    Find swing highs or lows in close prices.

    Returns (indices, values) arrays.
    """
    n = len(closes)
    if n < 2 * window + 1:
        return np.array([]), np.array([])

    indices = []
    values = []

    for i in range(window, n - window):
        if mode == 'high':
            if closes[i] == np.max(closes[i - window:i + window + 1]):
                indices.append(i)
                values.append(closes[i])
        else:  # low
            if closes[i] == np.min(closes[i - window:i + window + 1]):
                indices.append(i)
                values.append(closes[i])

    return np.array(indices), np.array(values)


def _fit_trendline(indices: np.ndarray, values: np.ndarray) -> tuple:
    """
    Fit linear regression trendline to swing points.

    Returns (slope, r_squared, intercept).
    """
    if len(indices) < 2:
        return np.nan, np.nan, np.nan

    x = indices.astype(np.float64)
    y = values.astype(np.float64)

    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-10:
        return 0.0, 0.0, y_mean

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    ss_yy = np.sum((y - y_mean) ** 2)
    if ss_yy < 1e-10:
        r2 = 1.0
    else:
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1.0 - ss_res / ss_yy

    return float(slope), float(max(r2, 0.0)), float(intercept)


def _all_feature_names() -> list:
    """Return all feature names for nan-filling."""
    return [
        'tl_5d_resist_slope', 'tl_5d_resist_r2', 'tl_5d_resist_dist_pct', 'tl_5d_resist_touches',
        'tl_5d_support_slope', 'tl_5d_support_r2', 'tl_5d_support_dist_pct', 'tl_5d_support_touches',
        'tl_1d_resist_slope', 'tl_1d_support_slope',
        'tl_1d_resist_dist_pct', 'tl_1d_support_dist_pct',
        'tl_convergence_rate', 'tl_channel_slope', 'tl_angle_diff',
        'tl_bars_to_apex', 'tl_mtf_agreement',
        'tl_symmetry_score', 'tl_triangle_convergence',
        'tl_avg_bounce_magnitude', 'tl_touch_recency_decay',
    ]
