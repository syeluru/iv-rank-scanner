"""
Support/Resistance features for v8 model (~30 features).

Computes prior-day levels, multi-day range analysis, peak/valley S/R zones,
Fibonacci retracements, Camarilla pivots, and swing structure from
SPX 1-min bars. Pure numpy/scipy.
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def compute_support_resistance_features(
    spx_1m: pd.DataFrame,
    prior_days_1m: list = None,
    spx_price: float = 0.0,
) -> dict:
    """
    Compute ~30 support/resistance features.

    Args:
        spx_1m: Today's 1-min SPX bars [timestamp, open, high, low, close]
        prior_days_1m: List of prior day DataFrames (most recent first), up to 5
        spx_price: Current SPX price
    Returns:
        dict of feature_name -> float
    """
    if spx_1m is None or spx_1m.empty or len(spx_1m) < 3:
        return {k: np.nan for k in _all_feature_names()}

    if prior_days_1m is None:
        prior_days_1m = []

    features = {}
    price = spx_price if spx_price > 0 else spx_1m['close'].iloc[-1]

    # ---- Prior day levels (6 features) ----
    if len(prior_days_1m) >= 1 and prior_days_1m[0] is not None and not prior_days_1m[0].empty:
        pd1 = prior_days_1m[0]
        pd_high = float(pd1['high'].max())
        pd_low = float(pd1['low'].min())
        pd_close = float(pd1['close'].iloc[-1])
        pd_open = float(pd1['open'].iloc[0])

        features['sr_prior_day_high'] = pd_high
        features['sr_prior_day_low'] = pd_low
        features['sr_prior_day_close'] = pd_close
        features['sr_prior_day_open'] = pd_open

        # Distance to prior day high/low (%)
        if price > 0:
            features['sr_dist_prior_high_pct'] = float((price - pd_high) / price * 100)
            features['sr_dist_prior_low_pct'] = float((price - pd_low) / price * 100)
        else:
            features['sr_dist_prior_high_pct'] = np.nan
            features['sr_dist_prior_low_pct'] = np.nan
    else:
        for k in ['sr_prior_day_high', 'sr_prior_day_low', 'sr_prior_day_close',
                   'sr_prior_day_open', 'sr_dist_prior_high_pct', 'sr_dist_prior_low_pct']:
            features[k] = np.nan

    # ---- 5-day range analysis (7 features) ----
    n_prior = min(5, len(prior_days_1m))
    all_highs = []
    all_lows = []
    daily_ranges = []

    for i in range(n_prior):
        if prior_days_1m[i] is not None and not prior_days_1m[i].empty:
            dh = float(prior_days_1m[i]['high'].max())
            dl = float(prior_days_1m[i]['low'].min())
            all_highs.append(dh)
            all_lows.append(dl)
            daily_ranges.append(dh - dl)

    if all_highs and all_lows:
        five_day_high = max(all_highs)
        five_day_low = min(all_lows)

        features['sr_5day_high'] = five_day_high
        features['sr_5day_low'] = five_day_low

        # Breakout flag: price outside 5-day range
        if price > five_day_high:
            features['sr_5day_breakout'] = 1.0
        elif price < five_day_low:
            features['sr_5day_breakout'] = -1.0
        else:
            features['sr_5day_breakout'] = 0.0

        # Days price has been in current range
        in_range_count = 0
        for i in range(len(all_highs)):
            if all_lows[i] <= price <= all_highs[i]:
                in_range_count += 1
        features['sr_days_in_range'] = float(in_range_count)

        # Consolidation tightness (avg daily range / 5-day range)
        five_day_range = five_day_high - five_day_low
        if five_day_range > 1e-10 and daily_ranges:
            features['sr_consolidation_tightness'] = float(np.mean(daily_ranges) / five_day_range)
        else:
            features['sr_consolidation_tightness'] = np.nan

        # Range contraction streak (consecutive narrowing daily ranges)
        if len(daily_ranges) >= 2:
            streak = 0
            # daily_ranges[0] is most recent
            for i in range(len(daily_ranges) - 1):
                if daily_ranges[i] < daily_ranges[i + 1]:
                    streak += 1
                else:
                    break
            features['sr_range_contraction_streak'] = float(streak)
        else:
            features['sr_range_contraction_streak'] = np.nan

        # 5-day range width (%)
        if price > 0:
            features['sr_5day_range_pct'] = float(five_day_range / price * 100)
        else:
            features['sr_5day_range_pct'] = np.nan
    else:
        for k in ['sr_5day_high', 'sr_5day_low', 'sr_5day_breakout', 'sr_days_in_range',
                   'sr_consolidation_tightness', 'sr_range_contraction_streak', 'sr_5day_range_pct']:
            features[k] = np.nan

    # ---- Peak/Valley S/R from 5-day 5-min bars (5 features) ----
    # Concatenate prior days + today, resample to 5-min, find peaks
    all_closes_5m = []
    for i in range(min(5, len(prior_days_1m)) - 1, -1, -1):  # oldest first
        if prior_days_1m[i] is not None and not prior_days_1m[i].empty:
            day_c = prior_days_1m[i]['close'].values.astype(np.float64)
            # Resample to 5-min
            n_bars = len(day_c) // 5
            if n_bars >= 1:
                c5 = day_c[:n_bars * 5].reshape(n_bars, 5)[:, -1]
                all_closes_5m.extend(c5.tolist())

    # Add today
    today_c = spx_1m['close'].values.astype(np.float64)
    n_bars_today = len(today_c) // 5
    if n_bars_today >= 1:
        c5_today = today_c[:n_bars_today * 5].reshape(n_bars_today, 5)[:, -1]
        all_closes_5m.extend(c5_today.tolist())

    sr_levels = _find_sr_levels(np.array(all_closes_5m), price) if len(all_closes_5m) >= 20 else []

    # Top 5 S/R levels: distances and strength
    for i in range(5):
        dist_key = f'sr_level_{i+1}_dist_pct'
        strength_key = f'sr_level_{i+1}_strength'
        if i < len(sr_levels):
            level, strength = sr_levels[i]
            if price > 0:
                features[dist_key] = float((price - level) / price * 100)
            else:
                features[dist_key] = np.nan
            features[strength_key] = float(strength)
        else:
            features[dist_key] = np.nan
            features[strength_key] = np.nan

    # ---- Fibonacci retracements from 5-day swing (3 features) ----
    if all_highs and all_lows:
        swing_high = max(all_highs)
        swing_low = min(all_lows)
        swing_range = swing_high - swing_low

        if swing_range > 1e-10 and price > 0:
            fib_382 = swing_high - 0.382 * swing_range
            fib_500 = swing_high - 0.500 * swing_range
            fib_618 = swing_high - 0.618 * swing_range

            features['sr_fib_382_dist_pct'] = float((price - fib_382) / price * 100)
            features['sr_fib_500_dist_pct'] = float((price - fib_500) / price * 100)
            features['sr_fib_618_dist_pct'] = float((price - fib_618) / price * 100)
        else:
            features['sr_fib_382_dist_pct'] = np.nan
            features['sr_fib_500_dist_pct'] = np.nan
            features['sr_fib_618_dist_pct'] = np.nan
    else:
        features['sr_fib_382_dist_pct'] = np.nan
        features['sr_fib_500_dist_pct'] = np.nan
        features['sr_fib_618_dist_pct'] = np.nan

    # ---- Camarilla pivots (4 features: R1, R3, S1, S3 distances) ----
    if len(prior_days_1m) >= 1 and prior_days_1m[0] is not None and not prior_days_1m[0].empty:
        pd1 = prior_days_1m[0]
        pd_h = float(pd1['high'].max())
        pd_l = float(pd1['low'].min())
        pd_c = float(pd1['close'].iloc[-1])
        pd_range = pd_h - pd_l

        cam_r1 = pd_c + pd_range * 1.1 / 12
        cam_r3 = pd_c + pd_range * 1.1 / 4
        cam_s1 = pd_c - pd_range * 1.1 / 12
        cam_s3 = pd_c - pd_range * 1.1 / 4

        if price > 0:
            features['sr_cam_r1_dist_pct'] = float((price - cam_r1) / price * 100)
            features['sr_cam_r3_dist_pct'] = float((price - cam_r3) / price * 100)
            features['sr_cam_s1_dist_pct'] = float((price - cam_s1) / price * 100)
            features['sr_cam_s3_dist_pct'] = float((price - cam_s3) / price * 100)
        else:
            features['sr_cam_r1_dist_pct'] = np.nan
            features['sr_cam_r3_dist_pct'] = np.nan
            features['sr_cam_s1_dist_pct'] = np.nan
            features['sr_cam_s3_dist_pct'] = np.nan
    else:
        for k in ['sr_cam_r1_dist_pct', 'sr_cam_r3_dist_pct',
                   'sr_cam_s1_dist_pct', 'sr_cam_s3_dist_pct']:
            features[k] = np.nan

    # ---- Multi-day swing structure (3 features) ----
    if len(all_highs) >= 2 and len(all_lows) >= 2:
        # Higher highs count
        hh_count = sum(1 for i in range(1, len(all_highs)) if all_highs[i] > all_highs[i - 1])
        ll_count = sum(1 for i in range(1, len(all_lows)) if all_lows[i] < all_lows[i - 1])

        features['sr_swing_hh_count'] = float(hh_count)
        features['sr_swing_ll_count'] = float(ll_count)

        # Swing trend score: (HH - LL) / total_days
        features['sr_swing_trend_score'] = float((hh_count - ll_count) / len(all_highs))
    else:
        features['sr_swing_hh_count'] = np.nan
        features['sr_swing_ll_count'] = np.nan
        features['sr_swing_trend_score'] = np.nan

    return features


def _find_sr_levels(closes_5m: np.ndarray, current_price: float, min_prominence_pct: float = 0.15) -> list:
    """
    Find S/R levels from 5-min closes using peak detection + clustering.

    Returns list of (level, strength) tuples sorted by proximity to current price.
    """
    n = len(closes_5m)
    if n < 10:
        return []

    # Find peaks and valleys using simple prominence check
    prominence_threshold = current_price * min_prominence_pct / 100 if current_price > 0 else 1.0

    peaks = []
    valleys = []
    window = 3

    for i in range(window, n - window):
        if closes_5m[i] == np.max(closes_5m[i - window:i + window + 1]):
            # Check prominence
            left_min = np.min(closes_5m[max(0, i - 10):i])
            right_min = np.min(closes_5m[i + 1:min(n, i + 11)])
            prominence = closes_5m[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
                peaks.append(closes_5m[i])

        if closes_5m[i] == np.min(closes_5m[i - window:i + window + 1]):
            left_max = np.max(closes_5m[max(0, i - 10):i])
            right_max = np.max(closes_5m[i + 1:min(n, i + 11)])
            prominence = min(left_max, right_max) - closes_5m[i]
            if prominence >= prominence_threshold:
                valleys.append(closes_5m[i])

    # Combine peaks and valleys
    all_levels = peaks + valleys
    if not all_levels:
        return []

    # Cluster levels within 0.05% of each other
    cluster_threshold = current_price * 0.0005 if current_price > 0 else 5.0
    all_levels.sort()
    clusters = []
    current_cluster = [all_levels[0]]

    for i in range(1, len(all_levels)):
        if all_levels[i] - current_cluster[-1] <= cluster_threshold:
            current_cluster.append(all_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [all_levels[i]]
    clusters.append(current_cluster)

    # Each cluster: (avg_level, touch_count)
    sr_results = [(np.mean(c), len(c)) for c in clusters]

    # Sort by proximity to current price
    sr_results.sort(key=lambda x: abs(x[0] - current_price))

    return sr_results[:5]


def _all_feature_names() -> list:
    """Return all feature names for nan-filling."""
    names = [
        'sr_prior_day_high', 'sr_prior_day_low', 'sr_prior_day_close', 'sr_prior_day_open',
        'sr_dist_prior_high_pct', 'sr_dist_prior_low_pct',
        'sr_5day_high', 'sr_5day_low', 'sr_5day_breakout', 'sr_days_in_range',
        'sr_consolidation_tightness', 'sr_range_contraction_streak', 'sr_5day_range_pct',
    ]
    for i in range(1, 6):
        names.append(f'sr_level_{i}_dist_pct')
        names.append(f'sr_level_{i}_strength')
    names.extend([
        'sr_fib_382_dist_pct', 'sr_fib_500_dist_pct', 'sr_fib_618_dist_pct',
        'sr_cam_r1_dist_pct', 'sr_cam_r3_dist_pct', 'sr_cam_s1_dist_pct', 'sr_cam_s3_dist_pct',
        'sr_swing_hh_count', 'sr_swing_ll_count', 'sr_swing_trend_score',
    ])
    return names
