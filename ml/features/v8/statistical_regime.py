"""
Statistical & regime detection features for v8 model (~14 features).

Computes Hurst exponent, entropy measures, autocorrelation, HMM regime
probabilities, change-point detection, and distribution statistics
from SPX 1-min bars. Pure numpy (no external ML dependencies).
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def compute_statistical_regime_features(
    spx_1m: pd.DataFrame,
    prior_days_1m: list = None,
) -> dict:
    """
    Compute ~14 statistical/regime features.

    Args:
        spx_1m: Today's 1-min SPX bars
        prior_days_1m: Optional prior day DataFrames for regime fitting
    Returns:
        dict of feature_name -> float
    """
    if spx_1m is None or spx_1m.empty or len(spx_1m) < 10:
        return {k: np.nan for k in _all_feature_names()}

    if prior_days_1m is None:
        prior_days_1m = []

    features = {}
    c = spx_1m['close'].values.astype(np.float64)
    h = spx_1m['high'].values.astype(np.float64)
    lo = spx_1m['low'].values.astype(np.float64)
    n = len(c)

    # Build 5-min returns for most calculations
    n5 = n // 5
    if n5 >= 2:
        c5 = c[:n5 * 5].reshape(n5, 5)[:, -1]
        rets_5m = np.diff(np.log(np.maximum(c5, 1e-10)))
    else:
        rets_5m = np.array([])

    # ---- 1. Hurst exponent (R/S method on 5-min returns) ----
    if len(rets_5m) >= 20:
        features['stat_hurst_exponent'] = _hurst_rs(rets_5m)
    else:
        features['stat_hurst_exponent'] = np.nan

    # ---- 2. Sample entropy ----
    if len(rets_5m) >= 20:
        features['stat_sample_entropy'] = _sample_entropy(rets_5m, m=2, r_mult=0.2)
    else:
        features['stat_sample_entropy'] = np.nan

    # ---- 3. Permutation entropy ----
    if len(rets_5m) >= 20:
        features['stat_permutation_entropy'] = _permutation_entropy(rets_5m, order=3)
    else:
        features['stat_permutation_entropy'] = np.nan

    # ---- 4-5. Autocorrelation at lag 1, 5 ----
    if len(rets_5m) >= 10:
        features['stat_autocorr_lag1'] = _autocorrelation(rets_5m, 1)
        features['stat_autocorr_lag5'] = _autocorrelation(rets_5m, 5) if len(rets_5m) >= 10 else np.nan
    else:
        features['stat_autocorr_lag1'] = np.nan
        features['stat_autocorr_lag5'] = np.nan

    # ---- 6. Partial autocorrelation at lag 1 ----
    if len(rets_5m) >= 10:
        # PACF(1) = ACF(1) for lag 1
        features['stat_pacf_lag1'] = _autocorrelation(rets_5m, 1)
    else:
        features['stat_pacf_lag1'] = np.nan

    # ---- 7-9. HMM-like regime probabilities (simplified 3-state) ----
    # Use rolling volatility to classify regime instead of full HMM
    if len(rets_5m) >= 12:
        # Rolling 6-bar RV
        rolling_rv = []
        for i in range(6, len(rets_5m) + 1):
            rolling_rv.append(np.sum(rets_5m[i - 6:i] ** 2))
        rolling_rv = np.array(rolling_rv)

        if len(rolling_rv) >= 3:
            # Classify into 3 regimes by percentile
            p33 = np.percentile(rolling_rv, 33)
            p67 = np.percentile(rolling_rv, 67)
            current_rv = rolling_rv[-1]

            # Regime probabilities (soft assignment using distance from boundaries)
            if current_rv <= p33:
                features['stat_regime_low_vol'] = 1.0
                features['stat_regime_med_vol'] = 0.0
                features['stat_regime_high_vol'] = 0.0
            elif current_rv <= p67:
                features['stat_regime_low_vol'] = 0.0
                features['stat_regime_med_vol'] = 1.0
                features['stat_regime_high_vol'] = 0.0
            else:
                features['stat_regime_low_vol'] = 0.0
                features['stat_regime_med_vol'] = 0.0
                features['stat_regime_high_vol'] = 1.0
        else:
            features['stat_regime_low_vol'] = np.nan
            features['stat_regime_med_vol'] = np.nan
            features['stat_regime_high_vol'] = np.nan
    else:
        features['stat_regime_low_vol'] = np.nan
        features['stat_regime_med_vol'] = np.nan
        features['stat_regime_high_vol'] = np.nan

    # ---- 10-11. Change-point detection (CUSUM-based) ----
    if len(rets_5m) >= 10:
        lookback = min(30, len(rets_5m))
        window_rets = rets_5m[-lookback:]
        mean_ret = np.mean(window_rets)

        # CUSUM
        cusum = np.cumsum(window_rets - mean_ret)
        cusum_abs = np.abs(cusum)

        # Change-point count (peaks in |CUSUM|)
        cp_count = 0
        for i in range(1, len(cusum_abs) - 1):
            if cusum_abs[i] > cusum_abs[i - 1] and cusum_abs[i] > cusum_abs[i + 1]:
                cp_count += 1
        features['stat_changepoint_count'] = float(cp_count)

        # Change-point recency (bars since last significant deviation)
        threshold = 2 * np.std(window_rets) * np.sqrt(lookback)
        above_threshold = np.where(cusum_abs > threshold)[0]
        if len(above_threshold) > 0:
            features['stat_changepoint_recency'] = float(lookback - above_threshold[-1])
        else:
            features['stat_changepoint_recency'] = float(lookback)
    else:
        features['stat_changepoint_count'] = np.nan
        features['stat_changepoint_recency'] = np.nan

    # ---- 12-13. Rolling skewness and kurtosis (30 5-min bars) ----
    if len(rets_5m) >= 10:
        lookback = min(30, len(rets_5m))
        window_rets = rets_5m[-lookback:]
        n_w = len(window_rets)
        mean_w = np.mean(window_rets)
        std_w = np.std(window_rets, ddof=1)

        if std_w > 1e-10 and n_w >= 3:
            features['stat_rolling_skewness'] = float(
                np.mean(((window_rets - mean_w) / std_w) ** 3) * n_w / ((n_w - 1) * (n_w - 2)) * n_w
                if n_w > 2 else np.nan
            )
            features['stat_rolling_kurtosis'] = float(
                np.mean(((window_rets - mean_w) / std_w) ** 4) - 3.0
            )
        else:
            features['stat_rolling_skewness'] = np.nan
            features['stat_rolling_kurtosis'] = np.nan
    else:
        features['stat_rolling_skewness'] = np.nan
        features['stat_rolling_kurtosis'] = np.nan

    # ---- 14. Range expansion ratio ----
    if n >= 30:
        recent_range = np.max(h[-30:]) - np.min(lo[-30:])
        prior_range = np.max(h[:30]) - np.min(lo[:30]) if n >= 60 else recent_range
        if prior_range > 1e-10:
            features['stat_range_expansion_ratio'] = float(recent_range / prior_range)
        else:
            features['stat_range_expansion_ratio'] = np.nan
    else:
        features['stat_range_expansion_ratio'] = np.nan

    return features


def _hurst_rs(data: np.ndarray) -> float:
    """
    Hurst exponent using R/S (rescaled range) method.
    H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
    """
    n = len(data)
    if n < 8:
        return np.nan

    # Use multiple window sizes
    max_k = min(n // 2, 50)
    sizes = []
    rs_values = []

    for size in [8, 12, 16, 24, 32, 48]:
        if size > max_k:
            break
        n_windows = n // size
        if n_windows < 1:
            continue

        rs_list = []
        for w in range(n_windows):
            window = data[w * size:(w + 1) * size]
            mean = np.mean(window)
            deviations = np.cumsum(window - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(window, ddof=1)
            if s > 1e-15:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(np.log(size))
            rs_values.append(np.log(np.mean(rs_list)))

    if len(sizes) < 2:
        return np.nan

    # Linear regression: log(R/S) = H * log(n)
    x = np.array(sizes)
    y = np.array(rs_values)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-10:
        return np.nan

    hurst = ss_xy / ss_xx
    return float(np.clip(hurst, 0.0, 1.0))


def _sample_entropy(data: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Sample entropy. Higher = more complex/unpredictable.
    """
    n = len(data)
    if n < m + 2:
        return np.nan

    r = r_mult * np.std(data, ddof=1)
    if r < 1e-15:
        return np.nan

    def _count_matches(length):
        count = 0
        templates = n - length
        for i in range(templates):
            for j in range(i + 1, templates):
                if np.max(np.abs(data[i:i + length] - data[j:j + length])) < r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan
    if a == 0:
        return float(np.log(b))

    return float(-np.log(a / b))


def _permutation_entropy(data: np.ndarray, order: int = 3) -> float:
    """
    Permutation entropy. Normalized to [0, 1].
    """
    n = len(data)
    if n < order + 1:
        return np.nan

    import math
    max_patterns = math.factorial(order)
    pattern_counts = {}

    for i in range(n - order + 1):
        window = data[i:i + order]
        pattern = tuple(np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    total = sum(pattern_counts.values())
    if total == 0:
        return np.nan

    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    # Normalize
    max_entropy = np.log2(max_patterns)
    if max_entropy > 0:
        return float(entropy / max_entropy)
    return np.nan


def _autocorrelation(data: np.ndarray, lag: int) -> float:
    """Autocorrelation at specified lag."""
    n = len(data)
    if n < lag + 2:
        return np.nan

    mean = np.mean(data)
    var = np.var(data, ddof=0)
    if var < 1e-15:
        return np.nan

    cov = np.mean((data[:n - lag] - mean) * (data[lag:] - mean))
    return float(cov / var)


def _all_feature_names() -> list:
    """Return all feature names for nan-filling."""
    return [
        'stat_hurst_exponent', 'stat_sample_entropy', 'stat_permutation_entropy',
        'stat_autocorr_lag1', 'stat_autocorr_lag5', 'stat_pacf_lag1',
        'stat_regime_low_vol', 'stat_regime_med_vol', 'stat_regime_high_vol',
        'stat_changepoint_count', 'stat_changepoint_recency',
        'stat_rolling_skewness', 'stat_rolling_kurtosis',
        'stat_range_expansion_ratio',
    ]
