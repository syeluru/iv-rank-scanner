"""
Realized volatility features from SPX 1-minute OHLC bars (v8 model).

All functions are pure numpy, operating on 1-minute SPX candle arrays.
Each returns a scalar value. Returns np.nan when insufficient data.
5-minute returns are derived by sampling every 5th bar from the 1-min data.
"""

import numpy as np
import pandas as pd


def _get_5min_log_returns(close: np.ndarray) -> np.ndarray:
    """
    Compute 5-minute log returns by sampling every 5th bar from 1-min closes.

    Returns array of log(close[i] / close[i-5]) for i = 5, 10, 15, ...
    """
    if len(close) < 6:
        return np.array([])

    indices = np.arange(0, len(close), 5)
    if len(indices) < 2:
        return np.array([])

    sampled = close[indices]
    # Guard against zero/negative prices
    mask = (sampled[:-1] > 0) & (sampled[1:] > 0)
    if not np.all(mask):
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return np.array([])
        returns = np.log(sampled[1:][mask] / sampled[:-1][mask])
    else:
        returns = np.log(sampled[1:] / sampled[:-1])

    return returns


def _get_5min_ohlc(
    open_arr: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> tuple:
    """
    Aggregate 1-min OHLC into 5-min OHLC bars.

    Returns (open_5m, high_5m, low_5m, close_5m) arrays.
    """
    n = len(close)
    n_bars = n // 5
    if n_bars < 1:
        return np.array([]), np.array([]), np.array([]), np.array([])

    trimmed = n_bars * 5
    o = open_arr[:trimmed].reshape(n_bars, 5)
    h = high[:trimmed].reshape(n_bars, 5)
    l = low[:trimmed].reshape(n_bars, 5)
    c = close[:trimmed].reshape(n_bars, 5)

    return o[:, 0], np.max(h, axis=1), np.min(l, axis=1), c[:, -1]


def _compute_rv_from_returns(returns: np.ndarray) -> float:
    """Sum of squared returns (realized variance)."""
    if len(returns) == 0:
        return np.nan
    return float(np.sum(returns ** 2))


def _rv_5min(close: np.ndarray) -> float:
    """Realized variance from 5-min log returns."""
    returns = _get_5min_log_returns(close)
    return _compute_rv_from_returns(returns)


def _bipower_variation(close: np.ndarray) -> float:
    """
    Bipower variation (Barndorff-Nielsen & Shephard).

    BPV = (pi/2) * sum(|r_i| * |r_{i-1}|) for adjacent 5-min returns.
    Estimates the continuous (diffusion) component of variance.
    """
    returns = _get_5min_log_returns(close)
    n = len(returns)
    if n < 2:
        return np.nan

    abs_ret = np.abs(returns)
    bpv = (np.pi / 2.0) * np.sum(abs_ret[1:] * abs_ret[:-1])
    return float(bpv)


def _realized_quarticity(close: np.ndarray) -> float:
    """
    Realized quarticity: (n/3) * sum(r_i^4).

    Needed for jump test standard error.
    """
    returns = _get_5min_log_returns(close)
    n = len(returns)
    if n < 2:
        return np.nan

    rq = (n / 3.0) * np.sum(returns ** 4)
    return float(rq)


def _full_day_rv(day_df: pd.DataFrame) -> float:
    """Compute full-session RV for a prior day DataFrame."""
    if day_df is None or day_df.empty or len(day_df) < 6:
        return np.nan

    close = day_df["close"].values.astype(float)
    return _rv_5min(close)


def compute_realized_vol_features(
    spx_1m: pd.DataFrame, prior_days_1m: list = None
) -> dict:
    """
    Compute ~30 realized volatility features from SPX 1-minute OHLC bars.

    Args:
        spx_1m: DataFrame with columns [timestamp, open, high, low, close]
                for today's session so far (1-min bars).
        prior_days_1m: Optional list of prior day DataFrames (most recent first)
                       for multi-day features.

    Returns:
        dict of feature_name -> float
    """
    nan_result = {
        "rv_5min": np.nan,
        "rv_annualized": np.nan,
        "bipower_variation": np.nan,
        "jump_variation": np.nan,
        "jump_test_statistic": np.nan,
        "jump_detected": np.nan,
        "relative_jump_contribution": np.nan,
        "realized_upside_semivariance": np.nan,
        "realized_downside_semivariance": np.nan,
        "semivariance_ratio": np.nan,
        "positive_jump_variation": np.nan,
        "negative_jump_variation": np.nan,
        "realized_quarticity": np.nan,
        "iq_ratio": np.nan,
        "parkinson_vol": np.nan,
        "garman_klass_vol": np.nan,
        "rogers_satchell_vol": np.nan,
        "yang_zhang_vol": np.nan,
        "gk_rv_vs_realized": np.nan,
        "rv_1bar": np.nan,
        "rv_6bar": np.nan,
        "rv_12bar": np.nan,
        "rv_daily_prior": np.nan,
        "rv_weekly": np.nan,
        "rv_monthly": np.nan,
        "rv_acceleration": np.nan,
        "intraday_vol_of_vol": np.nan,
        "rv_surprise": np.nan,
        "max_5min_return_abs": np.nan,
        "overnight_to_intraday_ratio": np.nan,
    }

    if spx_1m is None or spx_1m.empty or len(spx_1m) < 6:
        return nan_result

    if prior_days_1m is None:
        prior_days_1m = []

    close = spx_1m["close"].values.astype(float)
    open_arr = spx_1m["open"].values.astype(float)
    high = spx_1m["high"].values.astype(float)
    low = spx_1m["low"].values.astype(float)

    returns_5m = _get_5min_log_returns(close)
    n_ret = len(returns_5m)

    if n_ret < 2:
        return nan_result

    result = {}

    # --- 1. rv_5min ---
    rv_5min = float(np.sum(returns_5m ** 2))
    result["rv_5min"] = rv_5min

    # --- 2. rv_annualized ---
    # 78 five-min bars per day, 252 trading days
    result["rv_annualized"] = float(np.sqrt(rv_5min * 78 * 252) * 100)

    # --- 3. bipower_variation ---
    abs_ret = np.abs(returns_5m)
    bpv = (np.pi / 2.0) * np.sum(abs_ret[1:] * abs_ret[:-1])
    result["bipower_variation"] = float(bpv)

    # --- 4. jump_variation ---
    jump_var = max(0.0, rv_5min - bpv)
    result["jump_variation"] = float(jump_var)

    # --- 5. jump_test_statistic ---
    rq = (n_ret / 3.0) * np.sum(returns_5m ** 4)
    se = np.sqrt(max((1.0 / n_ret) * rq, 1e-20))
    jump_z = (rv_5min - bpv) / se
    result["jump_test_statistic"] = float(jump_z)

    # --- 6. jump_detected ---
    result["jump_detected"] = 1.0 if jump_z > 2.0 else 0.0

    # --- 7. relative_jump_contribution ---
    result["relative_jump_contribution"] = float(jump_var / max(rv_5min, 1e-10))

    # --- 8-9. semivariance ---
    pos_mask = returns_5m > 0
    neg_mask = returns_5m < 0
    upside_sv = float(np.sum(returns_5m[pos_mask] ** 2)) if np.any(pos_mask) else 0.0
    downside_sv = float(np.sum(returns_5m[neg_mask] ** 2)) if np.any(neg_mask) else 0.0
    result["realized_upside_semivariance"] = upside_sv
    result["realized_downside_semivariance"] = downside_sv

    # --- 10. semivariance_ratio ---
    result["semivariance_ratio"] = float(downside_sv / max(upside_sv, 1e-10))

    # --- 11-12. positive/negative jump variation ---
    result["positive_jump_variation"] = float(upside_sv - bpv / 2.0)
    result["negative_jump_variation"] = float(downside_sv - bpv / 2.0)

    # --- 13. realized_quarticity ---
    result["realized_quarticity"] = float(rq)

    # --- 14. iq_ratio ---
    result["iq_ratio"] = float(rq / max(rv_5min ** 2, 1e-20))

    # --- 15. parkinson_vol ---
    # Aggregate to 5-min bars for range-based estimators
    o5, h5, l5, c5 = _get_5min_ohlc(open_arr, high, low, close)
    n5 = len(h5)

    if n5 >= 2:
        mask = (h5 > 0) & (l5 > 0) & (h5 >= l5)
        if np.sum(mask) >= 2:
            log_hl = np.log(h5[mask] / l5[mask])
            n_valid = int(np.sum(mask))
            park_var = (1.0 / (4.0 * n_valid * np.log(2))) * np.sum(log_hl ** 2)
            result["parkinson_vol"] = float(np.sqrt(park_var * 78 * 252) * 100)
        else:
            result["parkinson_vol"] = np.nan
    else:
        result["parkinson_vol"] = np.nan

    # --- 16. garman_klass_vol ---
    if n5 >= 2:
        mask = (o5 > 0) & (h5 > 0) & (l5 > 0) & (c5 > 0) & (h5 >= l5)
        if np.sum(mask) >= 2:
            log_hl = np.log(h5[mask] / l5[mask])
            log_co = np.log(c5[mask] / o5[mask])
            gk_var = np.mean(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)
            if gk_var < 0:
                gk_var = 0.0
            gk_vol = float(np.sqrt(gk_var * 78 * 252) * 100)
            result["garman_klass_vol"] = gk_vol
        else:
            result["garman_klass_vol"] = np.nan
            gk_vol = np.nan
    else:
        result["garman_klass_vol"] = np.nan
        gk_vol = np.nan

    # --- 17. rogers_satchell_vol ---
    if n5 >= 2:
        mask = (o5 > 0) & (h5 > 0) & (l5 > 0) & (c5 > 0) & (h5 >= l5)
        if np.sum(mask) >= 2:
            h_m = h5[mask]
            l_m = l5[mask]
            o_m = o5[mask]
            c_m = c5[mask]
            rs_var = np.mean(
                np.log(h_m / c_m) * np.log(h_m / o_m)
                + np.log(l_m / c_m) * np.log(l_m / o_m)
            )
            if rs_var < 0:
                rs_var = 0.0
            result["rogers_satchell_vol"] = float(np.sqrt(rs_var * 78 * 252) * 100)
        else:
            result["rogers_satchell_vol"] = np.nan
    else:
        result["rogers_satchell_vol"] = np.nan

    # --- 18. yang_zhang_vol ---
    if n5 >= 3:
        mask = (o5 > 0) & (h5 > 0) & (l5 > 0) & (c5 > 0) & (h5 >= l5)
        if np.sum(mask) >= 3:
            o_m = o5[mask]
            h_m = h5[mask]
            l_m = l5[mask]
            c_m = c5[mask]
            n_v = len(o_m)

            # Overnight returns (open-to-prior-close)
            log_oc = np.log(o_m[1:] / c_m[:-1])
            overnight_var = np.var(log_oc, ddof=1) if len(log_oc) > 1 else 0.0

            # Close-to-open returns
            log_co = np.log(c_m / o_m)
            close_open_var = np.var(log_co, ddof=1) if len(log_co) > 1 else 0.0

            # Rogers-Satchell component
            rs_vals = (
                np.log(h_m / c_m) * np.log(h_m / o_m)
                + np.log(l_m / c_m) * np.log(l_m / o_m)
            )
            rs_var = np.mean(rs_vals)
            if rs_var < 0:
                rs_var = 0.0

            k = 0.34 / (1.34 + (n_v + 1) / (n_v - 1))
            yz_var = overnight_var + k * close_open_var + (1 - k) * rs_var
            if yz_var < 0:
                yz_var = 0.0
            result["yang_zhang_vol"] = float(np.sqrt(yz_var * 78 * 252) * 100)
        else:
            result["yang_zhang_vol"] = np.nan
    else:
        result["yang_zhang_vol"] = np.nan

    # --- 19. gk_rv_vs_realized ---
    rv_ann = result["rv_annualized"]
    gk_v = result.get("garman_klass_vol", np.nan)
    if np.isfinite(gk_v) and np.isfinite(rv_ann):
        result["gk_rv_vs_realized"] = float(gk_v / max(rv_ann, 1e-10))
    else:
        result["gk_rv_vs_realized"] = np.nan

    # --- 20. rv_1bar ---
    result["rv_1bar"] = float(returns_5m[-1] ** 2)

    # --- 21. rv_6bar ---
    if n_ret >= 6:
        result["rv_6bar"] = float(np.sum(returns_5m[-6:] ** 2))
    elif n_ret >= 2:
        result["rv_6bar"] = float(np.sum(returns_5m ** 2))
    else:
        result["rv_6bar"] = np.nan

    # --- 22. rv_12bar ---
    if n_ret >= 12:
        result["rv_12bar"] = float(np.sum(returns_5m[-12:] ** 2))
    elif n_ret >= 2:
        result["rv_12bar"] = float(np.sum(returns_5m ** 2))
    else:
        result["rv_12bar"] = np.nan

    # --- 23. rv_daily_prior ---
    if len(prior_days_1m) >= 1 and prior_days_1m[0] is not None:
        result["rv_daily_prior"] = _full_day_rv(prior_days_1m[0])
    else:
        result["rv_daily_prior"] = np.nan

    # --- 24. rv_weekly ---
    prior_rvs = []
    for i in range(min(5, len(prior_days_1m))):
        rv_val = _full_day_rv(prior_days_1m[i])
        if np.isfinite(rv_val):
            prior_rvs.append(rv_val)
    result["rv_weekly"] = float(np.mean(prior_rvs)) if prior_rvs else np.nan

    # --- 25. rv_monthly ---
    prior_rvs_month = []
    for i in range(min(22, len(prior_days_1m))):
        rv_val = _full_day_rv(prior_days_1m[i])
        if np.isfinite(rv_val):
            prior_rvs_month.append(rv_val)
    result["rv_monthly"] = float(np.mean(prior_rvs_month)) if prior_rvs_month else np.nan

    # --- 26. rv_acceleration ---
    # Change in rolling 30-min RV: rv_6bar(current) - rv_6bar(30 min ago)
    if n_ret >= 12:
        rv_6bar_now = float(np.sum(returns_5m[-6:] ** 2))
        rv_6bar_prev = float(np.sum(returns_5m[-12:-6] ** 2))
        result["rv_acceleration"] = rv_6bar_now - rv_6bar_prev
    else:
        result["rv_acceleration"] = np.nan

    # --- 27. intraday_vol_of_vol ---
    # Std dev of rolling 6-bar RV windows within the session
    if n_ret >= 8:
        rolling_rv = []
        for i in range(6, n_ret + 1):
            rolling_rv.append(float(np.sum(returns_5m[i - 6: i] ** 2)))
        result["intraday_vol_of_vol"] = float(np.std(rolling_rv, ddof=1)) if len(rolling_rv) > 1 else np.nan
    else:
        result["intraday_vol_of_vol"] = np.nan

    # --- 28. rv_surprise ---
    # Today's partial RV vs average of prior days' same-time-of-day RV
    if prior_days_1m and n_ret >= 2:
        same_time_rvs = []
        n_bars_today = len(close)
        for day_df in prior_days_1m[:10]:
            if day_df is None or day_df.empty:
                continue
            day_close = day_df["close"].values.astype(float)
            # Truncate prior day to same number of bars as today
            truncated = day_close[:n_bars_today]
            if len(truncated) >= 6:
                ret = _get_5min_log_returns(truncated)
                if len(ret) >= 2:
                    same_time_rvs.append(float(np.sum(ret ** 2)))
        if same_time_rvs:
            avg_prior_rv = float(np.mean(same_time_rvs))
            result["rv_surprise"] = rv_5min / max(avg_prior_rv, 1e-10)
        else:
            result["rv_surprise"] = np.nan
    else:
        result["rv_surprise"] = np.nan

    # --- 29. max_5min_return_abs ---
    result["max_5min_return_abs"] = float(np.max(np.abs(returns_5m)))

    # --- 30. overnight_to_intraday_ratio ---
    if len(prior_days_1m) >= 1 and prior_days_1m[0] is not None and not prior_days_1m[0].empty:
        prior_close = prior_days_1m[0]["close"].values.astype(float)
        if len(prior_close) > 0 and prior_close[-1] > 0 and open_arr[0] > 0:
            overnight_ret = np.log(open_arr[0] / prior_close[-1])
            result["overnight_to_intraday_ratio"] = float(
                overnight_ret ** 2 / max(rv_5min, 1e-10)
            )
        else:
            result["overnight_to_intraday_ratio"] = np.nan
    else:
        result["overnight_to_intraday_ratio"] = np.nan

    return result
