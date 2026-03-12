"""Volatility surface features from an option chain snapshot.

Computes ~30 features across skew, term structure, SVI-like parameterization,
variance risk premium, and IV level/dynamics.
"""

import numpy as np
import pandas as pd


def _normalize_right(right: str) -> str:
    """Normalize option right to single character.

    Args:
        right: Option type string ('C', 'CALL', 'call', 'P', 'PUT', 'put').

    Returns:
        'C' for calls, 'P' for puts.
    """
    r = str(right).strip().upper()
    if r in ('C', 'CALL'):
        return 'C'
    if r in ('P', 'PUT'):
        return 'P'
    return r


def _get_atm_iv(chain: pd.DataFrame, spx_price: float, right: str = None) -> float:
    """Get ATM implied vol as the IV at the strike nearest to spot.

    Args:
        chain: Option chain DataFrame with strike, right, implied_vol columns.
        spx_price: Current underlying price.
        right: Optional filter to 'C' or 'P'. If None, average across both.

    Returns:
        ATM implied volatility, or np.nan if unavailable.
    """
    df = chain.copy()
    df['_right'] = df['right'].apply(_normalize_right)

    if right is not None:
        right = _normalize_right(right)
        df = df[df['_right'] == right]

    df = df.dropna(subset=['strike', 'implied_vol'])
    if df.empty:
        return np.nan

    df = df.copy()
    df['_dist'] = (df['strike'] - spx_price).abs()
    nearest = df.loc[df['_dist'] == df['_dist'].min()]
    iv_vals = nearest['implied_vol'].values
    if len(iv_vals) == 0:
        return np.nan
    return float(np.nanmean(iv_vals))


def _get_iv_at_delta(chain: pd.DataFrame, target_delta: float, right: str) -> float:
    """Interpolate IV at a specific delta for a given option right.

    Uses linear interpolation between the two strikes whose deltas bracket
    the target. Put deltas are expected negative; call deltas positive.

    Args:
        chain: Option chain with delta, implied_vol, right columns.
        target_delta: Target delta value (e.g., -0.25 for 25-delta put).
        right: 'C' or 'P'.

    Returns:
        Interpolated IV at the target delta, or np.nan.
    """
    df = chain.copy()
    df['_right'] = df['right'].apply(_normalize_right)
    right = _normalize_right(right)
    df = df[df['_right'] == right].dropna(subset=['delta', 'implied_vol'])

    if len(df) < 2:
        return np.nan

    df = df.sort_values('delta')
    deltas = df['delta'].values
    ivs = df['implied_vol'].values

    # Check if target is within range
    if target_delta < deltas.min() or target_delta > deltas.max():
        # Use nearest point if out of range
        idx = np.argmin(np.abs(deltas - target_delta))
        return float(ivs[idx])

    return float(np.interp(target_delta, deltas, ivs))


def _fit_skew_slope(chain: pd.DataFrame, right: str,
                    target_deltas: list) -> float:
    """Fit a linear regression of IV vs delta at specified delta targets.

    Args:
        chain: Option chain DataFrame.
        right: 'C' or 'P'.
        target_deltas: List of delta values to sample.

    Returns:
        Slope of the linear fit, or np.nan.
    """
    points = []
    for d in target_deltas:
        iv = _get_iv_at_delta(chain, d, right)
        if not np.isnan(iv):
            points.append((d, iv))

    if len(points) < 2:
        return np.nan

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Linear regression: y = mx + b
    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return np.nan
    slope = (n * sxy - sx * sy) / denom
    return float(slope)


def _get_atm_iv_by_expiry(chain: pd.DataFrame, spx_price: float) -> dict:
    """Compute ATM IV grouped by expiration.

    Args:
        chain: Option chain with an 'expiration' column.
        spx_price: Current underlying price.

    Returns:
        Dict mapping expiration -> ATM IV.
    """
    if 'expiration' not in chain.columns:
        return {}

    result = {}
    for exp, grp in chain.groupby('expiration'):
        iv = _get_atm_iv(grp, spx_price)
        if not np.isnan(iv):
            result[exp] = iv
    return result


def _classify_expiry_dte(chain: pd.DataFrame) -> dict:
    """Map expirations to approximate DTE buckets (0, 1, 7, etc.).

    Expects an 'expiration' column that is datetime-like. Uses the minimum
    expiration as 0DTE reference.

    Returns:
        Dict mapping DTE bucket -> expiration value.
    """
    if 'expiration' not in chain.columns:
        return {}

    expirations = pd.to_datetime(chain['expiration'].unique())
    if len(expirations) == 0:
        return {}

    expirations = sorted(expirations)
    base = expirations[0]
    dte_map = {}
    for exp in expirations:
        days = (exp - base).days
        dte_map[days] = exp
    return dte_map


def compute_vol_surface_features(chain: pd.DataFrame, spx_price: float,
                                  prior_chain: pd.DataFrame = None,
                                  rv_annualized: float = None) -> dict:
    """Compute ~30 volatility surface features from an option chain snapshot.

    Args:
        chain: DataFrame with columns [strike, right, bid, ask, implied_vol,
               delta, gamma, theta, vega, underlying_price].
               right is 'C'/'CALL' for calls, 'P'/'PUT' for puts.
        spx_price: Current SPX price.
        prior_chain: Optional prior snapshot for change features.
        rv_annualized: Optional realized vol for VRP calculation.

    Returns:
        Dict of feature_name -> float. Missing features are np.nan.
    """
    features = {}

    # --- Skew (7) ---

    put_25d_iv = _get_iv_at_delta(chain, -0.25, 'P')
    call_25d_iv = _get_iv_at_delta(chain, 0.25, 'C')
    put_10d_iv = _get_iv_at_delta(chain, -0.10, 'P')
    call_10d_iv = _get_iv_at_delta(chain, 0.10, 'C')
    atm_iv = _get_atm_iv(chain, spx_price)

    skew_25d = put_25d_iv - call_25d_iv if not (np.isnan(put_25d_iv) or np.isnan(call_25d_iv)) else np.nan
    skew_10d = put_10d_iv - call_10d_iv if not (np.isnan(put_10d_iv) or np.isnan(call_10d_iv)) else np.nan

    features['skew_25d'] = skew_25d
    features['skew_10d'] = skew_10d

    # Skew slope: IV vs delta for puts at specified deltas
    put_slope_deltas = [-0.10, -0.15, -0.20, -0.25, -0.35]
    skew_slope = _fit_skew_slope(chain, 'P', put_slope_deltas)
    features['skew_slope'] = skew_slope

    # Skew slope z-score using prior chain
    if prior_chain is not None and not prior_chain.empty:
        prior_skew_slope = _fit_skew_slope(prior_chain, 'P', put_slope_deltas)
        if not np.isnan(skew_slope) and not np.isnan(prior_skew_slope):
            mean_slope = (skew_slope + prior_skew_slope) / 2.0
            std_slope = abs(skew_slope - prior_skew_slope) / np.sqrt(2)
            if std_slope > 1e-10:
                features['skew_slope_zscore'] = (skew_slope - mean_slope) / std_slope
            else:
                features['skew_slope_zscore'] = 0.0
        else:
            features['skew_slope_zscore'] = np.nan
    else:
        features['skew_slope_zscore'] = np.nan

    # Skew convexity (butterfly)
    if not (np.isnan(put_25d_iv) or np.isnan(call_25d_iv) or np.isnan(atm_iv)):
        features['skew_convexity'] = (put_25d_iv + call_25d_iv) / 2.0 - atm_iv
    else:
        features['skew_convexity'] = np.nan

    # Skew change
    if prior_chain is not None and not prior_chain.empty:
        prior_put_25d = _get_iv_at_delta(prior_chain, -0.25, 'P')
        prior_call_25d = _get_iv_at_delta(prior_chain, 0.25, 'C')
        if not (np.isnan(prior_put_25d) or np.isnan(prior_call_25d)) and not np.isnan(skew_25d):
            prior_skew_25d = prior_put_25d - prior_call_25d
            features['skew_change'] = skew_25d - prior_skew_25d
        else:
            features['skew_change'] = np.nan
    else:
        features['skew_change'] = np.nan

    # Put/call skew asymmetry
    call_slope_deltas = [0.10, 0.15, 0.20, 0.25, 0.35]
    call_skew_slope = _fit_skew_slope(chain, 'C', call_slope_deltas)
    if not (np.isnan(skew_slope) or np.isnan(call_skew_slope)):
        features['put_call_skew_asymmetry'] = abs(skew_slope) - abs(call_skew_slope)
    else:
        features['put_call_skew_asymmetry'] = np.nan

    # --- Term Structure (5) ---

    dte_map = _classify_expiry_dte(chain)
    expiry_ivs = _get_atm_iv_by_expiry(chain, spx_price)

    iv_0d = expiry_ivs.get(dte_map.get(0)) if 0 in dte_map else np.nan
    iv_1d = expiry_ivs.get(dte_map.get(1)) if 1 in dte_map else np.nan
    iv_7d = expiry_ivs.get(dte_map.get(7)) if 7 in dte_map else np.nan

    if iv_0d is None:
        iv_0d = np.nan
    if iv_1d is None:
        iv_1d = np.nan
    if iv_7d is None:
        iv_7d = np.nan

    term_slope_0d_1d = iv_0d - iv_1d if not (np.isnan(iv_0d) or np.isnan(iv_1d)) else np.nan
    features['term_slope_0d_1d'] = term_slope_0d_1d

    features['term_slope_0d_7d'] = (
        iv_0d - iv_7d if not (np.isnan(iv_0d) or np.isnan(iv_7d)) else np.nan
    )

    features['term_ratio_0d_1d'] = (
        iv_0d / iv_1d if not (np.isnan(iv_0d) or np.isnan(iv_1d)) and abs(iv_1d) > 1e-10 else np.nan
    )

    # Term structure regime based on term_slope_0d_1d
    if np.isnan(term_slope_0d_1d):
        features['term_structure_regime'] = np.nan
    elif term_slope_0d_1d < -0.05:
        features['term_structure_regime'] = 0.0  # steep contango
    elif term_slope_0d_1d < -0.01:
        features['term_structure_regime'] = 1.0  # mild contango
    elif term_slope_0d_1d <= 0.01:
        features['term_structure_regime'] = 2.0  # flat
    elif term_slope_0d_1d <= 0.05:
        features['term_structure_regime'] = 3.0  # mild backwardation
    else:
        features['term_structure_regime'] = 4.0  # steep backwardation

    # Term slope change
    if prior_chain is not None and not prior_chain.empty and 'expiration' in prior_chain.columns:
        prior_dte_map = _classify_expiry_dte(prior_chain)
        prior_expiry_ivs = _get_atm_iv_by_expiry(prior_chain, spx_price)
        prior_iv_0d = prior_expiry_ivs.get(prior_dte_map.get(0)) if 0 in prior_dte_map else np.nan
        prior_iv_1d = prior_expiry_ivs.get(prior_dte_map.get(1)) if 1 in prior_dte_map else np.nan
        if prior_iv_0d is None:
            prior_iv_0d = np.nan
        if prior_iv_1d is None:
            prior_iv_1d = np.nan
        prior_term = prior_iv_0d - prior_iv_1d if not (np.isnan(prior_iv_0d) or np.isnan(prior_iv_1d)) else np.nan
        if not np.isnan(term_slope_0d_1d) and not np.isnan(prior_term):
            features['term_slope_change'] = term_slope_0d_1d - prior_term
        else:
            features['term_slope_change'] = np.nan
    else:
        features['term_slope_change'] = np.nan

    # --- SVI-like Parameters (6) ---

    df_svi = chain.dropna(subset=['strike', 'implied_vol']).copy()
    df_svi = df_svi[df_svi['strike'] > 0]
    if len(df_svi) >= 3 and spx_price > 0:
        k = np.log(df_svi['strike'].values / spx_price)
        iv = df_svi['implied_vol'].values

        # Fit quadratic: IV = a + b*k + c*k^2
        try:
            coeffs = np.polyfit(k, iv, 2)  # [c, b, a]
            svi_c = float(coeffs[0])
            svi_b = float(coeffs[1])
            svi_a = float(coeffs[2])

            fitted = svi_a + svi_b * k + svi_c * k * k
            rmse = float(np.sqrt(np.mean((iv - fitted) ** 2)))

            # Strike of minimum IV: derivative = b + 2ck = 0 => k_min = -b/(2c)
            if abs(svi_c) > 1e-12:
                k_min = -svi_b / (2.0 * svi_c)
                svi_min_strike = float(k_min)  # log-moneyness relative to spot
            else:
                svi_min_strike = np.nan

            features['svi_a'] = svi_a
            features['svi_b'] = svi_b
            features['svi_c'] = svi_c
            features['svi_fit_rmse'] = rmse
            features['svi_min_strike'] = svi_min_strike
            features['svi_curvature_ratio'] = svi_c / max(abs(svi_b), 1e-6)
        except (np.linalg.LinAlgError, ValueError):
            for f in ['svi_a', 'svi_b', 'svi_c', 'svi_fit_rmse',
                       'svi_min_strike', 'svi_curvature_ratio']:
                features[f] = np.nan
    else:
        for f in ['svi_a', 'svi_b', 'svi_c', 'svi_fit_rmse',
                   'svi_min_strike', 'svi_curvature_ratio']:
            features[f] = np.nan

    # --- VRP (5) ---

    if rv_annualized is not None and not np.isnan(rv_annualized) and not np.isnan(atm_iv):
        features['vrp'] = atm_iv ** 2 - rv_annualized ** 2
        features['vrp_sqrt'] = atm_iv - rv_annualized
        features['iv_rv_ratio'] = atm_iv / max(rv_annualized, 1e-6)
        features['vrp_sign'] = 1.0 if features['vrp'] > 0 else -1.0
    else:
        features['vrp'] = np.nan
        features['vrp_sqrt'] = np.nan
        features['iv_rv_ratio'] = np.nan
        features['vrp_sign'] = np.nan

    # Intraday VRP: 0DTE ATM IV vs partial intraday RV
    # Without intraday candle data, use rv_annualized as proxy when available
    if not np.isnan(iv_0d if not isinstance(iv_0d, type(None)) else np.nan) and rv_annualized is not None and not np.isnan(rv_annualized):
        features['intraday_vrp'] = iv_0d - rv_annualized
    else:
        features['intraday_vrp'] = np.nan

    # --- IV Level & Dynamics (7) ---

    # ATM IV for 0DTE specifically
    if 0 in dte_map:
        exp_0d = dte_map[0]
        chain_0d = chain[chain['expiration'] == exp_0d] if 'expiration' in chain.columns else chain
        features['atm_iv_0dte'] = _get_atm_iv(chain_0d, spx_price)
    else:
        # Fall back to overall ATM IV if no expiration column
        features['atm_iv_0dte'] = atm_iv

    features['atm_iv_put'] = _get_atm_iv(chain, spx_price, right='P')
    features['atm_iv_call'] = _get_atm_iv(chain, spx_price, right='C')

    atm_put = features['atm_iv_put']
    atm_call = features['atm_iv_call']
    features['put_call_iv_spread'] = (
        atm_put - atm_call if not (np.isnan(atm_put) or np.isnan(atm_call)) else np.nan
    )

    # IV change from prior
    if prior_chain is not None and not prior_chain.empty:
        prior_atm = _get_atm_iv(prior_chain, spx_price)
        if not np.isnan(atm_iv) and not np.isnan(prior_atm):
            features['iv_change_from_prior'] = atm_iv - prior_atm
        else:
            features['iv_change_from_prior'] = np.nan
    else:
        features['iv_change_from_prior'] = np.nan

    # Short strike IVs at 10-delta
    features['short_put_strike_iv'] = put_10d_iv
    features['short_call_strike_iv'] = call_10d_iv

    return features
