"""
Cross-asset features for v8 model (~25 features).

Computes VIX complex, rates/credit, currency/commodity, and cross-market
features from daily data. Uses numpy/pandas only.
Returns np.nan when insufficient data.
"""

import numpy as np


def compute_cross_asset_features(cross_data: dict) -> dict:
    """
    Compute ~25 cross-asset features from daily market data.

    Args:
        cross_data: dict with keys:
            'vix': float - Current VIX level
            'vix_prior': float - Prior day VIX close
            'vix_history': list[float] - Last 30 VIX closes (most recent first)
            'vvix': float - VVIX level (nan if unavailable)
            'vix9d': float - VIX9D level (nan if unavailable)
            'vix3m': float - VIX3M level (nan if unavailable)
            'vix_future_1': float - VX1 front-month VIX future (nan if unavailable)
            'vix_future_2': float - VX2 second-month VIX future (nan if unavailable)
            'yield_10y': float - 10-year treasury yield
            'yield_10y_prior': float - Prior day 10Y yield
            'yield_2y': float - 2-year treasury yield
            'hy_oas': float - High-yield OAS (nan if unavailable)
            'hy_oas_prior': float - Prior day HY OAS
            'dxy': float - Dollar index (nan if unavailable)
            'dxy_prior': float
            'gold_change_pct': float - Gold % change today
            'oil_change_pct': float - Oil % change today
            'spx_overnight_return': float - (open - prior_close) / prior_close
            'es_volume_ratio': float - Today's ES futures volume / 20-day avg (nan if unavailable)
    Returns:
        dict of feature_name -> float
    """
    features = {}

    # Safe getter with nan default
    def g(key: str, default=np.nan) -> float:
        val = cross_data.get(key, default)
        if val is None:
            return np.nan
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    vix = g('vix')
    vix_prior = g('vix_prior')
    vix_history = cross_data.get('vix_history', [])
    if vix_history is None:
        vix_history = []
    # Convert to numpy, filtering out None/nan
    vix_hist = []
    for v in vix_history:
        try:
            fv = float(v)
            if not np.isnan(fv):
                vix_hist.append(fv)
        except (TypeError, ValueError):
            pass
    vix_arr = np.array(vix_hist, dtype=np.float64) if vix_hist else np.array([], dtype=np.float64)

    vvix = g('vvix')
    vix9d = g('vix9d')
    vix3m = g('vix3m')
    vf1 = g('vix_future_1')
    vf2 = g('vix_future_2')
    yield_10y = g('yield_10y')
    yield_10y_prior = g('yield_10y_prior')
    yield_2y = g('yield_2y')
    hy_oas = g('hy_oas')
    hy_oas_prior = g('hy_oas_prior')
    dxy = g('dxy')
    dxy_prior = g('dxy_prior')
    gold_chg = g('gold_change_pct')
    oil_chg = g('oil_change_pct')
    spx_overnight = g('spx_overnight_return')
    es_vol_ratio = g('es_volume_ratio')

    # ---- VIX Complex (12) ----

    # 1. vix_level
    features['vix_level'] = vix

    # 2. vix_percentile_1y
    if len(vix_arr) >= 1 and not np.isnan(vix):
        lookback = min(len(vix_arr), 252)
        window = vix_arr[:lookback]
        features['vix_percentile_1y'] = float(np.sum(window < vix) / len(window) * 100)
    else:
        features['vix_percentile_1y'] = np.nan

    # 3. vix_change_1d
    features['vix_change_1d'] = _safe_sub(vix, vix_prior)

    # 4. vix_change_1d_pct
    if not np.isnan(vix) and not np.isnan(vix_prior):
        features['vix_change_1d_pct'] = (vix - vix_prior) / max(vix_prior, 1e-6) * 100
    else:
        features['vix_change_1d_pct'] = np.nan

    # 5. vix_sma_10d_dist
    if len(vix_arr) >= 10 and not np.isnan(vix):
        sma_10 = np.mean(vix_arr[:10])
        features['vix_sma_10d_dist'] = (vix - sma_10) / max(sma_10, 1e-6) * 100
    else:
        features['vix_sma_10d_dist'] = np.nan

    # 6. vix_sma_20d_dist
    if len(vix_arr) >= 20 and not np.isnan(vix):
        sma_20 = np.mean(vix_arr[:20])
        features['vix_sma_20d_dist'] = (vix - sma_20) / max(sma_20, 1e-6) * 100
    else:
        features['vix_sma_20d_dist'] = np.nan

    # 7. vvix_level
    features['vvix_level'] = vvix

    # 8. vvix_vix_ratio
    if not np.isnan(vvix) and not np.isnan(vix):
        features['vvix_vix_ratio'] = vvix / max(vix, 1e-6)
    else:
        features['vvix_vix_ratio'] = np.nan

    # 9. vix9d_vs_vix
    if not np.isnan(vix9d) and not np.isnan(vix):
        features['vix9d_vs_vix'] = vix9d - vix
    else:
        features['vix9d_vs_vix'] = np.nan

    # 10. vix_term_slope
    if not np.isnan(vf1) and not np.isnan(vf2):
        features['vix_term_slope'] = vf2 - vf1
    else:
        features['vix_term_slope'] = np.nan

    # 11. vix_term_ratio
    if not np.isnan(vf1) and not np.isnan(vf2):
        features['vix_term_ratio'] = vf1 / max(vf2, 1e-6)
    else:
        features['vix_term_ratio'] = np.nan

    # 12. vix_term_regime
    slope = features.get('vix_term_slope', np.nan)
    if not np.isnan(slope):
        if slope > 2:
            features['vix_term_regime'] = 0.0  # steep contango
        elif slope > 0:
            features['vix_term_regime'] = 1.0  # mild contango
        elif slope > -0.5:
            features['vix_term_regime'] = 2.0  # flat
        elif slope > -2:
            features['vix_term_regime'] = 3.0  # mild backwardation
        else:
            features['vix_term_regime'] = 4.0  # steep backwardation
    else:
        features['vix_term_regime'] = np.nan

    # ---- Rates & Credit (6) ----

    # 13. yield_10y
    features['yield_10y'] = yield_10y

    # 14. yield_10y_change
    features['yield_10y_change'] = _safe_sub(yield_10y, yield_10y_prior)

    # 15. yield_2s10s_spread
    if not np.isnan(yield_10y) and not np.isnan(yield_2y):
        features['yield_2s10s_spread'] = yield_10y - yield_2y
    else:
        features['yield_2s10s_spread'] = np.nan

    # 16. hy_oas_level
    features['hy_oas_level'] = hy_oas

    # 17. hy_oas_change
    features['hy_oas_change'] = _safe_sub(hy_oas, hy_oas_prior)

    # 18. hy_oas_zscore - placeholder, requires history
    features['hy_oas_zscore'] = np.nan

    # ---- Currency & Commodities (4) ----

    # 19. dxy_change_pct
    if not np.isnan(dxy) and not np.isnan(dxy_prior):
        features['dxy_change_pct'] = (dxy - dxy_prior) / max(dxy_prior, 1e-6) * 100
    else:
        features['dxy_change_pct'] = np.nan

    # 20. gold_change_pct
    features['gold_change_pct'] = gold_chg

    # 21. oil_change_pct
    features['oil_change_pct'] = oil_chg

    # 22. risk_off_composite
    vix_chg_pct = features.get('vix_change_1d_pct', np.nan)
    components = [vix_chg_pct, gold_chg, oil_chg, features.get('dxy_change_pct', np.nan)]
    if all(not np.isnan(c) for c in components):
        # risk_off = vix_change_pct + gold_change_pct - oil_change_pct + dxy_change_pct
        features['risk_off_composite'] = components[0] + components[1] - components[2] + components[3]
    else:
        features['risk_off_composite'] = np.nan

    # ---- Cross-Market (3) ----

    # 23. spx_overnight_return
    features['spx_overnight_return'] = spx_overnight

    # 24. spx_overnight_abs
    if not np.isnan(spx_overnight):
        features['spx_overnight_abs'] = abs(spx_overnight)
    else:
        features['spx_overnight_abs'] = np.nan

    # 25. es_volume_ratio
    features['es_volume_ratio'] = es_vol_ratio

    return features


def _safe_sub(a: float, b: float) -> float:
    """Subtract two floats, returning nan if either is nan."""
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b
