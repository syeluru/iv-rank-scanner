"""
V8 Feature Store — Master orchestrator for all feature families.

Combines 10 feature modules into a single feature vector per entry-time slot.
Total: ~200+ features for short IC and long IC prediction.

Usage:
    from ml.features.v8.feature_store import compute_all_features
    features = compute_all_features(data)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from ml.features.v8.realized_vol import compute_realized_vol_features
from ml.features.v8.time_event import compute_time_event_features
from ml.features.v8.vol_surface import compute_vol_surface_features
from ml.features.v8.microstructure import compute_microstructure_features
from ml.features.v8.greeks_aggregate import compute_greeks_aggregate_features
from ml.features.v8.positioning import compute_positioning_features
from ml.features.v8.order_flow import compute_order_flow_features
from ml.features.v8.cross_asset import compute_cross_asset_features
from ml.features.v8.market_internals import compute_market_internals_features
from ml.features.v8.price_action import compute_price_action_features
from ml.features.v8.support_resistance import compute_support_resistance_features
from ml.features.v8.trendlines import compute_trendline_features
from ml.features.v8.statistical_regime import compute_statistical_regime_features
from ml.features.technical_indicators import compute_all_technical_indicators


def compute_all_features(data: dict) -> dict:
    """
    Compute all ~200+ features from raw data.

    Args:
        data: dict with keys:
            'spx_1m': pd.DataFrame - SPX 1-min OHLC [timestamp, open, high, low, close]
            'vix_1m': pd.DataFrame - VIX 1-min OHLC (optional)
            'chain': pd.DataFrame - Option chain snapshot [strike, right, bid, ask, delta,
                     gamma, theta, vega, implied_vol, underlying_price]
                     Optional higher-order greek columns: [vanna, charm, vomma, veta,
                     speed, zomma, color, ultima]
            'chain_oi': pd.DataFrame - Open interest [strike, right, open_interest]
            'chain_volume': pd.DataFrame - Option volume [strike, right, volume] (optional)
            'chain_trades': pd.DataFrame - Trade data [strike, right, price, size,
                            timestamp] (optional)
            'prior_chain': pd.DataFrame - Prior chain snapshot for change features (optional)
            'prior_oi': pd.DataFrame - Prior day OI (optional)
            'prior_days_1m': list[pd.DataFrame] - Prior days' 1-min data, most recent
                             first (optional)
            'prior_greeks': dict - Prior aggregate greeks for change features (optional)
            'cross_data': dict - Cross-asset data (see cross_asset.py for schema)
            'spx_price': float - Current SPX price
            'entry_time': datetime - Current timestamp (ET, timezone-aware or naive)
            'prior_ic_results': list[float] - Recent IC P&L results (optional)

    Returns:
        dict of feature_name -> float (~200+ features)
    """
    features = {}

    spx_1m = data.get('spx_1m', pd.DataFrame())
    vix_1m = data.get('vix_1m')
    chain = data.get('chain', pd.DataFrame())
    chain_oi = data.get('chain_oi', pd.DataFrame())
    spx_price = data.get('spx_price', 0.0)
    entry_time = data.get('entry_time', datetime.now())

    # 1. Realized Volatility (~30 features)
    rv_features = compute_realized_vol_features(
        spx_1m=spx_1m,
        prior_days_1m=data.get('prior_days_1m'),
    )
    features.update(rv_features)

    # 2. Time & Event (~25 features)
    time_features = compute_time_event_features(
        entry_time=entry_time,
        prior_ic_results=data.get('prior_ic_results'),
    )
    features.update(time_features)

    # 3. Volatility Surface (~30 features)
    vol_features = compute_vol_surface_features(
        chain=chain,
        spx_price=spx_price,
        prior_chain=data.get('prior_chain'),
        rv_annualized=rv_features.get('rv_annualized'),
    )
    features.update(vol_features)

    # 4. Microstructure (~25 features)
    micro_features = compute_microstructure_features(
        chain=chain,
        spx_price=spx_price,
        chain_trades=data.get('chain_trades'),
    )
    features.update(micro_features)

    # 5. Greeks Aggregate (~30 features)
    minutes_to_close = time_features.get('minutes_to_close', 390)
    greeks_features = compute_greeks_aggregate_features(
        chain=chain,
        chain_oi=chain_oi,
        spx_price=spx_price,
        minutes_to_close=minutes_to_close,
        prior_greeks=data.get('prior_greeks'),
    )
    features.update(greeks_features)

    # 6. Positioning (~20 features)
    pos_features = compute_positioning_features(
        chain=chain,
        chain_oi=chain_oi,
        spx_price=spx_price,
        minutes_to_close=minutes_to_close,
    )
    features.update(pos_features)

    # 7. Order Flow (~25 features)
    flow_features = compute_order_flow_features(
        chain=chain,
        chain_oi=chain_oi,
        spx_price=spx_price,
        prior_oi=data.get('prior_oi'),
        chain_volume=data.get('chain_volume'),
    )
    features.update(flow_features)

    # 8. Cross Asset (~25 features)
    cross_features = compute_cross_asset_features(
        cross_data=data.get('cross_data', {}),
    )
    features.update(cross_features)

    # 9. Market Internals (~15 features)
    internals_features = compute_market_internals_features(
        spx_1m=spx_1m,
        vix_1m=vix_1m,
    )
    features.update(internals_features)

    # 10. Price Action (~32 features)
    pa_features = compute_price_action_features(
        spx_1m=spx_1m,
        vix_1m=vix_1m,
    )
    features.update(pa_features)

    # 11. Support & Resistance (~30 features)
    sr_features = compute_support_resistance_features(
        spx_1m=spx_1m,
        prior_days_1m=data.get('prior_days_1m'),
        spx_price=spx_price,
    )
    features.update(sr_features)

    # 12. Trendlines (~21 features)
    tl_features = compute_trendline_features(
        spx_1m=spx_1m,
        prior_days_1m=data.get('prior_days_1m'),
        spx_price=spx_price,
    )
    features.update(tl_features)

    # 13. Statistical & Regime (~14 features)
    stat_features = compute_statistical_regime_features(
        spx_1m=spx_1m,
        prior_days_1m=data.get('prior_days_1m'),
    )
    features.update(stat_features)

    # 14. Technical Indicators (11 features from existing v4 module)
    # Resample 1-min to 5-min for the technical indicators
    spx_5m = _resample_to_5min(spx_1m) if not spx_1m.empty else pd.DataFrame()
    tech_features = compute_all_technical_indicators(spx_5m)
    # Prefix to avoid collisions with market_internals
    features.update({f"tech_{k}": v for k, v in tech_features.items()})

    # 11. Interaction & Regime Features (~15 additional)
    interaction_features = _compute_interaction_features(features)
    features.update(interaction_features)

    return features


def _resample_to_5min(spx_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLC to 5-min OHLC."""
    if spx_1m.empty:
        return pd.DataFrame()

    df = spx_1m.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).dropna()

    return resampled.reset_index()


def _compute_interaction_features(features: dict) -> dict:
    """
    Compute ~15 interaction and derived features from the base feature set.
    These capture non-linear relationships between feature families.
    """
    result = {}

    def _safe_get(key, default=np.nan):
        v = features.get(key, default)
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    def _is_valid(v) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        return True

    # VRP x GEX sign: High VRP + positive GEX = ideal short IC entry
    vrp = _safe_get('vrp', 0)
    gex_sign = _safe_get('gex_sign', 0)
    result['vrp_x_gex_sign'] = vrp * gex_sign if _is_valid(vrp) else np.nan

    # Skew x VIX percentile: Steep skew in high-VIX = extreme tail pricing
    skew = _safe_get('skew_25d', 0)
    vix_pct = _safe_get('vix_percentile_1y', 50)
    result['skew_x_vix_percentile'] = skew * (vix_pct / 100) if _is_valid(skew) else np.nan

    # Time remaining x Gamma: How dangerous is gamma given time left
    minutes_to_close = _safe_get('minutes_to_close', 390)
    ic_gamma = _safe_get('ic_net_gamma', 0)
    if _is_valid(minutes_to_close):
        time_fraction = max(1 - minutes_to_close / 390, 0)
    else:
        time_fraction = np.nan
    result['time_x_gamma'] = (
        time_fraction * abs(ic_gamma) if _is_valid(time_fraction) else np.nan
    )

    # RV/IV x Term structure: Multi-factor risk composite
    iv_rv = _safe_get('iv_rv_ratio', 1)
    term_slope = _safe_get('term_slope_0d_1d', 0)
    result['iv_rv_x_term_slope'] = iv_rv * term_slope if _is_valid(iv_rv) else np.nan

    # FOMC x VRP: Event-specific premium capture
    is_fomc = _safe_get('is_fomc_day', 0)
    result['fomc_x_vrp'] = is_fomc * vrp if _is_valid(vrp) else np.nan

    # Semivariance ratio x GEX sign: Downside-dominated + negative GEX = danger
    semi_ratio = _safe_get('semivariance_ratio', 1)
    result['semivar_x_gex'] = semi_ratio * gex_sign if _is_valid(semi_ratio) else np.nan

    # Jump detected x VIX change: Jumps in rising VIX = cascading risk
    jump = _safe_get('jump_detected', 0)
    vix_change = _safe_get('vix_change_1d_pct', 0)
    result['jump_x_vix_change'] = (
        jump * abs(vix_change) if _is_valid(vix_change) else np.nan
    )

    # Opening range x ADX: Wide range + high trend strength = trending day
    orb_width = _safe_get('opening_range_width', 0)
    adx = _safe_get('tech_adx', 0)
    result['orb_width_x_adx'] = orb_width * (adx / 100) if _is_valid(adx) else np.nan

    # Quote imbalance x Volume: Strong imbalance with high volume = directional pressure
    imbalance = _safe_get('imbalance_asymmetry', 0)
    vol_ratio = _safe_get('pc_ratio_volume', 1)
    result['imbalance_x_pc_ratio'] = (
        imbalance * vol_ratio if _is_valid(imbalance) else np.nan
    )

    # Prob stay in IC x VRP: High prob + high premium = best setup
    prob_stay = _safe_get('prob_stay_in_ic', 0.8)
    result['prob_stay_x_vrp'] = (
        prob_stay * max(vrp, 0) if _is_valid(prob_stay) else np.nan
    )

    # Charm pressure x Time: Charm intensifies near close
    charm_pressure = _safe_get('charm_pressure', 0)
    result['charm_x_time_fraction'] = (
        charm_pressure * time_fraction if _is_valid(charm_pressure) else np.nan
    )

    # Return autocorrelation x Efficiency ratio: Mean-reversion confirmation
    autocorr = _safe_get('return_autocorrelation', 0)
    eff_ratio = _safe_get('intraday_efficiency_ratio', 0.5)
    result['autocorr_x_efficiency'] = (
        autocorr * (1 - eff_ratio) if _is_valid(autocorr) else np.nan
    )

    # Max pain distance x OI concentration: Pin effect strength
    mp_dist = _safe_get('max_pain_abs_distance', 0)
    oi_conc = _safe_get('oi_concentration', 0)
    result['max_pain_x_oi_conc'] = (
        (1 - mp_dist) * oi_conc if _is_valid(mp_dist) else np.nan
    )

    # Overnight gap x VIX level: Large gaps in high-VIX = continuation risk
    overnight = _safe_get('spx_overnight_abs', 0)
    vix = _safe_get('vix_level', 20)
    result['gap_x_vix'] = overnight * (vix / 20) if _is_valid(overnight) else np.nan

    # Composite regime score: Synthesize multiple signals into regime context
    # Positive = favorable for short IC, negative = favorable for long IC
    safe_gex = gex_sign if _is_valid(gex_sign) else 0
    safe_vrp = 1 if (_is_valid(vrp) and vrp > 0) else (-1 if _is_valid(vrp) else 0)
    safe_autocorr = (
        -1 if (_is_valid(autocorr) and autocorr < -0.1)
        else (1 if (_is_valid(autocorr) and autocorr > 0.1) else 0)
    )
    safe_jump = -1 if (_is_valid(jump) and jump > 0) else 0
    result['regime_composite'] = safe_gex + safe_vrp + safe_autocorr + safe_jump

    return result


def get_feature_names() -> list[str]:
    """Return ordered list of all feature names produced by compute_all_features."""
    import warnings
    warnings.filterwarnings('ignore')

    # Build empty DataFrames with correct columns so modules that check
    # column existence don't raise KeyError on empty input.
    dummy_chain = pd.DataFrame(columns=[
        'strike', 'right', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega',
        'implied_vol', 'underlying_price', 'bid_size', 'ask_size',
        'vanna', 'charm', 'vomma', 'veta', 'speed', 'zomma', 'color', 'ultima',
    ])
    dummy_chain_oi = pd.DataFrame(columns=['strike', 'right', 'open_interest'])
    dummy_data = {
        'spx_1m': pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close']),
        'chain': dummy_chain,
        'chain_oi': dummy_chain_oi,
        'cross_data': {},
        'spx_price': 0.0,
        'entry_time': datetime(2024, 1, 2, 12, 0),
    }
    features = compute_all_features(dummy_data)
    return sorted(features.keys())


def count_features() -> int:
    """Return total number of features."""
    return len(get_feature_names())
