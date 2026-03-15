"""
Order flow features for v8 model (~25 features).

Computes put-call ratios, volume analysis, OI changes, and flow indicators
from options chain data. All functions use numpy/pandas only.
Returns np.nan when insufficient data.
"""

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division returning nan on zero denominator."""
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _pc_ratio(df: pd.DataFrame, oi_col: str = 'open_interest') -> float:
    """Put/call ratio from a DataFrame with 'right' and an OI/volume column."""
    if df.empty or oi_col not in df.columns:
        return np.nan
    puts = df.loc[df['right'].str.upper() == 'P', oi_col].sum()
    calls = df.loc[df['right'].str.upper() == 'C', oi_col].sum()
    return puts / max(calls, 1)


def _merge_chain_volume(chain: pd.DataFrame, chain_volume: pd.DataFrame) -> pd.DataFrame:
    """Merge volume data onto chain by strike and right."""
    merged = chain.merge(
        chain_volume[['strike', 'right', 'volume']],
        on=['strike', 'right'],
        how='left',
    )
    merged['volume'] = merged['volume'].fillna(0)
    return merged


def compute_order_flow_features(chain: pd.DataFrame, chain_oi: pd.DataFrame,
                                spx_price: float,
                                prior_oi: pd.DataFrame = None,
                                chain_volume: pd.DataFrame = None) -> dict:
    """
    Compute ~25 order flow features from options chain data.

    Args:
        chain: DataFrame with [strike, right, bid, ask, delta, implied_vol, underlying_price]
        chain_oi: DataFrame with [strike, right, open_interest]
        spx_price: Current SPX price
        prior_oi: Optional prior day's OI for change features
        chain_volume: Optional DataFrame with [strike, right, volume] for intraday volume
    Returns:
        dict of feature_name -> float
    """
    features = {}

    # Validate inputs
    if chain is None or chain.empty or chain_oi is None or chain_oi.empty:
        return {k: np.nan for k in _all_feature_names()}

    # Normalize right column to uppercase
    chain = chain.copy()
    chain['right'] = chain['right'].str.upper()
    chain_oi = chain_oi.copy()
    chain_oi['right'] = chain_oi['right'].str.upper()

    has_volume = chain_volume is not None and not chain_volume.empty
    has_prior_oi = prior_oi is not None and not prior_oi.empty

    if has_volume:
        chain_volume = chain_volume.copy()
        chain_volume['right'] = chain_volume['right'].str.upper()

    if has_prior_oi:
        prior_oi = prior_oi.copy()
        prior_oi['right'] = prior_oi['right'].str.upper()

    # ---- Put-Call Ratios (5) ----

    # 1. pc_ratio_oi
    features['pc_ratio_oi'] = _pc_ratio(chain_oi, 'open_interest')

    # 2. pc_ratio_volume
    if has_volume:
        features['pc_ratio_volume'] = _pc_ratio(chain_volume, 'volume')
    else:
        features['pc_ratio_volume'] = np.nan

    # 3. pc_ratio_0dte - P/C ratio for 0DTE options
    # If chain has an 'expiry' or 'dte' column we can filter; otherwise treat all as 0DTE
    if 'dte' in chain_oi.columns:
        oi_0dte = chain_oi[chain_oi['dte'] == 0]
    else:
        oi_0dte = chain_oi  # assume all are 0DTE if no dte column
    features['pc_ratio_0dte'] = _pc_ratio(oi_0dte, 'open_interest')

    # 4. pc_ratio_near_money - within ±2% of spot
    near_money_mask = (chain_oi['strike'] >= spx_price * 0.98) & (chain_oi['strike'] <= spx_price * 1.02)
    features['pc_ratio_near_money'] = _pc_ratio(chain_oi[near_money_mask], 'open_interest')

    # 5. pc_ratio_far_otm - beyond ±5% of spot
    far_otm_mask = (chain_oi['strike'] < spx_price * 0.95) | (chain_oi['strike'] > spx_price * 1.05)
    features['pc_ratio_far_otm'] = _pc_ratio(chain_oi[far_otm_mask], 'open_interest')

    # ---- Volume Analysis (6) ----

    if has_volume:
        vol_merged = _merge_chain_volume(chain, chain_volume)
        total_vol = vol_merged['volume'].sum()

        features['total_option_volume'] = float(total_vol)
        features['call_volume_total'] = float(vol_merged.loc[vol_merged['right'] == 'C', 'volume'].sum())
        features['put_volume_total'] = float(vol_merged.loc[vol_merged['right'] == 'P', 'volume'].sum())

        # volume_near_money_pct - within ±1% of spot
        near_1pct = (vol_merged['strike'] >= spx_price * 0.99) & (vol_merged['strike'] <= spx_price * 1.01)
        features['volume_near_money_pct'] = (
            vol_merged.loc[near_1pct, 'volume'].sum() / max(total_vol, 1) * 100
        )

        # volume_otm_put_pct - puts with delta between -0.30 and -0.05
        if 'delta' in vol_merged.columns:
            otm_put_mask = (vol_merged['right'] == 'P') & (vol_merged['delta'] >= -0.30) & (vol_merged['delta'] <= -0.05)
            features['volume_otm_put_pct'] = (
                vol_merged.loc[otm_put_mask, 'volume'].sum() / max(total_vol, 1) * 100
            )

            # volume_otm_call_pct - calls with delta between 0.05 and 0.30
            otm_call_mask = (vol_merged['right'] == 'C') & (vol_merged['delta'] >= 0.05) & (vol_merged['delta'] <= 0.30)
            features['volume_otm_call_pct'] = (
                vol_merged.loc[otm_call_mask, 'volume'].sum() / max(total_vol, 1) * 100
            )
        else:
            features['volume_otm_put_pct'] = np.nan
            features['volume_otm_call_pct'] = np.nan
    else:
        for k in ['total_option_volume', 'call_volume_total', 'put_volume_total',
                   'volume_near_money_pct', 'volume_otm_put_pct', 'volume_otm_call_pct']:
            features[k] = np.nan

    # ---- OI Changes (6) ----

    if has_prior_oi:
        # Merge current and prior OI
        oi_merged = chain_oi.merge(
            prior_oi[['strike', 'right', 'open_interest']],
            on=['strike', 'right'],
            how='outer',
            suffixes=('_curr', '_prior'),
        )
        oi_merged['open_interest_curr'] = oi_merged['open_interest_curr'].fillna(0)
        oi_merged['open_interest_prior'] = oi_merged['open_interest_prior'].fillna(0)
        oi_merged['oi_change'] = oi_merged['open_interest_curr'] - oi_merged['open_interest_prior']

        features['oi_change_total'] = float(oi_merged['oi_change'].sum())
        features['oi_change_calls'] = float(
            oi_merged.loc[oi_merged['right'] == 'C', 'oi_change'].sum()
        )
        features['oi_change_puts'] = float(
            oi_merged.loc[oi_merged['right'] == 'P', 'oi_change'].sum()
        )

        # oi_change_near_money - within ±2% of spot
        near_money_oi = (oi_merged['strike'] >= spx_price * 0.98) & (oi_merged['strike'] <= spx_price * 1.02)
        features['oi_change_near_money'] = float(oi_merged.loc[near_money_oi, 'oi_change'].sum())

        # oi_change_at_short_put - nearest strike to 10-delta put
        _compute_oi_change_at_delta(chain, oi_merged, features, 'P', -0.10, 'oi_change_at_short_put')

        # oi_change_at_short_call - nearest strike to 10-delta call
        _compute_oi_change_at_delta(chain, oi_merged, features, 'C', 0.10, 'oi_change_at_short_call')
    else:
        for k in ['oi_change_total', 'oi_change_calls', 'oi_change_puts',
                   'oi_change_near_money', 'oi_change_at_short_put', 'oi_change_at_short_call']:
            features[k] = np.nan

    # ---- Flow Indicators (8) ----

    # 18. net_delta_of_new_positions
    # 19. net_gamma_of_new_positions
    if has_prior_oi and 'delta' in chain.columns:
        # Merge delta/gamma onto oi_merged
        chain_deltas = chain[['strike', 'right', 'delta']].copy()
        if 'gamma' in chain.columns:
            chain_deltas['gamma'] = chain['gamma']
        else:
            chain_deltas['gamma'] = np.nan

        flow = oi_merged.merge(chain_deltas, on=['strike', 'right'], how='left')
        new_pos = flow[flow['oi_change'] > 0]

        delta_vals = new_pos['delta'].values
        oi_chg_vals = new_pos['oi_change'].values
        valid_delta = ~np.isnan(delta_vals)
        features['net_delta_of_new_positions'] = float(
            np.nansum(delta_vals[valid_delta] * oi_chg_vals[valid_delta])
        ) if valid_delta.any() else np.nan

        gamma_vals = new_pos['gamma'].values
        valid_gamma = ~np.isnan(gamma_vals)
        features['net_gamma_of_new_positions'] = float(
            np.nansum(gamma_vals[valid_gamma] * oi_chg_vals[valid_gamma])
        ) if valid_gamma.any() else np.nan
    else:
        features['net_delta_of_new_positions'] = np.nan
        features['net_gamma_of_new_positions'] = np.nan

    # 20. unusual_activity_score
    if has_volume:
        vol_oi = chain_volume.merge(chain_oi, on=['strike', 'right'], how='inner')
        strikes_with_oi = vol_oi[vol_oi['open_interest'] > 0]
        if len(strikes_with_oi) > 0:
            unusual_count = (strikes_with_oi['volume'] > 2 * strikes_with_oi['open_interest']).sum()
            features['unusual_activity_score'] = float(unusual_count) / len(strikes_with_oi)
        else:
            features['unusual_activity_score'] = 0.0
    else:
        features['unusual_activity_score'] = np.nan

    # 21-23. Premium features
    if has_volume:
        vol_chain = _merge_chain_volume(chain, chain_volume)
        vol_chain['mid'] = (vol_chain['bid'] + vol_chain['ask']) / 2.0

        put_mask = vol_chain['right'] == 'P'
        call_mask = vol_chain['right'] == 'C'

        features['premium_spent_puts'] = float(
            (vol_chain.loc[put_mask, 'mid'] * vol_chain.loc[put_mask, 'volume'] * 100).sum()
        )
        features['premium_spent_calls'] = float(
            (vol_chain.loc[call_mask, 'mid'] * vol_chain.loc[call_mask, 'volume'] * 100).sum()
        )
        features['premium_put_call_ratio'] = (
            features['premium_spent_puts'] / max(features['premium_spent_calls'], 1)
        )
    else:
        features['premium_spent_puts'] = np.nan
        features['premium_spent_calls'] = np.nan
        features['premium_put_call_ratio'] = np.nan

    # 24. skew_flow_signal
    # Change in 25d skew * put volume concentration
    if 'delta' in chain.columns and 'implied_vol' in chain.columns and has_volume:
        puts = chain[chain['right'] == 'P'].copy()
        calls = chain[chain['right'] == 'C'].copy()

        # Find 25-delta put IV
        put_25d_iv = _find_iv_at_delta(puts, -0.25)
        # Find 25-delta call IV
        call_25d_iv = _find_iv_at_delta(calls, 0.25)

        if not np.isnan(put_25d_iv) and not np.isnan(call_25d_iv):
            skew_25d = put_25d_iv - call_25d_iv
            # Put volume concentration = put_volume / total_volume
            total_vol = features.get('total_option_volume', 0)
            put_vol = features.get('put_volume_total', 0)
            put_concentration = put_vol / max(total_vol, 1)
            features['skew_flow_signal'] = skew_25d * put_concentration
        else:
            features['skew_flow_signal'] = np.nan
    else:
        features['skew_flow_signal'] = np.nan

    # 25. oi_weighted_avg_delta
    if 'delta' in chain.columns:
        chain_with_oi = chain.merge(chain_oi[['strike', 'right', 'open_interest']],
                                    on=['strike', 'right'], how='inner')
        valid = chain_with_oi.dropna(subset=['delta'])
        if len(valid) > 0 and valid['open_interest'].sum() > 0:
            features['oi_weighted_avg_delta'] = float(
                (valid['delta'] * valid['open_interest']).sum() / valid['open_interest'].sum()
            )
        else:
            features['oi_weighted_avg_delta'] = np.nan
    else:
        features['oi_weighted_avg_delta'] = np.nan

    return features


def _compute_oi_change_at_delta(chain: pd.DataFrame, oi_merged: pd.DataFrame,
                                features: dict, right: str, target_delta: float,
                                feature_name: str):
    """Find the strike nearest to target_delta and get its OI change."""
    if 'delta' not in chain.columns:
        features[feature_name] = np.nan
        return

    side = chain[chain['right'] == right].copy()
    if side.empty:
        features[feature_name] = np.nan
        return

    side = side.dropna(subset=['delta'])
    if side.empty:
        features[feature_name] = np.nan
        return

    # Find strike closest to target delta
    side['delta_dist'] = (side['delta'] - target_delta).abs()
    best_strike = side.loc[side['delta_dist'].idxmin(), 'strike']

    # Look up OI change for this strike
    match = oi_merged[(oi_merged['strike'] == best_strike) & (oi_merged['right'] == right)]
    if not match.empty:
        features[feature_name] = float(match['oi_change'].iloc[0])
    else:
        features[feature_name] = np.nan


def _find_iv_at_delta(options: pd.DataFrame, target_delta: float) -> float:
    """Find implied vol at the strike nearest to target_delta."""
    if options.empty or 'delta' not in options.columns or 'implied_vol' not in options.columns:
        return np.nan

    valid = options.dropna(subset=['delta', 'implied_vol'])
    if valid.empty:
        return np.nan

    idx = (valid['delta'] - target_delta).abs().idxmin()
    return float(valid.loc[idx, 'implied_vol'])


def _all_feature_names() -> list:
    """Return all feature names for nan-filling on bad input."""
    return [
        'pc_ratio_oi', 'pc_ratio_volume', 'pc_ratio_0dte',
        'pc_ratio_near_money', 'pc_ratio_far_otm',
        'total_option_volume', 'call_volume_total', 'put_volume_total',
        'volume_near_money_pct', 'volume_otm_put_pct', 'volume_otm_call_pct',
        'oi_change_total', 'oi_change_calls', 'oi_change_puts',
        'oi_change_near_money', 'oi_change_at_short_put', 'oi_change_at_short_call',
        'net_delta_of_new_positions', 'net_gamma_of_new_positions',
        'unusual_activity_score',
        'premium_spent_puts', 'premium_spent_calls', 'premium_put_call_ratio',
        'skew_flow_signal', 'oi_weighted_avg_delta',
    ]
