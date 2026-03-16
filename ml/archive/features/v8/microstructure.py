"""Microstructure features from option chain quotes and trade flow.

Computes ~25 features across spread dynamics, quote imbalance, liquidity,
and trade flow analysis.
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


def _get_atm_rows(chain: pd.DataFrame, spx_price: float) -> pd.DataFrame:
    """Select ATM rows: nearest 2 strikes to spot, both puts and calls.

    Args:
        chain: Option chain DataFrame with strike column.
        spx_price: Current underlying price.

    Returns:
        DataFrame filtered to ATM strikes.
    """
    df = chain.dropna(subset=['strike']).copy()
    if df.empty:
        return df

    unique_strikes = np.sort(df['strike'].unique())
    if len(unique_strikes) == 0:
        return df.iloc[:0]

    dists = np.abs(unique_strikes - spx_price)
    n_strikes = min(2, len(unique_strikes))
    if n_strikes <= 1:
        nearest_idx = np.array([np.argmin(dists)]) if len(dists) > 0 else np.array([], dtype=int)
    else:
        nearest_idx = np.argpartition(dists, n_strikes)[:n_strikes]
    nearest_strikes = unique_strikes[nearest_idx]

    return df[df['strike'].isin(nearest_strikes)]


def _get_rows_near_delta(chain: pd.DataFrame, target_delta: float,
                         right: str, tol: float = 0.05) -> pd.DataFrame:
    """Get rows near a target delta for a given right.

    Args:
        chain: Option chain with delta and right columns.
        target_delta: Target delta value.
        right: 'C' or 'P'.
        tol: Delta tolerance window.

    Returns:
        Filtered DataFrame near the target delta.
    """
    df = chain.copy()
    df['_right'] = df['right'].apply(_normalize_right)
    df = df[df['_right'] == right].dropna(subset=['delta'])

    mask = (df['delta'] - target_delta).abs() <= tol
    result = df[mask]

    # If nothing within tolerance, take the single nearest
    if result.empty and not df.empty:
        idx = (df['delta'] - target_delta).abs().idxmin()
        result = df.loc[[idx]]

    return result


def _safe_spread(row: pd.Series) -> float:
    """Compute bid-ask spread from a row, returning nan if data missing."""
    bid = row.get('bid', np.nan)
    ask = row.get('ask', np.nan)
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    return ask - bid


def _safe_midpoint(row: pd.Series) -> float:
    """Compute midpoint from a row."""
    bid = row.get('bid', np.nan)
    ask = row.get('ask', np.nan)
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    return (bid + ask) / 2.0


def compute_microstructure_features(chain: pd.DataFrame, spx_price: float,
                                     chain_trades: pd.DataFrame = None) -> dict:
    """Compute ~25 microstructure features from option chain quotes.

    Args:
        chain: DataFrame with columns [strike, right, bid, ask, implied_vol,
               delta, bid_size, ask_size]. bid_size/ask_size may be NaN.
        spx_price: Current SPX price.
        chain_trades: Optional DataFrame with [strike, right, price, size,
                      timestamp] for trade flow analysis.

    Returns:
        Dict of feature_name -> float. Missing features are np.nan.
    """
    features = {}
    df = chain.copy()
    df['_right'] = df['right'].apply(_normalize_right)
    df['_spread'] = df.apply(_safe_spread, axis=1)
    df['_mid'] = df.apply(_safe_midpoint, axis=1)

    has_sizes = ('bid_size' in df.columns and 'ask_size' in df.columns and
                 df[['bid_size', 'ask_size']].notna().any().all())

    # --- Spread Dynamics (6) ---

    atm = _get_atm_rows(df, spx_price)
    atm_spreads = atm['_spread'].dropna()
    atm_mids = atm['_mid'].dropna()

    atm_spread_abs = float(atm_spreads.mean()) if len(atm_spreads) > 0 else np.nan
    atm_mid_val = float(atm_mids.mean()) if len(atm_mids) > 0 else np.nan

    features['atm_spread_abs'] = atm_spread_abs
    features['atm_spread_relative'] = (
        atm_spread_abs / max(atm_mid_val, 1e-6)
        if not (np.isnan(atm_spread_abs) or np.isnan(atm_mid_val))
        else np.nan
    )

    # Short put spread (10-delta put)
    sp_rows = _get_rows_near_delta(df, -0.10, 'P')
    sp_spreads = sp_rows['_spread'].dropna() if '_spread' in sp_rows.columns else pd.Series(dtype=float)
    features['short_put_spread'] = float(sp_spreads.mean()) if len(sp_spreads) > 0 else np.nan

    # Short call spread (10-delta call)
    sc_rows = _get_rows_near_delta(df, 0.10, 'C')
    sc_spreads = sc_rows['_spread'].dropna() if '_spread' in sc_rows.columns else pd.Series(dtype=float)
    features['short_call_spread'] = float(sc_spreads.mean()) if len(sc_spreads) > 0 else np.nan

    # Spread slope OTM: linear regression of spread vs abs(delta)
    valid = df.dropna(subset=['delta', '_spread'])
    if len(valid) >= 3:
        x = valid['delta'].abs().values
        y = valid['_spread'].values
        n = len(x)
        sx = x.sum()
        sy = y.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        denom = n * sxx - sx * sx
        if abs(denom) > 1e-12:
            features['spread_slope_otm'] = float((n * sxy - sx * sy) / denom)
        else:
            features['spread_slope_otm'] = np.nan
    else:
        features['spread_slope_otm'] = np.nan

    # Wing spread ratio: avg spread at 5-delta / avg spread at 25-delta
    wing5_puts = _get_rows_near_delta(df, -0.05, 'P', tol=0.03)
    wing5_calls = _get_rows_near_delta(df, 0.05, 'C', tol=0.03)
    wing5 = pd.concat([wing5_puts, wing5_calls])
    wing5_spread = wing5['_spread'].dropna().mean() if '_spread' in wing5.columns else np.nan

    wing25_puts = _get_rows_near_delta(df, -0.25, 'P', tol=0.05)
    wing25_calls = _get_rows_near_delta(df, 0.25, 'C', tol=0.05)
    wing25 = pd.concat([wing25_puts, wing25_calls])
    wing25_spread = wing25['_spread'].dropna().mean() if '_spread' in wing25.columns else np.nan

    if not np.isnan(wing5_spread) and not np.isnan(wing25_spread) and abs(wing25_spread) > 1e-6:
        features['wing_spread_ratio'] = float(wing5_spread / wing25_spread)
    else:
        features['wing_spread_ratio'] = np.nan

    # --- Quote Imbalance (5) ---

    def _quote_imbalance(rows: pd.DataFrame) -> float:
        """Compute average (bid_size - ask_size) / (bid_size + ask_size)."""
        if not has_sizes:
            return 0.0
        valid_rows = rows.dropna(subset=['bid_size', 'ask_size'])
        if valid_rows.empty:
            return 0.0
        bs = valid_rows['bid_size'].values.astype(float)
        aks = valid_rows['ask_size'].values.astype(float)
        total = bs + aks
        mask = total > 0
        if not mask.any():
            return 0.0
        imb = (bs[mask] - aks[mask]) / total[mask]
        return float(np.mean(imb))

    features['atm_quote_imbalance'] = _quote_imbalance(atm)

    # OTM puts: delta -0.05 to -0.30
    otm_puts = df[(df['_right'] == 'P') & (df['delta'] >= -0.30) & (df['delta'] <= -0.05)].copy()
    features['put_side_imbalance'] = _quote_imbalance(otm_puts) if not otm_puts.empty else 0.0

    # OTM calls: delta 0.05 to 0.30
    otm_calls = df[(df['_right'] == 'C') & (df['delta'] >= 0.05) & (df['delta'] <= 0.30)].copy()
    features['call_side_imbalance'] = _quote_imbalance(otm_calls) if not otm_calls.empty else 0.0

    features['imbalance_asymmetry'] = features['put_side_imbalance'] - features['call_side_imbalance']

    # Microprice vs mid at ATM
    if has_sizes and not atm.empty:
        atm_valid = atm.dropna(subset=['bid', 'ask', 'bid_size', 'ask_size'])
        if not atm_valid.empty:
            bids = atm_valid['bid'].values.astype(float)
            asks = atm_valid['ask'].values.astype(float)
            bsz = atm_valid['bid_size'].values.astype(float)
            asz = atm_valid['ask_size'].values.astype(float)
            total_sz = bsz + asz
            mask = total_sz > 0
            if mask.any():
                microprice = np.mean((bids[mask] * asz[mask] + asks[mask] * bsz[mask]) / total_sz[mask])
                midpoint = np.mean((bids[mask] + asks[mask]) / 2.0)
                spread = np.mean(asks[mask] - bids[mask])
                features['microprice_vs_mid'] = float((microprice - midpoint) / max(spread, 1e-6))
            else:
                features['microprice_vs_mid'] = np.nan
        else:
            features['microprice_vs_mid'] = np.nan
    else:
        features['microprice_vs_mid'] = np.nan

    # --- Liquidity (6) ---

    features['atm_bid'] = float(atm['bid'].dropna().mean()) if 'bid' in atm.columns and atm['bid'].notna().any() else np.nan
    features['atm_ask'] = float(atm['ask'].dropna().mean()) if 'ask' in atm.columns and atm['ask'].notna().any() else np.nan
    features['atm_midpoint'] = atm_mid_val

    # Put wing liquidity: avg midpoint at 5-10 delta puts
    put_wing = df[(df['_right'] == 'P') & (df['delta'] >= -0.10) & (df['delta'] <= -0.05)]
    put_wing_mid = put_wing['_mid'].dropna()
    features['put_wing_liquidity'] = float(put_wing_mid.mean()) if len(put_wing_mid) > 0 else np.nan

    # Call wing liquidity: avg midpoint at 5-10 delta calls
    call_wing = df[(df['_right'] == 'C') & (df['delta'] >= 0.05) & (df['delta'] <= 0.10)]
    call_wing_mid = call_wing['_mid'].dropna()
    features['call_wing_liquidity'] = float(call_wing_mid.mean()) if len(call_wing_mid) > 0 else np.nan

    pwl = features['put_wing_liquidity']
    cwl = features['call_wing_liquidity']
    features['liquidity_asymmetry'] = (
        pwl / max(cwl, 1e-6)
        if not (np.isnan(pwl) or np.isnan(cwl))
        else np.nan
    )

    # --- Trade Flow (8) ---

    if chain_trades is not None and not chain_trades.empty:
        trades = chain_trades.copy()
        trades['_right'] = trades['right'].apply(_normalize_right)

        features['trade_count'] = float(len(trades))

        sizes = trades['size'].values.astype(float) if 'size' in trades.columns else np.zeros(len(trades))
        total_vol = float(sizes.sum())
        features['total_volume'] = total_vol

        # Classify trades using quote rule: merge chain quotes onto trades
        # Match by strike and right to get bid/ask at trade time
        if 'price' in trades.columns and 'bid' in df.columns and 'ask' in df.columns:
            quote_lookup = df.groupby(['strike', '_right']).agg(
                _bid=('bid', 'first'),
                _ask=('ask', 'first')
            ).reset_index()

            trades = trades.merge(
                quote_lookup,
                left_on=['strike', '_right'],
                right_on=['strike', '_right'],
                how='left'
            )

            buy_mask = trades['price'] >= trades['_ask']
            sell_mask = trades['price'] <= trades['_bid']

            buy_vol = float(sizes[buy_mask.values].sum()) if buy_mask.any() else 0.0
            sell_vol = float(sizes[sell_mask.values].sum()) if sell_mask.any() else 0.0
            signed_vol = buy_vol - sell_vol
            features['signed_volume'] = signed_vol
            features['signed_volume_ratio'] = signed_vol / max(total_vol, 1.0)

            # Aggressive trade share
            aggressive_count = int(buy_mask.sum() + sell_mask.sum())
            features['aggressive_trade_share'] = float(aggressive_count) / max(len(trades), 1)
        else:
            features['signed_volume'] = np.nan
            features['signed_volume_ratio'] = np.nan
            features['aggressive_trade_share'] = np.nan

        # Put volume ratio
        put_vol = float(sizes[trades['_right'].values == 'P'].sum())
        features['put_volume_ratio'] = put_vol / max(total_vol, 1.0)

        # Large trade fraction (>50 contracts)
        large_mask = sizes > 50
        features['large_trade_fraction'] = float(sizes[large_mask].sum()) / max(total_vol, 1.0)

        # Volume concentration: Herfindahl index across strikes
        if total_vol > 0:
            vol_by_strike = trades.groupby('strike')['size'].sum()
            shares = vol_by_strike.values / total_vol
            hhi = float((shares ** 2).sum())
            features['volume_concentration'] = hhi
        else:
            features['volume_concentration'] = np.nan
    else:
        for f in ['trade_count', 'total_volume', 'signed_volume',
                   'signed_volume_ratio', 'put_volume_ratio',
                   'large_trade_fraction', 'volume_concentration',
                   'aggressive_trade_share']:
            features[f] = np.nan

    return features
