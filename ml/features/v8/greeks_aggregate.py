"""
Aggregate Greeks features (~30) computed from the full option chain.

Covers GEX structure, delta/vanna/charm exposures, IC-specific Greeks,
flow-of-Greeks changes, and regime indicators.
"""

import numpy as np
import pandas as pd


def _normalize_right(s: pd.Series) -> pd.Series:
    """Map right column to uppercase single-char: 'C' or 'P'."""
    return s.astype(str).str.upper().str[0]


def _merge_oi(chain: pd.DataFrame, chain_oi: pd.DataFrame) -> pd.DataFrame:
    """Left-join OI onto chain by (strike, right), filling missing OI with 0."""
    chain = chain.copy()
    chain["_right"] = _normalize_right(chain["right"])

    oi = chain_oi.copy()
    oi["_right"] = _normalize_right(oi["right"])
    oi = oi.rename(columns={"open_interest": "oi"})

    merged = chain.merge(oi[["strike", "_right", "oi"]], on=["strike", "_right"], how="left")
    merged["oi"] = merged["oi"].fillna(0).astype(float)
    return merged


def _find_nearest_delta_strike(df: pd.DataFrame, target_abs_delta: float, right_char: str):
    """Find the row closest to a target |delta| for a given right ('C' or 'P')."""
    subset = df[df["_right"] == right_char].copy()
    if subset.empty:
        return None
    subset = subset.copy()
    subset["_abs_delta_diff"] = (subset["delta"].abs() - target_abs_delta).abs()
    return subset.loc[subset["_abs_delta_diff"].idxmin()]


def _herfindahl(values: np.ndarray) -> float:
    """Herfindahl index of shares. 1 = all concentrated, ~0 = spread."""
    total = np.sum(np.abs(values))
    if total == 0:
        return np.nan
    shares = np.abs(values) / total
    return float(np.sum(shares ** 2))


def compute_greeks_aggregate_features(
    chain: pd.DataFrame,
    chain_oi: pd.DataFrame,
    spx_price: float,
    minutes_to_close: float,
    prior_greeks: dict = None,
) -> dict:
    """
    Compute ~30 aggregate Greeks features from the full option chain.

    Args:
        chain: DataFrame with columns [strike, right, bid, ask, delta, gamma, theta, vega,
               implied_vol, underlying_price].
               Optional higher-order columns: [vanna, charm, vomma, veta, speed, zomma, color, ultima]
        chain_oi: DataFrame with columns [strike, right, open_interest]
        spx_price: Current SPX price
        minutes_to_close: Minutes until market close
        prior_greeks: Optional dict of prior aggregate Greeks for change features

    Returns:
        dict of feature_name -> float
    """
    features: dict = {}

    # --- Validation -----------------------------------------------------------
    required_chain_cols = {"strike", "right", "delta", "gamma", "theta", "vega"}
    required_oi_cols = {"strike", "right", "open_interest"}

    if chain is None or chain.empty or not required_chain_cols.issubset(chain.columns):
        return {k: np.nan for k in _all_feature_names()}

    if chain_oi is None or chain_oi.empty or not required_oi_cols.issubset(chain_oi.columns):
        return {k: np.nan for k in _all_feature_names()}

    # --- Merge OI onto chain --------------------------------------------------
    df = _merge_oi(chain, chain_oi)

    calls = df[df["_right"] == "C"]
    puts = df[df["_right"] == "P"]

    spot = float(spx_price)

    # =========================================================================
    # GEX Structure (10)
    # =========================================================================
    # contract_gamma = gamma * OI * 100 * spot
    df["contract_gamma"] = df["gamma"].astype(float) * df["oi"] * 100.0 * spot

    # Dealer GEX: +contract_gamma for calls, -contract_gamma for puts
    call_mask = df["_right"] == "C"
    put_mask = df["_right"] == "P"

    df["gex"] = np.where(call_mask, df["contract_gamma"], np.where(put_mask, -df["contract_gamma"], 0.0))

    gex_total = float(df["gex"].sum())
    gex_call = float(df.loc[call_mask, "contract_gamma"].sum())
    gex_put = float(-df.loc[put_mask, "contract_gamma"].sum())

    features["gex_total"] = gex_total
    features["gex_sign"] = 1.0 if gex_total >= 0 else -1.0
    features["gex_normalized"] = gex_total / (spot ** 2 * 1e6) if spot > 0 else np.nan
    features["gex_call"] = gex_call
    features["gex_put"] = gex_put
    features["gex_put_call_ratio"] = abs(gex_put) / max(abs(gex_call), 1e-6)

    # GEX flip strike: cumulative GEX from lowest to highest strike
    strike_gex = df.groupby("strike")["gex"].sum().sort_index()
    cum_gex = strike_gex.cumsum()
    flip_strike = np.nan
    for i in range(1, len(cum_gex)):
        prev_val = cum_gex.iloc[i - 1]
        curr_val = cum_gex.iloc[i]
        if prev_val * curr_val < 0:  # sign change
            flip_strike = float(cum_gex.index[i])
            break
    features["gex_flip_strike"] = flip_strike
    features["gex_flip_distance"] = (spot - flip_strike) / spot if not np.isnan(flip_strike) and spot > 0 else np.nan

    # 0DTE share: if 'expiration' or 'dte' column exists, otherwise 1.0
    if "dte" in df.columns:
        zero_dte_mask = df["dte"] == 0
        total_abs_gex = df["gex"].abs().sum()
        if total_abs_gex > 0:
            features["gex_0dte_share"] = float(df.loc[zero_dte_mask, "gex"].abs().sum() / total_abs_gex)
        else:
            features["gex_0dte_share"] = np.nan
    else:
        features["gex_0dte_share"] = 1.0

    # GEX concentration (Herfindahl of per-strike abs GEX)
    per_strike_gex = df.groupby("strike")["gex"].sum().values
    features["gex_concentration"] = _herfindahl(per_strike_gex)

    # =========================================================================
    # DEX and Exposures (5)
    # =========================================================================
    # delta exposure: call delta positive, put delta negative (delta already has correct sign)
    df["dex"] = df["delta"].astype(float) * df["oi"] * 100.0
    dex_total = float(df["dex"].sum())
    features["dex_total"] = dex_total
    features["dex_normalized"] = dex_total / (spot * 1e6) if spot > 0 else np.nan

    for col_name, feat_name in [("vanna", "vanna_exposure"), ("charm", "charm_exposure"), ("vomma", "vomma_exposure")]:
        if col_name in df.columns:
            features[feat_name] = float((df[col_name].astype(float) * df["oi"] * 100.0).sum())
        else:
            features[feat_name] = np.nan

    # =========================================================================
    # IC-Specific Greeks (5)
    # =========================================================================
    short_put = _find_nearest_delta_strike(df, 0.10, "P")
    short_call = _find_nearest_delta_strike(df, 0.10, "C")

    if short_put is not None and short_call is not None:
        sp_strike = float(short_put["strike"])
        sc_strike = float(short_call["strike"])

        # Long wings: 25 points wide
        wing_width = 25.0
        lp_strike = sp_strike - wing_width
        lc_strike = sc_strike + wing_width

        def _get_leg(strike_val, right_char):
            subset = df[(df["strike"] == strike_val) & (df["_right"] == right_char)]
            if subset.empty:
                # Find nearest available strike
                subset_r = df[df["_right"] == right_char]
                if subset_r.empty:
                    return None
                idx = (subset_r["strike"] - strike_val).abs().idxmin()
                return subset_r.loc[idx]
            return subset.iloc[0]

        legs = [
            _get_leg(sp_strike, "P"),   # short put
            _get_leg(sc_strike, "C"),    # short call
            _get_leg(lp_strike, "P"),    # long put
            _get_leg(lc_strike, "C"),    # long call
        ]
        # Signs: short legs negative, long legs positive for gamma/vega; theta opposite
        signs = [-1, -1, 1, 1]

        if all(leg is not None for leg in legs):
            ic_delta = sum(s * float(leg["delta"]) for s, leg in zip(signs, legs))
            ic_gamma = sum(s * float(leg["gamma"]) for s, leg in zip(signs, legs))
            ic_theta = sum(s * float(leg["theta"]) for s, leg in zip(signs, legs))
            ic_vega = sum(s * float(leg["vega"]) for s, leg in zip(signs, legs))

            features["ic_net_delta"] = ic_delta
            features["ic_net_gamma"] = ic_gamma
            features["ic_net_theta"] = ic_theta
            features["ic_net_vega"] = ic_vega
            features["ic_gamma_theta_ratio"] = abs(ic_gamma) / max(abs(ic_theta), 1e-10)
        else:
            for k in ["ic_net_delta", "ic_net_gamma", "ic_net_theta", "ic_net_vega", "ic_gamma_theta_ratio"]:
                features[k] = np.nan
    else:
        for k in ["ic_net_delta", "ic_net_gamma", "ic_net_theta", "ic_net_vega", "ic_gamma_theta_ratio"]:
            features[k] = np.nan

    # =========================================================================
    # Flow-of-Greeks Changes (5)
    # =========================================================================
    change_pairs = [
        ("gex_change_1h", "gex_total"),
        ("dex_change_1h", "dex_total"),
        ("vanna_flow", "vanna_exposure"),
        ("charm_flow_rate", "charm_exposure"),
        ("gamma_imbalance_change", "gex_put_call_ratio"),
    ]
    for feat_name, source_key in change_pairs:
        if prior_greeks is not None and source_key in prior_greeks and not np.isnan(prior_greeks[source_key]):
            current_val = features.get(source_key, np.nan)
            if not np.isnan(current_val):
                features[feat_name] = current_val - prior_greeks[source_key]
            else:
                features[feat_name] = np.nan
        else:
            features[feat_name] = np.nan

    # =========================================================================
    # Regime Indicators (5)
    # =========================================================================
    gex_norm = features.get("gex_normalized", np.nan)
    NEUTRAL_THRESHOLD = 0.01  # abs(gex_normalized) < 0.01 is neutral
    if np.isnan(gex_norm):
        features["gamma_regime"] = np.nan
    elif abs(gex_norm) < NEUTRAL_THRESHOLD:
        features["gamma_regime"] = 1.0  # neutral
    elif gex_norm > 0:
        features["gamma_regime"] = 2.0  # positive
    else:
        features["gamma_regime"] = 0.0  # negative

    # Pin strike proximity: distance from SPX to highest abs GEX strike
    if len(per_strike_gex) > 0:
        strike_gex_abs = df.groupby("strike")["gex"].apply(lambda x: abs(x.sum()))
        max_gex_strike = float(strike_gex_abs.idxmax())
        pin_prox = abs(spot - max_gex_strike) / spot if spot > 0 else np.nan
    else:
        pin_prox = np.nan
    features["pin_strike_proximity"] = pin_prox

    # GEX skew: GEX above spot minus GEX below spot
    above = df.loc[df["strike"] > spot, "gex"].sum()
    below = df.loc[df["strike"] < spot, "gex"].sum()
    features["gex_skew"] = float(above - below)

    # Charm pressure: charm_exposure * (minutes_to_close / 390)
    charm_exp = features.get("charm_exposure", np.nan)
    if not np.isnan(charm_exp):
        features["charm_pressure"] = charm_exp * (minutes_to_close / 390.0)
    else:
        features["charm_pressure"] = np.nan

    # Hedging flow intensity: abs(gex_normalized) * (1 / max(pin_strike_proximity, 0.001))
    if not np.isnan(gex_norm) and not np.isnan(pin_prox):
        features["hedging_flow_intensity"] = abs(gex_norm) * (1.0 / max(pin_prox, 0.001))
    else:
        features["hedging_flow_intensity"] = np.nan

    return features


def _all_feature_names() -> list:
    """Return list of all feature names for nan-filling on bad input."""
    return [
        # GEX Structure (10)
        "gex_total", "gex_sign", "gex_normalized", "gex_call", "gex_put",
        "gex_put_call_ratio", "gex_flip_strike", "gex_flip_distance",
        "gex_0dte_share", "gex_concentration",
        # DEX and Exposures (5)
        "dex_total", "dex_normalized", "vanna_exposure", "charm_exposure", "vomma_exposure",
        # IC-Specific Greeks (5)
        "ic_net_delta", "ic_net_gamma", "ic_net_theta", "ic_net_vega", "ic_gamma_theta_ratio",
        # Flow-of-Greeks Changes (5)
        "gex_change_1h", "dex_change_1h", "vanna_flow", "charm_flow_rate", "gamma_imbalance_change",
        # Regime Indicators (5)
        "gamma_regime", "pin_strike_proximity", "gex_skew", "charm_pressure", "hedging_flow_intensity",
    ]
