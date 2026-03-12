"""
Positioning and open-interest-based features (~20) computed from the option chain.

Covers max pain, OI structure, implied probabilities, and pin risk.
"""

import numpy as np
import pandas as pd


def _normalize_right(s: pd.Series) -> pd.Series:
    """Map right column to uppercase single-char: 'C' or 'P'."""
    return s.astype(str).str.upper().str[0]


def _herfindahl(values: np.ndarray) -> float:
    """Herfindahl index of shares. 1 = all concentrated, ~0 = spread."""
    total = np.sum(np.abs(values))
    if total == 0:
        return np.nan
    shares = np.abs(values) / total
    return float(np.sum(shares ** 2))


def compute_positioning_features(
    chain: pd.DataFrame,
    chain_oi: pd.DataFrame,
    spx_price: float,
    minutes_to_close: float = 390.0,
) -> dict:
    """
    Compute ~20 positioning and OI-based features from the option chain.

    Args:
        chain: DataFrame with columns [strike, right, bid, ask, delta, implied_vol]
        chain_oi: DataFrame with columns [strike, right, open_interest]
        spx_price: Current SPX price
        minutes_to_close: Minutes until market close (default 390 = full day)

    Returns:
        dict of feature_name -> float
    """
    features: dict = {}

    # --- Validation -----------------------------------------------------------
    required_chain_cols = {"strike", "right", "bid", "ask", "delta"}
    required_oi_cols = {"strike", "right", "open_interest"}

    if chain is None or chain.empty or not required_chain_cols.issubset(chain.columns):
        return {k: np.nan for k in _all_feature_names()}

    if chain_oi is None or chain_oi.empty or not required_oi_cols.issubset(chain_oi.columns):
        return {k: np.nan for k in _all_feature_names()}

    spot = float(spx_price)
    if spot <= 0:
        return {k: np.nan for k in _all_feature_names()}

    # Normalize right columns
    oi = chain_oi.copy()
    oi["_right"] = _normalize_right(oi["right"])
    oi = oi.rename(columns={"open_interest": "oi"})

    ch = chain.copy()
    ch["_right"] = _normalize_right(ch["right"])

    # Merge OI onto chain
    merged = ch.merge(oi[["strike", "_right", "oi"]], on=["strike", "_right"], how="left")
    merged["oi"] = merged["oi"].fillna(0).astype(float)

    call_oi = oi[oi["_right"] == "C"]
    put_oi = oi[oi["_right"] == "P"]

    # =========================================================================
    # Max Pain & Pin Risk (7)
    # =========================================================================

    # Collect unique strikes from OI data
    all_strikes = np.sort(oi["strike"].unique())

    if len(all_strikes) == 0:
        for k in ["max_pain_strike", "max_pain_distance", "max_pain_abs_distance",
                   "highest_call_oi_strike", "highest_put_oi_strike",
                   "call_wall_distance", "put_wall_distance"]:
            features[k] = np.nan
    else:
        # Build per-strike OI arrays for calls and puts
        call_strikes = call_oi["strike"].values
        call_ois = call_oi["oi"].values
        put_strikes = put_oi["strike"].values
        put_ois = put_oi["oi"].values

        # Max pain: for each candidate K, total payout =
        #   sum_calls[ max(0, K - call_strike) * call_OI ] +
        #   sum_puts[ max(0, put_strike - K) * put_OI ]
        best_payout = np.inf
        best_k = np.nan

        for K in all_strikes:
            # Call holder payout at K: calls are ITM when K > strike
            call_payout = np.sum(np.maximum(0, K - call_strikes) * call_ois)
            # Put holder payout at K: puts are ITM when strike > K
            put_payout = np.sum(np.maximum(0, put_strikes - K) * put_ois)
            total = call_payout + put_payout
            if total < best_payout:
                best_payout = total
                best_k = float(K)

        features["max_pain_strike"] = best_k
        features["max_pain_distance"] = (spot - best_k) / spot
        features["max_pain_abs_distance"] = abs(features["max_pain_distance"])

        # Highest OI strikes
        if not call_oi.empty:
            highest_call_idx = call_oi["oi"].idxmax()
            features["highest_call_oi_strike"] = float(call_oi.loc[highest_call_idx, "strike"])
        else:
            features["highest_call_oi_strike"] = np.nan

        if not put_oi.empty:
            highest_put_idx = put_oi["oi"].idxmax()
            features["highest_put_oi_strike"] = float(put_oi.loc[highest_put_idx, "strike"])
        else:
            features["highest_put_oi_strike"] = np.nan

        hc = features["highest_call_oi_strike"]
        hp = features["highest_put_oi_strike"]
        features["call_wall_distance"] = (hc - spot) / spot if not np.isnan(hc) else np.nan
        features["put_wall_distance"] = (spot - hp) / spot if not np.isnan(hp) else np.nan

    # =========================================================================
    # OI Structure (7)
    # =========================================================================
    total_call_oi = float(call_oi["oi"].sum()) if not call_oi.empty else 0.0
    total_put_oi = float(put_oi["oi"].sum()) if not put_oi.empty else 0.0
    oi_total = total_call_oi + total_put_oi

    features["oi_total"] = oi_total
    features["oi_put_call_ratio"] = total_put_oi / max(total_call_oi, 1.0)

    # OI concentration (Herfindahl across all strikes, combining put+call OI per strike)
    strike_oi = oi.groupby("strike")["oi"].sum().values
    features["oi_concentration"] = _herfindahl(strike_oi)

    # OI near money: within +/-1% of spot
    lower = spot * 0.99
    upper = spot * 1.01
    near_money_mask = (oi["strike"] >= lower) & (oi["strike"] <= upper)
    oi_near = float(oi.loc[near_money_mask, "oi"].sum())
    features["oi_near_money"] = oi_near / max(oi_total, 1.0)

    # OI at round numbers (divisible by 50 or 100)
    round_mask = (oi["strike"] % 50 == 0)
    oi_round = float(oi.loc[round_mask, "oi"].sum())
    features["oi_at_round_numbers"] = oi_round / max(oi_total, 1.0)

    # SPX position in OI range
    hc = features.get("highest_call_oi_strike", np.nan)
    hp = features.get("highest_put_oi_strike", np.nan)
    if not np.isnan(hc) and not np.isnan(hp):
        denom = max(hc - hp, 1.0)
        features["spx_in_oi_range"] = (spot - hp) / denom
    else:
        features["spx_in_oi_range"] = np.nan

    # OI-weighted gamma
    if "gamma" in merged.columns:
        gamma_oi = (merged["gamma"].astype(float) * merged["oi"]).sum()
        features["oi_weighted_gamma"] = gamma_oi / max(oi_total, 1.0)
    else:
        features["oi_weighted_gamma"] = np.nan

    # =========================================================================
    # Implied Probabilities (6)
    # =========================================================================

    # Find 10-delta strikes
    calls_m = merged[merged["_right"] == "C"]
    puts_m = merged[merged["_right"] == "P"]

    short_call_row = None
    short_put_row = None

    if not calls_m.empty:
        abs_diff = (calls_m["delta"].astype(float).abs() - 0.10).abs()
        short_call_row = calls_m.loc[abs_diff.idxmin()]

    if not puts_m.empty:
        abs_diff = (puts_m["delta"].astype(float).abs() - 0.10).abs()
        short_put_row = puts_m.loc[abs_diff.idxmin()]

    if short_put_row is not None and short_call_row is not None:
        delta_sp = abs(float(short_put_row["delta"]))
        delta_sc = abs(float(short_call_row["delta"]))

        features["prob_stay_in_ic"] = 1.0 - delta_sp - delta_sc
        features["prob_touch_short_put"] = 2.0 * delta_sp
        features["prob_touch_short_call"] = 2.0 * delta_sc

        sc_strike = float(short_call_row["strike"])
        sp_strike = float(short_put_row["strike"])
        ic_width = sc_strike - sp_strike
    else:
        features["prob_stay_in_ic"] = np.nan
        features["prob_touch_short_put"] = np.nan
        features["prob_touch_short_call"] = np.nan
        ic_width = np.nan

    # Expected move: ATM straddle price / spot
    # ATM = nearest strike to spot
    if not merged.empty:
        atm_strike_idx = (merged["strike"] - spot).abs().idxmin()
        atm_strike = float(merged.loc[atm_strike_idx, "strike"])

        atm_calls = merged[(merged["strike"] == atm_strike) & (merged["_right"] == "C")]
        atm_puts = merged[(merged["strike"] == atm_strike) & (merged["_right"] == "P")]

        if not atm_calls.empty and not atm_puts.empty:
            call_mid = (float(atm_calls.iloc[0]["bid"]) + float(atm_calls.iloc[0]["ask"])) / 2.0
            put_mid = (float(atm_puts.iloc[0]["bid"]) + float(atm_puts.iloc[0]["ask"])) / 2.0
            straddle = call_mid + put_mid
            features["expected_move_pct"] = (straddle / spot) * 100.0

            if not np.isnan(ic_width) and ic_width > 0:
                features["expected_move_vs_ic_width"] = straddle / ic_width
            else:
                features["expected_move_vs_ic_width"] = np.nan
        else:
            features["expected_move_pct"] = np.nan
            features["expected_move_vs_ic_width"] = np.nan
    else:
        features["expected_move_pct"] = np.nan
        features["expected_move_vs_ic_width"] = np.nan

    # Pin risk score: oi_near_money * (1 - minutes_to_close / 390)
    mtc_norm = minutes_to_close / 390.0
    features["pin_risk_score"] = features["oi_near_money"] * (1.0 - mtc_norm)

    return features


def _all_feature_names() -> list:
    """Return list of all feature names for nan-filling on bad input."""
    return [
        # Max Pain & Pin Risk (7)
        "max_pain_strike", "max_pain_distance", "max_pain_abs_distance",
        "highest_call_oi_strike", "highest_put_oi_strike",
        "call_wall_distance", "put_wall_distance",
        # OI Structure (7)
        "oi_total", "oi_put_call_ratio", "oi_concentration",
        "oi_near_money", "oi_at_round_numbers", "spx_in_oi_range", "oi_weighted_gamma",
        # Implied Probabilities (6)
        "prob_stay_in_ic", "prob_touch_short_put", "prob_touch_short_call",
        "expected_move_pct", "expected_move_vs_ic_width", "pin_risk_score",
    ]
