"""Gamma Exposure (GEX) calculator module.

Computes net dealer gamma exposure across strikes from options chain data.
Identifies key structural levels: zero-gamma flip point, call/put walls.

GEX Formula:
    GEX_per_strike = sign * Gamma * OI * 100 * Spot^2 * 0.01
    Sign convention: calls +1 (dealer long gamma), puts -1 (dealer short gamma)

Features produced:
    net_gex              — total net dealer GEX across all strikes
    net_gex_normalized   — net_gex / spot (scale-independent)
    gex_imbalance        — call_gex / (call_gex + |put_gex|), 0.5 = balanced
    zero_gamma_level     — strike where cumulative GEX flips sign
    call_wall            — strike with highest call GEX (upside resistance)
    put_wall             — strike with highest |put GEX| (downside support)
    call_wall_distance   — (call_wall - spot) / spot as %
    put_wall_distance    — (spot - put_wall) / spot as %
"""

import numpy as np
import pandas as pd

# Default NaN result when computation isn't possible
_NAN_RESULT = {
    "net_gex": np.nan, "net_gex_normalized": np.nan,
    "gex_imbalance": np.nan, "zero_gamma_level": np.nan,
    "call_wall": np.nan, "put_wall": np.nan,
    "call_wall_distance": np.nan, "put_wall_distance": np.nan,
}

GEX_FEATURE_NAMES = list(_NAN_RESULT.keys())


def compute_per_strike_gex(strikes, rights, gammas, open_interests, spot):
    """Compute GEX per strike row.

    Args:
        strikes: array of strike prices
        rights: array of 'call'/'put' labels
        gammas: array of Black-Scholes gamma values
        open_interests: array of open interest counts
        spot: underlying spot price

    Returns:
        DataFrame with columns [strike, right, gex]
    """
    df = pd.DataFrame({
        "strike": strikes,
        "right": rights,
        "gamma": gammas,
        "open_interest": open_interests,
    })
    df = df.dropna(subset=["gamma"])
    df = df[df["open_interest"] > 0]

    if df.empty:
        return df

    sign = np.where(df["right"] == "call", 1.0, -1.0)
    df["gex"] = sign * df["gamma"] * df["open_interest"] * 100 * spot**2 * 0.01
    return df


def find_zero_gamma(strike_gex_series):
    """Find the strike where cumulative GEX crosses zero.

    Uses linear interpolation between the two strikes bracketing the
    sign change in cumulative GEX (summed from lowest strike upward).

    Args:
        strike_gex_series: Series indexed by strike with net GEX per strike,
                          sorted by strike ascending.

    Returns:
        float: interpolated zero-gamma strike, or NaN if no crossing found.
    """
    cum_gex = strike_gex_series.cumsum()
    if len(cum_gex) < 2:
        return np.nan

    signs = np.sign(cum_gex.values)
    for j in range(1, len(signs)):
        if signs[j - 1] != signs[j] and signs[j - 1] != 0:
            s1, s2 = cum_gex.index[j - 1], cum_gex.index[j]
            g1, g2 = cum_gex.values[j - 1], cum_gex.values[j]
            return s1 + (s2 - s1) * (-g1) / (g2 - g1)

    return np.nan


def compute_gex_features(greeks_day, oi_day, spot):
    """Compute GEX features for a single trading day.

    Args:
        greeks_day: DataFrame with columns [strike, right, gamma]
                   (EOD greeks for the day's 0DTE options)
        oi_day: DataFrame with columns [strike, right, open_interest]
        spot: SPX spot price (typically the open)

    Returns:
        dict with GEX feature values (NaN if data insufficient)
    """
    if greeks_day.empty or oi_day.empty or pd.isna(spot) or spot <= 0:
        return dict(_NAN_RESULT)

    # Merge gamma with OI on (strike, right)
    merged = greeks_day[["strike", "right", "gamma"]].merge(
        oi_day[["strike", "right", "open_interest"]],
        on=["strike", "right"], how="inner"
    )
    merged = merged.dropna(subset=["gamma"])
    merged = merged[merged["open_interest"] > 0]

    if merged.empty:
        return dict(_NAN_RESULT)

    # GEX per contract
    sign = np.where(merged["right"] == "call", 1.0, -1.0)
    merged["gex"] = sign * merged["gamma"] * merged["open_interest"] * 100 * spot**2 * 0.01

    net_gex = merged["gex"].sum()

    # Split by right for walls
    calls = merged[merged["right"] == "call"]
    puts = merged[merged["right"] == "put"]

    total_call_gex = calls["gex"].sum() if not calls.empty else 0.0
    total_put_gex_abs = puts["gex"].abs().sum() if not puts.empty else 0.0

    # GEX imbalance: call_gex / (call_gex + |put_gex|)
    denom = total_call_gex + total_put_gex_abs
    gex_imbalance = total_call_gex / denom if denom > 0 else np.nan

    # Call wall: strike with highest call GEX (upside resistance)
    call_wall = calls.loc[calls["gex"].idxmax(), "strike"] if not calls.empty else np.nan
    # Put wall: strike with highest |put GEX| (downside support)
    put_wall = puts.loc[puts["gex"].abs().idxmax(), "strike"] if not puts.empty else np.nan

    # Wall distances as % of spot
    call_wall_dist = (call_wall - spot) / spot * 100 if not pd.isna(call_wall) else np.nan
    put_wall_dist = (spot - put_wall) / spot * 100 if not pd.isna(put_wall) else np.nan

    # Zero gamma level
    strike_gex = merged.groupby("strike")["gex"].sum().sort_index()
    zero_gamma = find_zero_gamma(strike_gex)

    return {
        "net_gex": net_gex,
        "net_gex_normalized": net_gex / spot if spot > 0 else np.nan,
        "gex_imbalance": gex_imbalance,
        "zero_gamma_level": zero_gamma,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_wall_distance": call_wall_dist,
        "put_wall_distance": put_wall_dist,
    }


def compute_gex_profile(greeks_df, oi_df, spot):
    """Compute full GEX profile across all strikes (for visualization).

    Args:
        greeks_df: DataFrame with [strike, right, gamma]
        oi_df: DataFrame with [strike, right, open_interest]
        spot: spot price

    Returns:
        DataFrame with columns [strike, call_gex, put_gex, net_gex] sorted by strike
    """
    merged = greeks_df[["strike", "right", "gamma"]].merge(
        oi_df[["strike", "right", "open_interest"]],
        on=["strike", "right"], how="inner"
    )
    merged = merged.dropna(subset=["gamma"])
    merged = merged[merged["open_interest"] > 0]

    if merged.empty:
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex"])

    sign = np.where(merged["right"] == "call", 1.0, -1.0)
    merged["gex"] = sign * merged["gamma"] * merged["open_interest"] * 100 * spot**2 * 0.01

    calls = merged[merged["right"] == "call"].groupby("strike")["gex"].sum().rename("call_gex")
    puts = merged[merged["right"] == "put"].groupby("strike")["gex"].sum().rename("put_gex")

    profile = pd.DataFrame({"call_gex": calls, "put_gex": puts}).fillna(0)
    profile["net_gex"] = profile["call_gex"] + profile["put_gex"]
    profile = profile.reset_index().sort_values("strike")
    return profile
