"""Vanna and Charm exposure calculator module.

Computes second-order dealer flow signals from options greeks data.

Vanna (dDelta/dIV):
    How much dealer delta changes when implied volatility moves.
    Net positive vanna = IV decline forces dealers to buy underlying (bullish flow).
    BS formula: vanna = -e^(-qT) * N'(d1) * d2 / sigma

Charm (dDelta/dTime):
    How much dealer delta decays as time passes (theta of delta).
    Critical for 0DTE where charm accelerates dramatically in final hours.
    BS formula (call): charm = -e^(-qT) * [N'(d1) * (2rT - d2*sigma*sqrt(T)) / (2T*sigma*sqrt(T))]

Features produced:
    net_vanna_exposure      — aggregate dealer vanna across all strikes
    net_vanna_normalized    — net_vanna / spot (scale-independent)
    vanna_imbalance         — call_vanna / (call_vanna + |put_vanna|)
    net_charm_exposure      — aggregate dealer charm across all strikes
    net_charm_normalized    — net_charm / spot (scale-independent)
    charm_imbalance         — call_charm / (call_charm + |put_charm|)
    vanna_charm_ratio       — |net_vanna| / (|net_vanna| + |net_charm|)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# Default NaN result
_NAN_RESULT = {
    "net_vanna_exposure": np.nan,
    "net_vanna_normalized": np.nan,
    "vanna_imbalance": np.nan,
    "net_charm_exposure": np.nan,
    "net_charm_normalized": np.nan,
    "charm_imbalance": np.nan,
    "vanna_charm_ratio": np.nan,
}

VANNA_CHARM_FEATURES = list(_NAN_RESULT.keys())


def bs_vanna(S, K, T, r, sigma, q=0.0):
    """Black-Scholes Vanna (dDelta/dIV) per share.

    Args:
        S: spot price
        K: strike price
        T: time to expiry in years (must be > 0)
        r: risk-free rate
        sigma: implied volatility
        q: dividend yield

    Returns:
        Vanna value per share. Positive for both calls and puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Vanna = -e^(-qT) * N'(d1) * d2 / sigma
    # This is the same for calls and puts (vanna is symmetric)
    vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    return vanna


def bs_charm(S, K, T, r, sigma, right="call", q=0.0):
    """Black-Scholes Charm (dDelta/dTime) per share.

    Charm measures how delta changes as time passes. For 0DTE options,
    charm is very large — delta decays rapidly toward expiry.

    Args:
        S: spot price
        K: strike price
        T: time to expiry in years (must be > 0)
        r: risk-free rate
        sigma: implied volatility
        right: 'call' or 'put'
        q: dividend yield

    Returns:
        Charm value per share.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Charm for call = -q * e^(-qT) * N(d1) + e^(-qT) * N'(d1) * (2(r-q)T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
    # For simplicity with q=0:
    # Charm_call = e^(-qT) * N'(d1) * (2*r*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
    npdf_d1 = norm.pdf(d1)
    exp_qt = np.exp(-q * T)

    numerator = 2 * (r - q) * T - d2 * sigma * sqrt_T
    denominator = 2 * T * sigma * sqrt_T

    if abs(denominator) < 1e-12:
        return np.nan

    charm_call = exp_qt * npdf_d1 * numerator / denominator
    if q != 0:
        charm_call -= q * exp_qt * norm.cdf(d1)

    if right == "call":
        return charm_call
    else:
        # Put charm = call charm + q * e^(-qT) (from put-call parity on delta)
        # For q=0: put charm = call charm
        return charm_call + q * exp_qt


def compute_vanna_charm_features(greeks_day, oi_day, spot, dte_years=1/365,
                                  risk_free_rate=0.05):
    """Compute Vanna and Charm exposure features for a single trading day.

    Args:
        greeks_day: DataFrame with columns [strike, right, iv, delta, gamma]
                    (EOD greeks for the day's options)
        oi_day: DataFrame with columns [strike, right, open_interest]
        spot: underlying spot price
        dte_years: time to expiry in years (default: 1 day)
        risk_free_rate: risk-free rate

    Returns:
        dict with Vanna/Charm feature values
    """
    if greeks_day.empty or oi_day.empty or pd.isna(spot) or spot <= 0:
        return dict(_NAN_RESULT)

    # Merge greeks with OI
    required_cols = ["strike", "right"]
    greeks_cols = [c for c in ["strike", "right", "iv", "delta", "gamma"] if c in greeks_day.columns]
    merged = greeks_day[greeks_cols].merge(
        oi_day[["strike", "right", "open_interest"]],
        on=["strike", "right"], how="inner"
    )

    # Filter valid rows
    if "iv" not in merged.columns:
        return dict(_NAN_RESULT)

    merged = merged[
        (merged["open_interest"] > 0) &
        (merged["iv"] > 0.01) &
        (merged["iv"] < 5.0)
    ].copy()

    if len(merged) < 2:
        return dict(_NAN_RESULT)

    # Compute Vanna and Charm per strike
    vannas = []
    charms = []
    for _, row in merged.iterrows():
        v = bs_vanna(spot, row["strike"], dte_years, risk_free_rate, row["iv"])
        c = bs_charm(spot, row["strike"], dte_years, risk_free_rate, row["iv"], row["right"])
        vannas.append(v)
        charms.append(c)

    merged["vanna"] = vannas
    merged["charm"] = charms
    merged = merged.dropna(subset=["vanna", "charm"])

    if merged.empty:
        return dict(_NAN_RESULT)

    # Dealer exposure: sign convention same as GEX
    # Dealers assumed long calls, short puts at index level
    sign = np.where(merged["right"] == "call", 1.0, -1.0)
    merged["vanna_exposure"] = sign * merged["vanna"] * merged["open_interest"] * 100
    merged["charm_exposure"] = sign * merged["charm"] * merged["open_interest"] * 100

    # Aggregate
    net_vanna = merged["vanna_exposure"].sum()
    net_charm = merged["charm_exposure"].sum()

    # Split by side
    calls = merged[merged["right"] == "call"]
    puts = merged[merged["right"] == "put"]

    call_vanna = calls["vanna_exposure"].sum() if not calls.empty else 0.0
    put_vanna_abs = puts["vanna_exposure"].abs().sum() if not puts.empty else 0.0
    vanna_denom = abs(call_vanna) + put_vanna_abs
    vanna_imbalance = abs(call_vanna) / vanna_denom if vanna_denom > 0 else np.nan

    call_charm = calls["charm_exposure"].sum() if not calls.empty else 0.0
    put_charm_abs = puts["charm_exposure"].abs().sum() if not puts.empty else 0.0
    charm_denom = abs(call_charm) + put_charm_abs
    charm_imbalance = abs(call_charm) / charm_denom if charm_denom > 0 else np.nan

    # Vanna/Charm ratio: relative importance of vol-sensitivity vs time-sensitivity
    abs_vanna = abs(net_vanna)
    abs_charm = abs(net_charm)
    vc_denom = abs_vanna + abs_charm
    vc_ratio = abs_vanna / vc_denom if vc_denom > 0 else np.nan

    return {
        "net_vanna_exposure": net_vanna,
        "net_vanna_normalized": net_vanna / spot if spot > 0 else np.nan,
        "vanna_imbalance": vanna_imbalance,
        "net_charm_exposure": net_charm,
        "net_charm_normalized": net_charm / spot if spot > 0 else np.nan,
        "charm_imbalance": charm_imbalance,
        "vanna_charm_ratio": vc_ratio,
    }
