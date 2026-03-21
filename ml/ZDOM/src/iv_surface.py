"""Implied Volatility Surface feature extractor module.

Extracts IV surface features from options greeks data:
  - ATM implied volatility (from delta-matched options)
  - Skew: 25-delta and 10-delta put-call IV spreads
  - IV smile curvature (wing avg minus ATM)
  - Term structure slopes (0DTE vs 7DTE vs 14DTE)
  - Rolling statistics and z-scores

Features produced:
    atm_iv           — ATM IV (avg of ~50-delta call/put)
    skew_25d         — 25-delta put IV - 25-delta call IV
    skew_10d         — 10-delta put IV - 10-delta call IV (tail risk)
    iv_25d_put       — 25-delta put IV
    iv_25d_call      — 25-delta call IV
    iv_10d_put       — 10-delta put IV
    iv_10d_call      — 10-delta call IV
    iv_smile_curve   — (25d_put + 25d_call) / 2 - atm_iv (convexity)
    iv_ts_slope_0_7  — atm_iv / ts_iv_7dte (>1 = inverted = stress)
    iv_ts_slope_0_14 — atm_iv / ts_iv_14dte
    ts_slope_7_14    — ts_iv_7dte / ts_iv_14dte
    ts_inverted_7_14 — 1 if 7DTE IV > 14DTE IV
    + rolling means, stds, z-scores at 5d and 21d windows
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Black-Scholes IV inversion
# ---------------------------------------------------------------------------

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_from_price(price, S, K, T, r, right="call"):
    """Invert Black-Scholes to get implied vol from market price.

    Args:
        price: observed option market price (mid)
        S: underlying spot price
        K: strike price
        T: time to expiration in years
        r: risk-free rate (annualized)
        right: 'call' or 'put'

    Returns:
        Implied volatility, or NaN if inversion fails.
    """
    if T <= 0 or price <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0) if right == "call" else max(K - S, 0.0)
    if price <= intrinsic + 0.01:
        return np.nan

    price_fn = bs_call_price if right == "call" else bs_put_price

    def objective(sigma):
        return price_fn(S, K, T, r, sigma) - price

    try:
        iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Delta-based IV extraction from greeks snapshots
# ---------------------------------------------------------------------------

def find_delta_iv(snapshot, target_delta, right, tolerance=0.10):
    """Find the IV of the option closest to target_delta in a snapshot.

    Args:
        snapshot: DataFrame with columns [strike, right, delta, implied_vol, iv_error]
        target_delta: target delta (positive for calls, negative for puts)
        right: 'call' or 'put'
        tolerance: max acceptable delta distance

    Returns:
        (iv, actual_delta, strike) tuple, or (NaN, NaN, NaN) if not found.
    """
    side = snapshot[snapshot["right"] == right].copy()
    if side.empty:
        return np.nan, np.nan, np.nan

    side = side[(side["implied_vol"] > 0.01) & (side["implied_vol"] < 5.0)
                & (side["iv_error"] == 0)]
    if side.empty:
        return np.nan, np.nan, np.nan

    side["delta_dist"] = (side["delta"] - target_delta).abs()
    best = side.loc[side["delta_dist"].idxmin()]

    if best["delta_dist"] > tolerance:
        return np.nan, np.nan, np.nan

    return best["implied_vol"], best["delta"], best["strike"]


def extract_snapshot_iv_features(snapshot):
    """Extract IV surface features from a single-time options snapshot.

    Args:
        snapshot: DataFrame with columns [strike, right, delta, implied_vol,
                 iv_error, underlying_price, mid]. One row per strike/right.

    Returns:
        dict of IV surface features for this snapshot.
    """
    feats = {}

    # ATM IV
    atm_call_iv, _, _ = find_delta_iv(snapshot, 0.50, "call", tolerance=0.15)
    atm_put_iv, _, _ = find_delta_iv(snapshot, -0.50, "put", tolerance=0.15)

    if not np.isnan(atm_call_iv) and not np.isnan(atm_put_iv):
        feats["atm_iv"] = (atm_call_iv + atm_put_iv) / 2
    elif not np.isnan(atm_call_iv):
        feats["atm_iv"] = atm_call_iv
    elif not np.isnan(atm_put_iv):
        feats["atm_iv"] = atm_put_iv
    else:
        feats["atm_iv"] = np.nan

    # 25-delta skew
    iv_25d_call, _, _ = find_delta_iv(snapshot, 0.25, "call")
    iv_25d_put, _, _ = find_delta_iv(snapshot, -0.25, "put")
    feats["iv_25d_call"] = iv_25d_call
    feats["iv_25d_put"] = iv_25d_put
    feats["skew_25d"] = (iv_25d_put - iv_25d_call
                         if not np.isnan(iv_25d_put) and not np.isnan(iv_25d_call)
                         else np.nan)

    # 10-delta skew (tail risk)
    iv_10d_call, _, _ = find_delta_iv(snapshot, 0.10, "call")
    iv_10d_put, _, _ = find_delta_iv(snapshot, -0.10, "put")
    feats["iv_10d_call"] = iv_10d_call
    feats["iv_10d_put"] = iv_10d_put
    feats["skew_10d"] = (iv_10d_put - iv_10d_call
                         if not np.isnan(iv_10d_put) and not np.isnan(iv_10d_call)
                         else np.nan)

    # IV smile curvature: wing average minus ATM
    if not np.isnan(iv_25d_call) and not np.isnan(iv_25d_put) and not np.isnan(feats["atm_iv"]):
        feats["iv_smile_curve"] = (iv_25d_put + iv_25d_call) / 2 - feats["atm_iv"]
    else:
        feats["iv_smile_curve"] = np.nan

    return feats


def extract_term_structure_iv(ts_day, spx_close, risk_free_rate=0.05):
    """Extract ATM IV at different DTEs from term structure data.

    Args:
        ts_day: DataFrame of term structure for one day with columns
               [dte, moneyness, right, mid, strike]
        spx_close: SPX close price for BS inversion
        risk_free_rate: risk-free rate for BS model

    Returns:
        dict with ts_iv_7dte, ts_iv_14dte
    """
    feats = {}

    if ts_day.empty:
        for label in ["7dte", "14dte"]:
            feats[f"ts_iv_{label}"] = np.nan
        return feats

    # Normalize right column to lowercase (ThetaData may return CALL/PUT)
    ts_day = ts_day.copy()
    ts_day["right"] = ts_day["right"].str.lower()

    for target_dte, label in [(7, "7dte"), (14, "14dte")]:
        bucket = ts_day[
            (ts_day["dte"] >= target_dte - 3) & (ts_day["dte"] <= target_dte + 3)
        ]

        if bucket.empty:
            feats[f"ts_iv_{label}"] = np.nan
            continue

        bucket = bucket.copy()
        bucket["abs_moneyness"] = bucket["moneyness"].abs()
        calls = bucket[bucket["right"] == "call"].sort_values("abs_moneyness")
        puts = bucket[bucket["right"] == "put"].sort_values("abs_moneyness")

        ivs = []
        for side, right_label in [(calls, "call"), (puts, "put")]:
            if side.empty:
                continue
            row = side.iloc[0]
            mid = row["mid"]
            strike = row["strike"]
            dte = row["dte"]
            T = max(dte / 365.0, 1 / (365.0 * 24))

            iv = implied_vol_from_price(mid, spx_close, strike, T, risk_free_rate, right_label)
            if not np.isnan(iv) and 0.01 < iv < 3.0:
                ivs.append(iv)

        feats[f"ts_iv_{label}"] = np.mean(ivs) if ivs else np.nan

    return feats


def add_rolling_features(df, col, windows=(5, 21)):
    """Add rolling mean, std, and z-score features for a column.

    Args:
        df: DataFrame (modified in place)
        col: column name to compute rolling stats for
        windows: tuple of window sizes in days
    """
    for w in windows:
        min_p = max(w // 2, 2)
        df[f"{col}_{w}d_mean"] = df[col].rolling(w, min_periods=min_p).mean()
        df[f"{col}_{w}d_std"] = df[col].rolling(w, min_periods=min_p).std()

    # Z-score relative to 21-day window (if 21 in windows)
    if 21 in windows:
        mean_21 = df[f"{col}_21d_mean"]
        std_21 = df[f"{col}_21d_std"]
        df[f"{col}_zscore_21d"] = (df[col] - mean_21) / std_21.replace(0, np.nan)


def compute_term_structure_slopes(df):
    """Compute IV term structure slopes from ATM IV and term structure IVs.

    Expects df to have: atm_iv, ts_iv_7dte, ts_iv_14dte columns.
    Adds: iv_ts_slope_0_7, iv_ts_slope_0_14, ts_slope_7_14, ts_inverted_7_14.
    """
    df["iv_ts_slope_0_7"] = df["atm_iv"] / df["ts_iv_7dte"]
    df["iv_ts_slope_0_14"] = df["atm_iv"] / df["ts_iv_14dte"]
    df["ts_slope_7_14"] = df["ts_iv_7dte"] / df["ts_iv_14dte"]
    df["ts_inverted_7_14"] = np.where(
        df["ts_slope_7_14"].notna(),
        (df["ts_slope_7_14"] > 1.0).astype(float),
        np.nan,
    )
    return df


# Convenience: list all features this module produces
IV_SURFACE_FEATURES = [
    "atm_iv", "iv_25d_call", "iv_25d_put", "skew_25d",
    "iv_10d_call", "iv_10d_put", "skew_10d", "iv_smile_curve",
    "ts_iv_7dte", "ts_iv_14dte",
    "iv_ts_slope_0_7", "iv_ts_slope_0_14", "ts_slope_7_14", "ts_inverted_7_14",
    "atm_iv_5d_mean", "atm_iv_5d_std", "atm_iv_21d_mean", "atm_iv_21d_std",
    "atm_iv_zscore_21d",
    "skew_25d_5d_mean", "skew_25d_5d_std", "skew_25d_21d_mean", "skew_25d_21d_std",
    "skew_25d_zscore_21d",
]
