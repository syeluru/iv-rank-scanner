"""
Compute Black-Scholes greeks for the SPXW 0DTE EOD option chain.

Uses bid/ask mid-price + SPX close to solve for implied volatility,
then computes delta, gamma, theta, vega for each contract.

Input:  data/spxw_0dte_eod.parquet  (336K rows — full chain, 754 days)
        data/spx_daily.parquet       (SPX daily OHLC — for underlying price)
Output: data/spxw_0dte_greeks.parquet

Columns: date, strike, right, mid, iv, delta, gamma, theta, vega, underlying

Notes:
  - Uses 0DTE time-to-expiry: T = (16:00 ET - 09:30 ET) / 252 / 6.5h = 1/252
    (approximated as a single trading day fraction)
  - Risk-free rate: Fed Funds effective rate proxy (flat ~5.0% for 2023-2026)
  - Dividend yield: SPX ~1.3% continuous
  - IV solver uses Brent's method via scipy. Contracts with no solution
    (zero-price, deep ITM/OTM with no market) are skipped.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Constants
RISK_FREE = 0.050   # ~5% Fed Funds effective (2023-2026 average)
DIV_YIELD = 0.013   # SPX dividend yield ~1.3%
T_0DTE = 1.0 / 252  # 0DTE = 1 trading day, expressed in years


# ── Black-Scholes functions ───────────────────────────────────────────────────

def bs_price(S, K, T, r, q, sigma, right):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if right == "call" else (K - S))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if right == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_iv(price, S, K, T, r, q, right, tol=1e-6):
    """Solve for implied volatility using Brent's method."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(0.0, (S - K) if right == "call" else (K - S))
    if price < intrinsic - 0.01:
        return np.nan
    try:
        iv = brentq(lambda sigma: bs_price(S, K, T, r, q, sigma, right) - price,
                    1e-6, 20.0, xtol=tol, maxiter=100)
        return iv
    except Exception:
        return np.nan


def bs_greeks(S, K, T, r, q, sigma, right):
    """Compute delta, gamma, theta, vega given sigma."""
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan, np.nan, np.nan
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * pdf_d1 * sqrt_T / 100  # per 1% move in IV

    if right == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 252
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 252

    return delta, gamma, theta, vega


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading EOD chain and SPX daily data...")
    eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")

    eod["date"] = pd.to_datetime(eod["date"])
    spx["date"] = pd.to_datetime(spx["date"])

    # Use SPX close as underlying price for EOD greeks
    price_map = dict(zip(spx["date"], spx["spx_close"]))
    eod["underlying"] = eod["date"].map(price_map)
    eod = eod.dropna(subset=["underlying"])

    # Compute mid-price
    eod["mid"] = (eod["bid"] + eod["ask"]) / 2

    # Filter: skip zero-price contracts (far OTM, no market)
    eod = eod[eod["mid"] > 0.05].copy()

    print(f"  Computing greeks for {len(eod):,} contracts ({eod['date'].nunique()} days)...")

    results = []
    for i, (_, row) in enumerate(eod.iterrows()):
        if i % 10000 == 0:
            print(f"  {i:,}/{len(eod):,} ({i/len(eod)*100:.1f}%)...", flush=True)

        S = row["underlying"]
        K = row["strike"]
        right = row["right"]
        mid = row["mid"]

        iv = bs_iv(mid, S, K, T_0DTE, RISK_FREE, DIV_YIELD, right)
        if np.isnan(iv):
            continue

        delta, gamma, theta, vega = bs_greeks(S, K, T_0DTE, RISK_FREE, DIV_YIELD, iv, right)

        results.append({
            "date": row["date"],
            "strike": K,
            "right": right,
            "underlying": S,
            "mid": mid,
            "iv": round(iv, 6),
            "delta": round(delta, 6) if not np.isnan(delta) else np.nan,
            "gamma": round(gamma, 6) if not np.isnan(gamma) else np.nan,
            "theta": round(theta, 6) if not np.isnan(theta) else np.nan,
            "vega": round(vega, 6) if not np.isnan(vega) else np.nan,
        })

    df = pd.DataFrame(results)
    outfile = DATA_DIR / "spxw_0dte_greeks.parquet"
    df.to_parquet(outfile, index=False)

    print(f"\nDone. {len(df):,} rows, {df['date'].nunique()} days")
    print(f"IV range: {df['iv'].min():.3f} — {df['iv'].max():.3f}")
    print(f"Delta range: {df['delta'].min():.3f} — {df['delta'].max():.3f}")
    print(f"Saved: {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
