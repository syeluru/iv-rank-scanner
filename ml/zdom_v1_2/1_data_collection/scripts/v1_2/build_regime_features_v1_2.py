"""
Build regime_features_v1_2.parquet — 2-state HMM vol regime from SPX + VIX.

Source: Built from raw daily data:
  - raw/v1_1/spx_daily.parquet (SPX daily OHLC for log returns)
  - raw/v1_1/vix_daily.parquet (VIX daily close)
  - src/regime.py (VolRegimeDetector — GaussianHMM wrapper)

The HMM is fit on all available SPX+VIX history (expanding window).
It uses the Viterbi algorithm to decode the most likely state sequence,
which is causal (uses observations up to time t only).

Output: data/v1_2/regime_features_v1_2.parquet
  One row per trading day, keyed by date.
  Coverage: first valid SPX+VIX day through 2026-03-13.

Columns:
  date, regime_state, regime_prob_highvol, regime_prob_lowvol,
  regime_duration, regime_switch

  regime_state: 0 = low-vol, 1 = high-vol (relabeled so state 1 = higher vol)

Join lag: T-1 (regime is computed from daily closes, not known until after close).

Note: hmmlearn is required (pip install hmmlearn).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
DATA_V1_2 = BASE / "data" / "v1_2"

CLIP = pd.Timestamp("2026-03-13")

sys.path.insert(0, str(PROJECT))


def main():
    DATA_V1_2.mkdir(parents=True, exist_ok=True)

    from src.regime import VolRegimeDetector

    # Load SPX daily
    spx = pd.read_parquet(RAW_V1_1 / "spx_daily.parquet")
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.sort_values("date")
    spx = spx[spx["spx_close"] > 0].copy()
    spx["spx_log_return"] = np.log(spx["spx_close"] / spx["spx_close"].shift(1))

    # Load VIX daily
    vix = pd.read_parquet(RAW_V1_1 / "vix_daily.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["vix_close"] > 0]

    # Merge and clean
    df = spx.merge(vix[["date", "vix_close"]], on="date", how="inner")
    df = df.dropna(subset=["spx_log_return"])
    df = df[np.isfinite(df["spx_log_return"])].reset_index(drop=True)

    print(f"Input: {len(df)} trading days ({df['date'].min().date()} to {df['date'].max().date()})")

    # Fit HMM
    print("Fitting 2-state GaussianHMM...")
    detector = VolRegimeDetector(n_restarts=25, n_iter=1000)
    detector.fit(df["spx_log_return"].values, df["vix_close"].values)

    # Predict
    result = detector.predict(df["spx_log_return"].values, df["vix_close"].values)

    df["regime_state"] = result["regime_state"]
    df["regime_prob_highvol"] = result["regime_prob_highvol"]
    df["regime_prob_lowvol"] = result["regime_prob_lowvol"]
    df["regime_duration"] = result["regime_duration"]
    df["regime_switch"] = result["regime_switch"]

    # Select and clip
    out_cols = [
        "date", "regime_state", "regime_prob_highvol", "regime_prob_lowvol",
        "regime_duration", "regime_switch",
    ]
    out = df[out_cols].copy()
    out = out[out["date"] <= CLIP]

    outfile = DATA_V1_2 / "regime_features_v1_2.parquet"
    out.to_parquet(outfile, index=False)

    print(f"\nregime_features_v1_2.parquet")
    print(f"  Shape: {out.shape}")
    print(f"  Range: {out['date'].min().date()} to {out['date'].max().date()}")
    print(f"  Nulls: {out.isna().sum().sum()}")
    for s in [0, 1]:
        n = (out["regime_state"] == s).sum()
        label = "LOW-VOL" if s == 0 else "HIGH-VOL"
        print(f"  State {s} ({label}): {n} days ({n / len(out) * 100:.1f}%)")
    print(f"  Switches: {out['regime_switch'].sum()}")
    print(f"  Saved to: {outfile}")


if __name__ == "__main__":
    main()
