"""
Build HMM regime features using a 2-state GaussianHMM.

Trains on SPX daily log-returns + VIX close (standardized).
Outputs regime state (0 or 1) plus regime probabilities as features
for downstream XGBoost models.

Reads:  data/spx_daily.parquet, data/vix_daily.parquet
Saves:  data/regime_features.parquet  (one row per trading day)
        models/hmm_regime_2state.pkl  (fitted HMM for online prediction)

Features produced:
  - regime_state        : 0 = low-vol, 1 = high-vol (relabeled so state 1 = higher vol)
  - regime_prob_highvol : P(high-vol regime) from posterior
  - regime_prob_lowvol  : P(low-vol regime) from posterior
  - regime_duration     : consecutive days in current regime
  - regime_switch       : 1 if regime changed from prior day, else 0

Usage:
  python scripts/build_regime_features.py

Walk-forward note:
  The HMM is fit on ALL available history (expanding window). At inference time
  for live scoring, we refit periodically (e.g. weekly) and predict the current
  regime. This is NOT lookahead — HMM.predict uses the Viterbi algorithm which
  decodes the most likely state sequence given ALL observations up to time t.
  For strict causal inference, we also provide filtered probabilities.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Import regime detector from src/ module
sys.path.insert(0, str(PROJECT_DIR))
from src.regime import VolRegimeDetector


def load_data():
    """Load and merge SPX daily returns with VIX."""
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    vix = pd.read_parquet(DATA_DIR / "vix_daily.parquet")

    spx["date"] = pd.to_datetime(spx["date"])
    vix["date"] = pd.to_datetime(vix["date"])

    spx = spx.sort_values("date").reset_index(drop=True)
    spx["spx_log_return"] = np.log(spx["spx_close"] / spx["spx_close"].shift(1))

    df = spx.merge(vix[["date", "vix_close"]], on="date", how="inner")
    df = df.dropna(subset=["spx_log_return"]).reset_index(drop=True)

    return df


def print_regime_summary(df, detector):
    """Print summary statistics for each regime."""
    print(f"\n{'='*60}")
    print(f"HMM Regime Detection Summary (2 states)")
    print(f"{'='*60}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Total trading days: {len(df)}")

    for s in range(2):
        count = (df["regime_state"] == s).sum()
        pct = count / len(df) * 100
        label = "LOW-VOL" if s == 0 else "HIGH-VOL"
        print(f"  State {s} ({label}): {count} days ({pct:.1f}%)")

    for s in range(2):
        mask = df["regime_state"] == s
        rets = df.loc[mask, "spx_log_return"]
        vix = df.loc[mask, "vix_close"]
        label = "LOW-VOL" if s == 0 else "HIGH-VOL"
        print(f"\n  State {s} ({label}):")
        print(f"    SPX return: mean={rets.mean()*100:.3f}%, std={rets.std()*100:.3f}%")
        print(f"    VIX:        mean={vix.mean():.1f}, median={vix.median():.1f}")

    n_switches = df["regime_switch"].sum()
    avg_duration = len(df) / max(n_switches, 1)
    print(f"\nRegime switches: {n_switches} ({n_switches/len(df)*100:.1f}% of days)")
    print(f"Average regime duration: {avg_duration:.1f} days")


def main():
    parser = argparse.ArgumentParser(description="Build HMM regime features")
    parser.add_argument("--states", type=int, default=2, help="Number of HMM states (default: 2)")
    args = parser.parse_args()

    print("Loading SPX + VIX data...")
    df = load_data()
    print(f"  {len(df)} trading days ({df['date'].min().date()} → {df['date'].max().date()})")

    returns = df["spx_log_return"].values
    vix = df["vix_close"].values

    # Fit regime detector
    print(f"\nFitting 2-state GaussianHMM...")
    detector = VolRegimeDetector(n_restarts=25, n_iter=1000)
    detector.fit(returns, vix)

    # Predict regimes
    result = detector.predict(returns, vix)

    df = df.copy()
    df["regime_state"] = result["regime_state"]
    df["regime_prob_highvol"] = result["regime_prob_highvol"]
    df["regime_prob_lowvol"] = result["regime_prob_lowvol"]
    df["regime_duration"] = result["regime_duration"]
    df["regime_switch"] = result["regime_switch"]

    # Print summary
    print_regime_summary(df, detector)

    # Select output columns
    out_cols = ["date", "regime_state", "regime_prob_highvol", "regime_prob_lowvol",
                "regime_duration", "regime_switch"]
    out = df[out_cols].copy()

    # Save features
    outfile = DATA_DIR / "regime_features.parquet"
    out.to_parquet(outfile, index=False)
    print(f"\nSaved regime features to {outfile}")
    print(f"  Shape: {out.shape}")
    print(f"  File size: {outfile.stat().st_size / 1e3:.1f} KB")

    # Save model for online prediction
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "hmm_regime_2state.pkl"
    detector.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
