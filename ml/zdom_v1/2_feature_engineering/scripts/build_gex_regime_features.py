"""
Build GEX-based HMM regime features using net_gex_normalized and VIX.

Classifies market into 3 regimes:
  - Positive GEX (dealers long gamma -> mean-reverting, low vol)
  - Negative GEX (dealers short gamma -> trend-following, high vol)
  - Transition (mixed / regime uncertainty)

Reads:  data/options_features.parquet (net_gex_normalized)
        data/vix_daily.parquet        (vix_close)
Saves:  data/gex_regime_features.parquet  (one row per trading day)
        models/hmm_gex_regime_3state.pkl  (fitted HMM for online prediction)

Features produced:
  - gex_regime           : 0 = positive_gex, 1 = transition, 2 = negative_gex
  - gex_regime_prob_pos  : P(positive GEX regime)
  - gex_regime_prob_neg  : P(negative GEX regime)
  - gex_regime_prob_trans: P(transition regime)
  - gex_regime_duration  : consecutive days in current regime
  - gex_regime_switch    : 1 if regime changed from prior day

Walk-forward note:
  Same as build_regime_features.py -- HMM is fit on all available history
  (expanding window). Viterbi decoding is causal given the full observation
  sequence up to time t. For strict online use, refit periodically.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Import GEX regime detector from src/ module
sys.path.insert(0, str(PROJECT_DIR))
from src.regime import GEXRegimeDetector, GEX_REGIME_LABELS


def load_data():
    """Load and merge net_gex_normalized with VIX."""
    opts = pd.read_parquet(DATA_DIR / "options_features.parquet")
    vix = pd.read_parquet(DATA_DIR / "vix_daily.parquet")

    opts["date"] = pd.to_datetime(opts["date"])
    vix["date"] = pd.to_datetime(vix["date"])

    df = opts[["date", "net_gex_normalized"]].merge(
        vix[["date", "vix_close"]], on="date", how="inner"
    )
    df = df.dropna(subset=["net_gex_normalized", "vix_close"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def print_regime_summary(df, detector):
    """Print summary statistics for each GEX regime."""
    n_states = detector.n_states

    print(f"\n{'='*60}")
    print(f"GEX Regime Detection Summary ({n_states} states)")
    print(f"{'='*60}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Total trading days: {len(df)}")

    print(f"\nState distribution:")
    for s in range(n_states):
        count = (df["gex_regime"] == s).sum()
        pct = count / len(df) * 100
        label = GEX_REGIME_LABELS[s]
        print(f"  State {s} ({label:>12s}): {count:4d} days ({pct:.1f}%)")

    print(f"\nPer-regime statistics:")
    for s in range(n_states):
        mask = df["gex_regime"] == s
        gex = df.loc[mask, "net_gex_normalized"]
        vix = df.loc[mask, "vix_close"]
        label = GEX_REGIME_LABELS[s]
        print(f"\n  State {s} ({label}):")
        print(f"    GEX normalized: mean={gex.mean():,.0f}, median={gex.median():,.0f}, std={gex.std():,.0f}")
        print(f"    VIX:            mean={vix.mean():.1f}, median={vix.median():.1f}")

    n_switches = df["gex_regime_switch"].sum()
    avg_duration = len(df) / max(n_switches, 1)
    print(f"\nRegime switches: {n_switches} ({n_switches/len(df)*100:.1f}% of days)")
    print(f"Average regime duration: {avg_duration:.1f} days")


def main():
    parser = argparse.ArgumentParser(description="Build GEX-based HMM regime features")
    parser.add_argument("--states", type=int, default=3,
                        help="Number of HMM states (default: 3)")
    args = parser.parse_args()
    n_states = args.states

    print("Loading net_gex_normalized + VIX data...")
    df = load_data()
    print(f"  {len(df)} trading days ({df['date'].min().date()} -> {df['date'].max().date()})")

    gex = df["net_gex_normalized"].values
    vix = df["vix_close"].values

    # Fit GEX regime detector
    print(f"\nFitting {n_states}-state GaussianHMM on [GEX_normalized, VIX]...")
    detector = GEXRegimeDetector(n_states=n_states, n_restarts=25, n_iter=1000)
    detector.fit(gex, vix)

    # Predict regimes
    result = detector.predict(gex, vix)

    df = df.copy()
    df["gex_regime"] = result["gex_regime"]
    df["gex_regime_prob_pos"] = result["gex_regime_prob_pos"]
    df["gex_regime_prob_trans"] = result["gex_regime_prob_trans"]
    df["gex_regime_prob_neg"] = result["gex_regime_prob_neg"]
    df["gex_regime_duration"] = result["gex_regime_duration"]
    df["gex_regime_switch"] = result["gex_regime_switch"]

    # Print summary
    print_regime_summary(df, detector)

    # Select output columns
    out_cols = [
        "date", "gex_regime",
        "gex_regime_prob_pos", "gex_regime_prob_trans", "gex_regime_prob_neg",
        "gex_regime_duration", "gex_regime_switch",
    ]
    out = df[out_cols].copy()

    # Save features
    DATA_DIR.mkdir(exist_ok=True)
    outfile = DATA_DIR / "gex_regime_features.parquet"
    out.to_parquet(outfile, index=False)
    print(f"\nSaved GEX regime features to {outfile}")
    print(f"  Shape: {out.shape}")
    print(f"  File size: {outfile.stat().st_size / 1e3:.1f} KB")

    # Save model for online prediction
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"hmm_gex_regime_{n_states}state.pkl"
    detector.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
