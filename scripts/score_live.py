"""
Score today's market conditions for live trading decision.

Outputs a single trade/skip recommendation with confidence.

Data sources used (what must be current):
  - SPY price (via yfinance, free)
  - VIX (via yfinance, free)
  - Upcoming econ events (hardcoded dates in fetch_events_data.py)
  - Options chain (requires Theta Terminal running for live IV data)

Usage:
  python3 scripts/score_live.py              # score today
  python3 scripts/score_live.py --date 2025-06-15  # score a specific date
  python3 scripts/score_live.py --threshold 0.6    # custom confidence threshold
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"


def get_latest_spy_data():
    """Fetch current SPY and VIX data via yfinance."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        vix = yf.download("^VIX", period="1y", interval="1d", progress=False, auto_adjust=True)
        return spy, vix
    except ImportError:
        print("  yfinance not installed. pip install yfinance")
        return None, None
    except Exception as e:
        print(f"  yfinance error: {e}")
        return None, None


def build_live_features(target_date, feature_cols):
    """
    Build features for a specific date using existing parquet files.
    Falls back to historical data when live data is unavailable.
    """
    target_dt = pd.Timestamp(target_date)

    features = {}

    # ── SPY technical features ──
    spy_file = DATA_DIR / "spy_features.parquet"
    if spy_file.exists():
        spy = pd.read_parquet(spy_file)
        spy["date"] = pd.to_datetime(spy["date"])

        # Find the row for this date (or most recent prior date)
        spy_day = spy[spy["date"] == target_dt]
        if spy_day.empty:
            spy_day = spy[spy["date"] <= target_dt].tail(1)
        if not spy_day.empty:
            # Take first row (daily features are broadcast across all minutes)
            for col in spy_day.columns:
                if col in feature_cols:
                    features[col] = spy_day[col].iloc[0]

    # ── Options features ──
    opts_file = DATA_DIR / "options_features.parquet"
    if opts_file.exists():
        opts = pd.read_parquet(opts_file)
        opts["date"] = pd.to_datetime(opts["date"])

        opts_day = opts[opts["date"] == target_dt]
        if opts_day.empty:
            opts_day = opts[opts["date"] <= target_dt].tail(1)
        if not opts_day.empty:
            for col in opts_day.columns:
                if col in feature_cols:
                    features[col] = opts_day[col].iloc[0]

    return features


def check_hard_blockers(target_date):
    """
    Check rules-based hard blockers regardless of model score.
    Returns (should_skip, reasons).
    """
    target_dt = pd.Timestamp(target_date)
    blockers = []

    # Load event files
    for fname, label in [
        ("fomc_dates.parquet",      "FOMC"),
        ("econ_calendar.parquet",   "Econ"),
        ("mag7_earnings.parquet",   "MAG7 Earnings"),
    ]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"])

        if "event" in df.columns:
            # econ_calendar has event types
            hits = df[df["date"] == target_dt]
            if not hits.empty:
                for _, row in hits.iterrows():
                    blockers.append(f"{label}: {row.get('event', 'event')}")
        else:
            if (df["date"] == target_dt).any():
                blockers.append(label)

    # Check if 1 day before FOMC
    fomc_path = DATA_DIR / "fomc_dates.parquet"
    if fomc_path.exists():
        fomc = pd.read_parquet(fomc_path)
        fomc["date"] = pd.to_datetime(fomc["date"])
        fomc_dates = set(fomc["date"])
        eve = target_dt + pd.Timedelta(days=1)
        if eve in fomc_dates:
            blockers.append("FOMC Eve (FOMC tomorrow)")

    return len(blockers) > 0, blockers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date to score (YYYY-MM-DD, default: today)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Probability threshold for TRADE recommendation")
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    print(f"{'='*60}")
    print(f"  0DTE IC Trade Scorer — {target_date}")
    print(f"{'='*60}")

    # ── Load model ──
    model_file = MODELS_DIR / "xgb_model.pkl"
    if not model_file.exists():
        print(f"\n❌ No model found at {model_file}. Run 'make train' first.")
        exit(1)

    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    model        = model_data["model"]
    feature_cols = model_data["feature_cols"]
    target_col   = model_data["target_col"]
    cv_auc       = model_data["cv_auc"]

    print(f"\nModel: {target_col} | CV AUC: {cv_auc:.4f}")

    # ── Hard blockers ──
    print(f"\nChecking hard blockers...")
    should_skip, blockers = check_hard_blockers(target_date)

    if should_skip:
        print(f"\n🚫 SKIP — Hard blockers triggered:")
        for b in blockers:
            print(f"   • {b}")
        print(f"\n   Recommendation: DO NOT TRADE TODAY")
        return

    print(f"   ✓ No hard blockers")

    # ── Build features ──
    print(f"\nBuilding features for {target_date}...")
    features = build_live_features(target_date, feature_cols)

    # Build feature vector
    X_row = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
    feature_coverage = X_row.notna().mean().item()

    if feature_coverage < 0.5:
        print(f"\n⚠️  Low feature coverage ({feature_coverage:.0%}) — prediction unreliable")
        print(f"   Ensure data files are up to date (run 'make fetch')")

    # Fill NaNs
    X_row = X_row.fillna(0)

    # ── Predict ──
    prob = model.predict_proba(X_row)[0, 1]
    print(f"\nFeature coverage: {feature_coverage:.0%}")

    print(f"\n{'─'*60}")
    if prob >= args.threshold:
        verdict = "✅ TRADE"
        action  = "Enter Iron Condor"
    else:
        verdict = "⏭️  SKIP"
        action  = "No trade today"

    print(f"\n  {verdict}")
    print(f"  Probability (IC profitable): {prob:.1%}")
    print(f"  Threshold:                   {args.threshold:.0%}")
    print(f"  Action:                      {action}")

    # Confidence bar
    bar_filled = int(prob * 40)
    bar = "█" * bar_filled + "░" * (40 - bar_filled)
    print(f"\n  [{bar}] {prob:.1%}")
    thresh_marker = " " * int(args.threshold * 40) + "▲ threshold"
    print(f"   {thresh_marker}")

    # ── Feature context ──
    print(f"\n{'─'*60}")
    print(f"  Key indicators:")
    key_features = ["vix_close", "rsi_14", "bb_pct_b", "gap_pct", "intraday_range_pct",
                    "pc_volume_ratio", "atm_straddle_open", "ic_credit_100"]
    for feat in key_features:
        if feat in features and not pd.isna(features[feat]):
            print(f"    {feat:30s}: {features[feat]:.4f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
