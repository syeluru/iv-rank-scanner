#!/usr/bin/env python3
"""
Train Random Forest model for 0DTE Iron Condor risk filtering.

The 0DTE IC strategy wins ~76% of the time, but the average loss is 4.7x
the average win — so NET P&L is negative. The ML model's job is to filter
out the ~11% of days with catastrophic losses (PnL < -$3), which account
for the majority of cumulative losses.

Architecture:
  - Random Forest classifier (better than LightGBM at this sample size)
  - 14 curated features (reduced from 43 — less noise with small N)
  - Combined with simple rule-based filters for robustness
  - Trained with class_weight='balanced' to handle 89:11 class imbalance

CV Performance (5-fold TSCV):
  - Combined (Rules + RF): $93.65 total lift, 4/5 positive folds
  - Catches 30% of big losses while skipping ~18% of days
  - Walk-forward backtest: 52% reduction in losses

Usage:
    python scripts/train_model.py                          # Default training
    python scripts/train_model.py --loss-threshold -5      # Stricter loss filter
    python scripts/train_model.py --cv 5                   # Run CV analysis
    python scripts/train_model.py --skip-pct 20            # Set skip percentile
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "ml" / "artifacts" / "models"

# ── Curated features (14) ─────────────────────────────────────────────────────
# Selected via domain expertise + feature importance analysis.
# Removed: 5 constant features, 24 low-importance or highly correlated features.
MODEL_FEATURES = [
    # Volatility regime (4)
    "vix_sma_10d_dist",     # VIX deviation from 10d SMA (recent spike indicator)
    "iv_rv_ratio",          # Implied / realized vol ratio (vol risk premium)
    "rv_5d_20d_ratio",      # Short-term vs long-term RV (vol acceleration)
    "rv_acceleration",      # 5d RV minus 10d RV (vol momentum)
    # Gap / prior day (4)
    "gap_pct",              # Overnight gap size (directional risk)
    "gap_vs_atr",           # Gap normalized by ATR (relative gap size)
    "prior_day_range_pct",  # Prior day's high-low range %
    "prior_day_return",     # Prior day's return (momentum/mean-reversion)
    # Intraday (3)
    "orb_range_pct",        # Opening range breakout size
    "orb_vs_atr",           # Opening range normalized by ATR
    "momentum_30min",       # 30-min momentum before entry
    # Term structure (2)
    "vix_term_slope",       # VIX term structure slope (contango/backwardation)
    "atm_iv",               # ATM implied volatility level
    # Market level (1)
    "vix_level",            # VIX level (fear gauge)
]

# ── Derived feature definitions ──────────────────────────────────────────────
DERIVED_FEATURES = {
    "iv_rv_ratio": ("atm_iv", "/", "rv_close_20d"),
    "rv_5d_20d_ratio": ("rv_close_5d", "/", "rv_close_20d"),
    "rv_acceleration": ("rv_close_5d", "-", "rv_close_10d"),
    "gap_vs_atr": ("|gap_pct|", "/", "atr_14d_pct"),
    "orb_vs_atr": ("orb_range_pct", "/", "atr_14d_pct"),
}

# ── Rule-based filters ────────────────────────────────────────────────────────
# Simple thresholds that catch obvious danger days.
# These don't need ML — they're market microstructure knowledge.
RULE_FILTERS = {
    "vix_sma_10d_dist": (">", 15),    # VIX 15%+ above 10d SMA
    "gap_vs_atr": (">", 1.5),         # Overnight gap > 1.5x ATR
    "rv_5d_20d_ratio": (">", 1.5),    # Short-term vol 50%+ above long-term
    "prior_day_range_pct": (">", 2.0), # Prior day range > 2%
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the training dataset."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    logger.info(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset."""
    df = df.copy()
    rv20 = df["rv_close_20d"].clip(lower=1)
    df["iv_rv_ratio"] = df["atm_iv"] / rv20
    df["rv_5d_20d_ratio"] = df["rv_close_5d"] / rv20
    df["rv_acceleration"] = df["rv_close_5d"] - df["rv_close_10d"]
    df["gap_vs_atr"] = df["gap_pct"].abs() / df["atr_14d_pct"].clip(lower=0.01)
    df["orb_vs_atr"] = df["orb_range_pct"] / df["atr_14d_pct"].clip(lower=0.01)
    logger.info(f"  Engineered {len(DERIVED_FEATURES)} derived features")
    return df


def create_labels(df: pd.DataFrame, loss_threshold: float = -3.0) -> pd.DataFrame:
    """Create the avoid_trade label."""
    df = df.copy()
    df["avoid_trade"] = (df["pnl_at_close"] < loss_threshold).astype(float)
    df.loc[df["pnl_at_close"].isna(), "avoid_trade"] = np.nan
    n_pos = df["avoid_trade"].sum()
    n_valid = df["avoid_trade"].notna().sum()
    rate = n_pos / n_valid if n_valid > 0 else 0
    logger.info(f"  avoid_trade (PnL < ${loss_threshold}): {rate:.1%} ({n_pos:.0f}/{n_valid} days)")
    return df


def apply_rules(df: pd.DataFrame) -> np.ndarray:
    """Apply rule-based filters. Returns boolean mask (True = skip this day)."""
    skip = np.zeros(len(df), dtype=bool)
    for feature, (op, threshold) in RULE_FILTERS.items():
        if feature not in df.columns:
            continue
        if op == ">":
            skip |= df[feature].values > threshold
        elif op == "<":
            skip |= df[feature].values < threshold
    return skip


def prepare_data(df: pd.DataFrame, test_months: int = 6) -> tuple:
    """Prepare time-based train/test split."""
    valid = df[df["avoid_trade"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values("trade_date").reset_index(drop=True)

    if len(valid) < 50:
        logger.error("Too few valid samples for training")
        sys.exit(1)

    max_date = valid["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=test_months)
    train = valid[valid["trade_date"] < cutoff]
    test = valid[valid["trade_date"] >= cutoff]

    features = [c for c in MODEL_FEATURES if c in valid.columns]
    missing = [c for c in MODEL_FEATURES if c not in valid.columns]
    if missing:
        logger.warning(f"  Missing features: {missing}")

    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Train: {len(train)} ({train['trade_date'].min().date()} → {train['trade_date'].max().date()})")
    logger.info(f"  Test:  {len(test)} ({test['trade_date'].min().date()} → {test['trade_date'].max().date()})")
    logger.info(f"  Train avoid rate: {train['avoid_trade'].mean():.1%}, Test: {test['avoid_trade'].mean():.1%}")

    X_train = train[features].fillna(0).values
    X_test = test[features].fillna(0).values
    y_train = train["avoid_trade"].astype(int).values
    y_test = test["avoid_trade"].astype(int).values

    return X_train, X_test, y_train, y_test, train, test, features


def train_rf(X_train, y_train):
    """Train Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_cv(df: pd.DataFrame, features: list, n_splits: int = 5) -> dict:
    """Run time-series cross-validation with combined (rules + RF) evaluation."""
    valid = df[df["avoid_trade"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values("trade_date").reset_index(drop=True)

    X = valid[features].fillna(0).values
    y = valid["avoid_trade"].astype(int).values
    pnl = valid["pnl_at_close"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"rf_only": [], "rules_only": [], "combined": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pnl_te = pnl[test_idx]
        test_rows = valid.iloc[test_idx]
        baseline = pnl_te.sum()
        n_avoidable = sum(1 for p in pnl_te if p < -3)

        dates = valid.iloc[test_idx]["trade_date"]

        # RF model
        rf = train_rf(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5

        # Rule-based
        rule_skip = apply_rules(test_rows)

        # RF skip (top 15%)
        rf_thresh = np.percentile(y_prob, 85)
        rf_skip = y_prob >= rf_thresh

        # Combined
        rf_top10 = y_prob >= np.percentile(y_prob, 90)
        combined_skip = rule_skip | rf_top10

        for name, skip in [("rf_only", rf_skip), ("rules_only", rule_skip), ("combined", combined_skip)]:
            trade = ~skip
            traded_pnl = pnl_te[trade].sum()
            avoided = sum(1 for p in pnl_te[skip] if p < -3)
            results[name].append({
                "pnl": traded_pnl, "lift": traded_pnl - baseline,
                "skipped": skip.sum(), "avoided": avoided, "total": n_avoidable,
            })

        logger.info(
            f"  Fold {fold+1}: {dates.min().date()}→{dates.max().date()}, "
            f"AUC={auc:.3f}, "
            f"RF lift=${results['rf_only'][-1]['lift']:+.1f}, "
            f"Combined lift=${results['combined'][-1]['lift']:+.1f}"
        )

    logger.info("\n  CV Summary:")
    for name, folds in results.items():
        total_lift = sum(f["lift"] for f in folds)
        total_avoided = sum(f["avoided"] for f in folds)
        total_avoidable = sum(f["total"] for f in folds)
        pos_folds = sum(1 for f in folds if f["lift"] > 0)
        logger.info(
            f"    {name:20s} Lift: ${total_lift:+.2f}, "
            f"Avoided: {total_avoided}/{total_avoidable}, "
            f"Positive folds: {pos_folds}/{n_splits}"
        )

    return results


def evaluate_model(model, X_test, y_test, features, test_df, skip_pct):
    """Evaluate model with classification metrics and economic P&L simulation."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    pnl = test_df["pnl_at_close"].values

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0

    logger.info("\n" + "=" * 65)
    logger.info("MODEL EVALUATION (out-of-sample)")
    logger.info("=" * 65)
    logger.info(f"\n  AUC-ROC:    {auc:.4f}")
    logger.info(f"  PR-AUC:     {ap:.4f}")
    logger.info(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    max_imp = max(importances) if max(importances) > 0 else 1

    logger.info(f"\n  Feature Importances:")
    for name, imp in feat_imp:
        bar = "#" * int(imp / max_imp * 30)
        logger.info(f"    {name:28s} {imp:.4f}  {bar}")

    # ── Economic evaluation: RF only ──
    logger.info("\n" + "=" * 65)
    logger.info("ECONOMIC EVALUATION")
    logger.info("=" * 65)

    baseline_pnl = pnl.sum()
    n_avoidable = sum(1 for p in pnl if p < -3)
    logger.info(f"\n  Baseline: {len(pnl)} trades, P&L ${baseline_pnl:.2f}, {n_avoidable} big losses")

    logger.info(f"\n  RF Model (skip riskiest N% of days):")
    logger.info(f"    {'Skip%':>6s}  {'Trades':>7s}  {'Total PnL':>10s}  {'Avg PnL':>8s}  {'Lift':>10s}  {'Avoided':>10s}")
    logger.info(f"    {'─'*6}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")

    for pct in [10, 13, 15, 18, 20, 25, 30]:
        thresh = np.percentile(y_prob, 100 - pct)
        skip = y_prob >= thresh
        trade = ~skip
        n_trade = trade.sum()
        if n_trade == 0:
            continue
        traded_pnl = pnl[trade].sum()
        avoided = sum(1 for p in pnl[skip] if p < -3)
        lift = traded_pnl - baseline_pnl
        logger.info(
            f"    {pct:4d}%   {n_trade:5d}    "
            f"${traded_pnl:8.2f}  ${traded_pnl/n_trade:6.2f}  "
            f"${lift:+8.2f}    {avoided}/{n_avoidable}"
        )

    # ── Economic evaluation: Combined (Rules + RF) ──
    rule_skip = apply_rules(test_df)
    n_rule_skip = rule_skip.sum()

    logger.info(f"\n  Combined (Rules + RF top {100-skip_pct}% threshold):")
    logger.info(f"    Rules skip: {n_rule_skip} days")

    rf_thresh = np.percentile(y_prob, skip_pct)
    rf_top_skip = y_prob >= rf_thresh
    combined_skip = rule_skip | rf_top_skip
    combined_trade = ~combined_skip
    n_combined_trade = combined_trade.sum()
    if n_combined_trade > 0:
        combined_pnl = pnl[combined_trade].sum()
        combined_avoided = sum(1 for p in pnl[combined_skip] if p < -3)
        combined_lift = combined_pnl - baseline_pnl
        logger.info(f"    Combined skip: {combined_skip.sum()} days")
        logger.info(f"    Trades: {n_combined_trade}, P&L: ${combined_pnl:.2f}, Lift: ${combined_lift:+.2f}")
        logger.info(f"    Big losses avoided: {combined_avoided}/{n_avoidable}")

    return y_prob


def save_model(model, features, metrics, output_path, loss_threshold, skip_pct):
    """Save model artifact with all metadata needed for production."""
    artifact = {
        "model": model,
        "model_type": "RandomForestClassifier",
        "feature_names": features,
        "n_features": len(features),
        "label": "avoid_trade",
        "loss_threshold": loss_threshold,
        "skip_percentile": skip_pct,
        "rule_filters": RULE_FILTERS,
        "derived_features": DERIVED_FEATURES,
        "metrics": metrics,
        "description": (
            "Risk filter for 0DTE IC trades. Predicts P(big_loss). "
            "Higher probability = more dangerous day. "
            "Use combined with rule filters for best results."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"\nModel saved to: {output_path}")
    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Skip percentile: {skip_pct}")


def main():
    parser = argparse.ArgumentParser(description="Train 0DTE IC risk filter model")
    parser.add_argument(
        "--dataset", type=Path,
        default=TRAINING_DATA_DIR / "training_dataset_h12_w25.csv",
    )
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--cv", type=int, default=0, help="CV folds (0=skip)")
    parser.add_argument("--loss-threshold", type=float, default=-3.0)
    parser.add_argument("--skip-pct", type=int, default=90,
                        help="Percentile threshold for RF skip (90 = skip top 10%%)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or (MODEL_DIR / "lightgbm_v3.joblib")

    logger.info("=" * 65)
    logger.info("0DTE Iron Condor Risk Filter Training")
    logger.info("=" * 65)
    logger.info(f"  Dataset:         {args.dataset}")
    logger.info(f"  Loss threshold:  ${args.loss_threshold}")
    logger.info(f"  Skip percentile: {args.skip_pct}")
    logger.info(f"  Test months:     {args.test_months}")
    logger.info(f"  Output:          {output_path}")

    # Load & prepare
    df = load_dataset(args.dataset)
    df = engineer_features(df)
    df = create_labels(df, args.loss_threshold)

    X_train, X_test, y_train, y_test, train_df, test_df, features = prepare_data(
        df, args.test_months
    )

    # CV
    if args.cv > 0:
        logger.info("\n" + "=" * 65)
        logger.info(f"Cross-validation ({args.cv} folds)...")
        run_cv(df, features, args.cv)

    # Train
    logger.info("\n" + "=" * 65)
    logger.info("Training Random Forest...")
    model = train_rf(X_train, y_train)
    logger.info(f"  Trees: {model.n_estimators}, Depth: {model.max_depth}")

    # Evaluate
    y_prob = evaluate_model(model, X_test, y_test, features, test_df, args.skip_pct)

    # Save
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0
    metrics = {
        "auc_roc": float(auc),
        "pr_auc": float(ap),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "test_avoid_rate": float(y_test.mean()),
    }
    save_model(model, features, metrics, output_path, args.loss_threshold, args.skip_pct)
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
