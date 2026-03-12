#!/usr/bin/env python3
"""
Train v7 model: predicts P(long_IC_profitable) for 0DTE iron condors.

When the SHORT iron condor loses money, the LONG iron condor wins.
This model identifies days where buying the IC (instead of selling it) is
the profitable play — high-volatility, trending days where short ICs get
crushed.

Labels:
  - long_profitable: pnl_at_close < 0  (short IC lost → long IC won)
  - long_big_win:    pnl_at_close < -3  (short IC had catastrophic loss)

Architecture:
  - RandomForest classifier per delta (d10, d15, d20)
  - ~55 features (full feature set + derived + v3 meta-feature)
  - Time-based train/test split
  - Economic evaluation with threshold sweep

Usage:
    python scripts/train_v7_long_ic.py                    # Train all deltas
    python scripts/train_v7_long_ic.py --delta 10         # Train 10-delta only
    python scripts/train_v7_long_ic.py --delta all --cv 5 # With cross-validation
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
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "ml" / "artifacts" / "models" / "v7"
V3_MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "models" / "lightgbm_v3.joblib"

# ── Dataset paths per delta ────────────────────────────────────────────────────
DELTA_DATASETS = {
    10: TRAINING_DATA_DIR / "training_dataset_multi_w25.csv",
    15: TRAINING_DATA_DIR / "training_dataset_multi_w25_d15.csv",
    20: TRAINING_DATA_DIR / "training_dataset_multi_w25_d20.csv",
}

# ── Base features (from training data) ─────────────────────────────────────────
VOLATILITY_FEATURES = [
    "rv_close_5d", "rv_close_10d", "rv_close_20d", "atr_14d_pct",
    "vix_level", "vix_change_1d", "vix_change_5d", "vix_rank_30d",
    "vix_sma_10d_dist", "vix_term_slope", "atm_iv", "iv_skew_10d",
]

PRICE_ACTION_FEATURES = [
    "gap_pct", "prior_day_range_pct", "prior_day_return", "sma_20d_dist",
]

INTRADAY_FEATURES = [
    "intraday_rv", "move_from_open_pct", "orb_range_pct", "orb_contained",
    "range_exhaustion", "high_low_range_pct", "momentum_30min", "trend_slope_norm",
]

TECHNICAL_FEATURES = [
    "adx", "efficiency_ratio", "choppiness_index", "rsi_dist_50", "linreg_r2",
    "atr_ratio", "bbw_percentile", "ttm_squeeze", "bars_in_squeeze",
    "garman_klass_rv", "orb_failure",
]

TIME_FEATURES = [
    "time_sin", "time_cos", "dow_sin", "dow_cos", "minutes_to_close",
]

CALENDAR_FEATURES = [
    "is_fomc_day", "is_fomc_week", "is_friday", "days_since_fomc",
]

OPTIONS_FEATURES = [
    "short_put_delta", "short_call_delta",
]

# Features already in the dataset (no need to compute)
PRECOMPUTED_DERIVED = [
    "gk_rv_vs_atm_iv", "range_vs_vix_range", "range_exhaustion_pct",
]

# ── Derived features we compute ───────────────────────────────────────────────
COMPUTED_DERIVED = [
    "iv_rv_ratio", "rv_5d_20d_ratio", "rv_acceleration",
    "gap_vs_atr", "orb_vs_atr",
]

# Meta-feature
META_FEATURES = ["v3_confidence"]

# ── v3 model features and rule filters ─────────────────────────────────────────
V3_FEATURES = [
    "vix_sma_10d_dist", "iv_rv_ratio", "rv_5d_20d_ratio", "rv_acceleration",
    "gap_pct", "gap_vs_atr", "prior_day_range_pct", "prior_day_return",
    "orb_range_pct", "orb_vs_atr", "momentum_30min", "vix_term_slope",
    "atm_iv", "vix_level",
]

V3_RULE_FILTERS = {
    "vix_sma_10d_dist": (">", 15),
    "gap_vs_atr": (">", 1.5),
    "rv_5d_20d_ratio": (">", 1.5),
    "prior_day_range_pct": (">", 2.0),
}

# Slippage cost for long IC (commissions + spread crossing)
LONG_IC_SLIPPAGE = 0.80


def build_feature_list() -> list[str]:
    """Build the full ordered feature list."""
    features = (
        VOLATILITY_FEATURES
        + PRICE_ACTION_FEATURES
        + INTRADAY_FEATURES
        + TECHNICAL_FEATURES
        + TIME_FEATURES
        + CALENDAR_FEATURES
        + OPTIONS_FEATURES
        + PRECOMPUTED_DERIVED
        + COMPUTED_DERIVED
        + META_FEATURES
    )
    return features


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the training dataset."""
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    logger.info(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    logger.info(f"  Columns: {len(df.columns)}")
    return df


def load_v3_model():
    """Load the v3 risk filter model for meta-feature scoring."""
    if not V3_MODEL_PATH.exists():
        logger.warning(f"v3 model not found at {V3_MODEL_PATH}, v3_confidence will be 0.5")
        return None
    artifact = joblib.load(V3_MODEL_PATH)
    model = artifact["model"]
    logger.info(f"Loaded v3 model from {V3_MODEL_PATH.name}")
    return model


def score_v3_confidence(df: pd.DataFrame, v3_model) -> np.ndarray:
    """Score each row with v3 model + rule filters to produce v3_confidence meta-feature.

    v3 predicts P(big_loss). We apply rule filters: if any rule triggers,
    confidence is set to 0 (meaning v3 says "skip this day").
    """
    if v3_model is None:
        return np.full(len(df), 0.5)

    # Ensure derived features exist for v3 scoring
    v3_features_present = [f for f in V3_FEATURES if f in df.columns]
    missing = [f for f in V3_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"  Missing v3 features: {missing} — filling with 0")

    X_v3 = pd.DataFrame(index=df.index)
    for f in V3_FEATURES:
        if f in df.columns:
            X_v3[f] = df[f].fillna(0)
        else:
            X_v3[f] = 0.0

    # Get raw v3 probability of big loss
    raw_prob = v3_model.predict_proba(X_v3.values)[:, 1]

    # Apply rule filters: if any rule triggers, set confidence to 0
    # (v3 confidence = 1 - P(big_loss), but 0 if rule-filtered)
    confidence = 1.0 - raw_prob

    for feature, (op, threshold) in V3_RULE_FILTERS.items():
        if feature not in df.columns:
            continue
        vals = df[feature].values
        if op == ">":
            rule_triggered = vals > threshold
        elif op == "<":
            rule_triggered = vals < threshold
        else:
            continue
        confidence[rule_triggered] = 0.0

    logger.info(f"  v3_confidence: mean={confidence.mean():.3f}, "
                f"zeroed={np.sum(confidence == 0)}/{len(confidence)}")
    return confidence


def engineer_features(df: pd.DataFrame, v3_model) -> pd.DataFrame:
    """Add derived features and v3 meta-feature to the dataset."""
    df = df.copy()

    # Computed derived features
    rv20 = df["rv_close_20d"].clip(lower=1)
    atr = df["atr_14d_pct"].clip(lower=0.01)

    df["iv_rv_ratio"] = df["atm_iv"] / rv20
    df["rv_5d_20d_ratio"] = df["rv_close_5d"] / rv20
    df["rv_acceleration"] = df["rv_close_5d"] - df["rv_close_10d"]
    df["gap_vs_atr"] = df["gap_pct"].abs() / atr
    df["orb_vs_atr"] = df["orb_range_pct"] / atr

    # v3 meta-feature
    df["v3_confidence"] = score_v3_confidence(df, v3_model)

    n_derived = len(COMPUTED_DERIVED) + 1  # +1 for v3_confidence
    logger.info(f"  Engineered {n_derived} features (5 derived + v3_confidence)")
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create long IC labels from short IC P&L."""
    df = df.copy()

    # Primary: long IC profitable when short IC loses
    df["long_profitable"] = (df["pnl_at_close"] < 0).astype(float)
    df.loc[df["pnl_at_close"].isna(), "long_profitable"] = np.nan

    # Secondary: long IC big win when short IC has catastrophic loss
    df["long_big_win"] = (df["pnl_at_close"] < -3.0).astype(float)
    df.loc[df["pnl_at_close"].isna(), "long_big_win"] = np.nan

    n_valid = df["long_profitable"].notna().sum()
    n_profitable = df["long_profitable"].sum()
    n_big_win = df["long_big_win"].sum()
    rate_profitable = n_profitable / n_valid if n_valid > 0 else 0
    rate_big_win = n_big_win / n_valid if n_valid > 0 else 0

    logger.info(f"  long_profitable (PnL < $0): {rate_profitable:.1%} "
                f"({n_profitable:.0f}/{n_valid})")
    logger.info(f"  long_big_win (PnL < -$3):   {rate_big_win:.1%} "
                f"({n_big_win:.0f}/{n_valid})")
    return df


def prepare_data(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    test_cutoff: str,
) -> tuple:
    """Prepare time-based train/test split."""
    valid = df[df[label].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values("trade_date").reset_index(drop=True)

    if len(valid) < 50:
        logger.error(f"Too few valid samples for training: {len(valid)}")
        sys.exit(1)

    cutoff = pd.Timestamp(test_cutoff)
    train = valid[valid["trade_date"] < cutoff]
    test = valid[valid["trade_date"] >= cutoff]

    # Filter to features actually present
    available = [f for f in features if f in valid.columns]
    missing = [f for f in features if f not in valid.columns]
    if missing:
        logger.warning(f"  Missing features (dropped): {missing}")

    logger.info(f"  Features: {len(available)}")
    logger.info(f"  Train: {len(train)} rows "
                f"({train['trade_date'].min().date()} to {train['trade_date'].max().date()})")
    logger.info(f"  Test:  {len(test)} rows "
                f"({test['trade_date'].min().date()} to {test['trade_date'].max().date()})")
    logger.info(f"  Train label rate: {train[label].mean():.1%}, "
                f"Test: {test[label].mean():.1%}")

    X_train = train[available].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test[available].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train[label].astype(int).values
    y_test = test[label].astype(int).values

    return X_train, X_test, y_train, y_test, train, test, available


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_cv(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    n_splits: int = 5,
) -> dict:
    """Run time-series cross-validation."""
    valid = df[df[label].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values("trade_date").reset_index(drop=True)

    available = [f for f in features if f in valid.columns]
    X = valid[available].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = valid[label].astype(int).values
    pnl = valid["pnl_at_close"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pnl_te = pnl[test_idx]
        dates = valid.iloc[test_idx]["trade_date"]

        rf = train_rf(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
        aucs.append(auc)

        # Economic: compute long IC P&L for trades where v7 score >= 0.50
        long_pnl = -pnl_te - LONG_IC_SLIPPAGE
        trade_mask = y_prob >= 0.50
        n_trades = trade_mask.sum()
        if n_trades > 0:
            total_long_pnl = long_pnl[trade_mask].sum()
            avg_long_pnl = long_pnl[trade_mask].mean()
        else:
            total_long_pnl = 0
            avg_long_pnl = 0

        logger.info(
            f"  Fold {fold+1}: {dates.min().date()} to {dates.max().date()}, "
            f"AUC={auc:.3f}, trades={n_trades}, "
            f"long P&L=${total_long_pnl:+.2f} (avg ${avg_long_pnl:+.2f})"
        )

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    logger.info(f"\n  CV AUC: {mean_auc:.3f} +/- {std_auc:.3f}")
    return {"aucs": aucs, "mean_auc": mean_auc, "std_auc": std_auc}


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Evaluate model: classification metrics + economic P&L simulation."""
    y_prob = model.predict_proba(X_test)[:, 1]
    pnl = test_df["pnl_at_close"].values

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0
    acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))

    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION (out-of-sample)")
    logger.info("=" * 70)
    logger.info(f"  AUC-ROC:    {auc:.4f}")
    logger.info(f"  PR-AUC:     {ap:.4f}")
    logger.info(f"  Accuracy:   {acc:.4f}")
    logger.info(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    max_imp = max(importances) if max(importances) > 0 else 1

    logger.info(f"\n  Top 20 Feature Importances:")
    for i, (name, imp) in enumerate(feat_imp[:20]):
        bar = "#" * int(imp / max_imp * 30)
        logger.info(f"    {name:28s} {imp:.4f}  {bar}")

    # ── Economic evaluation ──
    logger.info("\n" + "=" * 70)
    logger.info("ECONOMIC EVALUATION — Long IC P&L by Threshold")
    logger.info("=" * 70)

    # Long IC P&L: when short IC loses X, long IC gains X minus slippage
    long_pnl = -pnl - LONG_IC_SLIPPAGE

    # Baseline: always buy the IC
    baseline_n = len(pnl)
    baseline_total = long_pnl.sum()
    baseline_avg = long_pnl.mean()
    baseline_wins = (long_pnl > 0).sum()
    baseline_wr = baseline_wins / baseline_n if baseline_n > 0 else 0

    logger.info(f"\n  Baseline (always buy IC): {baseline_n} trades, "
                f"WR={baseline_wr:.1%}, Avg=${baseline_avg:.2f}, "
                f"Total=${baseline_total:.2f}")

    # Naive v3-inversion baseline: trade when v3 confidence < 0.50
    v3_conf = test_df["v3_confidence"].values if "v3_confidence" in test_df.columns else None
    if v3_conf is not None:
        naive_mask = v3_conf < 0.50
        naive_n = naive_mask.sum()
        if naive_n > 0:
            naive_total = long_pnl[naive_mask].sum()
            naive_avg = long_pnl[naive_mask].mean()
            naive_wins = (long_pnl[naive_mask] > 0).sum()
            naive_wr = naive_wins / naive_n
            logger.info(f"  v3-inversion (conf < 0.50): {naive_n} trades, "
                        f"WR={naive_wr:.1%}, Avg=${naive_avg:.2f}, "
                        f"Total=${naive_total:.2f}")
        else:
            logger.info("  v3-inversion (conf < 0.50): 0 trades")

    # Threshold sweep
    logger.info(f"\n  {'Thresh':>6s}  {'Trades':>7s}  {'WinRate':>8s}  {'AvgPnL':>8s}  "
                f"{'TotalPnL':>10s}  {'ProfitFctr':>11s}")
    logger.info(f"  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*11}")

    best_threshold = 0.50
    best_total_pnl = -np.inf

    for thresh in np.arange(0.40, 0.75, 0.05):
        trade_mask = y_prob >= thresh
        n_trades = trade_mask.sum()
        if n_trades == 0:
            logger.info(f"  {thresh:6.2f}  {0:7d}  {'N/A':>8s}  {'N/A':>8s}  "
                        f"{'N/A':>10s}  {'N/A':>11s}")
            continue

        traded_long_pnl = long_pnl[trade_mask]
        total = traded_long_pnl.sum()
        avg = traded_long_pnl.mean()
        wins = (traded_long_pnl > 0).sum()
        losses_sum = abs(traded_long_pnl[traded_long_pnl < 0].sum())
        gains_sum = traded_long_pnl[traded_long_pnl > 0].sum()
        wr = wins / n_trades
        pf = gains_sum / losses_sum if losses_sum > 0 else float("inf")

        if total > best_total_pnl:
            best_total_pnl = total
            best_threshold = thresh

        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        logger.info(f"  {thresh:6.2f}  {n_trades:7d}  {wr:7.1%}  ${avg:7.2f}  "
                    f"${total:9.2f}  {pf_str:>11s}")

    logger.info(f"\n  Best threshold: {best_threshold:.2f} "
                f"(total P&L: ${best_total_pnl:.2f})")

    metrics = {
        "auc_roc": float(auc),
        "pr_auc": float(ap),
        "accuracy": float(acc),
        "test_samples": int(len(y_test)),
        "label_rate": float(y_test.mean()),
        "best_threshold": float(best_threshold),
        "best_total_pnl": float(best_total_pnl),
        "baseline_total_pnl": float(baseline_total),
    }

    return y_prob, metrics


def save_model(
    model: RandomForestClassifier,
    features: list[str],
    label: str,
    metrics: dict,
    delta: int,
) -> Path:
    """Save model artifact with metadata."""
    output_path = MODEL_DIR / f"long_ic_d{delta}.joblib"
    artifact = {
        "model": model,
        "model_type": "RandomForestClassifier",
        "feature_names": features,
        "n_features": len(features),
        "label": label,
        "delta": delta,
        "slippage": LONG_IC_SLIPPAGE,
        "metrics": metrics,
        "v3_rule_filters": V3_RULE_FILTERS,
        "description": (
            f"v7 Long IC model for {delta}-delta. Predicts P(long_IC_profitable). "
            f"Higher score = more likely the long IC profits. "
            f"Uses v3_confidence as meta-feature."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"\n  Model saved to: {output_path}")
    logger.info(f"  Features: {len(features)}, Label: {label}")
    return output_path


def train_delta(
    delta: int,
    v3_model,
    test_cutoff: str,
    cv_folds: int,
) -> None:
    """Train and evaluate v7 model for a single delta."""
    dataset_path = DELTA_DATASETS.get(delta)
    if dataset_path is None:
        logger.error(f"No dataset configured for delta {delta}")
        return
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING v7 — {delta}-DELTA LONG IC MODEL")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_path.name}")
    logger.info(f"  Test cutoff: {test_cutoff}")

    # Load & prepare
    df = load_dataset(dataset_path)
    df = engineer_features(df, v3_model)
    df = create_labels(df)

    features = build_feature_list()
    label = "long_profitable"

    # Cross-validation
    if cv_folds > 0:
        logger.info(f"\n{'─'*70}")
        logger.info(f"Cross-validation ({cv_folds} folds) — {label}")
        cv_results = run_cv(df, features, label, cv_folds)

    # Train/test split
    logger.info(f"\n{'─'*70}")
    logger.info("Preparing train/test split...")
    X_train, X_test, y_train, y_test, train_df, test_df, used_features = prepare_data(
        df, features, label, test_cutoff
    )

    # Train
    logger.info(f"\nTraining RandomForest (500 trees, depth=6)...")
    model = train_rf(X_train, y_train)
    logger.info(f"  Fit complete. Trees={model.n_estimators}, Depth={model.max_depth}")

    # Evaluate
    y_prob, metrics = evaluate_model(model, X_test, y_test, used_features, test_df)

    # Add CV metrics if available
    if cv_folds > 0:
        metrics["cv_mean_auc"] = cv_results["mean_auc"]
        metrics["cv_std_auc"] = cv_results["std_auc"]
        metrics["cv_folds"] = cv_folds

    metrics["train_samples"] = int(len(y_train))

    # Also train secondary label (long_big_win) for reporting only
    logger.info(f"\n{'─'*70}")
    logger.info("Secondary label: long_big_win (PnL < -$3)")
    X_train_2, X_test_2, y_train_2, y_test_2, _, test_df_2, feats_2 = prepare_data(
        df, features, "long_big_win", test_cutoff
    )
    model_2 = train_rf(X_train_2, y_train_2)
    y_prob_2 = model_2.predict_proba(X_test_2)[:, 1]
    auc_2 = roc_auc_score(y_test_2, y_prob_2) if len(np.unique(y_test_2)) > 1 else 0.5
    logger.info(f"  long_big_win AUC: {auc_2:.4f} (informational only, not saved)")

    # Save primary model
    save_model(model, used_features, label, metrics, delta)


def main():
    parser = argparse.ArgumentParser(
        description="Train v7 Long IC model for 0DTE iron condors"
    )
    parser.add_argument(
        "--delta", type=str, default="all",
        help="Delta to train: 10, 15, 20, or 'all' (default: all)",
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of CV folds (default: 5, 0 to skip)",
    )
    parser.add_argument(
        "--test-cutoff", type=str, default="2025-06-01",
        help="Date cutoff for test set (default: 2025-06-01)",
    )
    args = parser.parse_args()

    # Determine deltas to train
    if args.delta == "all":
        deltas = [10, 15, 20]
    else:
        try:
            deltas = [int(args.delta)]
        except ValueError:
            logger.error(f"Invalid delta: {args.delta}. Use 10, 15, 20, or 'all'.")
            sys.exit(1)
        if deltas[0] not in DELTA_DATASETS:
            logger.error(f"No dataset for delta {deltas[0]}. Available: {list(DELTA_DATASETS.keys())}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("v7 LONG IRON CONDOR MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Deltas:      {deltas}")
    logger.info(f"  CV folds:    {args.cv}")
    logger.info(f"  Test cutoff: {args.test_cutoff}")
    logger.info(f"  Output dir:  {MODEL_DIR}")

    # Load v3 model once (shared across deltas)
    logger.info(f"\nLoading v3 model for meta-feature...")
    v3_model = load_v3_model()

    for delta in deltas:
        train_delta(delta, v3_model, args.test_cutoff, args.cv)

    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE")
    logger.info("=" * 70)
    for delta in deltas:
        path = MODEL_DIR / f"long_ic_d{delta}.joblib"
        status = "saved" if path.exists() else "MISSING"
        logger.info(f"  v7_d{delta}: {path} [{status}]")


if __name__ == "__main__":
    main()
