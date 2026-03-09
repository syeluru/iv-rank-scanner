#!/usr/bin/env python3
"""
Train v6 baseline take-profit prediction models for 0DTE Iron Condor trades.

v6 baseline changes from v5:
  1. XGBoost instead of Random Forest
  2. Expanded from 26 to 41 features:
     - VIX extended: vix_level, vix_change_1d, vix_change_5d, vix_rank_30d, vix_term_slope
     - Options: iv_skew_10d, short_put_delta, short_call_delta
     - Calendar: days_since_fomc, dow_sin, dow_cos, is_friday
     - Intraday: orb_contained, orb_range_pct, momentum_30min

Usage:
    python scripts/train_entry_model_v6_baseline.py --target hit_25pct --cv 5
    python scripts/train_entry_model_v6_baseline.py --target hit_50pct --cv 5
    python scripts/train_entry_model_v6_baseline.py --both --cv 5
"""

import argparse
import sys
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "ml" / "artifacts" / "models"
V6_DIR = MODEL_DIR / "v6"

# ── v6 Features (41) ────────────────────────────────────────────────────────
# Expanded from v5's 26 features to 41 features
V6_FEATURES = [
    # VIX extended (8) - expanded from 2 to 8
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
    "vix_rank_30d",
    "vix_sma_10d_dist",
    "vix_term_slope",
    # Volatility (6) - retained from v5
    "rv_close_5d",
    "rv_close_10d",
    "rv_close_20d",
    "iv_rv_ratio",
    "atm_iv",
    "garman_klass_rv",
    # Options (3) - new
    "iv_skew_10d",
    "short_put_delta",
    "short_call_delta",
    # Price action (3) - retained from v5
    "gap_pct",
    "prior_day_range_pct",
    "prior_day_return",
    # Technical indicators (11) - retained from v5
    "adx",
    "efficiency_ratio",
    "choppiness_index",
    "rsi_dist_50",
    "linreg_r2",
    "atr_ratio",
    "bbw_percentile",
    "ttm_squeeze",
    "bars_in_squeeze",
    "orb_failure",
    "sma_20d_dist",
    # Intraday (6) - expanded from 2 to 6
    "intraday_rv",
    "high_low_range_pct",
    "orb_contained",
    "orb_range_pct",
    "momentum_30min",
    "range_exhaustion_pct",
    # Time features (3) - retained from v5
    "time_sin",
    "time_cos",
    "minutes_to_close",
    # Calendar (6) - expanded from 2 to 6
    "is_fomc_week",
    "is_fomc_day",
    "days_since_fomc",
    "dow_sin",
    "dow_cos",
    "is_friday",
]

# ── Derived feature definitions ───────────────────────────────────────────────
DERIVED_FEATURES = {
    "iv_rv_ratio": ("atm_iv", "/", "rv_close_20d"),
}

# Valid target columns
VALID_TARGETS = {"hit_25pct", "hit_50pct"}

# Output filename mapping
TARGET_OUTPUT_NAMES = {
    "hit_25pct": "entry_timing_v6_baseline_tp25.joblib",
    "hit_50pct": "entry_timing_v6_baseline_tp50.joblib",
}


def log_info(msg):
    """Simple logger replacement."""
    print(f"[INFO] {msg}")


def log_warning(msg):
    """Simple logger replacement."""
    print(f"[WARNING] {msg}")


def log_error(msg):
    """Simple logger replacement."""
    print(f"[ERROR] {msg}")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the multi-entry training dataset."""
    df = pd.read_csv(path)
    log_info(f"Loaded {len(df)} rows from {path.name}")
    log_info(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    if "entry_time" in df.columns:
        n_times = df["entry_time"].nunique()
        log_info(f"  Entry times: {n_times} unique slots")
    else:
        log_warning("  No 'entry_time' column — treating as single-entry dataset")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset."""
    df = df.copy()
    rv20 = df["rv_close_20d"].clip(lower=1)
    df["iv_rv_ratio"] = df["atm_iv"] / rv20
    log_info(f"  Engineered {len(DERIVED_FEATURES)} derived features")
    return df


def create_labels(df: pd.DataFrame, target: str = "hit_25pct") -> pd.DataFrame:
    """Create target label from existing columns (hit_25pct or hit_50pct)."""
    if target not in VALID_TARGETS:
        raise ValueError(f"Invalid target '{target}'. Must be one of: {VALID_TARGETS}")

    df = df.copy()
    df["target"] = df[target].astype(float)
    df.loc[df[target].isna(), "target"] = np.nan

    n_pos = df["target"].sum()
    n_valid = df["target"].notna().sum()
    rate = n_pos / n_valid if n_valid > 0 else 0
    log_info(f"  Target '{target}': {rate:.1%} base rate ({n_pos:.0f}/{n_valid} rows)")
    return df


class GroupTimeSeriesSplit:
    """
    Time-series cross-validation that splits by group (date).

    All rows for a given date stay in the same fold, preventing
    same-day data leakage when multiple entry times per day exist.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided (trade dates)")
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        min_train = n_groups // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Walk-forward: expanding training window
            train_end = min_train + fold * ((n_groups - min_train) // self.n_splits)
            test_end = min(train_end + (n_groups - min_train) // self.n_splits, n_groups)

            if train_end >= n_groups or test_end <= train_end:
                continue

            train_groups = set(unique_groups[:train_end])
            test_groups = set(unique_groups[train_end:test_end])

            train_idx = np.where(np.isin(groups, list(train_groups)))[0]
            test_idx = np.where(np.isin(groups, list(test_groups)))[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self):
        return self.n_splits


def prepare_data(df: pd.DataFrame, target: str, test_months: int = 6) -> tuple:
    """Prepare time-based train/test split."""
    valid = df[df["target"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values(
        ["trade_date", "entry_time"] if "entry_time" in valid.columns else ["trade_date"]
    )
    valid = valid.reset_index(drop=True)

    if len(valid) < 100:
        log_error(f"Too few valid samples ({len(valid)}) for training")
        sys.exit(1)

    max_date = valid["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=test_months)
    train = valid[valid["trade_date"] < cutoff]
    test = valid[valid["trade_date"] >= cutoff]

    features = [c for c in V6_FEATURES if c in valid.columns]
    missing = [c for c in V6_FEATURES if c not in valid.columns]
    if missing:
        log_warning(f"  Missing features: {missing}")

    n_train_days = train["trade_date"].dt.date.nunique()
    n_test_days = test["trade_date"].dt.date.nunique()

    log_info(f"  Features: {len(features)}")
    log_info(f"  Train: {len(train)} rows ({n_train_days} days) "
                f"({train['trade_date'].min().date()} -> {train['trade_date'].max().date()})")
    log_info(f"  Test:  {len(test)} rows ({n_test_days} days) "
                f"({test['trade_date'].min().date()} -> {test['trade_date'].max().date()})")
    log_info(f"  Train hit rate: {train['target'].mean():.1%}, "
                f"Test: {test['target'].mean():.1%}")

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train["target"].astype(int).values
    y_test = test["target"].astype(int).values

    return X_train, X_test, y_train, y_test, train, test, features


def train_xgb(X_train, y_train):
    """Train XGBoost for TP prediction."""
    # Calculate scale_pos_weight from class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=25,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    return model


def run_cv(df: pd.DataFrame, target: str, features: list, n_splits: int = 5) -> dict:
    """Run group time-series cross-validation with TP-focused economic eval."""
    valid = df[df["target"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values(
        ["trade_date", "entry_time"] if "entry_time" in valid.columns else ["trade_date"]
    )
    valid = valid.reset_index(drop=True)

    # Time-to-TP column name
    time_col = "time_to_25pct_min" if target == "hit_25pct" else "time_to_50pct_min"
    has_time_col = time_col in valid.columns

    X = valid[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = valid["target"].astype(int).values
    groups = valid["trade_date"].dt.date.values

    cv = GroupTimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        dates = valid.iloc[test_idx]["trade_date"]
        n_test_days = len(np.unique(groups[test_idx]))

        xgb = train_xgb(X_tr, y_tr)
        y_prob = xgb.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5

        # Economic eval: noon hit rate vs model-guided hit rate
        test_df = valid.iloc[test_idx].copy()
        test_df["v6_score"] = y_prob

        noon_hits = 0
        noon_total = 0
        noon_time_sum = 0.0
        for day, group in test_df.groupby(test_df["trade_date"].dt.date):
            noon_rows = group[group["entry_time"] == "12:00"] if "entry_time" in group.columns else group
            if len(noon_rows) > 0:
                noon_total += 1
                hit = noon_rows["target"].values[0]
                noon_hits += int(hit)
                if has_time_col and hit:
                    noon_time_sum += noon_rows[time_col].values[0]

        results.append({
            "fold": fold + 1,
            "auc": auc,
            "n_test_rows": len(test_idx),
            "n_test_days": n_test_days,
            "noon_hit_rate": noon_hits / noon_total if noon_total > 0 else 0,
            "noon_total": noon_total,
        })

        log_info(
            f"  Fold {fold+1}: {dates.min().date()}->{dates.max().date()}, "
            f"{n_test_days} days, AUC={auc:.3f}, "
            f"noon hit rate={noon_hits}/{noon_total}"
        )

    log_info("\n  CV Summary:")
    avg_auc = np.mean([r["auc"] for r in results])
    log_info(f"    Avg AUC: {avg_auc:.3f}")

    return results


def evaluate_model(model, X_test, y_test, features, test_df, target, threshold):
    """Evaluate model with TP-focused classification metrics and economic simulation."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Time-to-TP column name
    time_col = "time_to_25pct_min" if target == "hit_25pct" else "time_to_50pct_min"
    has_time_col = time_col in test_df.columns
    tp_label = "25% TP" if target == "hit_25pct" else "50% TP"

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0

    log_info("\n" + "=" * 65)
    log_info(f"MODEL EVALUATION — v6 {tp_label} (out-of-sample)")
    log_info("=" * 65)
    log_info(f"\n  AUC-ROC:    {auc:.4f}")
    log_info(f"  PR-AUC:     {ap:.4f}")
    log_info(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    log_info(f"  Accuracy @{threshold}: {accuracy_score(y_test, y_pred):.3f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    max_imp = max(importances) if max(importances) > 0 else 1

    log_info(f"\n  Feature Importances (top 20):")
    for name, imp in feat_imp[:20]:
        bar = "#" * int(imp / max_imp * 30)
        log_info(f"    {name:28s} {imp:.4f}  {bar}")

    # ── Economic evaluation (TP-focused) ──
    log_info("\n" + "=" * 65)
    log_info(f"ECONOMIC EVALUATION — {tp_label}")
    log_info("=" * 65)

    test_copy = test_df.copy()
    test_copy["v6_score"] = y_prob

    # Per-day analysis
    day_results = []
    for day, group in test_copy.groupby(pd.to_datetime(test_copy["trade_date"]).dt.date):
        # Baseline: noon entry
        noon_rows = group[group["entry_time"] == "12:00"] if "entry_time" in group.columns else group
        if len(noon_rows) > 0:
            noon_hit = int(noon_rows["target"].values[0])
            noon_time_to_tp = noon_rows[time_col].values[0] if has_time_col and noon_hit else np.nan
        else:
            noon_hit = 0
            noon_time_to_tp = np.nan

        # v6 threshold: enter at first time above threshold
        above = group[group["v6_score"] >= threshold]
        if len(above) > 0:
            first_above = above.iloc[0]
            thresh_hit = int(first_above["target"])
            thresh_time = first_above["entry_time"] if "entry_time" in group.columns else "12:00"
            thresh_time_to_tp = first_above[time_col] if has_time_col and thresh_hit else np.nan
            entered = True
        else:
            thresh_hit = 0
            thresh_time = None
            thresh_time_to_tp = np.nan
            entered = False

        day_results.append({
            "date": day,
            "noon_hit": noon_hit,
            "noon_time_to_tp": noon_time_to_tp,
            "thresh_hit": thresh_hit,
            "thresh_time": thresh_time,
            "thresh_time_to_tp": thresh_time_to_tp,
            "entered": entered,
        })

    day_df = pd.DataFrame(day_results)
    n_days = len(day_df)
    n_entered = day_df["entered"].sum()
    skip_rate = 1 - n_entered / n_days if n_days > 0 else 0

    noon_hits = day_df["noon_hit"].sum()
    noon_hit_rate = noon_hits / n_days if n_days > 0 else 0
    noon_avg_time = day_df["noon_time_to_tp"].dropna().mean()

    entered_df = day_df[day_df["entered"]]
    thresh_hits = entered_df["thresh_hit"].sum() if len(entered_df) > 0 else 0
    thresh_hit_rate = thresh_hits / n_entered if n_entered > 0 else 0
    thresh_avg_time = entered_df["thresh_time_to_tp"].dropna().mean()

    log_info(f"\n  {n_days} test days:")
    log_info(f"")
    log_info(f"  Strategy          {tp_label} Hit Rate   Avg Time-to-TP   Days Traded   Skip Rate")
    log_info(f"  {'─'*15}    {'─'*14}   {'─'*14}   {'─'*11}   {'─'*9}")
    log_info(
        f"  Noon (baseline)   {noon_hit_rate:>12.1%}   "
        f"{'%6.0f min' % noon_avg_time if not np.isnan(noon_avg_time) else '   N/A':>14s}   "
        f"{n_days:>11d}   {'0%':>9s}"
    )
    if n_entered > 0:
        log_info(
            f"  v6 thresh={threshold:.2f}   {thresh_hit_rate:>12.1%}   "
            f"{'%6.0f min' % thresh_avg_time if not np.isnan(thresh_avg_time) else '   N/A':>14s}   "
            f"{n_entered:>11d}   {skip_rate:>8.0%}"
        )
    else:
        log_info(f"  v6 thresh={threshold:.2f}   No entries (threshold too high)")

    # Threshold sweep
    log_info(f"\n  Threshold sweep:")
    log_info(f"    {'Thresh':>7s}  {'Entered':>8s}  {'Skip%':>6s}  {'Hit Rate':>9s}  {'Avg Time':>9s}  {'Lift':>6s}")
    log_info(f"    {'─'*7}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*6}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        entered_days = []
        for _, row in day_df.iterrows():
            group = test_copy[pd.to_datetime(test_copy["trade_date"]).dt.date == row["date"]]
            above = group[group["v6_score"] >= t]
            if len(above) > 0:
                first = above.iloc[0]
                entered_days.append({
                    "hit": int(first["target"]),
                    "time": first[time_col] if has_time_col and int(first["target"]) else np.nan,
                })
        n_e = len(entered_days)
        if n_e == 0:
            continue
        hits = sum(d["hit"] for d in entered_days)
        hit_rate = hits / n_e
        avg_t = np.nanmean([d["time"] for d in entered_days if d["hit"]])
        skip = 1 - n_e / n_days
        lift = hit_rate - noon_hit_rate
        time_str = f"{avg_t:7.0f}m" if not np.isnan(avg_t) else "    N/A"
        log_info(
            f"    {t:>6.2f}   {n_e:>6d}    {skip:>5.0%}   {hit_rate:>7.1%}   {time_str}  {lift:>+5.1%}"
        )

    return y_prob, {"auc_roc": float(auc), "pr_auc": float(ap)}


def save_model(model, features, metrics, target, output_path):
    """Save v6 model artifact."""
    artifact = {
        "model": model,
        "model_version": "v6_baseline",
        "model_type": "XGBClassifier",
        "feature_names": features,
        "n_features": len(features),
        "label": target,
        "derived_features": DERIVED_FEATURES,
        "metrics": metrics,
        "description": (
            f"v6 baseline TP prediction model for 0DTE IC trades. "
            f"Uses XGBoost with 41 features (expanded from v5's 26). "
            f"Predicts P({target}) at each 5-min interval. "
            f"Higher = more likely to hit take-profit target."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    log_info(f"\nModel saved to: {output_path}")
    log_info(f"  Version: v6_baseline")
    log_info(f"  Model type: XGBoost")
    log_info(f"  Target: {target}")
    log_info(f"  Features: {len(features)}")


def train_one_target(args, target: str, output_path: Path = None):
    """Train a single v6 model for the given target."""
    output_path = output_path or (V6_DIR / TARGET_OUTPUT_NAMES[target])
    tp_label = "25% TP" if target == "hit_25pct" else "50% TP"

    log_info("=" * 65)
    log_info(f"0DTE Iron Condor v6 Baseline Model — {tp_label}")
    log_info("=" * 65)
    log_info(f"  Dataset:    {args.dataset}")
    log_info(f"  Target:     {target}")
    log_info(f"  Threshold:  {args.threshold}")
    log_info(f"  Test months: {args.test_months}")
    log_info(f"  Output:     {output_path}")

    # Load & prepare
    df = load_dataset(args.dataset)
    df = engineer_features(df)
    df = create_labels(df, target=target)

    X_train, X_test, y_train, y_test, train_df, test_df, features = prepare_data(
        df, target, args.test_months
    )

    # CV
    if args.cv > 0:
        log_info("\n" + "=" * 65)
        log_info(f"Cross-validation ({args.cv} folds, grouped by date)...")
        run_cv(df, target, features, args.cv)

    # Train
    log_info("\n" + "=" * 65)
    log_info("Training XGBoost...")
    start_time = time.time()
    model = train_xgb(X_train, y_train)
    train_time = time.time() - start_time
    log_info(f"  Trees: {model.n_estimators}, Depth: {model.max_depth}, "
                f"Learning rate: {model.learning_rate}")
    log_info(f"  Training time: {train_time:.1f}s")

    # Evaluate
    y_prob, metrics = evaluate_model(model, X_test, y_test, features, test_df, target, args.threshold)
    metrics["test_samples"] = int(len(y_test))
    metrics["train_samples"] = int(len(y_train))
    metrics["test_hit_rate"] = float(y_test.mean())
    metrics["threshold"] = args.threshold
    metrics["target"] = target
    metrics["train_time_seconds"] = train_time

    # Save
    save_model(model, features, metrics, target, output_path)
    log_info(f"\n{'=' * 65}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train 0DTE IC v6 baseline XGBoost models")
    parser.add_argument(
        "--dataset", type=Path,
        default=TRAINING_DATA_DIR / "training_dataset_multi_w25.csv",
    )
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--cv", type=int, default=0, help="CV folds (0=skip)")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Entry confidence threshold for economic eval")
    parser.add_argument("--target", type=str, default="hit_25pct",
                        choices=["hit_25pct", "hit_50pct"],
                        help="Target label to train on")
    parser.add_argument("--both", action="store_true",
                        help="Train both v6.1 (25%% TP) and v6.2 (50%% TP) sequentially")
    parser.add_argument("--output", type=Path, default=None,
                        help="Custom output path (only for single-target mode)")
    args = parser.parse_args()

    if args.both:
        log_info(f"Training both TP25 and TP50...\n")
        m1 = train_one_target(args, "hit_25pct", V6_DIR / "entry_timing_v6_baseline_tp25.joblib")
        log_info("\n")
        m2 = train_one_target(args, "hit_50pct", V6_DIR / "entry_timing_v6_baseline_tp50.joblib")
        log_info("\nDone! Both models trained.")
        log_info(f"  TP25 AUC: {m1['auc_roc']:.4f}  |  TP50 AUC: {m2['auc_roc']:.4f}")
        log_info(f"  TP25 train time: {m1['train_time_seconds']:.1f}s  |  TP50 train time: {m2['train_time_seconds']:.1f}s")
    else:
        out = args.output or (V6_DIR / TARGET_OUTPUT_NAMES[args.target])
        train_one_target(args, args.target, out)
        log_info("\nDone!")


if __name__ == "__main__":
    main()
