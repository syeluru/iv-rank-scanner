#!/usr/bin/env python3
"""
Train v5 take-profit prediction models for 0DTE Iron Condor trades.

The v4 model predicts P(profitable at 3PM close), but the bot actually
uses dynamic take-profit orders (25% and 50% of credit). A trade that
hits 25% TP at 11:30 but is underwater at 3PM was a WIN that v4 labels
as a loss.

v5 fixes this mismatch by training two models:
  - v5.1: P(hit 25% take-profit before 3PM) — "will I get my quick exit?"
  - v5.2: P(hit 50% take-profit before 3PM) — "will I get a big win?"

The labels `hit_25pct` and `hit_50pct` already exist in the training
dataset. No dataset rebuild needed.

Additionally includes economic calendar features (is_fomc_day, is_fomc_week)
already present in the dataset.

Usage:
    python scripts/train_entry_model_v5.py --target hit_25pct --cv 5
    python scripts/train_entry_model_v5.py --target hit_50pct --cv 5
    python scripts/train_entry_model_v5.py --both --cv 5
    python scripts/train_entry_model_v5.py --both --threshold 0.80
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

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"
MODEL_DIR = PROJECT_ROOT / "ml" / "artifacts" / "models"

# ── v5 Features (26) ────────────────────────────────────────────────────────
# Same 24 as v4, plus 2 economic calendar features
V5_FEATURES = [
    # Retained v3 day-level features (5) — important for day-level context
    "vix_sma_10d_dist",
    "iv_rv_ratio",
    "gap_pct",
    "prior_day_range_pct",
    "atm_iv",
    # New technical indicators (11) — all time-varying within day
    "adx",
    "efficiency_ratio",
    "choppiness_index",
    "rsi_dist_50",
    "linreg_r2",
    "atr_ratio",
    "bbw_percentile",
    "ttm_squeeze",
    "bars_in_squeeze",
    "garman_klass_rv",
    "orb_failure",
    # Derived cross-features (3) — time-varying
    "gk_rv_vs_atm_iv",
    "range_vs_vix_range",
    "range_exhaustion_pct",
    # Time features (3) — time-varying
    "time_sin",
    "time_cos",
    "minutes_to_close",
    # Existing intraday (2) — time-varying
    "intraday_rv",
    "high_low_range_pct",
    # Economic calendar (2) — day-level
    "is_fomc_week",
    "is_fomc_day",
]

# ── Derived feature definitions ───────────────────────────────────────────────
DERIVED_FEATURES = {
    "iv_rv_ratio": ("atm_iv", "/", "rv_close_20d"),
}

# Valid target columns
VALID_TARGETS = {"hit_25pct", "hit_50pct"}

# Output filename mapping
TARGET_OUTPUT_NAMES = {
    "hit_25pct": "entry_timing_v5_tp25.joblib",
    "hit_50pct": "entry_timing_v5_tp50.joblib",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the multi-entry training dataset."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    logger.info(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    if "entry_time" in df.columns:
        n_times = df["entry_time"].nunique()
        logger.info(f"  Entry times: {n_times} unique slots")
    else:
        logger.warning("  No 'entry_time' column — treating as single-entry dataset")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset."""
    df = df.copy()
    rv20 = df["rv_close_20d"].clip(lower=1)
    df["iv_rv_ratio"] = df["atm_iv"] / rv20
    logger.info(f"  Engineered {len(DERIVED_FEATURES)} derived features")
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
    logger.info(f"  Target '{target}': {rate:.1%} base rate ({n_pos:.0f}/{n_valid} rows)")
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
        logger.error(f"Too few valid samples ({len(valid)}) for training")
        sys.exit(1)

    max_date = valid["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=test_months)
    train = valid[valid["trade_date"] < cutoff]
    test = valid[valid["trade_date"] >= cutoff]

    features = [c for c in V5_FEATURES if c in valid.columns]
    missing = [c for c in V5_FEATURES if c not in valid.columns]
    if missing:
        logger.warning(f"  Missing features: {missing}")

    n_train_days = train["trade_date"].dt.date.nunique()
    n_test_days = test["trade_date"].dt.date.nunique()

    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Train: {len(train)} rows ({n_train_days} days) "
                f"({train['trade_date'].min().date()} -> {train['trade_date'].max().date()})")
    logger.info(f"  Test:  {len(test)} rows ({n_test_days} days) "
                f"({test['trade_date'].min().date()} -> {test['trade_date'].max().date()})")
    logger.info(f"  Train hit rate: {train['target'].mean():.1%}, "
                f"Test: {test['target'].mean():.1%}")

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train["target"].astype(int).values
    y_test = test["target"].astype(int).values

    return X_train, X_test, y_train, y_test, train, test, features


def train_rf(X_train, y_train):
    """Train Random Forest for TP prediction."""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
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

        rf = train_rf(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5

        # Economic eval: noon hit rate vs model-guided hit rate
        test_df = valid.iloc[test_idx].copy()
        test_df["v5_score"] = y_prob

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

        logger.info(
            f"  Fold {fold+1}: {dates.min().date()}->{dates.max().date()}, "
            f"{n_test_days} days, AUC={auc:.3f}, "
            f"noon hit rate={noon_hits}/{noon_total}"
        )

    logger.info("\n  CV Summary:")
    avg_auc = np.mean([r["auc"] for r in results])
    logger.info(f"    Avg AUC: {avg_auc:.3f}")

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

    logger.info("\n" + "=" * 65)
    logger.info(f"MODEL EVALUATION — v5 {tp_label} (out-of-sample)")
    logger.info("=" * 65)
    logger.info(f"\n  AUC-ROC:    {auc:.4f}")
    logger.info(f"  PR-AUC:     {ap:.4f}")
    logger.info(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    logger.info(f"  Accuracy @{threshold}: {accuracy_score(y_test, y_pred):.3f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    max_imp = max(importances) if max(importances) > 0 else 1

    logger.info(f"\n  Feature Importances:")
    for name, imp in feat_imp:
        bar = "#" * int(imp / max_imp * 30)
        logger.info(f"    {name:28s} {imp:.4f}  {bar}")

    # ── Economic evaluation (TP-focused) ──
    logger.info("\n" + "=" * 65)
    logger.info(f"ECONOMIC EVALUATION — {tp_label}")
    logger.info("=" * 65)

    test_copy = test_df.copy()
    test_copy["v5_score"] = y_prob

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

        # v5 threshold: enter at first time above threshold
        above = group[group["v5_score"] >= threshold]
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

    logger.info(f"\n  {n_days} test days:")
    logger.info(f"")
    logger.info(f"  Strategy          {tp_label} Hit Rate   Avg Time-to-TP   Days Traded   Skip Rate")
    logger.info(f"  {'─'*15}    {'─'*14}   {'─'*14}   {'─'*11}   {'─'*9}")
    logger.info(
        f"  Noon (baseline)   {noon_hit_rate:>12.1%}   "
        f"{'%6.0f min' % noon_avg_time if not np.isnan(noon_avg_time) else '   N/A':>14s}   "
        f"{n_days:>11d}   {'0%':>9s}"
    )
    if n_entered > 0:
        logger.info(
            f"  v5 thresh={threshold:.2f}   {thresh_hit_rate:>12.1%}   "
            f"{'%6.0f min' % thresh_avg_time if not np.isnan(thresh_avg_time) else '   N/A':>14s}   "
            f"{n_entered:>11d}   {skip_rate:>8.0%}"
        )
    else:
        logger.info(f"  v5 thresh={threshold:.2f}   No entries (threshold too high)")

    # Threshold sweep
    logger.info(f"\n  Threshold sweep:")
    logger.info(f"    {'Thresh':>7s}  {'Entered':>8s}  {'Skip%':>6s}  {'Hit Rate':>9s}  {'Avg Time':>9s}  {'Lift':>6s}")
    logger.info(f"    {'─'*7}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*6}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        entered_days = []
        for _, row in day_df.iterrows():
            group = test_copy[pd.to_datetime(test_copy["trade_date"]).dt.date == row["date"]]
            above = group[group["v5_score"] >= t]
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
        logger.info(
            f"    {t:>6.2f}   {n_e:>6d}    {skip:>5.0%}   {hit_rate:>7.1%}   {time_str}  {lift:>+5.1%}"
        )

    return y_prob, {"auc_roc": float(auc), "pr_auc": float(ap)}


def save_model(model, features, metrics, target, output_path):
    """Save v5 model artifact."""
    artifact = {
        "model": model,
        "model_version": "v5",
        "model_type": "RandomForestClassifier",
        "feature_names": features,
        "n_features": len(features),
        "label": target,
        "derived_features": DERIVED_FEATURES,
        "metrics": metrics,
        "description": (
            f"v5 TP prediction model for 0DTE IC trades. "
            f"Predicts P({target}) at each 5-min interval. "
            f"Higher = more likely to hit take-profit target. "
            f"Use with v3 risk filter as day-level pre-screen."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"\nModel saved to: {output_path}")
    logger.info(f"  Version: v5")
    logger.info(f"  Target: {target}")
    logger.info(f"  Features: {len(features)}")


def train_one_target(args, target: str, output_path: Path = None):
    """Train a single v5 model for the given target."""
    output_path = output_path or (MODEL_DIR / TARGET_OUTPUT_NAMES[target])
    tp_label = "25% TP" if target == "hit_25pct" else "50% TP"

    logger.info("=" * 65)
    logger.info(f"0DTE Iron Condor v5 Model — {tp_label}")
    logger.info("=" * 65)
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Target:     {target}")
    logger.info(f"  Threshold:  {args.threshold}")
    logger.info(f"  Test months: {args.test_months}")
    logger.info(f"  Output:     {output_path}")

    # Load & prepare
    df = load_dataset(args.dataset)
    df = engineer_features(df)
    df = create_labels(df, target=target)

    # ── Intraday variance check ──────────────────────────────────────────────
    day_level = {"vix_sma_10d_dist", "iv_rv_ratio", "gap_pct", "prior_day_range_pct",
                 "atm_iv", "is_fomc_week", "is_fomc_day"}
    time_varying = [f for f in V5_FEATURES if f in df.columns and f not in day_level]
    if "entry_time" in df.columns and len(time_varying) > 0:
        logger.info("\n  Intraday variance check (should be > 0 for time-varying features):")
        daily_std = df.groupby("trade_date")[time_varying].std()
        mean_std = daily_std.mean()
        for feat in time_varying:
            std_val = mean_std.get(feat, 0)
            status = "OK" if std_val > 0 else "ZERO-VARIANCE"
            logger.info(f"    {feat:28s} avg_within_day_std={std_val:.6f}  [{status}]")

    X_train, X_test, y_train, y_test, train_df, test_df, features = prepare_data(
        df, target, args.test_months
    )

    # CV
    if args.cv > 0:
        logger.info("\n" + "=" * 65)
        logger.info(f"Cross-validation ({args.cv} folds, grouped by date)...")
        run_cv(df, target, features, args.cv)

    # Train
    logger.info("\n" + "=" * 65)
    logger.info("Training Random Forest...")
    model = train_rf(X_train, y_train)
    logger.info(f"  Trees: {model.n_estimators}, Depth: {model.max_depth}, "
                f"MinLeaf: {model.min_samples_leaf}")

    # Evaluate
    y_prob, metrics = evaluate_model(model, X_test, y_test, features, test_df, target, args.threshold)
    metrics["test_samples"] = int(len(y_test))
    metrics["train_samples"] = int(len(y_train))
    metrics["test_hit_rate"] = float(y_test.mean())
    metrics["threshold"] = args.threshold
    metrics["target"] = target

    # Save
    save_model(model, features, metrics, target, output_path)
    logger.info(f"\n{'=' * 65}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train 0DTE IC v5 take-profit models")
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
                        help="Train both v5.1 (25%% TP) and v5.2 (50%% TP) sequentially")
    parser.add_argument("--output", type=Path, default=None,
                        help="Custom output path (only for single-target mode)")
    args = parser.parse_args()

    if args.both:
        logger.info("Training both v5.1 (25%% TP) and v5.2 (50%% TP)...\n")
        m1 = train_one_target(args, "hit_25pct")
        logger.info("\n")
        m2 = train_one_target(args, "hit_50pct")
        logger.info("\nDone! Both models trained.")
        logger.info(f"  v5.1 AUC: {m1['auc_roc']:.4f}  |  v5.2 AUC: {m2['auc_roc']:.4f}")
    else:
        train_one_target(args, args.target, args.output)
        logger.info("\nDone!")


if __name__ == "__main__":
    main()
