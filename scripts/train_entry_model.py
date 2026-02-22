#!/usr/bin/env python3
"""
Train entry timing model (v4) for 0DTE Iron Condor trades.

The 0DTE bot currently enters at a fixed time (noon ET). This v4 model
predicts P(profitable) at each 5-min interval from 10:00-14:00, enabling
dynamic entry timing — enter at the first slot exceeding a confidence
threshold instead of blindly entering at noon.

Architecture:
  - Random Forest classifier (500 trees, depth=6)
  - 17 features: 14 curated v3 features + 3 time features
  - Target: profitable = (pnl_at_close > 0)
  - GroupTimeSeriesSplit by DATE to prevent same-day leakage

The v3 risk filter remains as a day-level pre-screen. This v4 model
only controls WHEN to enter, not WHETHER to enter.

Usage:
    python scripts/train_entry_model.py                          # Default
    python scripts/train_entry_model.py --cv 5                   # Run CV
    python scripts/train_entry_model.py --threshold 0.55         # Entry threshold
    python scripts/train_entry_model.py --dataset path/to/data.csv
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

# ── v4 Features (24) ─────────────────────────────────────────────────────────
# 5 retained v3 day-level + 11 new technical indicators + 3 derived cross-features
# + 3 time features + 2 existing intraday
V4_FEATURES = [
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
]

# ── Derived feature definitions ───────────────────────────────────────────────
DERIVED_FEATURES = {
    "iv_rv_ratio": ("atm_iv", "/", "rv_close_20d"),
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


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create profitable label (binary: pnl_at_close > 0)."""
    df = df.copy()
    df["profitable"] = (df["pnl_at_close"] > 0).astype(float)
    df.loc[df["pnl_at_close"].isna(), "profitable"] = np.nan
    n_pos = df["profitable"].sum()
    n_valid = df["profitable"].notna().sum()
    rate = n_pos / n_valid if n_valid > 0 else 0
    logger.info(f"  profitable (PnL > 0): {rate:.1%} ({n_pos:.0f}/{n_valid} rows)")
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


def prepare_data(df: pd.DataFrame, test_months: int = 6) -> tuple:
    """Prepare time-based train/test split."""
    valid = df[df["profitable"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values(["trade_date", "entry_time"] if "entry_time" in valid.columns else ["trade_date"])
    valid = valid.reset_index(drop=True)

    if len(valid) < 100:
        logger.error(f"Too few valid samples ({len(valid)}) for training")
        sys.exit(1)

    max_date = valid["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=test_months)
    train = valid[valid["trade_date"] < cutoff]
    test = valid[valid["trade_date"] >= cutoff]

    features = [c for c in V4_FEATURES if c in valid.columns]
    missing = [c for c in V4_FEATURES if c not in valid.columns]
    if missing:
        logger.warning(f"  Missing features: {missing}")

    n_train_days = train["trade_date"].dt.date.nunique()
    n_test_days = test["trade_date"].dt.date.nunique()

    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Train: {len(train)} rows ({n_train_days} days) "
                f"({train['trade_date'].min().date()} -> {train['trade_date'].max().date()})")
    logger.info(f"  Test:  {len(test)} rows ({n_test_days} days) "
                f"({test['trade_date'].min().date()} -> {test['trade_date'].max().date()})")
    logger.info(f"  Train win rate: {train['profitable'].mean():.1%}, "
                f"Test: {test['profitable'].mean():.1%}")

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train["profitable"].astype(int).values
    y_test = test["profitable"].astype(int).values

    return X_train, X_test, y_train, y_test, train, test, features


def train_rf(X_train, y_train):
    """Train Random Forest for entry timing."""
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


def run_cv(df: pd.DataFrame, features: list, n_splits: int = 5) -> dict:
    """Run group time-series cross-validation."""
    valid = df[df["profitable"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values(["trade_date", "entry_time"] if "entry_time" in valid.columns else ["trade_date"])
    valid = valid.reset_index(drop=True)

    X = valid[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = valid["profitable"].astype(int).values
    pnl = valid["pnl_at_close"].values
    groups = valid["trade_date"].dt.date.values

    cv = GroupTimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pnl_te = pnl[test_idx]
        dates = valid.iloc[test_idx]["trade_date"]
        n_test_days = len(np.unique(groups[test_idx]))

        rf = train_rf(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5

        # Economic eval: for each test day, compare noon entry vs best v4 entry
        test_df = valid.iloc[test_idx].copy()
        test_df["v4_score"] = y_prob

        day_results = []
        for day, group in test_df.groupby(test_df["trade_date"].dt.date):
            noon_rows = group[group["entry_time"] == "12:00"] if "entry_time" in group.columns else group
            noon_pnl = noon_rows["pnl_at_close"].values[0] if len(noon_rows) > 0 else 0

            best_idx = group["v4_score"].idxmax()
            best_pnl = group.loc[best_idx, "pnl_at_close"]
            best_time = group.loc[best_idx, "entry_time"] if "entry_time" in group.columns else "12:00"

            day_results.append({
                "date": day, "noon_pnl": noon_pnl,
                "best_pnl": best_pnl, "best_time": best_time,
            })

        day_df = pd.DataFrame(day_results)
        noon_total = day_df["noon_pnl"].sum()
        best_total = day_df["best_pnl"].sum()

        results.append({
            "fold": fold + 1,
            "auc": auc,
            "n_test_rows": len(test_idx),
            "n_test_days": n_test_days,
            "noon_pnl": noon_total,
            "best_pnl": best_total,
            "lift": best_total - noon_total,
        })

        logger.info(
            f"  Fold {fold+1}: {dates.min().date()}->{dates.max().date()}, "
            f"{n_test_days} days, AUC={auc:.3f}, "
            f"Noon=${noon_total:+.1f}, Best=${best_total:+.1f}, "
            f"Lift=${best_total - noon_total:+.1f}"
        )

    logger.info("\n  CV Summary:")
    total_lift = sum(r["lift"] for r in results)
    avg_auc = np.mean([r["auc"] for r in results])
    pos_folds = sum(1 for r in results if r["lift"] > 0)
    logger.info(f"    Avg AUC: {avg_auc:.3f}")
    logger.info(f"    Total lift vs noon: ${total_lift:+.2f}")
    logger.info(f"    Positive folds: {pos_folds}/{len(results)}")

    return results


def evaluate_model(model, X_test, y_test, features, test_df, threshold):
    """Evaluate model with classification metrics and economic simulation."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    pnl = test_df["pnl_at_close"].values

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    ap = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0

    logger.info("\n" + "=" * 65)
    logger.info("MODEL EVALUATION (out-of-sample)")
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

    # ── Economic evaluation ──
    logger.info("\n" + "=" * 65)
    logger.info("ECONOMIC EVALUATION")
    logger.info("=" * 65)

    test_copy = test_df.copy()
    test_copy["v4_score"] = y_prob

    # Per-day analysis
    day_results = []
    for day, group in test_copy.groupby(pd.to_datetime(test_copy["trade_date"]).dt.date):
        # Baseline: noon entry
        noon_rows = group[group["entry_time"] == "12:00"] if "entry_time" in group.columns else group
        noon_pnl = noon_rows["pnl_at_close"].values[0] if len(noon_rows) > 0 else 0

        # v4 best-score: enter at time with highest score
        best_idx = group["v4_score"].idxmax()
        best_pnl = group.loc[best_idx, "pnl_at_close"]
        best_time = group.loc[best_idx, "entry_time"] if "entry_time" in group.columns else "12:00"
        best_score = group.loc[best_idx, "v4_score"]

        # v4 threshold: enter at first time above threshold
        above = group[group["v4_score"] >= threshold]
        if len(above) > 0:
            first_above = above.iloc[0]
            thresh_pnl = first_above["pnl_at_close"]
            thresh_time = first_above["entry_time"] if "entry_time" in group.columns else "12:00"
            entered = True
        else:
            thresh_pnl = 0  # No entry
            thresh_time = None
            entered = False

        day_results.append({
            "date": day, "noon_pnl": noon_pnl,
            "best_pnl": best_pnl, "best_time": best_time, "best_score": best_score,
            "thresh_pnl": thresh_pnl, "thresh_time": thresh_time, "entered": entered,
        })

    day_df = pd.DataFrame(day_results)
    n_days = len(day_df)
    n_entered = day_df["entered"].sum()
    skip_rate = 1 - n_entered / n_days if n_days > 0 else 0

    noon_total = day_df["noon_pnl"].sum()
    best_total = day_df["best_pnl"].sum()
    thresh_total = day_df["thresh_pnl"].sum()

    noon_wins = (day_df["noon_pnl"] > 0).sum()
    thresh_wins = (day_df[day_df["entered"]]["thresh_pnl"] > 0).sum()

    logger.info(f"\n  {n_days} test days:")
    logger.info(f"")
    logger.info(f"  Strategy          Total PnL    Avg/Day    Win Rate   Days Traded")
    logger.info(f"  {'─'*15}    {'─'*9}    {'─'*7}    {'─'*8}   {'─'*11}")
    logger.info(
        f"  Noon (baseline)   ${noon_total:>8.2f}    ${noon_total/n_days:>6.2f}    "
        f"{noon_wins/n_days:>6.1%}   {n_days}"
    )
    logger.info(
        f"  v4 best-score     ${best_total:>8.2f}    ${best_total/n_days:>6.2f}    "
        f"{'—':>8}   {n_days}"
    )
    if n_entered > 0:
        logger.info(
            f"  v4 thresh={threshold:.2f}   ${thresh_total:>8.2f}    ${thresh_total/n_entered:>6.2f}    "
            f"{thresh_wins/n_entered:>6.1%}   {n_entered} ({skip_rate:.0%} skip)"
        )
    else:
        logger.info(f"  v4 thresh={threshold:.2f}   No entries (threshold too high)")

    logger.info(f"\n  Lift vs noon:")
    logger.info(f"    Best-score: ${best_total - noon_total:+.2f}")
    logger.info(f"    Threshold:  ${thresh_total - noon_total:+.2f}")

    # Threshold sweep
    logger.info(f"\n  Threshold sweep:")
    logger.info(f"    {'Thresh':>7s}  {'Entered':>8s}  {'Skip%':>6s}  {'Total PnL':>10s}  {'Avg PnL':>8s}  {'Win Rate':>9s}  {'Lift':>10s}")
    logger.info(f"    {'─'*7}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*10}")
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        entered_days = []
        for _, row in day_df.iterrows():
            group = test_copy[pd.to_datetime(test_copy["trade_date"]).dt.date == row["date"]]
            above = group[group["v4_score"] >= t]
            if len(above) > 0:
                entered_days.append(above.iloc[0]["pnl_at_close"])
        n_e = len(entered_days)
        if n_e == 0:
            continue
        total_p = sum(entered_days)
        wins = sum(1 for p in entered_days if p > 0)
        skip = 1 - n_e / n_days
        lift = total_p - noon_total
        logger.info(
            f"    {t:>6.2f}   {n_e:>6d}    {skip:>5.0%}   ${total_p:>8.2f}  ${total_p/n_e:>6.2f}    {wins/n_e:>7.1%}  ${lift:>+8.2f}"
        )

    return y_prob, {"auc_roc": float(auc), "pr_auc": float(ap)}


def save_model(model, features, metrics, output_path):
    """Save v4 model artifact."""
    artifact = {
        "model": model,
        "model_version": "v4",
        "model_type": "RandomForestClassifier",
        "feature_names": features,
        "n_features": len(features),
        "label": "profitable",
        "derived_features": DERIVED_FEATURES,
        "metrics": metrics,
        "description": (
            "Entry timing model for 0DTE IC trades. Predicts P(profitable) "
            "at each 5-min interval. Higher = better time to enter. "
            "Use with v3 risk filter as day-level pre-screen."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"\nModel saved to: {output_path}")
    logger.info(f"  Version: v4")
    logger.info(f"  Features: {len(features)}")


def main():
    parser = argparse.ArgumentParser(description="Train 0DTE IC entry timing model (v4)")
    parser.add_argument(
        "--dataset", type=Path,
        default=TRAINING_DATA_DIR / "training_dataset_multi_w25.csv",
    )
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--cv", type=int, default=0, help="CV folds (0=skip)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Entry confidence threshold for economic eval")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or (MODEL_DIR / "entry_timing_v4.joblib")

    logger.info("=" * 65)
    logger.info("0DTE Iron Condor Entry Timing Model (v4) Training")
    logger.info("=" * 65)
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Threshold:  {args.threshold}")
    logger.info(f"  Test months: {args.test_months}")
    logger.info(f"  Output:     {output_path}")

    # Load & prepare
    df = load_dataset(args.dataset)
    df = engineer_features(df)
    df = create_labels(df)

    # ── Intraday variance check ──────────────────────────────────────────────
    # Verify that time-varying features actually vary within each day
    time_varying = [f for f in V4_FEATURES if f in df.columns and f not in
                    ["vix_sma_10d_dist", "iv_rv_ratio", "gap_pct", "prior_day_range_pct", "atm_iv"]]
    if "entry_time" in df.columns and len(time_varying) > 0:
        logger.info("\n  Intraday variance check (should be > 0 for time-varying features):")
        daily_std = df.groupby("trade_date")[time_varying].std()
        mean_std = daily_std.mean()
        for feat in time_varying:
            std_val = mean_std.get(feat, 0)
            status = "OK" if std_val > 0 else "ZERO-VARIANCE"
            logger.info(f"    {feat:28s} avg_within_day_std={std_val:.6f}  [{status}]")

    X_train, X_test, y_train, y_test, train_df, test_df, features = prepare_data(
        df, args.test_months
    )

    # CV
    if args.cv > 0:
        logger.info("\n" + "=" * 65)
        logger.info(f"Cross-validation ({args.cv} folds, grouped by date)...")
        run_cv(df, features, args.cv)

    # Train
    logger.info("\n" + "=" * 65)
    logger.info("Training Random Forest...")
    model = train_rf(X_train, y_train)
    logger.info(f"  Trees: {model.n_estimators}, Depth: {model.max_depth}, "
                f"MinLeaf: {model.min_samples_leaf}")

    # Evaluate
    y_prob, metrics = evaluate_model(model, X_test, y_test, features, test_df, args.threshold)
    metrics["test_samples"] = int(len(y_test))
    metrics["train_samples"] = int(len(y_train))
    metrics["test_win_rate"] = float(y_test.mean())
    metrics["threshold"] = args.threshold

    # Save
    save_model(model, features, metrics, output_path)
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
