#!/usr/bin/env python3
"""
Feature selection for v8 model using Boruta-SHAP approach.

1. Loads full feature dataset (v4 parquet)
2. Runs iterative importance-based selection with XGBoost
3. Stability check across 5 non-overlapping time windows
4. Outputs selected feature list to JSON

Usage:
    python -m scripts.feature_selection_v8
    python -m scripts.feature_selection_v8 --features-file ml/features/v8/data/training_features_v4.parquet
    python -m scripts.feature_selection_v8 --target short_mid_tp25
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_selection")

OUTPUT_DIR = Path("ml/features/v8/data")
LABEL_FILE = OUTPUT_DIR / "short_ic_labels_v3.parquet"


def load_data(features_file: str, target: str) -> tuple:
    """Load features and labels, return aligned X, y, dates."""
    logger.info(f"Loading features from {features_file}...")
    features_df = pd.read_parquet(features_file)
    logger.info(f"  Features: {len(features_df):,} rows × {len(features_df.columns)} columns")

    logger.info(f"Loading labels from {LABEL_FILE}...")
    labels_df = pd.read_parquet(LABEL_FILE)
    logger.info(f"  Labels: {len(labels_df):,} rows × {len(labels_df.columns)} columns")

    if target not in labels_df.columns:
        available = [c for c in labels_df.columns if c not in ('date', 'entry_time')]
        logger.error(f"Target '{target}' not found. Available: {available}")
        sys.exit(1)

    # Align on date + entry_time
    merged = features_df.merge(
        labels_df[['date', 'entry_time', target]],
        on=['date', 'entry_time'],
        how='inner',
    )
    logger.info(f"  Merged: {len(merged):,} rows")

    # Drop metadata columns
    meta_cols = ['date', 'entry_time']
    feature_cols = [c for c in merged.columns if c not in meta_cols + [target]]

    X = merged[feature_cols].copy()
    y = merged[target].copy()
    dates = merged['date'].copy()

    # Drop rows with NaN target
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    dates = dates[valid].reset_index(drop=True)

    logger.info(f"  Valid samples: {len(X):,} | Features: {len(feature_cols)}")
    logger.info(f"  Target '{target}': {y.mean():.3f} positive rate")

    return X, y, dates, feature_cols


def boruta_shap_selection(X: pd.DataFrame, y: pd.Series, n_iterations: int = 20) -> list:
    """
    Simplified Boruta-SHAP: compare real features against shadow (shuffled) features.

    Features that consistently beat their shadows are selected.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost required. Install with: pip install xgboost")
        sys.exit(1)

    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    # Track how many times each feature beats shadow max
    hit_counts = np.zeros(n_features)

    for iteration in range(n_iterations):
        # Create shadow features (shuffled copies)
        X_shadow = X.sample(frac=1.0).reset_index(drop=True)
        X_shadow.columns = [f"shadow_{c}" for c in X_shadow.columns]

        X_combined = pd.concat([X.reset_index(drop=True), X_shadow], axis=1)

        # Fill NaN with median for XGBoost
        X_filled = X_combined.fillna(X_combined.median())

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=iteration,
            eval_metric='logloss',
            verbosity=0,
        )

        # Use subset for speed
        sample_size = min(50000, len(X_filled))
        idx = np.random.RandomState(iteration).choice(len(X_filled), sample_size, replace=False)
        model.fit(X_filled.iloc[idx], y.iloc[idx])

        # Get feature importances
        importances = model.feature_importances_
        real_importances = importances[:n_features]
        shadow_importances = importances[n_features:]
        shadow_max = np.max(shadow_importances)

        # Count hits
        hits = real_importances > shadow_max
        hit_counts += hits

        if (iteration + 1) % 5 == 0:
            confirmed = np.sum(hit_counts > iteration * 0.6)
            logger.info(f"  Iteration {iteration + 1}/{n_iterations}: {confirmed} features confirmed")

    # Select features that beat shadow in >60% of iterations
    threshold = n_iterations * 0.6
    selected_mask = hit_counts >= threshold
    selected = [f for f, s in zip(feature_names, selected_mask) if s]

    logger.info(f"Boruta-SHAP selected {len(selected)}/{n_features} features")
    return selected


def stability_check(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series,
    selected_features: list, n_windows: int = 5,
) -> list:
    """
    Run feature selection on n non-overlapping time windows.
    Keep features selected in majority (n-1) of windows.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return selected_features

    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    window_size = n_dates // n_windows

    feature_window_counts = {f: 0 for f in selected_features}

    for w in range(n_windows):
        start_idx = w * window_size
        end_idx = min((w + 1) * window_size, n_dates)
        window_dates = set(unique_dates[start_idx:end_idx])

        mask = dates.isin(window_dates)
        X_w = X[mask][selected_features].reset_index(drop=True)
        y_w = y[mask].reset_index(drop=True)

        if len(X_w) < 100:
            continue

        X_filled = X_w.fillna(X_w.median())

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.6,
            random_state=w, eval_metric='logloss', verbosity=0,
        )
        model.fit(X_filled, y_w)

        importances = dict(zip(selected_features, model.feature_importances_))
        median_imp = np.median(list(importances.values()))

        for f in selected_features:
            if importances.get(f, 0) > median_imp:
                feature_window_counts[f] += 1

    # Keep features selected in n_windows - 1 or more
    min_windows = max(n_windows - 1, 3)
    stable = [f for f, count in feature_window_counts.items() if count >= min_windows]

    logger.info(f"Stability check: {len(stable)}/{len(selected_features)} stable across {min_windows}/{n_windows} windows")
    return stable


def main():
    parser = argparse.ArgumentParser(description="V8 Feature Selection")
    parser.add_argument("--features-file", type=str,
                        default="ml/features/v8/data/training_features_v4.parquet")
    parser.add_argument("--target", type=str, default="short_mid_tp25",
                        help="Label column to select features for")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Boruta-SHAP iterations")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability check")
    args = parser.parse_args()

    X, y, dates, all_features = load_data(args.features_file, args.target)

    # Step 1: Boruta-SHAP
    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Boruta-SHAP feature selection")
    logger.info(f"{'='*60}")
    selected = boruta_shap_selection(X, y, n_iterations=args.iterations)

    # Step 2: Stability check
    if not args.skip_stability and len(selected) > 0:
        logger.info(f"\n{'='*60}")
        logger.info("Step 2: Stability check across time windows")
        logger.info(f"{'='*60}")
        stable = stability_check(X, y, dates, selected)
    else:
        stable = selected

    # Save results
    output = {
        "target": args.target,
        "n_total_features": len(all_features),
        "n_boruta_selected": len(selected),
        "n_stable": len(stable),
        "boruta_selected": sorted(selected),
        "stable_selected": sorted(stable),
    }

    output_file = OUTPUT_DIR / "selected_features_v4.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSaved: {output_file}")
    logger.info(f"  Boruta selected: {len(selected)} features")
    logger.info(f"  Stability filtered: {len(stable)} features")
    logger.info(f"\nTop stable features:")
    for f in sorted(stable)[:30]:
        logger.info(f"  {f}")


if __name__ == "__main__":
    main()
