#!/usr/bin/env python3
"""
V8 Model Training — XGBoost on 169-feature v2 backfill data.

Trains short IC and long IC models using time-series aware splits.
Supports Optuna hyperparameter tuning.

Usage:
    python -m scripts.train_v8_models
    python -m scripts.train_v8_models --tune --n-trials 50
    python -m scripts.train_v8_models --target short   # short IC only
    python -m scripts.train_v8_models --target long    # long IC only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ml" / "features" / "v8" / "data"
MODEL_DIR = BASE_DIR / "ml" / "artifacts" / "models" / "v8"

FEATURES_V3_PATH = DATA_DIR / "training_features_v3.parquet"
FEATURES_V4_PATH = DATA_DIR / "training_features_v4.parquet"
SELECTED_FEATURES_PATH = DATA_DIR / "selected_features_v4.json"
LABELS_V3_PATH = DATA_DIR / "labels_v3.parquet"
# Legacy separate label files (fallback)
SHORT_LABELS_PATH = DATA_DIR / "short_ic_labels_v2.parquet"
LONG_LABELS_PATH = DATA_DIR / "long_ic_labels_v2.parquet"


# ---------------------------------------------------------------------------
# Default XGBoost params
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "auc",
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(use_v4: bool = False):
    """Load features and v3 unified labels."""
    print("Loading data...")
    features_path = FEATURES_V4_PATH if (use_v4 and FEATURES_V4_PATH.exists()) else FEATURES_V3_PATH
    feat = pd.read_parquet(features_path)
    labels = pd.read_parquet(LABELS_V3_PATH)

    print(f"  Features: {feat.shape} (from {features_path.name})")
    print(f"  Labels (v3): {labels.shape}")

    return feat, labels


def select_features(feat: pd.DataFrame, use_selected: bool = False) -> list[str]:
    """Select good features (non-constant, <50% NaN).

    If use_selected=True and selected_features_v4.json exists, use the
    Boruta-SHAP selected feature list instead of heuristic filtering.
    """
    # Try loading pre-selected features from Boruta-SHAP
    if use_selected and SELECTED_FEATURES_PATH.exists():
        with open(SELECTED_FEATURES_PATH) as f:
            selected = json.load(f)
        feature_list = selected.get("stable_selected") or selected.get("boruta_selected", [])
        # Filter to features actually present in the dataframe
        available = [c for c in feature_list if c in feat.columns]
        print(f"  Using Boruta-SHAP selected features: {len(available)}/{len(feature_list)}")
        return available

    # Heuristic selection: non-constant, <50% NaN
    meta_cols = {"date", "entry_time"}
    feat_cols = [c for c in feat.columns if c not in meta_cols]

    good = []
    for c in feat_cols:
        pct_nan = feat[c].isna().mean()
        n_unique = feat[c].nunique(dropna=True)
        if n_unique > 1 and pct_nan <= 0.5:
            good.append(c)

    print(f"  Selected {len(good)}/{len(feat_cols)} features")
    return good


def time_series_split(dates, test_frac=0.15, val_frac=0.10):
    """
    Split dates chronologically: train | val | test.
    No shuffling — prevents future leakage.
    """
    sorted_dates = sorted(dates.unique())
    n = len(sorted_dates)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))

    train_dates = set(sorted_dates[:val_start])
    val_dates = set(sorted_dates[val_start:test_start])
    test_dates = set(sorted_dates[test_start:])

    print(f"  Train: {len(train_dates)} days ({min(train_dates)} to {max(train_dates)})")
    print(f"  Val:   {len(val_dates)} days ({min(val_dates)} to {max(val_dates)})")
    print(f"  Test:  {len(test_dates)} days ({min(test_dates)} to {max(test_dates)})")

    return train_dates, val_dates, test_dates


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost with early stopping on validation set."""
    params = params or DEFAULT_PARAMS.copy()
    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_model(model, X, y, label="Test"):
    """Evaluate and print metrics."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)

    print(f"\n  {label} AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"  {label} base rate: {y.mean():.4f}")

    # Threshold analysis
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        t_preds = (probs >= thresh).astype(int)
        if t_preds.sum() > 0:
            tp = ((t_preds == 1) & (y == 1)).sum()
            fp = ((t_preds == 1) & (y == 0)).sum()
            fn = ((t_preds == 0) & (y == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            coverage = t_preds.sum() / len(y)
            print(f"  @{thresh:.1f}: prec={precision:.3f} rec={recall:.3f} cov={coverage:.3f} n={t_preds.sum()}")

    return {"auc": auc, "ap": ap, "base_rate": float(y.mean())}


def get_feature_importance(model, feature_cols, top_n=25):
    """Get top N important features."""
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top {top_n} features:")
    for _, row in imp.head(top_n).iterrows():
        print(f"    {row['importance']:.4f}  {row['feature']}")

    return imp


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------
def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune XGBoost hyperparameters with Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  optuna not installed, using defaults")
        return DEFAULT_PARAMS.copy()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
            "tree_method": "hist",
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best = study.best_params.copy()
    best.update({"random_state": 42, "n_jobs": -1, "eval_metric": "auc", "tree_method": "hist"})
    return best


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_target(
    feat, labels, feature_cols,
    target_col, target_name,
    tune=False, n_trials=50,
):
    """Train a single target model (e.g., short IC TP25)."""
    print(f"\n{'='*70}")
    print(f"  Target: {target_name} ({target_col})")
    print(f"{'='*70}")

    # Merge features + labels
    merged = feat[["date", "entry_time"] + feature_cols].merge(
        labels[["date", "entry_time", target_col]],
        on=["date", "entry_time"],
    )

    # Drop NaN labels
    merged = merged.dropna(subset=[target_col])
    print(f"  Samples: {len(merged)} (after dropping NaN labels)")
    print(f"  Base rate: {merged[target_col].mean():.4f}")

    # Replace inf with NaN in features, then fill with column median
    X = merged[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Time-series split
    train_dates, val_dates, test_dates = time_series_split(merged["date"])

    train_mask = merged["date"].isin(train_dates)
    val_mask = merged["date"].isin(val_dates)
    test_mask = merged["date"].isin(test_dates)

    X_train, y_train = X[train_mask], merged.loc[train_mask, target_col]
    X_val, y_val = X[val_mask], merged.loc[val_mask, target_col]
    X_test, y_test = X[test_mask], merged.loc[test_mask, target_col]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Tune or use defaults
    if tune:
        print(f"\n  Tuning hyperparameters ({n_trials} trials)...")
        params = tune_hyperparams(X_train, y_train, X_val, y_val, n_trials)
    else:
        params = DEFAULT_PARAMS.copy()

    # Train final model
    print("\n  Training final model...")
    t0 = time.time()
    model = train_model(X_train, y_train, X_val, y_val, params)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Val")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Feature importance
    importance = get_feature_importance(model, feature_cols)

    return model, params, {
        "target": target_name,
        "target_col": target_col,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "params": params,
        "n_features": len(feature_cols),
        "train_time": train_time,
        "importance": importance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--target", choices=["short", "long", "long_nosl", "both"], default="both")
    parser.add_argument("--v4", action="store_true", help="Use v4 augmented features")
    parser.add_argument("--selected", action="store_true", help="Use Boruta-SHAP selected features")
    args = parser.parse_args()

    print("=" * 70)
    print("  V8 Model Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    feat, labels = load_data(use_v4=args.v4)
    feature_cols = select_features(feat, use_selected=args.selected)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Define all targets using v3 unified labels
    # Format: (target_col in labels_v3, model_name)
    targets = []
    if args.target in ("short", "both"):
        targets += [
            ("short_mid_tp25", "short_ic_d10_tp25"),
            ("short_mid_tp50", "short_ic_d10_tp50"),
            # big_loss = SL hit (IC value >= 2x credit)
            ("short_mid_pnl_3pm_raw", "short_ic_d10_big_loss"),
        ]
    if args.target in ("long", "long_nosl", "both"):
        targets += [
            ("long_mid_4pm_profitable_raw", "long_ic_d10_profitable"),
        ]
    if args.target in ("long_nosl", "both"):
        targets += [
            ("long_nosl_mid_tp25", "long_nosl_d10_tp25"),
            ("long_nosl_mid_tp50", "long_nosl_d10_tp50"),
        ]

    for target_col, model_name in targets:
        # For big_loss, derive binary label: P&L < -1.0 (lost more than 100% of credit)
        if model_name == "short_ic_d10_big_loss":
            labels_for_target = labels.copy()
            labels_for_target[target_col] = (labels_for_target["short_mid_pnl_3pm_raw"] < -1.0).astype(float)
        else:
            labels_for_target = labels

        model, params, metrics = train_target(
            feat, labels_for_target, feature_cols,
            target_col, model_name,
            tune=args.tune, n_trials=args.n_trials,
        )

        # Save model
        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"\n  Saved: {model_path}")

        # Save feature list
        meta = {
            "model_name": model_name,
            "features": feature_cols,
            "params": params,
            "metrics": {
                "train_auc": metrics["train"]["auc"],
                "val_auc": metrics["val"]["auc"],
                "test_auc": metrics["test"]["auc"],
                "test_ap": metrics["test"]["ap"],
                "base_rate": metrics["test"]["base_rate"],
            },
            "trained_at": datetime.now().isoformat(),
            "n_samples": {
                "total": len(feat),
            },
        }
        meta_path = MODEL_DIR / f"{model_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        results[model_name] = metrics

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, m in results.items():
        print(f"  {name}:")
        print(f"    Train AUC: {m['train']['auc']:.4f}")
        print(f"    Val AUC:   {m['val']['auc']:.4f}")
        print(f"    Test AUC:  {m['test']['auc']:.4f}")
        print(f"    Base rate: {m['test']['base_rate']:.4f}")
        print()


if __name__ == "__main__":
    main()
