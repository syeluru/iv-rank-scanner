#!/usr/bin/env python3
"""
Train V8 Long IC Model — XGBoost with 200+ features, 6-window ensemble.

Predicts:
  - P(profitable): probability that a long IC would have been profitable
  - P(big_move): probability that SPX moved enough to breach a short strike

Usage:
    ./venv/bin/python3 scripts/train_v8_long_ic.py
    ./venv/bin/python3 scripts/train_v8_long_ic.py --delta 10
    ./venv/bin/python3 scripts/train_v8_long_ic.py --delta 10 --target profitable
    ./venv/bin/python3 scripts/train_v8_long_ic.py --ensemble --model xgboost
    ./venv/bin/python3 scripts/train_v8_long_ic.py --tune --n-trials 50
    ./venv/bin/python3 scripts/train_v8_long_ic.py --ensemble --tune --n-trials 100
"""

import argparse
import gc
import json
import sys
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Try XGBoost first, fall back to sklearn GBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'

FEATURES_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'training_features.parquet'
LABELS_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'long_ic_labels.parquet'

DEFAULT_OUTPUT_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v8' / 'long_ic'

TARGETS = ['tp10', 'tp25', 'tp50']

# Target column mapping — maps CLI target name to label parquet column
TARGET_COL_MAP = {
    'tp10': 'hit_tp10',
    'tp25': 'hit_tp25',
    'tp50': 'hit_tp50',
}

# Ensemble windows: name -> trading days
WINDOWS = {
    '3y': 756,
    '2y': 504,
    '1y': 252,
    '6m': 126,
    '3m': 63,
    '1m': 21,
}

WEIGHTS = {
    '3y': 0.20,
    '2y': 0.20,
    '1y': 0.20,
    '6m': 0.15,
    '3m': 0.15,
    '1m': 0.10,
}

# Temporal embargo: 5 trading days gap between train/test splits
# to prevent pattern leakage across split boundaries.
EMBARGO_DAYS = 5

# XGBoost hyperparameters for Long IC — deeper trees, lower LR, more estimators
# Long IC profits are rarer events requiring more capacity to capture tail patterns.
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}

# LightGBM equivalent params
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# sklearn GBM fallback params
SKLEARN_GBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'min_samples_leaf': 5,
    'random_state': 42,
}


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(delta: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and labels, join on date+entry_time, filter to delta."""
    print(f"\nLoading features from {FEATURES_PATH}")
    features_df = pd.read_parquet(FEATURES_PATH)
    print(f"  Features shape: {features_df.shape}")

    print(f"Loading labels from {LABELS_PATH}")
    labels_df = pd.read_parquet(LABELS_PATH)
    print(f"  Labels shape: {labels_df.shape}")

    # Rename delta-prefixed label columns to generic names
    # e.g., d10_long_profitable -> long_profitable, d10_pnl_pct -> pnl_pct
    prefix = f"d{delta}_"
    rename_map = {}
    cols_to_keep = ['date', 'entry_time']
    for col in labels_df.columns:
        if col.startswith(prefix):
            new_name = col[len(prefix):]
            rename_map[col] = new_name
            cols_to_keep.append(col)
    labels_df = labels_df[cols_to_keep].rename(columns=rename_map)

    # Derive take-profit target columns from pnl_pct (preserve NaN)
    # Long IC pnl_pct measures profit as fraction of debit paid
    if 'pnl_pct' in labels_df.columns:
        for tp_name, tp_threshold in [('hit_tp10', 0.10), ('hit_tp25', 0.25), ('hit_tp50', 0.50)]:
            labels_df[tp_name] = labels_df['pnl_pct'].apply(
                lambda x, t=tp_threshold: float(x >= t) if pd.notna(x) else float('nan')
            )

    print(f"  Delta {delta} label columns: {[c for c in labels_df.columns if c not in ('date', 'entry_time')]}")
    for tp in ['hit_tp10', 'hit_tp25', 'hit_tp50']:
        if tp in labels_df.columns:
            print(f"  {tp} rate: {labels_df[tp].mean():.1%}")

    # Identify merge keys
    merge_keys = []
    date_col = 'date'
    if date_col in features_df.columns and date_col in labels_df.columns:
        merge_keys.append(date_col)
    if 'entry_time' in features_df.columns and 'entry_time' in labels_df.columns:
        merge_keys.append('entry_time')

    if not merge_keys:
        raise ValueError(
            "Cannot find common merge columns between features and labels. "
            f"Feature cols: {list(features_df.columns[:5])}, "
            f"Label cols: {list(labels_df.columns[:5])}"
        )

    # Merge
    df = features_df.merge(labels_df, on=merge_keys, how='inner')
    print(f"  Merged shape: {df.shape}")

    # Free memory from source dataframes
    del features_df, labels_df
    gc.collect()

    # Ensure date column is sortable
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values([date_col] + (['entry_time'] if 'entry_time' in df.columns else [])).reset_index(drop=True)

    return df, date_col


def preprocess(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """Drop NaN labels, fill NaN features with -999, remove constant features."""
    mask = df[target_col].notna()
    df = df[mask].copy()

    # Fill NaN features with -999 (XGBoost handles missing natively)
    df[feature_cols] = df[feature_cols].fillna(-999)

    # Replace inf
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], -999)

    # Remove constant features (zero variance)
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant features: {constant_cols[:5]}...")
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    return df, feature_cols


def get_feature_columns(df: pd.DataFrame, date_col: str, label_cols: list[str]) -> list[str]:
    """Identify feature columns by exclusion."""
    exclude = {date_col, 'entry_time', 'delta'} | set(label_cols)
    # Also exclude any raw label columns that came from the labels merge
    label_keywords = ('credit', 'pnl', 'pnl_pct', 'long_profitable', 'profitable', 'big_move')
    for col in df.columns:
        if col in label_keywords or any(col.endswith(f'_{s}') for s in label_keywords):
            exclude.add(col)
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    return feature_cols


def split_holdout(df: pd.DataFrame, date_col: str, holdout_months: int = 3):
    """Hold out the last N months as final test set, with EMBARGO_DAYS gap."""
    max_date = df[date_col].max()
    cutoff = max_date - pd.DateOffset(months=holdout_months)

    test_df = df[df[date_col] > cutoff].copy()

    # Find the first test date and remove EMBARGO_DAYS trading days before it
    first_test_date = test_df[date_col].min()
    all_train_dates = sorted(df[df[date_col] <= cutoff][date_col].dt.date.unique())

    if len(all_train_dates) > EMBARGO_DAYS:
        embargo_cutoff_date = all_train_dates[-EMBARGO_DAYS]
        train_df = df[df[date_col].dt.date < embargo_cutoff_date].copy()
        print(f"  Embargo: removed {EMBARGO_DAYS} trading days before test start "
              f"({embargo_cutoff_date} to {all_train_dates[-1]})")
    else:
        train_df = df[df[date_col] <= cutoff].copy()

    print(f"  Holdout split: {len(train_df)} train, {len(test_df)} test")
    print(f"  Train: {train_df[date_col].min().date()} to {train_df[date_col].max().date()}")
    print(f"  Test:  {test_df[date_col].min().date()} to {test_df[date_col].max().date()}")
    return train_df, test_df


def get_window_data(df: pd.DataFrame, date_col: str, lookback_days: int) -> pd.DataFrame:
    """Slice to the most recent N trading days."""
    all_dates = sorted(df[date_col].dt.date.unique())
    if lookback_days >= len(all_dates):
        print(f"    Window {lookback_days}d: using all {len(all_dates)} available days")
        return df
    window_dates = set(all_dates[-lookback_days:])
    filtered = df[df[date_col].dt.date.isin(window_dates)]
    print(f"    Window {lookback_days}d: {len(filtered)} rows "
          f"({min(window_dates)} to {max(window_dates)})")
    return filtered


def _calc_embargo_samples(n_samples: int, n_unique_days: int) -> int:
    """Calculate embargo gap in samples for TimeSeriesSplit."""
    if n_unique_days == 0:
        return 0
    avg_samples_per_day = n_samples / n_unique_days
    return int(EMBARGO_DAYS * avg_samples_per_day)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(model_type: str, scale_pos_weight: float, params_override: dict | None = None):
    """Build a classifier based on the requested model type."""
    if model_type == 'xgboost':
        if not HAS_XGBOOST:
            print("  WARNING: xgboost not installed, falling back to sklearn GBM")
            return _build_sklearn_gbm()
        params = XGBOOST_PARAMS.copy()
        if params_override:
            params.update(params_override)
        params['scale_pos_weight'] = scale_pos_weight
        return xgb.XGBClassifier(**params)

    elif model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            print("  WARNING: lightgbm not installed, falling back to sklearn GBM")
            return _build_sklearn_gbm()
        params = LIGHTGBM_PARAMS.copy()
        if params_override:
            params.update(params_override)
        params['scale_pos_weight'] = scale_pos_weight
        return lgb.LGBMClassifier(**params)

    else:  # sklearn
        return _build_sklearn_gbm()


def _build_sklearn_gbm():
    """Build sklearn GradientBoostingClassifier as fallback."""
    return GradientBoostingClassifier(**SKLEARN_GBM_PARAMS)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight for imbalanced classes."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_single_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    model_type: str,
    scale_pos_weight: float,
    early_stopping: bool = True,
    params_override: dict | None = None,
) -> tuple:
    """Train a single model with optional early stopping."""
    model = build_model(model_type, scale_pos_weight, params_override)

    # Convert to numpy arrays for XGBoost efficiency
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

    start = time.time()

    if model_type == 'xgboost' and HAS_XGBOOST and early_stopping:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False,
        )
    elif model_type == 'lightgbm' and HAS_LIGHTGBM and early_stopping:
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train_np, y_train_np)

    elapsed = time.time() - start
    return model, elapsed


def evaluate_model(model, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict:
    """Compute evaluation metrics."""
    X_np = X.values if hasattr(X, 'values') else X
    y_np = y.values if hasattr(y, 'values') else y

    probs = model.predict_proba(X_np)[:, 1]
    metrics = {}

    try:
        metrics['auc'] = float(roc_auc_score(y_np, probs))
    except ValueError:
        metrics['auc'] = 0.5

    metrics['brier'] = float(brier_score_loss(y_np, probs))

    # Optimal threshold via F1
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(y_np, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = thr

    preds_opt = (probs >= best_thr).astype(int)
    metrics['optimal_threshold'] = float(best_thr)
    metrics['precision'] = float(precision_score(y_np, preds_opt, zero_division=0))
    metrics['recall'] = float(recall_score(y_np, preds_opt, zero_division=0))
    metrics['f1'] = float(best_f1)

    return metrics


def calibration_data(model, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, n_bins: int = 10) -> list[dict]:
    """Compute calibration/reliability diagram data."""
    X_np = X.values if hasattr(X, 'values') else X
    y_np = y.values if hasattr(y, 'values') else y

    probs = model.predict_proba(X_np)[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    cal = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        cal.append({
            'bin_lower': float(bins[i]),
            'bin_upper': float(bins[i + 1]),
            'mean_predicted': float(probs[mask].mean()),
            'mean_actual': float(y_np[mask].mean()),
            'count': int(mask.sum()),
        })
    return cal


# ---------------------------------------------------------------------------
# Optuna Bayesian hyperparameter tuning
# ---------------------------------------------------------------------------

def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    scale_pos_weight: float,
    n_trials: int = 50,
) -> dict:
    """Run Bayesian hyperparameter tuning with Optuna. Returns best params dict."""
    if not HAS_OPTUNA:
        raise RuntimeError("optuna is not installed. Install with: pip install optuna")
    if model_type not in ('xgboost', 'lightgbm'):
        print("  Optuna tuning only supported for xgboost/lightgbm — skipping")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute embargo samples for CV gap
    n_unique_days = len(pd.Series(X_train.index).unique()) if hasattr(X_train, 'index') else len(X_train)
    # Estimate from the training data date count if available
    embargo_samples = _calc_embargo_samples(len(X_train), max(1, len(X_train) // 5))

    # Convert to numpy once for speed
    X_np = X_train.values
    y_np = y_train.values

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        tscv = TimeSeriesSplit(n_splits=5, gap=embargo_samples)
        fold_aucs = []

        for tr_idx, va_idx in tscv.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            y_tr, y_va = y_np[tr_idx], y_np[va_idx]

            spw = float((y_tr == 0).sum()) / max(1, float((y_tr == 1).sum()))

            if model_type == 'xgboost':
                full_params = XGBOOST_PARAMS.copy()
                full_params.update(params)
                full_params['scale_pos_weight'] = spw
                m = xgb.XGBClassifier(**full_params)
                m.set_params(early_stopping_rounds=30)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:  # lightgbm
                full_params = LIGHTGBM_PARAMS.copy()
                full_params.update(params)
                full_params['scale_pos_weight'] = spw
                m = lgb.LGBMClassifier(**full_params)
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
                )

            try:
                auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)

        return float(np.mean(fold_aucs))

    print(f"\n  Running Optuna Bayesian tuning ({n_trials} trials, 5-fold TSCV with gap={embargo_samples})...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  Best trial AUC: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best, indent=4)}")

    return best


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def ensemble_predict_batch(X: pd.DataFrame | np.ndarray, models: dict, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Batch ensemble prediction: weighted average + consensus count."""
    X_np = X.values if hasattr(X, 'values') else X
    all_probs = {}
    for window, model in models.items():
        all_probs[window] = model.predict_proba(X_np)[:, 1]

    # Weighted average
    ensemble_probs = np.zeros(len(X_np))
    for w, weight in weights.items():
        if w in all_probs:
            ensemble_probs += all_probs[w] * weight

    # Consensus: how many models predict >= 0.5
    consensus = np.zeros(len(X_np), dtype=int)
    for probs in all_probs.values():
        consensus += (probs >= 0.5).astype(int)

    return ensemble_probs, consensus


def evaluate_ensemble(X_test, y_test, models, weights, target_name) -> dict:
    """Evaluate ensemble on test set."""
    print(f"\n  Evaluating ensemble for {target_name}...")
    y_np = y_test.values if hasattr(y_test, 'values') else y_test
    ens_probs, consensus = ensemble_predict_batch(X_test, models, weights)

    try:
        ens_auc = float(roc_auc_score(y_np, ens_probs))
    except ValueError:
        ens_auc = 0.5

    ens_brier = float(brier_score_loss(y_np, ens_probs))

    # Consensus distribution
    consensus_dist = {}
    for v in range(len(weights) + 1):
        count = int((consensus == v).sum())
        if count > 0:
            consensus_dist[str(v)] = count

    # Strong consensus: majority agree
    majority = len(weights) // 2 + 1
    strong_mask = consensus >= majority
    strong_auc = None
    strong_pct = float(strong_mask.sum() / len(y_np) * 100)
    if strong_mask.sum() > 10:
        try:
            strong_auc = float(roc_auc_score(y_np[strong_mask], ens_probs[strong_mask]))
        except ValueError:
            pass

    return {
        'ensemble_auc': ens_auc,
        'ensemble_brier': ens_brier,
        'consensus_dist': consensus_dist,
        'strong_consensus_auc': strong_auc,
        'strong_consensus_pct': strong_pct,
    }


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X_test: pd.DataFrame, top_n: int = 30) -> list[dict]:
    """Compute SHAP values and return top features.

    # SHAP is used for post-training interpretation only, NOT for feature selection.
    # The model trains on ALL features; SHAP just tells us which ones mattered.
    """
    if not HAS_SHAP:
        print("  SHAP not installed — skipping analysis")
        return []

    print("  Computing SHAP values (this may take a minute)...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values may be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(
            zip(X_test.columns, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )

        top_features = []
        for rank, (feat, val) in enumerate(feature_importance[:top_n], 1):
            top_features.append({
                'rank': rank,
                'feature': feat,
                'mean_abs_shap': float(val),
            })
            print(f"    {rank:2d}. {feat:40s} {val:.6f}")

        return top_features
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    target_name: str,
    target_col: str,
    delta: int,
    model_type: str,
    do_ensemble: bool,
    do_shap: bool,
    do_tune: bool,
    n_trials: int,
    output_dir: Path,
) -> dict:
    """Train models for a single target, optionally as 6-window ensemble."""

    print(f"\n{'=' * 70}")
    print(f"  Target: {target_name.upper()} (column: {target_col}) | delta={delta}")
    print(f"{'=' * 70}")

    X_train_all = train_df[feature_cols]
    y_train_all = train_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    pos_rate = y_train_all.mean()
    scale_pos_weight = compute_scale_pos_weight(y_train_all)
    print(f"  Class balance: {pos_rate:.3f} positive | scale_pos_weight={scale_pos_weight:.2f}")
    print(f"  Train: {len(X_train_all)} | Test: {len(X_test)} | Features: {len(feature_cols)}")

    # Calculate embargo samples for TimeSeriesSplit
    n_unique_days_train = len(X_train_all.index.unique()) if hasattr(X_train_all.index, 'unique') else len(X_train_all)
    # Estimate unique trading days from the date column
    unique_train_days = train_df[date_col].dt.date.nunique()
    embargo_samples = _calc_embargo_samples(len(X_train_all), unique_train_days)
    print(f"  Embargo: {EMBARGO_DAYS} trading days = ~{embargo_samples} samples gap in CV")

    result = {
        'target': target_name,
        'delta': delta,
        'n_train': len(X_train_all),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
        'positive_rate': float(pos_rate),
        'scale_pos_weight': float(scale_pos_weight),
        'embargo_days': EMBARGO_DAYS,
        'embargo_samples': embargo_samples,
        'windows': {},
    }

    # Optuna tuning (run once, reuse best params for all windows)
    best_params = None
    if do_tune:
        best_params = run_optuna_tuning(
            X_train_all, y_train_all, model_type, scale_pos_weight, n_trials=n_trials,
        )
        result['optuna_best_params'] = best_params

    if do_ensemble:
        models = {}
        for window_name, lookback_days in WINDOWS.items():
            print(f"\n  --- Window: {window_name.upper()} ({lookback_days} trading days) ---")
            window_df = get_window_data(train_df, date_col, lookback_days)
            X_w = window_df[feature_cols]
            y_w = window_df[target_col].astype(int)

            if len(X_w) < 50:
                print(f"    Skipping — only {len(X_w)} samples")
                continue

            # Calculate window-specific embargo
            unique_window_days = window_df[date_col].dt.date.nunique()
            window_embargo = _calc_embargo_samples(len(X_w), unique_window_days)

            # TimeSeriesSplit CV with temporal gap
            tscv = TimeSeriesSplit(n_splits=5, gap=window_embargo)
            cv_aucs = []
            model = None
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_w)):
                X_tr, X_va = X_w.iloc[tr_idx], X_w.iloc[va_idx]
                y_tr, y_va = y_w.iloc[tr_idx], y_w.iloc[va_idx]
                spw = compute_scale_pos_weight(y_tr)
                m, _ = train_single_model(
                    X_tr, y_tr, X_va, y_va, model_type, spw,
                    early_stopping=True, params_override=best_params,
                )
                try:
                    auc = roc_auc_score(y_va.values, m.predict_proba(X_va.values)[:, 1])
                except ValueError:
                    auc = 0.5
                cv_aucs.append(auc)
                model = m

            # Retrain on full window data with early stopping against test
            spw = compute_scale_pos_weight(y_w)
            model, elapsed = train_single_model(
                X_w, y_w, X_test, y_test, model_type, spw,
                early_stopping=True, params_override=best_params,
            )

            test_metrics = evaluate_model(model, X_test, y_test)
            train_metrics = evaluate_model(model, X_w, y_w)

            print(f"    CV AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")
            print(f"    Final — Train AUC: {train_metrics['auc']:.4f} | "
                  f"Test AUC: {test_metrics['auc']:.4f} | "
                  f"Brier: {test_metrics['brier']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Save model
            artifact_name = f"v8_long_ic_d{delta}_{target_name}_{window_name}.joblib"
            joblib.dump(model, output_dir / artifact_name)
            print(f"    Saved: {artifact_name}")

            models[window_name] = model
            result['windows'][window_name] = {
                'n_samples': len(X_w),
                'cv_auc_mean': float(np.mean(cv_aucs)),
                'cv_auc_std': float(np.std(cv_aucs)),
                'train_auc': train_metrics['auc'],
                'test_auc': test_metrics['auc'],
                'test_brier': test_metrics['brier'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'optimal_threshold': test_metrics['optimal_threshold'],
                'train_time_s': elapsed,
            }

            # Clean up window intermediates
            del X_w, y_w, window_df
            gc.collect()

        # Ensemble evaluation
        if models:
            active_weights = {w: WEIGHTS[w] for w in models}
            total_w = sum(active_weights.values())
            active_weights = {w: v / total_w for w, v in active_weights.items()}

            ens_metrics = evaluate_ensemble(X_test, y_test, models, active_weights, target_name)
            result['ensemble'] = ens_metrics

            print(f"\n  Ensemble AUC: {ens_metrics['ensemble_auc']:.4f} | "
                  f"Brier: {ens_metrics['ensemble_brier']:.4f}")
            print(f"  Consensus dist: {ens_metrics['consensus_dist']}")
            if ens_metrics['strong_consensus_auc'] is not None:
                print(f"  Strong consensus AUC: {ens_metrics['strong_consensus_auc']:.4f} "
                      f"({ens_metrics['strong_consensus_pct']:.1f}% coverage)")

            # Save ensemble weights
            weights_artifact = f"v8_long_ic_d{delta}_{target_name}_ensemble_weights.json"
            with open(output_dir / weights_artifact, 'w') as f:
                json.dump(active_weights, f, indent=2)

            # Calibration data (from 1y model if available)
            cal_model = models.get('1y', list(models.values())[0])
            cal = calibration_data(cal_model, X_test, y_test)
            result['calibration'] = cal

            # SHAP on 1y model
            # SHAP is used for post-training interpretation only, NOT for feature selection.
            # The model trains on ALL features; SHAP just tells us which ones mattered.
            if do_shap:
                shap_model = models.get('1y', list(models.values())[0])
                shap_features = run_shap_analysis(shap_model, X_test)
                result['shap_top_features'] = shap_features
    else:
        # Single model (no ensemble) — train on full training data
        print("\n  Training single model on full training data...")
        tscv = TimeSeriesSplit(n_splits=5, gap=embargo_samples)
        cv_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_all)):
            X_tr, X_va = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
            y_tr, y_va = y_train_all.iloc[tr_idx], y_train_all.iloc[va_idx]
            spw = compute_scale_pos_weight(y_tr)
            m, _ = train_single_model(
                X_tr, y_tr, X_va, y_va, model_type, spw,
                params_override=best_params,
            )
            try:
                auc = roc_auc_score(y_va.values, m.predict_proba(X_va.values)[:, 1])
            except ValueError:
                auc = 0.5
            cv_aucs.append(auc)

        # Final model
        model, elapsed = train_single_model(
            X_train_all, y_train_all, X_test, y_test,
            model_type, scale_pos_weight, early_stopping=True,
            params_override=best_params,
        )

        test_metrics = evaluate_model(model, X_test, y_test)
        train_metrics = evaluate_model(model, X_train_all, y_train_all)

        print(f"  CV AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")
        print(f"  Train AUC: {train_metrics['auc']:.4f} | Test AUC: {test_metrics['auc']:.4f} | "
              f"Brier: {test_metrics['brier']:.4f} | Time: {elapsed:.1f}s")

        artifact_name = f"v8_long_ic_d{delta}_{target_name}_full.joblib"
        joblib.dump(model, output_dir / artifact_name)
        print(f"  Saved: {artifact_name}")

        result['single_model'] = {
            'cv_auc_mean': float(np.mean(cv_aucs)),
            'cv_auc_std': float(np.std(cv_aucs)),
            'train_auc': train_metrics['auc'],
            'test_auc': test_metrics['auc'],
            'test_brier': test_metrics['brier'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'optimal_threshold': test_metrics['optimal_threshold'],
            'train_time_s': elapsed,
        }

        cal = calibration_data(model, X_test, y_test)
        result['calibration'] = cal

        # SHAP is used for post-training interpretation only, NOT for feature selection.
        # The model trains on ALL features; SHAP just tells us which ones mattered.
        if do_shap:
            shap_features = run_shap_analysis(model, X_test)
            result['shap_top_features'] = shap_features

    # Clean up large arrays
    del X_train_all, y_train_all, X_test, y_test
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(results: dict, delta: int, output_dir: Path):
    """Save training report as JSON."""
    report = {
        'strategy': 'long_ic',
        'version': 'v8',
        'delta': delta,
        'trained_at': datetime.now().isoformat(),
        'targets': results,
    }

    report_path = output_dir / f'training_report_d{delta}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train V8 Long IC XGBoost models')
    parser.add_argument('--delta', type=int, default=10, choices=[10, 15, 20],
                        help='Short strike delta (default: 10)')
    parser.add_argument('--target', type=str, default='all',
                        choices=['tp10', 'tp25', 'tp50', 'all'],
                        help='Which target(s) to train (default: all)')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'sklearn'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Train 6-window ensemble (3y, 2y, 1y, 6m, 3m, 1m)')
    parser.add_argument('--tune', action='store_true',
                        help='Run Optuna Bayesian hyperparameter tuning before training')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP analysis')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter training data to start from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter training data to end at this date (YYYY-MM-DD)')
    parser.add_argument('--holdout-months', type=int, default=3,
                        help='Holdout period in months (default: 3)')
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = TARGETS if args.target == 'all' else [args.target]
    do_shap = not args.no_shap

    print("=" * 70)
    print("V8 Long IC Model Training")
    print("=" * 70)
    print(f"  Delta:    {args.delta}")
    print(f"  Targets:  {targets}")
    print(f"  Model:    {args.model}")
    print(f"  Ensemble: {args.ensemble}")
    print(f"  Tune:     {args.tune} (trials={args.n_trials}, optuna={'available' if HAS_OPTUNA else 'NOT installed'})")
    print(f"  Embargo:  {EMBARGO_DAYS} trading days between train/test splits")
    print(f"  SHAP:     {do_shap} (available={HAS_SHAP})")
    print(f"  Date:     {args.start_date or 'all'} to {args.end_date or 'all'}")
    print(f"  Output:   {output_dir}")
    print(f"  Note:     Using deeper trees (max_depth=8) and lower LR (0.03)")
    print(f"            for capturing rare tail-event patterns")

    if args.tune and not HAS_OPTUNA:
        print("\n  ERROR: --tune requires optuna. Install with: pip install optuna")
        sys.exit(1)

    total_start = time.time()

    # Load and prepare data
    df, date_col = load_data(args.delta)

    # Apply date filtering if specified
    if args.start_date:
        start_dt = pd.Timestamp(args.start_date)
        before = len(df)
        df = df[df[date_col] >= start_dt].copy()
        print(f"  Date filter (start={args.start_date}): {before} → {len(df)} rows")
    if args.end_date:
        end_dt = pd.Timestamp(args.end_date)
        before = len(df)
        df = df[df[date_col] <= end_dt].copy()
        print(f"  Date filter (end={args.end_date}): {before} → {len(df)} rows")

    if len(df) < 50:
        print(f"\n  ERROR: Only {len(df)} rows after date filtering — need at least 50. Skipping.")
        sys.exit(0)

    # Determine label columns present
    label_cols = [v for v in TARGET_COL_MAP.values() if v in df.columns]
    feature_cols = get_feature_columns(df, date_col, label_cols)
    print(f"\nIdentified {len(feature_cols)} feature columns")

    # Holdout split
    train_df, test_df = split_holdout(df, date_col, holdout_months=args.holdout_months)

    # Free the full dataframe
    del df
    gc.collect()

    # Preprocess
    train_df, feature_cols = preprocess(train_df, label_cols[0], feature_cols)
    test_df, _ = preprocess(test_df, label_cols[0], feature_cols)

    print(f"  Final feature count: {len(feature_cols)}")

    # Train each target
    all_results = {}
    for target_name in targets:
        target_col = TARGET_COL_MAP[target_name]
        if target_col not in train_df.columns:
            print(f"\n  WARNING: Column '{target_col}' not found in labels — skipping {target_name}")
            continue

        result = train_target(
            train_df=train_df,
            test_df=test_df,
            date_col=date_col,
            feature_cols=feature_cols,
            target_name=target_name,
            target_col=target_col,
            delta=args.delta,
            model_type=args.model,
            do_ensemble=args.ensemble,
            do_shap=do_shap,
            do_tune=args.tune,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        all_results[target_name] = result

    # Save report
    save_report(all_results, args.delta, output_dir)

    total_elapsed = time.time() - total_start
    print(f"\nTraining complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
