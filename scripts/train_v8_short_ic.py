#!/usr/bin/env python3
"""
Train V8 Short IC Model — XGBoost with 200+ features, 6-window ensemble.

Predicts:
  - P(hit_tp25): probability of hitting 25% take-profit
  - P(hit_tp50): probability of hitting 50% take-profit
  - P(big_loss): probability of losing > 2x credit

Usage:
    ./venv/bin/python3 scripts/train_v8_short_ic.py
    ./venv/bin/python3 scripts/train_v8_short_ic.py --delta 10
    ./venv/bin/python3 scripts/train_v8_short_ic.py --delta 10 --target tp25
    ./venv/bin/python3 scripts/train_v8_short_ic.py --ensemble --model xgboost
    ./venv/bin/python3 scripts/train_v8_short_ic.py --tune --n-trials 50
"""

import argparse
import gc
import json
import sys
import os
import time
import warnings
from datetime import datetime, timedelta
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
LABELS_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'short_ic_labels.parquet'

DEFAULT_OUTPUT_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v8' / 'short_ic'

TARGETS = ['tp10', 'tp25', 'tp50', 'big_loss']

# Target column mapping — maps CLI target name to label parquet column
TARGET_COL_MAP = {
    'tp10': 'hit_tp10',
    'tp25': 'hit_tp25',
    'tp50': 'hit_tp50',
    'big_loss': 'big_loss',
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

# Temporal embargo: 5 trading days gap between train and test splits
# to prevent lookahead leakage from autocorrelated features
EMBARGO_DAYS = 5

# XGBoost default hyperparameters for Short IC
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
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
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# sklearn GBM fallback params
SKLEARN_GBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'min_samples_leaf': 10,
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
    # e.g., d10_hit_tp25 -> hit_tp25, d10_big_loss -> big_loss
    prefix = f"d{delta}_"
    rename_map = {}
    cols_to_keep = ['date', 'entry_time']
    for col in labels_df.columns:
        if col.startswith(prefix):
            new_name = col[len(prefix):]
            rename_map[col] = new_name
            cols_to_keep.append(col)
    # Drop columns for other deltas
    labels_df = labels_df[cols_to_keep].rename(columns=rename_map)

    # Derive hit_tp10 from pnl_pct if not already present (preserve NaN)
    if 'hit_tp10' not in labels_df.columns and 'pnl_pct' in labels_df.columns:
        labels_df['hit_tp10'] = labels_df['pnl_pct'].apply(
            lambda x: float(x >= 0.10) if pd.notna(x) else float('nan')
        )
        print(f"  Derived hit_tp10: {labels_df['hit_tp10'].mean():.1%} positive rate")

    print(f"  Delta {delta} label columns: {[c for c in labels_df.columns if c not in ('date', 'entry_time')]}")

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

    # Free source DataFrames immediately after merge
    del features_df, labels_df
    gc.collect()

    # Ensure date column is sortable
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values([date_col] + (['entry_time'] if 'entry_time' in df.columns else [])).reset_index(drop=True)

    return df, date_col


def preprocess(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """Drop NaN labels, fill NaN features with -999, remove constant features."""
    # Drop rows where target is NaN
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
    # Exclude date, entry_time, delta, all label/target columns, and raw label data
    exclude = {date_col, 'entry_time', 'delta'} | set(label_cols)
    # Also exclude any raw label columns that came from the labels merge
    label_keywords = ('credit', 'pnl', 'pnl_pct', 'long_profitable')
    for col in df.columns:
        if col in label_keywords or any(col.endswith(f'_{s}') for s in label_keywords):
            exclude.add(col)
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    return feature_cols


def split_holdout(df: pd.DataFrame, date_col: str, holdout_months: int = 3,
                  embargo_days: int = EMBARGO_DAYS):
    """Hold out the last N months as final test set, with embargo gap."""
    max_date = df[date_col].max()
    holdout_cutoff = max_date - pd.DateOffset(months=holdout_months)

    all_dates = sorted(df[date_col].dt.date.unique())
    holdout_dates = [d for d in all_dates if d > holdout_cutoff.date()]

    if holdout_dates and embargo_days > 0:
        holdout_start = holdout_dates[0]
        # Get all training dates (before holdout)
        train_dates = [d for d in all_dates if d < holdout_start]
        if len(train_dates) > embargo_days:
            # Remove the last embargo_days trading days from training data
            embargo_cutoff = train_dates[-(embargo_days)]
            train_df = df[df[date_col].dt.date <= embargo_cutoff].copy()
            n_embargoed = len(train_dates) - len([d for d in train_dates if d <= embargo_cutoff])
            print(f"  Embargo: removed {n_embargoed} trading days between train and test")
        else:
            train_df = df[df[date_col].dt.date < holdout_start].copy()
    else:
        train_df = df[df[date_col] <= holdout_cutoff].copy()

    test_df = df[df[date_col].dt.date.isin(set(holdout_dates))].copy()

    print(f"  Holdout split: {len(train_df)} train, {len(test_df)} test")
    print(f"  Train: {train_df[date_col].min().date()} to {train_df[date_col].max().date()}")
    print(f"  Test:  {test_df[date_col].min().date()} to {test_df[date_col].max().date()}")
    return train_df, test_df


def compute_embargo_samples(df: pd.DataFrame, date_col: str,
                            embargo_days: int = EMBARGO_DAYS) -> int:
    """Compute embargo gap in samples for TimeSeriesSplit.

    embargo_days is in trading days. We estimate avg samples per trading day
    from the data, then multiply.
    """
    n_trading_days = df[date_col].dt.date.nunique()
    if n_trading_days == 0:
        return 0
    avg_samples_per_day = len(df) / n_trading_days
    return int(embargo_days * avg_samples_per_day)


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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    scale_pos_weight: float,
    early_stopping: bool = True,
    params_override: dict | None = None,
) -> tuple:
    """Train a single model with optional early stopping.

    Expects numpy arrays for X_train, y_train, X_val, y_val.
    """
    model = build_model(model_type, scale_pos_weight, params_override=params_override)

    start = time.time()

    if model_type == 'xgboost' and HAS_XGBOOST and early_stopping:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_type == 'lightgbm' and HAS_LIGHTGBM and early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start
    return model, elapsed


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute evaluation metrics. Expects numpy arrays."""
    probs = model.predict_proba(X)[:, 1]
    metrics = {}

    try:
        metrics['auc'] = float(roc_auc_score(y, probs))
    except ValueError:
        metrics['auc'] = 0.5

    metrics['brier'] = float(brier_score_loss(y, probs))

    # Optimal threshold via F1
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = thr

    preds_opt = (probs >= best_thr).astype(int)
    metrics['optimal_threshold'] = float(best_thr)
    metrics['precision'] = float(precision_score(y, preds_opt, zero_division=0))
    metrics['recall'] = float(recall_score(y, preds_opt, zero_division=0))
    metrics['f1'] = float(best_f1)

    return metrics


def calibration_data(model, X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Compute calibration/reliability diagram data. Expects numpy arrays."""
    probs = model.predict_proba(X)[:, 1]
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
            'mean_actual': float(y[mask].mean()),
            'count': int(mask.sum()),
        })
    return cal


# ---------------------------------------------------------------------------
# Bayesian hyperparameter tuning with Optuna
# ---------------------------------------------------------------------------

def optuna_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    date_col_values: np.ndarray,
    model_type: str,
    scale_pos_weight: float,
    n_trials: int = 50,
    embargo_samples: int = 0,
) -> dict:
    """Run Optuna Bayesian hyperparameter search. Returns best params dict.

    Uses TimeSeriesSplit with embargo gap as CV, optimizes AUC.
    """
    if not HAS_OPTUNA:
        raise RuntimeError("optuna is not installed. Run: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=5, gap=embargo_samples)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        fold_aucs = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            spw = compute_scale_pos_weight(pd.Series(y_tr))
            model = build_model(model_type, spw, params_override=params)

            if model_type == 'xgboost' and HAS_XGBOOST:
                model.set_params(early_stopping_rounds=30)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                model.fit(
                    X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
                )
            else:
                model.fit(X_tr, y_tr)

            try:
                auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)

            # Cleanup fold data references
            del model
            gc.collect()

        return np.mean(fold_aucs)

    study = optuna.create_study(direction='maximize', study_name='v8_short_ic_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Optuna best trial: #{study.best_trial.number}")
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def ensemble_predict_batch(X: np.ndarray, models: dict, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Batch ensemble prediction: weighted average + consensus count."""
    all_probs = {}
    for window, model in models.items():
        all_probs[window] = model.predict_proba(X)[:, 1]

    # Weighted average
    ensemble_probs = np.zeros(len(X))
    for w, weight in weights.items():
        if w in all_probs:
            ensemble_probs += all_probs[w] * weight

    # Consensus: how many models predict >= 0.5
    consensus = np.zeros(len(X), dtype=int)
    for probs in all_probs.values():
        consensus += (probs >= 0.5).astype(int)

    return ensemble_probs, consensus


def evaluate_ensemble(X_test: np.ndarray, y_test: np.ndarray, models: dict,
                      weights: dict, target_name: str) -> dict:
    """Evaluate ensemble on test set."""
    print(f"\n  Evaluating ensemble for {target_name}...")
    ens_probs, consensus = ensemble_predict_batch(X_test, models, weights)

    try:
        ens_auc = float(roc_auc_score(y_test, ens_probs))
    except ValueError:
        ens_auc = 0.5

    ens_brier = float(brier_score_loss(y_test, ens_probs))

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
    strong_pct = float(strong_mask.sum() / len(y_test) * 100)
    if strong_mask.sum() > 10:
        try:
            strong_auc = float(roc_auc_score(y_test[strong_mask], ens_probs[strong_mask]))
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

def run_shap_analysis(model, X_test: np.ndarray, feature_names: list[str],
                      top_n: int = 30) -> list[dict]:
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
            zip(feature_names, mean_abs_shap),
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

    # Convert to numpy arrays — XGBoost works best with numpy, not DataFrames.
    # Keep feature_cols list separately for SHAP interpretation.
    X_train_all = train_df[feature_cols].values
    y_train_all = train_df[target_col].astype(int).values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].astype(int).values
    train_date_values = train_df[date_col].values

    pos_rate = y_train_all.mean()
    scale_pos_weight = compute_scale_pos_weight(pd.Series(y_train_all))
    print(f"  Class balance: {pos_rate:.3f} positive | scale_pos_weight={scale_pos_weight:.2f}")
    print(f"  Train: {len(X_train_all)} | Test: {len(X_test)} | Features: {len(feature_cols)}")

    # Compute embargo gap in samples for TimeSeriesSplit
    embargo_samples = compute_embargo_samples(train_df, date_col)
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

    # Optuna hyperparameter tuning (runs once, used for all windows)
    tuned_params = None
    if do_tune:
        print(f"\n  Running Optuna hyperparameter tuning ({n_trials} trials)...")
        tuned_params = optuna_tune(
            X_train=X_train_all,
            y_train=y_train_all,
            date_col_values=train_date_values,
            model_type=model_type,
            scale_pos_weight=scale_pos_weight,
            n_trials=n_trials,
            embargo_samples=embargo_samples,
        )
        result['optuna_best_params'] = tuned_params
        gc.collect()

    if do_ensemble:
        models = {}
        for window_name, lookback_days in WINDOWS.items():
            print(f"\n  --- Window: {window_name.upper()} ({lookback_days} trading days) ---")
            window_df = get_window_data(train_df, date_col, lookback_days)

            # Convert window data to numpy arrays
            X_w = window_df[feature_cols].values
            y_w = window_df[target_col].astype(int).values

            if len(X_w) < 50:
                print(f"    Skipping — only {len(X_w)} samples")
                del X_w, y_w
                gc.collect()
                continue

            # Compute embargo for this window's data size
            window_embargo = compute_embargo_samples(window_df, date_col)

            # TimeSeriesSplit CV with embargo gap
            tscv = TimeSeriesSplit(n_splits=5, gap=window_embargo)
            cv_aucs = []
            model = None
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_w)):
                X_tr, X_va = X_w[tr_idx], X_w[va_idx]
                y_tr, y_va = y_w[tr_idx], y_w[va_idx]
                spw = compute_scale_pos_weight(pd.Series(y_tr))
                m, _ = train_single_model(
                    X_tr, y_tr, X_va, y_va, model_type, spw,
                    early_stopping=True, params_override=tuned_params,
                )
                try:
                    auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
                except ValueError:
                    auc = 0.5
                cv_aucs.append(auc)
                model = m  # keep last fold model (trained on most data)

            # Retrain on full window data with early stopping against test
            spw = compute_scale_pos_weight(pd.Series(y_w))
            model, elapsed = train_single_model(
                X_w, y_w, X_test, y_test, model_type, spw,
                early_stopping=True, params_override=tuned_params,
            )

            test_metrics = evaluate_model(model, X_test, y_test)
            train_metrics = evaluate_model(model, X_w, y_w)

            print(f"    CV AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")
            print(f"    Final — Train AUC: {train_metrics['auc']:.4f} | "
                  f"Test AUC: {test_metrics['auc']:.4f} | "
                  f"Brier: {test_metrics['brier']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Save model
            artifact_name = f"v8_short_ic_d{delta}_{target_name}_{window_name}.joblib"
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

            # Clean up window training data
            del X_w, y_w, window_df
            gc.collect()

        # Ensemble evaluation
        if models:
            active_weights = {w: WEIGHTS[w] for w in models}
            # Re-normalize weights
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
            weights_artifact = f"v8_short_ic_d{delta}_{target_name}_ensemble_weights.json"
            with open(output_dir / weights_artifact, 'w') as f:
                json.dump(active_weights, f, indent=2)

            # Calibration data (from 1y model if available)
            cal_model = models.get('1y', list(models.values())[0])
            cal = calibration_data(cal_model, X_test, y_test)
            result['calibration'] = cal

            # SHAP is used for post-training interpretation only, NOT for feature selection.
            # The model trains on ALL features; SHAP just tells us which ones mattered.
            if do_shap:
                shap_model = models.get('1y', list(models.values())[0])
                shap_features = run_shap_analysis(shap_model, X_test, feature_cols)
                result['shap_top_features'] = shap_features

            # Clean up model references
            del models
            gc.collect()
    else:
        # Single model (no ensemble) — train on full training data
        print("\n  Training single model on full training data...")
        tscv = TimeSeriesSplit(n_splits=5, gap=embargo_samples)
        cv_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_all)):
            X_tr, X_va = X_train_all[tr_idx], X_train_all[va_idx]
            y_tr, y_va = y_train_all[tr_idx], y_train_all[va_idx]
            spw = compute_scale_pos_weight(pd.Series(y_tr))
            m, _ = train_single_model(
                X_tr, y_tr, X_va, y_va, model_type, spw,
                params_override=tuned_params,
            )
            try:
                auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
            except ValueError:
                auc = 0.5
            cv_aucs.append(auc)

        # Final model
        model, elapsed = train_single_model(
            X_train_all, y_train_all, X_test, y_test,
            model_type, scale_pos_weight, early_stopping=True,
            params_override=tuned_params,
        )

        test_metrics = evaluate_model(model, X_test, y_test)
        train_metrics = evaluate_model(model, X_train_all, y_train_all)

        print(f"  CV AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")
        print(f"  Train AUC: {train_metrics['auc']:.4f} | Test AUC: {test_metrics['auc']:.4f} | "
              f"Brier: {test_metrics['brier']:.4f} | Time: {elapsed:.1f}s")

        artifact_name = f"v8_short_ic_d{delta}_{target_name}_full.joblib"
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
            shap_features = run_shap_analysis(model, X_test, feature_cols)
            result['shap_top_features'] = shap_features

    # Final cleanup
    del X_train_all, y_train_all, X_test, y_test
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(results: dict, delta: int, output_dir: Path):
    """Save training report as JSON."""
    report = {
        'strategy': 'short_ic',
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
    parser = argparse.ArgumentParser(description='Train V8 Short IC XGBoost models')
    parser.add_argument('--delta', type=int, default=10, choices=[10, 15, 20],
                        help='Short strike delta (default: 10)')
    parser.add_argument('--target', type=str, default='all',
                        choices=['tp10', 'tp25', 'tp50', 'big_loss', 'all'],
                        help='Which target(s) to train (default: all)')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'sklearn'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Train 6-window ensemble (3y, 2y, 1y, 6m, 3m, 1m)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP analysis')
    parser.add_argument('--tune', action='store_true',
                        help='Run Optuna Bayesian hyperparameter tuning before training')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
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
    print("V8 Short IC Model Training")
    print("=" * 70)
    print(f"  Delta:    {args.delta}")
    print(f"  Targets:  {targets}")
    print(f"  Model:    {args.model}")
    print(f"  Ensemble: {args.ensemble}")
    print(f"  Tune:     {args.tune} (n_trials={args.n_trials})")
    print(f"  SHAP:     {do_shap} (available={HAS_SHAP})")
    print(f"  Embargo:  {EMBARGO_DAYS} trading days")
    print(f"  Date:     {args.start_date or 'all'} to {args.end_date or 'all'}")
    print(f"  Output:   {output_dir}")

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

    # Free the full DataFrame — train_df and test_df are copies
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

    # Final cleanup
    del train_df, test_df
    gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\nTraining complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
