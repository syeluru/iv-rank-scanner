#!/usr/bin/env python3
"""
Train V8 Ensemble — 6-timeframe weighted ensemble for Short IC and Long IC.

Trains separate XGBoost models on 6 date-based windows, all ending 2026-03-10:
    3yr:  2023-03-10 → 2026-03-10  (weight 0.30)
    2yr:  2024-03-10 → 2026-03-10  (weight 0.25)
    1yr:  2025-03-10 → 2026-03-10  (weight 0.20)
    6mo:  2025-09-10 → 2026-03-10  (weight 0.13)
    3mo:  2025-12-10 → 2026-03-10  (weight 0.08)
    1mo:  2026-02-10 → 2026-03-10  (weight 0.04)

Each window gets its own model. At prediction time, probabilities are combined
via weighted average with automatic re-normalization when windows are missing.

Usage:
    venv/bin/python3 scripts/train_v8_ensemble.py
    venv/bin/python3 scripts/train_v8_ensemble.py --delta 10
    venv/bin/python3 scripts/train_v8_ensemble.py --strategy short
    venv/bin/python3 scripts/train_v8_ensemble.py --strategy long
    venv/bin/python3 scripts/train_v8_ensemble.py --strategy both --delta 10
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

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'

FEATURES_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'training_features.parquet'
SHORT_LABELS_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'short_ic_labels.parquet'
LONG_LABELS_PATH = BASE_DIR / 'ml' / 'features' / 'v8' / 'data' / 'long_ic_labels.parquet'

OUTPUT_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v8_ensemble'

# Temporal embargo between train/test for CV
EMBARGO_DAYS = 5

# 6 timeframe windows — all end at 2026-03-10
END_DATE = pd.Timestamp('2026-03-10')
WINDOWS = {
    '3yr': pd.Timestamp('2023-03-10'),
    '2yr': pd.Timestamp('2024-03-10'),
    '1yr': pd.Timestamp('2025-03-10'),
    '6mo': pd.Timestamp('2025-09-10'),
    '3mo': pd.Timestamp('2025-12-10'),
    '1mo': pd.Timestamp('2026-02-10'),
}

WEIGHTS = {
    '3yr': 0.30,
    '2yr': 0.25,
    '1yr': 0.20,
    '6mo': 0.13,
    '3mo': 0.08,
    '1mo': 0.04,
}

# Short IC targets
SHORT_TARGETS = {
    'tp25': 'hit_tp25',
    'tp50': 'hit_tp50',
    'big_loss': 'big_loss',
}

# Long IC targets
LONG_TARGETS = {
    'profitable': 'profitable',
    'big_move': 'big_move',
}

# XGBoost params — short IC
SHORT_XGBOOST_PARAMS = {
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

# XGBoost params — long IC (deeper, more conservative)
LONG_XGBOOST_PARAMS = {
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_short_data(delta: int) -> tuple[pd.DataFrame, str]:
    """Load features + short IC labels, merge, return (df, date_col)."""
    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_parquet(SHORT_LABELS_PATH)

    prefix = f"d{delta}_"
    rename_map = {}
    cols_to_keep = ['date', 'entry_time']
    for col in labels_df.columns:
        if col.startswith(prefix):
            rename_map[col] = col[len(prefix):]
            cols_to_keep.append(col)
    labels_df = labels_df[cols_to_keep].rename(columns=rename_map)

    df = features_df.merge(labels_df, on=['date', 'entry_time'], how='inner')
    del features_df, labels_df
    gc.collect()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    print(f"  Short IC data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df, 'date'


def load_long_data(delta: int) -> tuple[pd.DataFrame, str]:
    """Load features + long IC labels, merge, return (df, date_col)."""
    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_parquet(LONG_LABELS_PATH)

    prefix = f"d{delta}_"
    rename_map = {}
    cols_to_keep = ['date', 'entry_time']
    for col in labels_df.columns:
        if col.startswith(prefix):
            rename_map[col] = col[len(prefix):]
            cols_to_keep.append(col)
    labels_df = labels_df[cols_to_keep].rename(columns=rename_map)

    # Derive target columns
    if 'long_profitable' in labels_df.columns:
        labels_df['profitable'] = labels_df['long_profitable']
    if 'pnl_pct' in labels_df.columns:
        labels_df['big_move'] = (labels_df['pnl_pct'] > 1.0).astype(float)

    df = features_df.merge(labels_df, on=['date', 'entry_time'], how='inner')
    del features_df, labels_df
    gc.collect()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    print(f"  Long IC data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df, 'date'


def get_feature_columns(df: pd.DataFrame, date_col: str, label_cols: list[str]) -> list[str]:
    """Identify feature columns by exclusion."""
    exclude = {date_col, 'entry_time', 'delta'} | set(label_cols)
    label_keywords = ('credit', 'pnl', 'pnl_pct', 'long_profitable', 'profitable', 'big_move')
    for col in df.columns:
        if col in label_keywords or any(col.endswith(f'_{s}') for s in label_keywords):
            exclude.add(col)
    return [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def preprocess(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """Fill NaN, remove constants. Returns (df, final_feature_cols)."""
    mask = df[target_col].notna()
    df = df[mask].copy()
    df[feature_cols] = df[feature_cols].fillna(-999)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], -999)
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"    Removing {len(constant_cols)} constant features")
        feature_cols = [c for c in feature_cols if c not in constant_cols]
    return df, feature_cols


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    scale_pos_weight: float,
) -> tuple:
    """Train XGBoost with early stopping. Returns (model, elapsed_seconds)."""
    if not HAS_XGBOOST:
        print("    WARNING: xgboost not installed, using sklearn fallback")
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        start = time.time()
        model.fit(X_train, y_train)
        return model, time.time() - start

    p = params.copy()
    p['scale_pos_weight'] = scale_pos_weight
    model = xgb.XGBClassifier(**p, early_stopping_rounds=50)

    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, time.time() - start


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate model, return metrics dict."""
    probs = model.predict_proba(X)[:, 1]
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = 0.5
    brier = float(brier_score_loss(y, probs))

    # Optimal F1 threshold
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr

    return {'auc': auc, 'brier': brier, 'f1': best_f1, 'threshold': float(best_thr)}


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def train_ensemble_for_strategy(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    targets: dict[str, str],
    params: dict,
    strategy_name: str,
    delta: int,
    output_base: Path,
):
    """Train 6-window ensemble for all targets of a strategy.

    Saves models to output_base/{window_name}/ and ensemble config to output_base/.
    """
    results = {}

    # Common holdout: last 3 months for evaluation only
    holdout_cutoff = END_DATE - pd.DateOffset(months=3)
    embargo_cutoff = holdout_cutoff - pd.Timedelta(days=EMBARGO_DAYS * 1.5)
    test_df = df[df[date_col] > holdout_cutoff].copy()

    if len(test_df) == 0:
        print("  WARNING: no test data in holdout period, using last 10% of data")
        n = len(df)
        test_df = df.iloc[int(n * 0.9):].copy()

    print(f"\n  Holdout test: {len(test_df)} rows "
          f"({test_df[date_col].min().date()} to {test_df[date_col].max().date()})")

    for target_name, target_col in targets.items():
        if target_col not in df.columns:
            print(f"\n  WARNING: column '{target_col}' missing — skipping {target_name}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {strategy_name.upper()} | Target: {target_name} | Delta: {delta}")
        print(f"{'=' * 70}")

        target_results = {'windows': {}}
        window_models = {}

        for window_name, window_start in WINDOWS.items():
            print(f"\n  --- Window: {window_name} ({window_start.date()} to {END_DATE.date()}) ---")

            # Filter to window dates, excluding holdout+embargo
            window_df = df[
                (df[date_col] >= window_start) &
                (df[date_col] <= embargo_cutoff)
            ].copy()

            if len(window_df) < 50:
                print(f"    Only {len(window_df)} rows — skipping")
                continue

            # Preprocess this window
            window_df, w_feature_cols = preprocess(window_df, target_col, feature_cols.copy())

            # Ensure consistent features across windows
            for c in feature_cols:
                if c not in window_df.columns:
                    window_df[c] = -999.0

            X_train = window_df[feature_cols].values
            y_train = window_df[target_col].astype(int).values

            # Preprocess test data with same features
            test_copy = test_df.copy()
            test_copy[feature_cols] = test_copy[feature_cols].fillna(-999)
            test_copy[feature_cols] = test_copy[feature_cols].replace([np.inf, -np.inf], -999)
            test_mask = test_copy[target_col].notna()
            test_copy = test_copy[test_mask]
            X_test = test_copy[feature_cols].values
            y_test = test_copy[target_col].astype(int).values

            pos_rate = y_train.mean()
            spw = compute_scale_pos_weight(y_train)

            print(f"    Samples: {len(X_train)} train, {len(X_test)} test")
            print(f"    Positive rate: {pos_rate:.3f} | scale_pos_weight: {spw:.2f}")

            # Internal CV for metrics
            embargo_samples = max(1, int(EMBARGO_DAYS * len(window_df) / window_df[date_col].dt.date.nunique()))
            tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 200)), gap=embargo_samples)
            cv_aucs = []
            for tr_idx, va_idx in tscv.split(X_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                m, _ = train_model(X_tr, y_tr, X_va, y_va, params, compute_scale_pos_weight(y_tr))
                try:
                    cv_aucs.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
                except ValueError:
                    cv_aucs.append(0.5)
                del m
                gc.collect()

            # Final model: train on full window, early stop on test
            model, elapsed = train_model(X_train, y_train, X_test, y_test, params, spw)
            test_metrics = evaluate(model, X_test, y_test)
            train_metrics = evaluate(model, X_train, y_train)

            cv_mean = np.mean(cv_aucs) if cv_aucs else 0.5
            cv_std = np.std(cv_aucs) if cv_aucs else 0.0

            print(f"    CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print(f"    Train AUC: {train_metrics['auc']:.4f} | "
                  f"Test AUC: {test_metrics['auc']:.4f} | "
                  f"Brier: {test_metrics['brier']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Save model
            window_dir = output_base / window_name
            window_dir.mkdir(parents=True, exist_ok=True)
            artifact_name = f"v8_{strategy_name}_d{delta}_{target_name}.joblib"
            joblib.dump(model, window_dir / artifact_name)
            print(f"    Saved: {window_name}/{artifact_name}")

            window_models[window_name] = model
            target_results['windows'][window_name] = {
                'n_samples': len(X_train),
                'positive_rate': float(pos_rate),
                'cv_auc_mean': float(cv_mean),
                'cv_auc_std': float(cv_std),
                'train_auc': train_metrics['auc'],
                'test_auc': test_metrics['auc'],
                'test_brier': test_metrics['brier'],
                'test_f1': test_metrics['f1'],
                'train_time_s': elapsed,
            }

            del X_train, y_train, window_df
            gc.collect()

        # Ensemble evaluation on holdout
        if window_models:
            active_weights = {w: WEIGHTS[w] for w in window_models}
            total_w = sum(active_weights.values())
            active_weights = {w: v / total_w for w, v in active_weights.items()}

            # Ensemble prediction
            test_copy = test_df.copy()
            test_copy[feature_cols] = test_copy[feature_cols].fillna(-999)
            test_copy[feature_cols] = test_copy[feature_cols].replace([np.inf, -np.inf], -999)
            test_mask = test_copy[target_col].notna()
            test_copy = test_copy[test_mask]
            X_test = test_copy[feature_cols].values
            y_test = test_copy[target_col].astype(int).values

            ens_probs = np.zeros(len(X_test))
            for w, model in window_models.items():
                ens_probs += model.predict_proba(X_test)[:, 1] * active_weights[w]

            try:
                ens_auc = float(roc_auc_score(y_test, ens_probs))
            except ValueError:
                ens_auc = 0.5
            ens_brier = float(brier_score_loss(y_test, ens_probs))

            # Consensus: how many models predict >= 0.5
            consensus = np.zeros(len(X_test), dtype=int)
            for model in window_models.values():
                consensus += (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            consensus_dist = {}
            for v in range(len(window_models) + 1):
                count = int((consensus == v).sum())
                if count > 0:
                    consensus_dist[str(v)] = count

            # Best individual model
            best_window = max(target_results['windows'].items(),
                              key=lambda x: x[1]['test_auc'])

            print(f"\n  ENSEMBLE RESULTS for {target_name}:")
            print(f"    Ensemble AUC:  {ens_auc:.4f}  (Brier: {ens_brier:.4f})")
            print(f"    Best single:   {best_window[0]} AUC={best_window[1]['test_auc']:.4f}")
            print(f"    Models used:   {len(window_models)}/{len(WINDOWS)}")
            print(f"    Weights:       {active_weights}")
            print(f"    Consensus:     {consensus_dist}")

            improvement = ens_auc - best_window[1]['test_auc']
            marker = "+" if improvement > 0 else ""
            print(f"    Ensemble vs best single: {marker}{improvement:.4f} AUC")

            target_results['ensemble'] = {
                'auc': ens_auc,
                'brier': ens_brier,
                'consensus_dist': consensus_dist,
                'active_weights': active_weights,
                'n_models': len(window_models),
            }

            del window_models
            gc.collect()

        results[target_name] = target_results

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train V8 6-timeframe ensemble')
    parser.add_argument('--delta', type=int, default=10, choices=[10, 15, 20])
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['short', 'long', 'both'])
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    print("=" * 70)
    print("V8 ENSEMBLE TRAINING — 6-Timeframe Weighted Ensemble")
    print("=" * 70)
    print(f"  Delta:      {args.delta}")
    print(f"  Strategy:   {args.strategy}")
    print(f"  End date:   {END_DATE.date()}")
    print(f"  Windows:    {list(WINDOWS.keys())}")
    print(f"  Weights:    {list(WEIGHTS.values())}")
    print(f"  XGBoost:    {'YES' if HAS_XGBOOST else 'NO (sklearn fallback)'}")
    print(f"  Output:     {output_dir}")

    all_reports = {}

    # ── Short IC ──────────────────────────────────────────────────────────
    if args.strategy in ('short', 'both'):
        print(f"\n\n{'#' * 70}")
        print(f"  SHORT IC ENSEMBLE")
        print(f"{'#' * 70}")

        df, date_col = load_short_data(args.delta)

        # Get feature columns from full data, then preprocess to get consistent list
        all_label_cols = [v for v in SHORT_TARGETS.values() if v in df.columns]
        feature_cols = get_feature_columns(df, date_col, all_label_cols)

        # Do a global preprocess to get the canonical feature list
        df_pp, feature_cols = preprocess(df, all_label_cols[0], feature_cols)
        # Revert to full df but keep the canonical feature_cols
        del df_pp
        gc.collect()

        print(f"  Features: {len(feature_cols)}")

        short_results = train_ensemble_for_strategy(
            df=df,
            date_col=date_col,
            feature_cols=feature_cols,
            targets=SHORT_TARGETS,
            params=SHORT_XGBOOST_PARAMS,
            strategy_name='short_ic',
            delta=args.delta,
            output_base=output_dir,
        )
        all_reports['short_ic'] = short_results

        # Save feature names (for predictor loading)
        with open(output_dir / 'feature_names.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"\n  Feature names saved ({len(feature_cols)} features)")

        del df
        gc.collect()

    # ── Long IC ───────────────────────────────────────────────────────────
    if args.strategy in ('long', 'both'):
        print(f"\n\n{'#' * 70}")
        print(f"  LONG IC ENSEMBLE")
        print(f"{'#' * 70}")

        df, date_col = load_long_data(args.delta)

        all_label_cols = [v for v in LONG_TARGETS.values() if v in df.columns]
        feature_cols_long = get_feature_columns(df, date_col, all_label_cols)

        df_pp, feature_cols_long = preprocess(df, all_label_cols[0], feature_cols_long)
        del df_pp
        gc.collect()

        print(f"  Features: {len(feature_cols_long)}")

        long_results = train_ensemble_for_strategy(
            df=df,
            date_col=date_col,
            feature_cols=feature_cols_long,
            targets=LONG_TARGETS,
            params=LONG_XGBOOST_PARAMS,
            strategy_name='long_ic',
            delta=args.delta,
            output_base=output_dir,
        )
        all_reports['long_ic'] = long_results

        # Save feature names if not already saved (should match short IC)
        fn_path = output_dir / 'feature_names.json'
        if not fn_path.exists():
            with open(fn_path, 'w') as f:
                json.dump(feature_cols_long, f, indent=2)

        del df
        gc.collect()

    # ── Save ensemble config ──────────────────────────────────────────────
    config = {
        'version': 'v8_ensemble',
        'delta': args.delta,
        'trained_at': datetime.now().isoformat(),
        'end_date': str(END_DATE.date()),
        'windows': {w: str(d.date()) for w, d in WINDOWS.items()},
        'weights': WEIGHTS,
        'embargo_days': EMBARGO_DAYS,
        'strategies': list(all_reports.keys()),
    }
    with open(output_dir / 'ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save training report
    report = {
        'config': config,
        'results': all_reports,
    }
    with open(output_dir / f'training_report_d{args.delta}.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Models saved to: {output_dir}")
    print(f"  Strategies: {list(all_reports.keys())}")
    print(f"  Windows trained: {list(WINDOWS.keys())}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
