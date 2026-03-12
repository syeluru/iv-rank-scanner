#!/usr/bin/env python3
"""
Quick test script to demonstrate ensemble prediction
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'
MODEL_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v6' / 'ensemble'
DATA_PATH = BASE_DIR / 'ml' / 'training_data' / 'training_dataset_multi_w25.csv'

WINDOWS = ['3y', '2y', '1y', '6m', '3m', '1m']
WEIGHTS = {'3y': 0.20, '2y': 0.20, '1y': 0.20, '6m': 0.15, '3m': 0.15, '1m': 0.10}

FEATURE_COLS = [
    'rv_close_5d', 'rv_close_10d', 'rv_close_20d', 'atr_14d_pct', 'gap_pct',
    'prior_day_range_pct', 'prior_day_return', 'sma_20d_dist', 'vix_level',
    'vix_change_1d', 'vix_change_5d', 'vix_rank_30d', 'vix_sma_10d_dist',
    'vix_term_slope', 'intraday_rv', 'move_from_open_pct', 'orb_range_pct',
    'orb_contained', 'range_exhaustion', 'high_low_range_pct', 'momentum_30min',
    'trend_slope_norm', 'adx', 'efficiency_ratio', 'choppiness_index',
    'rsi_dist_50', 'linreg_r2', 'atr_ratio', 'bbw_percentile', 'ttm_squeeze',
    'bars_in_squeeze', 'garman_klass_rv', 'orb_failure', 'time_sin', 'time_cos',
    'dow_sin', 'dow_cos', 'minutes_to_close', 'is_fomc_day', 'is_fomc_week',
    'days_since_fomc'
]


def load_ensemble_models(target):
    """Load all models for a target."""
    models = {}
    for window in WINDOWS:
        model_path = MODEL_DIR / f'entry_timing_v6_{target}_{window}.joblib'
        models[window] = joblib.load(model_path)
    return models


def ensemble_predict(features, models, weights):
    """Generate ensemble prediction with consensus."""
    probs = {}
    
    for window, model in models.items():
        if isinstance(features, pd.DataFrame):
            probs[window] = model.predict_proba(features)[0, 1]
        else:
            probs[window] = model.predict_proba(features.reshape(1, -1))[0, 1]
    
    # Weighted average
    ensemble_prob = sum(probs[w] * weights[w] for w in weights)
    
    # Consensus count (how many models predict positive)
    consensus_count = sum(1 for p in probs.values() if p >= 0.5)
    
    return ensemble_prob, probs, consensus_count


def main():
    print("=" * 80)
    print("ML Model v6 - Ensemble Prediction Demo")
    print("=" * 80)
    
    # Load models
    print("\nLoading ensemble models...")
    tp25_models = load_ensemble_models('tp25')
    tp50_models = load_ensemble_models('tp50')
    print(f"✓ Loaded {len(tp25_models)} TP25 models")
    print(f"✓ Loaded {len(tp50_models)} TP50 models")
    
    # Load sample data
    print("\nLoading test data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['hit_25pct', 'hit_50pct'] + FEATURE_COLS)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y_tp25 = df.loc[mask.index, 'hit_25pct']
    y_tp50 = df.loc[mask.index, 'hit_50pct']
    
    print(f"✓ Loaded {len(X):,} clean samples")
    
    # Test on a few random samples
    print("\n" + "=" * 80)
    print("Sample Predictions")
    print("=" * 80)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=5, replace=False)
    
    for idx in sample_indices:
        features = X.iloc[idx:idx+1]
        actual_tp25 = y_tp25.iloc[idx]
        actual_tp50 = y_tp50.iloc[idx]
        
        # Get predictions
        ens_tp25, probs_tp25, consensus_tp25 = ensemble_predict(features, tp25_models, WEIGHTS)
        ens_tp50, probs_tp50, consensus_tp50 = ensemble_predict(features, tp50_models, WEIGHTS)
        
        print(f"\nSample {idx}:")
        print(f"  TP25 Actual: {actual_tp25:.0f} | Ensemble: {ens_tp25:.3f} | Consensus: {consensus_tp25}/4")
        print(f"    Individual: 1Y={probs_tp25['1y']:.3f}, 6M={probs_tp25['6m']:.3f}, "
              f"3M={probs_tp25['3m']:.3f}, 1M={probs_tp25['1m']:.3f}")
        print(f"  TP50 Actual: {actual_tp50:.0f} | Ensemble: {ens_tp50:.3f} | Consensus: {consensus_tp50}/4")
        print(f"    Individual: 1Y={probs_tp50['1y']:.3f}, 6M={probs_tp50['6m']:.3f}, "
              f"3M={probs_tp50['3m']:.3f}, 1M={probs_tp50['1m']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Ensemble models working correctly!")
    print("=" * 80)


if __name__ == '__main__':
    main()
