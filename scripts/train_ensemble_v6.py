#!/usr/bin/env python3
"""
ML Model v6 - Ensemble Training Script
Train 4 XGBoost models with different lookback windows for TP25 and TP50 targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'
DATA_PATH = BASE_DIR / 'ml' / 'training_data' / 'training_dataset_multi_w25.csv'
MODEL_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v6' / 'ensemble'
BASELINE_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v6'
REPORT_PATH = BASE_DIR / 'ML_V6_ENSEMBLE_REPORT.md'

# Ensemble configuration
WINDOWS = {
    '3y': 756,  # 3 years trading days
    '2y': 504,  # 2 years trading days
    '1y': 252,  # 1 year trading days
    '6m': 126,  # 6 months trading days
    '3m': 63,   # 3 months trading days
    '1m': 21    # 1 month trading days
}

WEIGHTS = {
    '3y': 0.20,
    '2y': 0.20,
    '1y': 0.20,
    '6m': 0.15,
    '3m': 0.15,
    '1m': 0.10
}

TARGETS = ['tp25', 'tp50']

# XGBoost hyperparameters (same as baseline)
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc'
}

# Feature columns (41 features from baseline)
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


def get_window_data(df, lookback_days):
    """Extract data for specific trading day window."""
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    all_dates = sorted(df['trade_date'].dt.date.unique())
    
    if lookback_days >= len(all_dates):
        print(f"  Warning: Requested {lookback_days} days but only {len(all_dates)} available. Using all data.")
        return df
    
    window_dates = all_dates[-lookback_days:]
    filtered = df[df['trade_date'].dt.date.isin(window_dates)]
    
    print(f"  Window: {window_dates[0]} to {window_dates[-1]}")
    print(f"  Rows: {len(filtered):,} / {len(df):,} ({len(filtered)/len(df)*100:.1f}%)")
    
    return filtered


def train_model(X_train, y_train, X_test, y_test, window_name, target_name):
    """Train a single XGBoost model."""
    print(f"\n  Training {window_name} model for {target_name}...")
    
    start_time = time.time()
    
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    
    print(f"  Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Time: {train_time:.1f}s")
    
    return model, {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_time': train_time,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


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


def evaluate_ensemble(X_test, y_test, models, weights, target_name):
    """Evaluate ensemble performance on test set."""
    print(f"\n  Evaluating ensemble for {target_name}...")
    
    ensemble_probs = []
    consensus_counts = []
    
    for i in range(len(X_test)):
        features = X_test.iloc[i:i+1]
        ens_prob, probs, consensus = ensemble_predict(features, models, weights)
        ensemble_probs.append(ens_prob)
        consensus_counts.append(consensus)
    
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    
    # Consensus statistics
    consensus_dist = pd.Series(consensus_counts).value_counts().sort_index()
    
    # Filtered predictions (3+ models agree)
    strong_consensus_mask = np.array(consensus_counts) >= (len(weights) // 2 + 1)
    if strong_consensus_mask.sum() > 0:
        strong_consensus_auc = roc_auc_score(
            y_test[strong_consensus_mask],
            np.array(ensemble_probs)[strong_consensus_mask]
        )
        strong_consensus_pct = strong_consensus_mask.sum() / len(y_test) * 100
    else:
        strong_consensus_auc = None
        strong_consensus_pct = 0.0
    
    return {
        'ensemble_auc': ensemble_auc,
        'consensus_dist': consensus_dist,
        'strong_consensus_auc': strong_consensus_auc,
        'strong_consensus_pct': strong_consensus_pct
    }


def get_feature_importance(models, feature_cols):
    """Get feature importance aggregated across all models."""
    importance_df = pd.DataFrame()
    
    for window, model in models.items():
        imp = pd.DataFrame({
            'feature': feature_cols,
            f'importance_{window}': model.feature_importances_
        })
        if importance_df.empty:
            importance_df = imp
        else:
            importance_df = importance_df.merge(imp, on='feature')
    
    # Average importance
    imp_cols = [c for c in importance_df.columns if c.startswith('importance_')]
    importance_df['avg_importance'] = importance_df[imp_cols].mean(axis=1)
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    return importance_df


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("ML Model v6 - Ensemble Training")
    print("=" * 80)
    
    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")
    
    # Verify feature columns
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    print(f"Using {len(FEATURE_COLS)} features")
    
    # Track all results
    all_results = {target: {} for target in TARGETS}
    all_models = {target: {} for target in TARGETS}
    total_start_time = time.time()
    
    # Train models for each target
    for target in TARGETS:
        print(f"\n{'='*80}")
        print(f"Target: {target.upper()}")
        print(f"{'='*80}")
        
        # Map target name to actual column name
        target_col_map = {'tp25': 'hit_25pct', 'tp50': 'hit_50pct'}
        target_col = target_col_map[target]
        
        # Prepare base dataset - drop rows with NaN in target or features
        df_clean = df.dropna(subset=[target_col] + FEATURE_COLS)
        print(f"After dropping NaN: {len(df_clean):,} rows (removed {len(df) - len(df_clean):,})")
        
        # Remove inf and very large values
        X_base = df_clean[FEATURE_COLS].copy()
        y_base = df_clean[target_col].copy()
        
        # Replace inf with NaN and then drop
        X_base = X_base.replace([np.inf, -np.inf], np.nan)
        mask = X_base.notna().all(axis=1)
        X_base = X_base[mask]
        y_base = y_base[mask]
        
        print(f"After removing inf: {len(X_base):,} rows (removed {len(df_clean) - len(X_base):,})")
        
        # Split for test set (held-out from all windows)
        X_train_base, X_test, y_train_base, y_test = train_test_split(
            X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
        )
        
        print(f"\nBase split: {len(X_train_base):,} train, {len(X_test):,} test")
        print(f"Target distribution - Train: {y_train_base.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # Train model for each window
        for window_name, lookback_days in WINDOWS.items():
            print(f"\n{'-'*80}")
            print(f"Window: {window_name.upper()} ({lookback_days} trading days)")
            print(f"{'-'*80}")
            
            # Get windowed training data
            train_df = df_clean.loc[X_train_base.index].copy()
            windowed_df = get_window_data(train_df, lookback_days)
            
            # Extract features and target for this window
            X_train_window = windowed_df[FEATURE_COLS]
            y_train_window = windowed_df[target_col]
            
            # Train model
            model, metrics = train_model(
                X_train_window, y_train_window,
                X_test, y_test,
                window_name, target
            )
            
            # Save model
            model_filename = f'entry_timing_v6_{target}_{window_name}.joblib'
            model_path = MODEL_DIR / model_filename
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path}")
            
            # Store results
            all_results[target][window_name] = metrics
            all_models[target][window_name] = model
        
        # Evaluate ensemble
        print(f"\n{'-'*80}")
        print(f"Ensemble Evaluation - {target.upper()}")
        print(f"{'-'*80}")
        
        ensemble_metrics = evaluate_ensemble(
            X_test, y_test,
            all_models[target], WEIGHTS, target
        )
        
        all_results[target]['ensemble'] = ensemble_metrics
        
        print(f"  Ensemble AUC: {ensemble_metrics['ensemble_auc']:.4f}")
        print(f"  Consensus distribution:")
        for count, freq in ensemble_metrics['consensus_dist'].items():
            pct = freq / len(X_test) * 100
            print(f"    {count}/{len(WINDOWS)} models agree: {freq:,} ({pct:.1f}%)")
        
        if ensemble_metrics['strong_consensus_auc']:
            print(f"  Strong consensus (3+): AUC={ensemble_metrics['strong_consensus_auc']:.4f}, "
                  f"Coverage={ensemble_metrics['strong_consensus_pct']:.1f}%")
    
    total_time = time.time() - total_start_time
    
    # Generate report
    print(f"\n{'='*80}")
    print("Generating Report")
    print(f"{'='*80}")
    
    generate_report(all_results, all_models, total_time)
    
    print(f"\n✅ Training complete! Total time: {total_time:.1f}s")
    print(f"📁 Models saved to: {MODEL_DIR}")
    print(f"📄 Report saved to: {REPORT_PATH}")


def generate_report(all_results, all_models, total_time):
    """Generate comprehensive performance report."""
    
    # Note: Baseline model uses different feature set (44 vs 41 features)
    # Skipping direct baseline comparison - focusing on ensemble performance
    
    report = []
    report.append("# ML Model v6 - Ensemble Training Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Total Training Time:** {total_time:.1f}s ({total_time/60:.1f} minutes)")
    report.append(f"\n**Models Trained:** {len(WINDOWS) * len(TARGETS)} ({len(WINDOWS)} windows × {len(TARGETS)} targets)")
    
    # Summary table
    report.append("\n## Performance Summary")
    report.append("\n### TP25 Target")
    report.append("\n| Model | Train Samples | Test AUC | Train Time |")
    report.append("|-------|--------------|----------|------------|")
    
    for window in WINDOWS.keys():
        metrics = all_results['tp25'][window]
        report.append(f"| {window.upper()} ({WINDOWS[window]}d) | {metrics['n_train']:,} | "
                     f"{metrics['test_auc']:.4f} | {metrics['train_time']:.1f}s |")

    ens_auc_tp25 = all_results['tp25']['ensemble']['ensemble_auc']
    report.append(f"| **Ensemble (Weighted)** | - | **{ens_auc_tp25:.4f}** | - |")

    report.append(f"\n**Ensemble AUC:** {ens_auc_tp25:.4f}")

    report.append("\n### TP50 Target")
    report.append("\n| Model | Train Samples | Test AUC | Train Time |")
    report.append("|-------|--------------|----------|------------|")

    for window in WINDOWS.keys():
        metrics = all_results['tp50'][window]
        report.append(f"| {window.upper()} ({WINDOWS[window]}d) | {metrics['n_train']:,} | "
                     f"{metrics['test_auc']:.4f} | {metrics['train_time']:.1f}s |")
    
    ens_auc_tp50 = all_results['tp50']['ensemble']['ensemble_auc']
    report.append(f"| **Ensemble (Weighted)** | - | **{ens_auc_tp50:.4f}** | - |")
    
    report.append(f"\n**Ensemble AUC:** {ens_auc_tp50:.4f}")
    
    # Consensus analysis
    report.append("\n## Consensus Analysis")
    
    for target in TARGETS:
        report.append(f"\n### {target.upper()} Consensus Distribution")
        report.append("\n| Agreement | Count | Percentage |")
        report.append("|-----------|-------|------------|")
        
        consensus_dist = all_results[target]['ensemble']['consensus_dist']
        total = consensus_dist.sum()
        
        for count in sorted(consensus_dist.index):
            freq = consensus_dist[count]
            pct = freq / total * 100
            report.append(f"| {count}/{len(WINDOWS)} models | {freq:,} | {pct:.1f}% |")
        
        strong_auc = all_results[target]['ensemble']['strong_consensus_auc']
        strong_pct = all_results[target]['ensemble']['strong_consensus_pct']
        
        if strong_auc:
            report.append(f"\n**Strong Consensus (3+ models):** AUC={strong_auc:.4f}, Coverage={strong_pct:.1f}%")
    
    # Feature importance
    report.append("\n## Feature Importance by Window")
    
    for target in TARGETS:
        report.append(f"\n### {target.upper()} - Top 10 Features (Average Across Windows)")
        
        importance_df = get_feature_importance(all_models[target], FEATURE_COLS)
        
        window_headers = " | ".join(w.upper() for w in WINDOWS.keys())
        report.append(f"\n| Rank | Feature | Avg Importance | {window_headers} |")
        report.append("|------|---------|----------------|" + "|".join(["----"] * len(WINDOWS)) + "|")

        for idx, row in importance_df.head(10).iterrows():
            window_vals = " | ".join(f"{row[f'importance_{w}']:.4f}" for w in WINDOWS.keys())
            report.append(f"| {idx+1} | {row['feature']} | {row['avg_importance']:.4f} | {window_vals} |")
    
    # Ensemble configuration
    report.append("\n## Ensemble Configuration")
    report.append("\n### Window Definitions")
    report.append("\n| Window | Trading Days | Weight |")
    report.append("|--------|--------------|--------|")
    for window, days in WINDOWS.items():
        report.append(f"| {window.upper()} | {days} | {WEIGHTS[window]:.0%} |")
    
    report.append("\n### Prediction Formula")
    report.append("\n```")
    formula_parts = " + ".join(f"({w.upper()} × {WEIGHTS[w]:.2f})" for w in WINDOWS.keys())
    report.append(f"ensemble_prob = {formula_parts}")
    report.append("consensus_count = number of models predicting prob ≥ 0.5")
    report.append(f"strong_consensus = consensus_count ≥ {len(WINDOWS) // 2 + 1}")
    report.append("```")
    
    # Model files
    report.append("\n## Saved Models")
    report.append("\n```")
    for target in TARGETS:
        for window in WINDOWS.keys():
            filename = f'entry_timing_v6_{target}_{window}.joblib'
            report.append(f"ml/artifacts/models/v6/ensemble/{filename}")
    report.append("```")
    
    # Key findings
    report.append("\n## Key Findings")
    report.append(f"\n1. **Ensemble Performance:**")
    report.append(f"   - TP25: {ens_auc_tp25:.4f}")
    report.append(f"   - TP50: {ens_auc_tp50:.4f}")
    
    report.append(f"\n2. **Window Performance:**")
    for window in WINDOWS.keys():
        tp25_auc = all_results['tp25'][window]['test_auc']
        tp50_auc = all_results['tp50'][window]['test_auc']
        report.append(f"   - {window.upper()}: TP25={tp25_auc:.4f}, TP50={tp50_auc:.4f}")
    
    report.append(f"\n3. **Consensus Insights:**")
    for target in TARGETS:
        strong_pct = all_results[target]['ensemble']['strong_consensus_pct']
        strong_auc = all_results[target]['ensemble']['strong_consensus_auc']
        if strong_auc:
            report.append(f"   - {target.upper()}: {strong_pct:.1f}% of predictions have 3+ model agreement (AUC={strong_auc:.4f})")
    
    # Write report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"  Report generated: {REPORT_PATH}")


if __name__ == '__main__':
    main()
