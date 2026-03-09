#!/usr/bin/env python3
"""
Compare v5 (Random Forest) vs v6 (XGBoost) models.
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "ml" / "artifacts" / "models"
V6_DIR = MODEL_DIR / "v6"
TRAINING_DATA_DIR = PROJECT_ROOT / "ml" / "training_data"

def load_model(path):
    """Load a model artifact."""
    return joblib.load(path)

def evaluate_on_test_set(model_artifact, X_test, y_test):
    """Evaluate model on test set."""
    model = model_artifact["model"]
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    
    # Calculate hit rate at different thresholds
    results = {
        "auc": auc,
        "prob_mean": y_prob.mean(),
        "prob_std": y_prob.std(),
        "prob_min": y_prob.min(),
        "prob_max": y_prob.max(),
    }
    
    for thresh in [0.70, 0.75, 0.80, 0.85]:
        preds = y_prob >= thresh
        if preds.sum() > 0:
            hit_rate = y_test[preds].mean()
            coverage = preds.mean()
            results[f"hit_rate_{int(thresh*100)}"] = hit_rate
            results[f"coverage_{int(thresh*100)}"] = coverage
        else:
            results[f"hit_rate_{int(thresh*100)}"] = np.nan
            results[f"coverage_{int(thresh*100)}"] = 0.0
    
    return results

def prepare_test_data(target="hit_25pct", v5_features=None, v6_features=None):
    """Load and prepare test set for both v5 and v6."""
    dataset_path = TRAINING_DATA_DIR / "training_dataset_multi_w25.csv"
    df = pd.read_csv(dataset_path)
    
    # Add derived features
    rv20 = df["rv_close_20d"].clip(lower=1)
    df["iv_rv_ratio"] = df["atm_iv"] / rv20
    
    # Create target
    df["target"] = df[target].astype(float)
    valid = df[df["target"].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    valid = valid.sort_values(["trade_date", "entry_time"] if "entry_time" in valid.columns else ["trade_date"])
    
    # Split into train/test (last 6 months = test)
    max_date = valid["trade_date"].max()
    cutoff = max_date - pd.DateOffset(months=6)
    test = valid[valid["trade_date"] >= cutoff]
    
    # Prepare features for v5 and v6
    X_test_v5 = test[v5_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test_v6 = test[v6_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = test["target"].astype(int).values
    
    return X_test_v5, X_test_v6, y_test, test

def main():
    print("=" * 80)
    print("V5 (Random Forest) vs V6 (XGBoost) Model Comparison")
    print("=" * 80)
    
    # Load v5 models
    v5_tp25 = load_model(MODEL_DIR / "entry_timing_v5_tp25.joblib")
    v5_tp50 = load_model(MODEL_DIR / "entry_timing_v5_tp50.joblib")
    
    # Load v6 models
    v6_tp25 = load_model(V6_DIR / "entry_timing_v6_baseline_tp25.joblib")
    v6_tp50 = load_model(V6_DIR / "entry_timing_v6_baseline_tp50.joblib")
    
    print("\n" + "=" * 80)
    print("TP25 (25% Take-Profit) Model Comparison")
    print("=" * 80)
    
    # Get features
    v5_features_tp25 = v5_tp25["feature_names"]
    v6_features_tp25 = v6_tp25["feature_names"]
    
    print(f"\nv5 Features: {len(v5_features_tp25)}")
    print(f"v6 Features: {len(v6_features_tp25)}")
    
    # Prepare test data for TP25
    X_test_v5, X_test_v6, y_test_25, test_df = prepare_test_data("hit_25pct", v5_features_tp25, v6_features_tp25)
    
    print(f"\nTest set: {len(y_test_25)} samples")
    print(f"Test hit rate: {y_test_25.mean():.1%}")
    
    # Evaluate v5 TP25
    v5_results_25 = evaluate_on_test_set(v5_tp25, X_test_v5, y_test_25)
    
    # Evaluate v6 TP25
    v6_results_25 = evaluate_on_test_set(v6_tp25, X_test_v6, y_test_25)
    
    print(f"\nv5 TP25 Results:")
    print(f"  AUC:           {v5_results_25['auc']:.4f}")
    print(f"  Prob range:    [{v5_results_25['prob_min']:.4f}, {v5_results_25['prob_max']:.4f}]")
    print(f"  Prob mean/std: {v5_results_25['prob_mean']:.4f} ± {v5_results_25['prob_std']:.4f}")
    print(f"\n  Threshold Performance:")
    for thresh in [70, 75, 80, 85]:
        hr = v5_results_25.get(f"hit_rate_{thresh}", np.nan)
        cov = v5_results_25.get(f"coverage_{thresh}", 0)
        print(f"    @{thresh/100:.2f}: Hit Rate={hr:.1%}, Coverage={cov:.1%}")
    
    print(f"\nv6 TP25 Results:")
    print(f"  AUC:           {v6_results_25['auc']:.4f}  (Δ: {v6_results_25['auc'] - v5_results_25['auc']:+.4f})")
    print(f"  Prob range:    [{v6_results_25['prob_min']:.4f}, {v6_results_25['prob_max']:.4f}]")
    print(f"  Prob mean/std: {v6_results_25['prob_mean']:.4f} ± {v6_results_25['prob_std']:.4f}")
    print(f"\n  Threshold Performance:")
    for thresh in [70, 75, 80, 85]:
        hr = v6_results_25.get(f"hit_rate_{thresh}", np.nan)
        hr_v5 = v5_results_25.get(f"hit_rate_{thresh}", np.nan)
        cov = v6_results_25.get(f"coverage_{thresh}", 0)
        delta = hr - hr_v5 if not (np.isnan(hr) or np.isnan(hr_v5)) else 0
        print(f"    @{thresh/100:.2f}: Hit Rate={hr:.1%} (Δ: {delta:+.1%}), Coverage={cov:.1%}")
    
    print("\n" + "=" * 80)
    print("TP50 (50% Take-Profit) Model Comparison")
    print("=" * 80)
    
    # Get features
    v5_features_tp50 = v5_tp50["feature_names"]
    v6_features_tp50 = v6_tp50["feature_names"]
    
    print(f"\nv5 Features: {len(v5_features_tp50)}")
    print(f"v6 Features: {len(v6_features_tp50)}")
    
    # Prepare test data for TP50
    X_test_v5, X_test_v6, y_test_50, test_df = prepare_test_data("hit_50pct", v5_features_tp50, v6_features_tp50)
    
    print(f"\nTest set: {len(y_test_50)} samples")
    print(f"Test hit rate: {y_test_50.mean():.1%}")
    
    # Evaluate v5 TP50
    v5_results_50 = evaluate_on_test_set(v5_tp50, X_test_v5, y_test_50)
    
    # Evaluate v6 TP50
    v6_results_50 = evaluate_on_test_set(v6_tp50, X_test_v6, y_test_50)
    
    print(f"\nv5 TP50 Results:")
    print(f"  AUC:           {v5_results_50['auc']:.4f}")
    print(f"  Prob range:    [{v5_results_50['prob_min']:.4f}, {v5_results_50['prob_max']:.4f}]")
    print(f"  Prob mean/std: {v5_results_50['prob_mean']:.4f} ± {v5_results_50['prob_std']:.4f}")
    print(f"\n  Threshold Performance:")
    for thresh in [70, 75, 80, 85]:
        hr = v5_results_50.get(f"hit_rate_{thresh}", np.nan)
        cov = v5_results_50.get(f"coverage_{thresh}", 0)
        print(f"    @{thresh/100:.2f}: Hit Rate={hr:.1%}, Coverage={cov:.1%}")
    
    print(f"\nv6 TP50 Results:")
    print(f"  AUC:           {v6_results_50['auc']:.4f}  (Δ: {v6_results_50['auc'] - v5_results_50['auc']:+.4f})")
    print(f"  Prob range:    [{v6_results_50['prob_min']:.4f}, {v6_results_50['prob_max']:.4f}]")
    print(f"  Prob mean/std: {v6_results_50['prob_mean']:.4f} ± {v6_results_50['prob_std']:.4f}")
    print(f"\n  Threshold Performance:")
    for thresh in [70, 75, 80, 85]:
        hr = v6_results_50.get(f"hit_rate_{thresh}", np.nan)
        hr_v5 = v5_results_50.get(f"hit_rate_{thresh}", np.nan)
        cov = v6_results_50.get(f"coverage_{thresh}", 0)
        delta = hr - hr_v5 if not (np.isnan(hr) or np.isnan(hr_v5)) else 0
        print(f"    @{thresh/100:.2f}: Hit Rate={hr:.1%} (Δ: {delta:+.1%}), Coverage={cov:.1%}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    auc_improvement_25 = v6_results_25['auc'] - v5_results_25['auc']
    auc_improvement_50 = v6_results_50['auc'] - v5_results_50['auc']
    
    print(f"\nTP25 Model:")
    print(f"  v5 AUC:  {v5_results_25['auc']:.4f}")
    print(f"  v6 AUC:  {v6_results_25['auc']:.4f}")
    print(f"  Change:  {auc_improvement_25:+.4f} ({auc_improvement_25/v5_results_25['auc']*100:+.1f}%)")
    
    print(f"\nTP50 Model:")
    print(f"  v5 AUC:  {v5_results_50['auc']:.4f}")
    print(f"  v6 AUC:  {v6_results_50['auc']:.4f}")
    print(f"  Change:  {auc_improvement_50:+.4f} ({auc_improvement_50/v5_results_50['auc']*100:+.1f}%)")
    
    print(f"\nModel Info:")
    print(f"  v5: {v5_tp25['model_type']} with {len(v5_features_tp25)} features")
    print(f"  v6: {v6_tp25['model_type']} with {len(v6_features_tp25)} features")
    
    # New features in v6
    v5_set = set(v5_features_tp25)
    v6_set = set(v6_features_tp25)
    new_features = sorted(v6_set - v5_set)
    
    if new_features:
        print(f"\nNew features in v6 ({len(new_features)}):")
        for feat in new_features:
            print(f"  - {feat}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
