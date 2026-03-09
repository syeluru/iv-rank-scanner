#!/usr/bin/env python3
"""
Monthly Retraining Script for v6 Ensemble Models

Retrains all 8 v6 ensemble models (4 windows × 2 targets) on latest data.
Archives previous month's models and generates performance report.

Usage:
    python scripts/monthly_retrain_v6.py --month auto      # Auto-detect current month
    python scripts/monthly_retrain_v6.py --month 202603    # Specific month (YYYYMM)
    python scripts/monthly_retrain_v6.py --dry-run         # Preview only
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from typing import Dict, Tuple
import sys
import os
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from ml.data.backtest_loader import BacktestLoader


def get_window_data(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Get most recent N days of data."""
    if len(df) <= days:
        return df
    return df.iloc[-days:].copy()


def train_xgboost_model(X_train, y_train, X_test, y_test, target: str, window: str):
    """
    Train XGBoost model with v6 hyperparameters.
    
    Returns: (model, metrics_dict)
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get precision/recall at threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    metrics = {
        'target': target,
        'window': window,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return model, metrics


def retrain_for_month(dataset_path: Path, month_label: str, dry_run: bool = False) -> Dict:
    """
    Retrain all v6 ensemble models for given month.
    
    Args:
        dataset_path: Path to training dataset CSV
        month_label: Month tag (YYYYMM format)
        dry_run: If True, don't save models
    
    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"=== v6 Monthly Retraining: {month_label} ===")
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"  Loaded {len(df)} trades")
    
    # Identify feature columns (exclude target and metadata columns)
    exclude_cols = ['trade_date', 'hit_tp25', 'hit_tp50', 'profit_pct', 
                    'exit_reason', 'entry_time', 'exit_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"  Features: {len(feature_cols)}")
    
    # Split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    
    # Windows configuration
    windows = [
        ('1y', 252),
        ('6m', 126),
        ('3m', 63),
        ('1m', 21)
    ]
    
    targets = ['tp25', 'tp50']
    
    all_metrics = []
    models_created = []
    
    # Create base path
    base_path = Path("ml/artifacts/models/v6/ensemble")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Archive existing models if not dry run
    if not dry_run:
        archive_path = base_path / "archive"
        archive_path.mkdir(exist_ok=True)
        
        # Move existing models to archive
        for target in targets:
            for window, _ in windows:
                pattern = f"entry_timing_v6_{target}_*_{window}.joblib"
                for existing_model in base_path.glob(pattern):
                    archive_dest = archive_path / existing_model.name
                    logger.info(f"  Archiving: {existing_model.name} -> archive/")
                    existing_model.rename(archive_dest)
    
    # Train all models
    for target in targets:
        target_col = f'hit_{target}'
        logger.info(f"\n=== Training {target.upper()} models ===")
        
        for window, days in windows:
            logger.info(f"\n  Window: {window} ({days} days)")
            
            # Get window data
            window_data = get_window_data(df, days)
            logger.info(f"    Data: {len(window_data)} trades")
            
            # Recalculate split for this window
            window_split_idx = int(len(window_data) * 0.8)
            train_df = window_data.iloc[:window_split_idx]
            test_df = window_data.iloc[window_split_idx:]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            logger.info(f"    Train: {len(X_train)} | Test: {len(X_test)}")
            logger.info(f"    Train TP rate: {y_train.mean():.1%} | Test TP rate: {y_test.mean():.1%}")
            
            # Train model
            model, metrics = train_xgboost_model(
                X_train, y_train, X_test, y_test,
                target, window
            )
            
            all_metrics.append(metrics)
            
            logger.info(f"    ✓ AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
            
            # Save model
            model_filename = f"entry_timing_v6_{target}_{month_label}_{window}.joblib"
            model_path = base_path / model_filename
            
            artifact = {
                "model": model,
                "feature_names": feature_cols,
                "label": target,
                "window": window,
                "month": month_label,
                "metrics": metrics,
                "trained_date": str(date.today())
            }
            
            if not dry_run:
                joblib.dump(artifact, model_path)
                logger.info(f"    Saved: {model_path}")
                models_created.append(model_filename)
            else:
                logger.info(f"    [DRY-RUN] Would save: {model_path}")
    
    # Health check: verify all 8 models exist
    if not dry_run:
        logger.info(f"\n=== Health Check ===")
        all_exist = True
        for target in targets:
            for window, _ in windows:
                pattern = f"entry_timing_v6_{target}_{month_label}_{window}.joblib"
                model_path = base_path / pattern
                exists = model_path.exists()
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {pattern}")
                if not exists:
                    all_exist = False
        
        if all_exist:
            logger.info("  ✓ All 8 models created successfully")
        else:
            logger.error("  ✗ Some models missing!")
    
    # Generate report
    report = {
        'month': month_label,
        'date': str(date.today()),
        'dataset': str(dataset_path),
        'total_trades': len(df),
        'models_created': len(models_created),
        'metrics': all_metrics,
        'dry_run': dry_run
    }
    
    return report


def create_monthly_report(report: Dict, report_path: Path):
    """Generate markdown report for monthly retraining."""
    lines = []
    lines.append(f"# v6 Ensemble Monthly Retraining Report")
    lines.append(f"")
    lines.append(f"**Month:** {report['month']}")
    lines.append(f"**Date:** {report['date']}")
    lines.append(f"**Dataset:** {report['dataset']}")
    lines.append(f"**Total Trades:** {report['total_trades']:,}")
    lines.append(f"")
    
    if report['dry_run']:
        lines.append(f"⚠️ **DRY RUN** - No models were saved")
        lines.append(f"")
    
    lines.append(f"## Models Created: {report['models_created']}/8")
    lines.append(f"")
    
    lines.append(f"## Performance Metrics")
    lines.append(f"")
    lines.append(f"### TP25 Models")
    lines.append(f"")
    lines.append(f"| Window | AUC | Precision | Recall | Train Size | Test Size |")
    lines.append(f"|--------|-----|-----------|--------|------------|-----------|")
    
    for m in report['metrics']:
        if m['target'] == 'tp25':
            lines.append(f"| {m['window']} | {m['auc']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['n_train']} | {m['n_test']} |")
    
    lines.append(f"")
    lines.append(f"### TP50 Models")
    lines.append(f"")
    lines.append(f"| Window | AUC | Precision | Recall | Train Size | Test Size |")
    lines.append(f"|--------|-----|-----------|--------|------------|-----------|")
    
    for m in report['metrics']:
        if m['target'] == 'tp50':
            lines.append(f"| {m['window']} | {m['auc']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['n_train']} | {m['n_test']} |")
    
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    
    # Average AUC per target
    tp25_metrics = [m for m in report['metrics'] if m['target'] == 'tp25']
    tp50_metrics = [m for m in report['metrics'] if m['target'] == 'tp50']
    
    avg_auc_tp25 = np.mean([m['auc'] for m in tp25_metrics])
    avg_auc_tp50 = np.mean([m['auc'] for m in tp50_metrics])
    
    lines.append(f"- **Average AUC (TP25):** {avg_auc_tp25:.4f}")
    lines.append(f"- **Average AUC (TP50):** {avg_auc_tp50:.4f}")
    lines.append(f"")
    lines.append(f"## Next Steps")
    lines.append(f"")
    lines.append(f"1. Review AUC scores - all should be > 0.60")
    lines.append(f"2. Test models in paper trading")
    lines.append(f"3. Monitor ensemble consensus during live scanning")
    lines.append(f"4. Check for high disagreement warnings")
    lines.append(f"")
    
    report_text = "\n".join(lines)
    
    if report_path:
        report_path.write_text(report_text)
        logger.info(f"\nReport saved: {report_path}")
    
    # Also print to console
    print("\n" + "="*70)
    print(report_text)
    print("="*70)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Monthly v6 ensemble retraining")
    parser.add_argument('--month', default='auto', help='Month label (YYYYMM) or "auto" for current month')
    parser.add_argument('--dataset', default='ml/training_data/latest_training_data.csv',
                        help='Path to training dataset CSV')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, do not save models')
    
    args = parser.parse_args()
    
    # Determine month label
    if args.month == 'auto':
        month_label = date.today().strftime("%Y%m")
        logger.info(f"Auto-detected month: {month_label}")
    else:
        month_label = args.month
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Available datasets:")
        data_dir = Path("ml/training_data")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                logger.info(f"  - {f}")
        sys.exit(1)
    
    # Run retraining
    try:
        report = retrain_for_month(dataset_path, month_label, dry_run=args.dry_run)
        
        # Generate report
        report_filename = f"v6_retraining_{month_label}_report.md"
        report_path = Path("ml/artifacts/models/v6") / report_filename
        if args.dry_run:
            report_path = None  # Don't save in dry-run mode
        
        create_monthly_report(report, report_path)
        
        if args.dry_run:
            logger.info("\n✓ Dry run complete - no changes made")
        else:
            logger.info(f"\n✓ Monthly retraining complete for {month_label}")
            logger.info(f"  Models: ml/artifacts/models/v6/ensemble/")
            logger.info(f"  Report: {report_path}")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
