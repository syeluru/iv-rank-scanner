"""
Evaluate the trained XGBoost model.

Outputs:
  - AUC, accuracy, precision, recall, F1
  - Confusion matrix
  - Calibration (predicted prob vs actual win rate)
  - Top feature importances
  - Threshold analysis (precision/recall at different cutoffs)
  - CV fold breakdown

Usage:
  python3 scripts/evaluate.py
  python3 scripts/evaluate.py --threshold 0.6   # custom trade threshold
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, accuracy_score,
        f1_score, confusion_matrix, classification_report
    )
    from sklearn.calibration import calibration_curve
except ImportError:
    print("Install: pip install scikit-learn")
    exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"


def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for trade entry")
    parser.add_argument("--test_pct", type=float, default=0.2,
                        help="Fraction of data to use as test set (most recent)")
    args = parser.parse_args()

    # ── Load model ──
    model_file = MODELS_DIR / "xgb_model.pkl"
    if not model_file.exists():
        print(f"Model not found at {model_file}. Run 'make train' first.")
        exit(1)

    from model_io import load_model
    model_data = load_model(model_file)

    model       = model_data["model"]
    feature_cols = model_data["feature_cols"]
    target_col  = model_data["target_col"]
    cv_auc      = model_data["cv_auc"]
    trained_at  = model_data["trained_at"]

    print(f"Model loaded:")
    print(f"  Target:     {target_col}")
    print(f"  CV AUC:     {cv_auc:.4f}")
    print(f"  Trained at: {trained_at}")
    print(f"  Features:   {len(feature_cols)}")

    # ── Load data ──
    df = pd.read_parquet(DATA_DIR / "model_table.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Test set = most recent X%
    n_test = max(50, int(len(df) * args.test_pct))
    test_df = df.iloc[-n_test:].copy()

    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y_test = test_df[target_col]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    print_section(f"Test Set Performance  (last {n_test} days, {args.threshold:.0%} threshold)")

    print(f"  Date range: {test_df['date'].min().date()} → {test_df['date'].max().date()}")
    print(f"  AUC:        {roc_auc_score(y_test, probs):.4f}")
    print(f"  Accuracy:   {accuracy_score(y_test, preds):.3f}")
    print(f"  Precision:  {precision_score(y_test, preds, zero_division=0):.3f}  "
          f"(of trades taken, % profitable)")
    print(f"  Recall:     {recall_score(y_test, preds, zero_division=0):.3f}  "
          f"(of profitable days, % captured)")
    print(f"  F1:         {f1_score(y_test, preds, zero_division=0):.3f}")
    print(f"\n  Actual win rate:   {y_test.mean():.1%}")
    print(f"  Trades taken:      {preds.sum():3d} / {len(preds)} ({preds.mean():.1%} of days)")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"            Pred 0    Pred 1")
    print(f"  Actual 0:  {cm[0,0]:6d}    {cm[0,1]:6d}   (losses: {cm[0,0]} avoided, {cm[0,1]} taken)")
    print(f"  Actual 1:  {cm[1,0]:6d}    {cm[1,1]:6d}   (wins: {cm[1,1]} taken, {cm[1,0]} missed)")

    # ── Threshold analysis ──
    print_section("Threshold Analysis")
    print(f"  {'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'Precision':>10} {'Recall':>10} {'Skip Rate':>10}")
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p = (probs >= thresh).astype(int)
        if p.sum() == 0:
            continue
        wr = y_test[p == 1].mean()
        prec = precision_score(y_test, p, zero_division=0)
        rec  = recall_score(y_test, p, zero_division=0)
        skip = 1 - p.mean()
        marker = " ◄" if thresh == args.threshold else ""
        print(f"  {thresh:>10.2f} {p.sum():>8d} {wr:>10.1%} {prec:>10.3f} {rec:>10.3f} {skip:>10.1%}{marker}")

    # ── Calibration ──
    print_section("Calibration (Predicted Probability vs Actual Win Rate)")
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    print(f"  {'Bin':>12} {'Count':>8} {'Avg Prob':>10} {'Actual Win%':>12}")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            avg_prob = probs[mask].mean()
            actual   = y_test[mask].mean()
            print(f"  [{lo:.1f}, {hi:.1f})  {mask.sum():>8d} {avg_prob:>10.3f} {actual:>12.1%}")

    # ── Feature importance ──
    fi_file = MODELS_DIR / "feature_importance.csv"
    if fi_file.exists():
        print_section("Top 25 Feature Importances")
        fi = pd.read_csv(fi_file).head(25)
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:40s} {row['importance']:.4f}  {bar}")

    # ── CV breakdown ──
    cv_file = MODELS_DIR / "cv_results.json"
    if cv_file.exists():
        print_section("Cross-Validation Fold Breakdown")
        with open(cv_file) as f:
            cv = json.load(f)
        print(f"  Mean AUC: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
        print(f"  {'Fold':>6} {'Train':>8} {'Val':>8} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Val Period'}")
        for fold in cv["folds"]:
            print(f"  {fold['fold']:>6d} {fold['n_train']:>8d} {fold['n_val']:>8d} "
                  f"{fold['auc']:>8.4f} {fold['accuracy']:>8.3f} "
                  f"{fold['precision']:>8.3f} {fold['recall']:>8.3f} "
                  f"  {fold['val_start']} → {fold['val_end']}")

    # ── P&L simulation ──
    print_section("Simulated P&L  (test set, hold-to-expiry)")
    if "ic_pnl_per_share" in test_df.columns:
        taken = test_df[preds == 1]
        skipped = test_df[preds == 0]

        if len(taken) > 0:
            avg_pnl = taken["ic_pnl_per_share"].mean()
            total_pnl = taken["ic_pnl_per_share"].sum()
            win_days  = (taken["ic_pnl_per_share"] > 0).sum()
            loss_days = (taken["ic_pnl_per_share"] <= 0).sum()

            print(f"  Trades taken: {len(taken)}")
            print(f"  Win days:     {win_days} ({win_days/len(taken):.1%})")
            print(f"  Loss days:    {loss_days} ({loss_days/len(taken):.1%})")
            print(f"  Avg PnL/share: ${avg_pnl:.2f}")
            print(f"  Total PnL/share: ${total_pnl:.2f}")
            print(f"  Avg credit: ${taken['ic_credit'].mean():.2f}")

            print(f"\n  Baseline (trade every day):")
            base_win  = (test_df["ic_pnl_per_share"] > 0).sum()
            base_pnl  = test_df["ic_pnl_per_share"].sum()
            print(f"  Total days: {len(test_df)}, Win: {base_win} ({base_win/len(test_df):.1%})")
            print(f"  Total PnL/share: ${base_pnl:.2f}")


if __name__ == "__main__":
    main()
