"""
Train XGBoost model to predict 0DTE Iron Condor profitability.

Target: ic_profitable (1 = SPX closed inside short strikes, 0 = loss)

Training approach:
  - Walk-forward time-series cross-validation (no shuffling, no lookahead)
  - N_SPLITS expanding windows
  - Final model trained on all data
  - Saves: models/xgb_model.pkl, models/cv_results.json, models/feature_importance.csv

Hyperparameter tuning:
  - Basic grid search on validation folds
  - Can extend with Optuna for better tuning (set TUNE=True)

Usage:
  python3 scripts/train_model.py
  python3 scripts/train_model.py --tune   # run Optuna tuning (slow)
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install xgboost scikit-learn")
    exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Columns to exclude from features ─────────────────────────────────────────
META_COLS = [
    "date", "spx_open", "spx_close", "atm_strike",
    "short_call_strike", "short_put_strike",
    "ic_credit", "ic_pnl_per_share", "ic_pnl_pct",
    "ic_profitable", "ic_tp25_hit",
]

TARGET_COL = "ic_profitable"

# ── Default XGBoost params ────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma":           0.1,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "scale_pos_weight": 1.0,   # adjust for class imbalance
    "random_state":    42,
    "eval_metric":     "auc",
    "early_stopping_rounds": 30,
    "verbosity":       0,
}

# ── Cross-validation config ───────────────────────────────────────────────────
N_SPLITS    = 5     # number of CV folds
MIN_TRAIN   = 180   # min training days before first validation


def walk_forward_cv(df, feature_cols, n_splits=N_SPLITS, min_train=MIN_TRAIN):
    """
    Walk-forward time-series CV.
    Each fold: train on [0..i], validate on [i..i+fold_size].
    """
    n = len(df)
    fold_size = (n - min_train) // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_start = train_end
        val_end   = val_start + fold_size

        if val_end > n:
            break

        train_idx = list(range(0, train_end))
        val_idx   = list(range(val_start, val_end))
        splits.append((train_idx, val_idx))

    return splits


def train_fold(X_train, y_train, X_val, y_val, params):
    """Train one XGBoost fold, return model + metrics."""
    model = xgb.XGBClassifier(**{k: v for k, v in params.items()
                                  if k != "early_stopping_rounds"},
                               early_stopping_rounds=params.get("early_stopping_rounds", 30))
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc":       roc_auc_score(y_val, probs),
        "accuracy":  accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall":    recall_score(y_val, preds, zero_division=0),
        "n_train":   len(X_train),
        "n_val":     len(X_val),
        "best_iteration": model.best_iteration,
    }
    return model, metrics, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--target", default=TARGET_COL,
                        choices=["ic_profitable", "ic_tp25_hit"],
                        help="Target variable to train on")
    args = parser.parse_args()

    target_col = args.target
    print(f"Target: {target_col}")
    print("Loading model table...")

    df = pd.read_parquet(DATA_DIR / "model_table.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Feature prep ──
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Drop high-null features (>50% missing)
    null_rates = df[feature_cols].isna().mean()
    drop_high_null = null_rates[null_rates > 0.50].index.tolist()
    if drop_high_null:
        print(f"\nDropping {len(drop_high_null)} high-null features (>50%): {drop_high_null}")
        feature_cols = [c for c in feature_cols if c not in drop_high_null]

    print(f"\nFeatures: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop non-numeric columns (e.g. mag7_earnings_symbols is a string list)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    # Fill remaining NaNs with median (XGBoost handles NaN natively, but be safe)
    X = X.fillna(X.median())

    print(f"Target distribution:")
    print(f"  Win  (1): {y.sum():4.0f} ({y.mean():.1%})")
    print(f"  Loss (0): {(1-y).sum():4.0f} ({(1-y.mean()):.1%})")

    # Adjust scale_pos_weight for class imbalance
    pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {**XGB_PARAMS, "scale_pos_weight": pos_weight}
    print(f"  scale_pos_weight: {pos_weight:.2f}")

    # ── Optuna tuning (optional) ──
    if args.tune:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                p = {
                    "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
                    "max_depth":       trial.suggest_int("max_depth", 3, 7),
                    "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "gamma":           trial.suggest_float("gamma", 0, 1.0),
                    "reg_alpha":       trial.suggest_float("reg_alpha", 0, 1.0),
                    "reg_lambda":      trial.suggest_float("reg_lambda", 0.1, 5.0),
                    "scale_pos_weight": pos_weight,
                    "random_state":    42,
                    "eval_metric":     "auc",
                    "early_stopping_rounds": 20,
                    "verbosity":       0,
                }
                splits = walk_forward_cv(df, feature_cols, n_splits=3, min_train=180)
                aucs = []
                for train_idx, val_idx in splits:
                    _, metrics, _ = train_fold(
                        X.iloc[train_idx], y.iloc[train_idx],
                        X.iloc[val_idx],   y.iloc[val_idx], p
                    )
                    aucs.append(metrics["auc"])
                return np.mean(aucs)

            print(f"\nRunning Optuna tuning (50 trials)...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            best = study.best_params
            print(f"Best params: {best}")
            params.update(best)

        except ImportError:
            print("Optuna not installed — skipping tuning. pip install optuna")

    # ── Walk-forward CV ──
    print(f"\nWalk-forward CV ({N_SPLITS} folds, min_train={MIN_TRAIN})...")
    splits = walk_forward_cv(df, feature_cols)

    cv_results = []
    fold_probs = pd.Series(index=df.index, dtype=float)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        # Skip folds where train or val has only one class (too imbalanced)
        if y.iloc[train_idx].nunique() < 2 or y.iloc[val_idx].nunique() < 2:
            print(f"  Fold {fold_i+1}: SKIPPED (single class in train or val — target too imbalanced)")
            continue

        _, metrics, probs = train_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx],   y.iloc[val_idx],
            params
        )
        fold_probs.iloc[val_idx] = probs

        val_dates = df["date"].iloc[[val_idx[0], val_idx[-1]]]
        train_dates = df["date"].iloc[[train_idx[0], train_idx[-1]]]
        metrics["fold"] = fold_i + 1
        metrics["train_start"] = str(train_dates.iloc[0].date())
        metrics["train_end"]   = str(train_dates.iloc[-1].date())
        metrics["val_start"]   = str(val_dates.iloc[0].date())
        metrics["val_end"]     = str(val_dates.iloc[-1].date())
        cv_results.append(metrics)

        print(f"  Fold {fold_i+1}: train={metrics['n_train']:4d}d "
              f"val={metrics['n_val']:4d}d  "
              f"AUC={metrics['auc']:.4f}  "
              f"Acc={metrics['accuracy']:.3f}  "
              f"Prec={metrics['precision']:.3f}  "
              f"Rec={metrics['recall']:.3f}  "
              f"({metrics['val_start']} → {metrics['val_end']})")

    if not cv_results:
        print("\nWARNING: All CV folds skipped — target is too imbalanced for walk-forward CV.")
        print("Consider using a different target variable (e.g. predicting dangerous days vs. safe days).")
        mean_auc, std_auc = float("nan"), float("nan")
    else:
        mean_auc = np.mean([r["auc"] for r in cv_results])
        std_auc  = np.std([r["auc"] for r in cv_results])
    print(f"\nCV AUC: {mean_auc} ± {std_auc}")

    # Save CV results
    cv_file = MODELS_DIR / "cv_results.json"
    with open(cv_file, "w") as f:
        json.dump({
            "target": target_col,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "n_features": len(feature_cols),
            "folds": cv_results,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"CV results saved to {cv_file}")

    # ── Final model (train on all data) ──
    print(f"\nTraining final model on all {len(X)} days...")
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_model = xgb.XGBClassifier(**final_params)

    # Use last 20% as holdout for final training early-stop reference
    holdout_n = max(30, int(len(X) * 0.2))
    final_model.fit(
        X.iloc[:-holdout_n], y.iloc[:-holdout_n],
        eval_set=[(X.iloc[-holdout_n:], y.iloc[-holdout_n:])],
        verbose=False,
    )

    # ── Feature importance ──
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\nTop 20 features:")
    for _, row_fi in importance_df.head(20).iterrows():
        print(f"  {row_fi['feature']:40s} {row_fi['importance']:.4f}")

    fi_file = MODELS_DIR / "feature_importance.csv"
    importance_df.to_csv(fi_file, index=False)
    print(f"\nFeature importance saved to {fi_file}")

    # ── Save model ──
    model_data = {
        "model": final_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cv_auc": mean_auc,
        "trained_at": datetime.now().isoformat(),
        "n_train_days": len(X),
        "params": params,
    }
    model_file = MODELS_DIR / "xgb_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_file}")

    print(f"\n{'='*60}")
    print(f"Done. CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Target: {target_col}")
    print(f"Run 'make eval' for detailed evaluation or 'make score' for today's prediction.")


if __name__ == "__main__":
    main()
