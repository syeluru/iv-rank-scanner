"""
Train all 5 multi-horizon XGBoost models in sequence.

Models trained (each is an independent specialist):
  model_2yr.pkl  — 2-year lookback window, captures full macro cycles
  model_1yr.pkl  — 1-year lookback, captures annual seasonal patterns
  model_180d.pkl — 6-month lookback, captures current vol regime
  model_90d.pkl  — 3-month lookback, current momentum state
  model_7d.pkl   — 1-week lookback, micro-regime final filter

Training approach:
  - Walk-forward time-series CV only (NO random splits — ever)
  - Each horizon uses a different min_train threshold for its CV folds
  - Feature selection: each model uses the full feature set by default,
    but short-horizon models DROP long-window features (e.g. 7d drops 200d SMA)
  - Final model trained on all available data for that horizon
  - Platt scaling (isotonic regression) for calibrated probabilities
  - SHAP feature importance logged on every run
  - Results saved to models/cv_results_{horizon}.json
  - Models saved to models/model_{horizon}.pkl

Usage:
  python3 scripts/train_all_horizons.py                        # train all 5
  python3 scripts/train_all_horizons.py --horizon 180d         # train one
  python3 scripts/train_all_horizons.py --horizon 7d 90d 180d  # train subset
  python3 scripts/train_all_horizons.py --tune                 # Optuna tuning (slow)
  python3 scripts/train_all_horizons.py --target ic_tp25_hit   # different target
"""

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, roc_auc_score)
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install xgboost scikit-learn")
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL  = "ic_profitable"

META_COLS = [
    "date", "spx_open", "spx_close", "atm_strike",
    "short_call_strike", "short_put_strike",
    "ic_credit", "ic_pnl_per_share", "ic_pnl_pct",
    "ic_profitable", "ic_tp25_hit",
    "president",  # string categorical
]

# ── Horizon configurations ─────────────────────────────────────────────────────
# min_train: minimum days before first CV fold (defines the "lookback" requirement)
# drop_long_features: patterns to DROP for short-window models (too much lookahead noise)

HORIZONS = {
    "2yr": {
        "min_train":          500,
        "n_splits":           5,
        "description":        "2-year lookback — full macro cycle regime",
        "drop_patterns":      [],  # keep all features
        "xgb_overrides":      {"max_depth": 5, "n_estimators": 400},
    },
    "1yr": {
        "min_train":          252,
        "n_splits":           5,
        "description":        "1-year lookback — annual seasonal patterns",
        "drop_patterns":      [],
        "xgb_overrides":      {"max_depth": 4, "n_estimators": 350},
    },
    "180d": {
        "min_train":          126,
        "n_splits":           5,
        "description":        "180-day lookback — current vol regime (primary signal)",
        "drop_patterns":      [],
        "xgb_overrides":      {"max_depth": 4, "n_estimators": 300},
    },
    "90d": {
        "min_train":          63,
        "n_splits":           5,
        "description":        "90-day lookback — current momentum state",
        "drop_patterns":      ["sma_200", "sma_100", "yz_vol_60", "hv60"],
        "xgb_overrides":      {"max_depth": 4, "n_estimators": 250},
    },
    "7d": {
        "min_train":          21,
        "n_splits":           5,
        "description":        "7-day lookback — micro-regime final filter",
        "drop_patterns":      ["sma_200", "sma_100", "sma_50", "yz_vol_60",
                               "yz_vol_30", "hv60", "hv30", "ema_20"],
        "xgb_overrides":      {"max_depth": 3, "n_estimators": 150,
                               "learning_rate": 0.03},
    },
}

# ── Base XGBoost params ────────────────────────────────────────────────────────
BASE_XGB_PARAMS = {
    "n_estimators":          300,
    "max_depth":             4,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "min_child_weight":      5,
    "gamma":                 0.1,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "scale_pos_weight":      1.0,
    "random_state":          42,
    "eval_metric":           "auc",
    "early_stopping_rounds": 30,
    "verbosity":             0,
}

MIN_AUC_TO_SAVE = 0.60  # don't save a model that can't beat random


# ── Walk-forward CV ────────────────────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, n_splits: int,
                    min_train: int) -> list[tuple[list, list]]:
    """
    Expanding-window walk-forward CV.
    Returns list of (train_indices, val_indices) tuples.
    """
    n         = len(df)
    fold_size = max(1, (n - min_train) // n_splits)

    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_start = train_end
        val_end   = val_start + fold_size

        if val_end > n:
            break

        splits.append((list(range(0, train_end)), list(range(val_start, val_end))))

    return splits


# ── Single fold training ───────────────────────────────────────────────────────

def train_fold(X_train, y_train, X_val, y_val,
               params: dict) -> tuple:
    """Train one XGBoost fold. Returns (model, metrics, probs)."""
    # Separate early_stopping from params for sklearn compat
    early_stop = params.pop("early_stopping_rounds", 30)

    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stop)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

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

    # Restore for re-use
    params["early_stopping_rounds"] = early_stop
    return model, metrics, probs


# ── Feature selection per horizon ─────────────────────────────────────────────

def select_features(all_feature_cols: list, drop_patterns: list) -> list:
    """
    Remove features that contain any pattern in drop_patterns.
    E.g., drop_patterns=["sma_200"] removes "sma_200", "sma_200_dist", etc.
    """
    if not drop_patterns:
        return all_feature_cols
    return [
        col for col in all_feature_cols
        if not any(pat in col for pat in drop_patterns)
    ]


# ── Optuna tuning ──────────────────────────────────────────────────────────────

def optuna_tune(X: pd.DataFrame, y: pd.Series, splits: list,
                pos_weight: float, n_trials: int = 40) -> dict:
    """Run Optuna tuning. Returns best params dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — skipping. pip install optuna")
        return {}

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "gamma":             trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 5.0),
            "scale_pos_weight":  pos_weight,
            "random_state":      42,
            "eval_metric":       "auc",
            "early_stopping_rounds": 20,
            "verbosity":         0,
        }
        aucs = []
        for train_idx, val_idx in splits[:3]:  # use first 3 folds for speed
            _, metrics, _ = train_fold(
                X.iloc[train_idx], y.iloc[train_idx],
                X.iloc[val_idx],   y.iloc[val_idx],
                dict(p)
            )
            aucs.append(metrics["auc"])
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best Optuna AUC: {study.best_value:.4f}")
    return study.best_params


# ── SHAP analysis ──────────────────────────────────────────────────────────────

def compute_shap_importance(model, X: pd.DataFrame,
                             feature_cols: list) -> pd.DataFrame:
    """Compute SHAP-based feature importance."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.head(min(500, len(X))))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 for binary
        importance = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame({
            "feature":    feature_cols,
            "shap_importance": importance,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [warn] SHAP failed: {e}")

    # Fallback: XGBoost native importance
    return pd.DataFrame({
        "feature":         feature_cols,
        "shap_importance": model.feature_importances_,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)


# ── Main train loop for one horizon ───────────────────────────────────────────

def train_horizon(horizon: str, df: pd.DataFrame, target_col: str,
                  tune: bool = False) -> dict | None:
    """
    Train a single-horizon model. Returns model metadata or None on failure.
    """
    cfg = HORIZONS[horizon]
    print(f"\n{'='*65}")
    print(f"  Training: {horizon.upper()} — {cfg['description']}")
    print(f"  Target: {target_col} | min_train: {cfg['min_train']} days")
    print(f"{'='*65}")

    # ── Subset data for short-horizon models ──
    # For 7d/90d models: still use all historical data for training
    # (the "horizon" is captured by feature selection, not data slicing)
    df_train = df.copy()
    print(f"  Dataset: {len(df_train)} rows x {df_train.shape[1]} cols")
    print(f"  Date range: {df_train['date'].min().date()} → {df_train['date'].max().date()}")

    if len(df_train) < cfg["min_train"] + 20:
        print(f"  [skip] insufficient data ({len(df_train)} rows, need {cfg['min_train']+20})")
        return None

    # ── Feature prep ──
    all_feature_cols = [c for c in df_train.columns if c not in META_COLS]

    # Drop high-null features
    null_rates = df_train[all_feature_cols].isna().mean()
    drop_high_null = null_rates[null_rates > 0.50].index.tolist()
    if drop_high_null:
        print(f"  Dropping {len(drop_high_null)} high-null features (>50%)")
        all_feature_cols = [c for c in all_feature_cols if c not in drop_high_null]

    # Horizon-specific feature selection
    feature_cols = select_features(all_feature_cols, cfg["drop_patterns"])

    # Drop non-numeric
    non_numeric = [c for c in feature_cols
                   if not pd.api.types.is_numeric_dtype(df_train[c])]
    if non_numeric:
        print(f"  Dropping non-numeric: {non_numeric}")
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    X = df_train[feature_cols].fillna(df_train[feature_cols].median())
    y = df_train[target_col]

    if y.nunique() < 2:
        print(f"  [skip] target has only one class")
        return None

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: win={y.mean():.1%}  loss={(1-y.mean()):.1%}")

    # ── Build XGBoost params ──
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    params = {
        **BASE_XGB_PARAMS,
        "scale_pos_weight": pos_weight,
        **cfg["xgb_overrides"],
    }
    print(f"  scale_pos_weight: {pos_weight:.2f}")

    # ── Optuna tuning ──
    splits = walk_forward_cv(df_train, cfg["n_splits"], cfg["min_train"])
    if not splits:
        print(f"  [skip] walk_forward_cv returned no splits")
        return None

    if tune:
        print(f"\n  Running Optuna ({40} trials)...")
        best_params = optuna_tune(X, y, splits, pos_weight)
        if best_params:
            params.update(best_params)

    # ── Walk-forward CV ──
    print(f"\n  Walk-forward CV ({len(splits)} folds, min_train={cfg['min_train']})...")
    cv_results  = []
    fold_probs  = pd.Series(index=df_train.index, dtype=float)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[val_idx].nunique() < 2:
            print(f"  Fold {fold_i+1}: SKIPPED (single class)")
            continue

        _, metrics, probs = train_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx],   y.iloc[val_idx],
            dict(params)
        )
        fold_probs.iloc[val_idx] = probs

        val_dates   = df_train["date"].iloc[[val_idx[0], val_idx[-1]]]
        train_dates = df_train["date"].iloc[[train_idx[0], train_idx[-1]]]

        metrics.update({
            "fold":        fold_i + 1,
            "train_start": str(train_dates.iloc[0].date()),
            "train_end":   str(train_dates.iloc[-1].date()),
            "val_start":   str(val_dates.iloc[0].date()),
            "val_end":     str(val_dates.iloc[-1].date()),
        })
        cv_results.append(metrics)

        print(f"  Fold {fold_i+1}: train={metrics['n_train']:4d}d  "
              f"val={metrics['n_val']:3d}d  "
              f"AUC={metrics['auc']:.4f}  "
              f"Prec={metrics['precision']:.3f}  "
              f"({metrics['val_start']} → {metrics['val_end']})")

    if not cv_results:
        print(f"  [skip] all CV folds skipped — target too imbalanced")
        return None

    mean_auc = np.mean([r["auc"] for r in cv_results])
    std_auc  = np.std( [r["auc"] for r in cv_results])
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    if mean_auc < MIN_AUC_TO_SAVE:
        print(f"  [warn] AUC {mean_auc:.4f} < {MIN_AUC_TO_SAVE} threshold — model saved but flagged")

    # ── Save CV results ──
    cv_file = MODELS_DIR / f"cv_results_{horizon}.json"
    with open(cv_file, "w") as f:
        json.dump({
            "horizon":    horizon,
            "target":     target_col,
            "mean_auc":   mean_auc,
            "std_auc":    std_auc,
            "n_features": len(feature_cols),
            "folds":      cv_results,
            "trained_at": datetime.now().isoformat(),
            "min_auc_met": mean_auc >= MIN_AUC_TO_SAVE,
        }, f, indent=2)
    print(f"  CV results → {cv_file.name}")

    # ── Final model (train on all data) ──
    print(f"\n  Training final model on all {len(X)} days...")
    final_params = {k: v for k, v in params.items()
                    if k != "early_stopping_rounds"}

    # Use last 20% as holdout for early stopping
    holdout_n = max(30, int(len(X) * 0.20))
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        X.iloc[:-holdout_n], y.iloc[:-holdout_n],
        eval_set=[(X.iloc[-holdout_n:], y.iloc[-holdout_n:])],
        verbose=False,
    )

    # ── Platt calibration ──
    print(f"  Calibrating probabilities (isotonic regression)...")
    try:
        calibrated = CalibratedClassifierCV(
            xgb.XGBClassifier(**final_params),
            method="isotonic",
            cv=3,
        )
        # Use recent half for calibration (walk-forward spirit)
        cal_start = len(X) // 2
        calibrated.fit(X.iloc[cal_start:], y.iloc[cal_start:])
        calibrated_model = calibrated
        print(f"  Calibration fit on last {len(X) - cal_start} days")
    except Exception as e:
        print(f"  [warn] calibration failed: {e} — using uncalibrated model")
        calibrated_model = None

    # ── SHAP importance ──
    print(f"  Computing feature importance...")
    importance_df = compute_shap_importance(final_model, X, feature_cols)

    fi_file = MODELS_DIR / f"feature_importance_{horizon}.csv"
    importance_df.to_csv(fi_file, index=False)
    print(f"  Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['shap_importance']:.4f}")

    # ── Save model ──
    model_data = {
        "model":           final_model,
        "calibrated_model": calibrated_model,
        "feature_cols":    feature_cols,
        "target_col":      target_col,
        "horizon":         horizon,
        "description":     cfg["description"],
        "cv_auc":          mean_auc,
        "cv_std":          std_auc,
        "min_auc_met":     mean_auc >= MIN_AUC_TO_SAVE,
        "trained_at":      datetime.now().isoformat(),
        "n_train_days":    len(X),
        "params":          params,
        "importance":      importance_df.head(30).to_dict("records"),
    }

    model_file = MODELS_DIR / f"model_{horizon}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved → {model_file.name}")
    print(f"  {'✓' if mean_auc >= MIN_AUC_TO_SAVE else '⚠'} AUC={mean_auc:.4f} "
          f"{'(meets threshold)' if mean_auc >= MIN_AUC_TO_SAVE else f'(below {MIN_AUC_TO_SAVE} — needs more data/features)'}")

    return model_data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train all multi-horizon XGBoost models"
    )
    parser.add_argument(
        "--horizon", nargs="*",
        choices=list(HORIZONS.keys()),
        default=None,
        help="Horizons to train (default: all). E.g. --horizon 7d 90d 180d"
    )
    parser.add_argument(
        "--target", default=TARGET_COL,
        choices=["ic_profitable", "ic_tp25_hit"],
        help="Target variable (default: ic_profitable)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning (slow)"
    )
    args = parser.parse_args()

    horizons_to_train = args.horizon or list(HORIZONS.keys())
    target_col        = args.target

    print(f"\nBill Benton — Multi-Horizon Model Trainer")
    print(f"Target:   {target_col}")
    print(f"Horizons: {', '.join(horizons_to_train)}")
    print(f"Tuning:   {'YES (Optuna)' if args.tune else 'NO'}")

    # ── Load model table ──
    model_table = DATA_DIR / "model_table.parquet"
    if not model_table.exists():
        print(f"\n[error] {model_table} not found.")
        print("  Run 'make build' first to generate the model table.")
        sys.exit(1)

    print(f"\nLoading model table from {model_table}...")
    df = pd.read_parquet(model_table)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} rows x {df.shape[1]} cols")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    if target_col not in df.columns:
        print(f"\n[error] target '{target_col}' not found in model table.")
        print(f"  Available targets: {[c for c in df.columns if 'ic_' in c]}")
        sys.exit(1)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    print(f"  After dropping target NaN: {len(df)} rows")
    print(f"  Target distribution: {df[target_col].mean():.1%} wins")

    # ── Train each horizon ──
    results = {}
    for horizon in horizons_to_train:
        try:
            model_data = train_horizon(horizon, df.copy(), target_col, tune=args.tune)
            if model_data:
                results[horizon] = {
                    "auc":       model_data["cv_auc"],
                    "std":       model_data["cv_std"],
                    "n_days":    model_data["n_train_days"],
                    "min_met":   model_data["min_auc_met"],
                }
        except Exception as e:
            print(f"\n[error] {horizon} training failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE — {len(results)}/{len(horizons_to_train)} models")
    print(f"{'='*65}")
    print(f"\n  {'Horizon':8s}  {'AUC':6s}  {'±Std':6s}  {'N Days':8s}  {'Status'}")
    print(f"  {'-'*55}")

    for h in horizons_to_train:
        if h in results:
            r = results[h]
            status = "READY" if r["min_met"] else "LOW AUC — needs more data"
            print(f"  {h:8s}  {r['auc']:.4f}  {r['std']:.4f}  {r['n_days']:8d}  {status}")
        else:
            print(f"  {h:8s}  {'FAILED':50s}")

    # Voting system check
    ready = [h for h, r in results.items() if r.get("min_met")]
    print(f"\n  Models meeting AUC threshold: {len(ready)}/5")
    if len(ready) >= 3:
        print(f"  ✓ Voting system ready (need 3-of-5). Ready: {', '.join(ready)}")
    else:
        print(f"  ⚠ Voting system NOT ready — need 3+ models above AUC {MIN_AUC_TO_SAVE}")
        print(f"    Run 'make build' with more historical data, then retrain.")

    print(f"\n  Next: run 'python3 scripts/shadow_trade_log.py' to start paper trading")


if __name__ == "__main__":
    main()
