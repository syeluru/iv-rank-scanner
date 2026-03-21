"""
ZDOM V1.2 Training: TP10 target with Optuna tuning, XGB + LGBM.

Input:  3_feature_engineering/v1_2/outputs/master_data_v1_2_final.parquet
Target: tp10_target
Split:  50% train / 25% test / 25% holdout (time-ordered, 7-day gaps)
Tuning: Optuna Bayesian search (50 trials), every trial printed
Models: XGBoost and LightGBM
Output: Per-strategy AUC on test and holdout

Usage:
  python3 train_tp10_v1_2.py                    # default 50 trials
  python3 train_tp10_v1_2.py --n-trials 30      # fewer trials
  python3 train_tp10_v1_2.py --no-tune           # skip Optuna, use defaults
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = (
    PROJECT_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final.parquet"
)
DICT_FILE = PROJECT_DIR / "data_dictionary_codex.csv"
MODELS_DIR = PROJECT_DIR / "4_models" / "v1_2"
RESULTS_DIR = PROJECT_DIR / "5_analysis" / "v1_2" / "results"

TRAIN_PCT = 0.50
GAP_DAYS = 7
TARGET = "tp10_target"
TUNE_SAMPLE = 500_000


# ── Feature selection from data dictionary ───────────────────────────────────

def get_predictor_cols(df: pd.DataFrame) -> list[str]:
    """Get predictor columns: In Model=Yes, exclude all tp* and meta columns."""
    # Load data dictionary
    if DICT_FILE.exists():
        dd = pd.read_csv(DICT_FILE)
        yes_cols = set(dd[dd["In Model"].str.strip().str.lower() == "yes"]["Final Name"])
        print(f"  Data dictionary: {len(yes_cols)} In Model=Yes columns")
    else:
        print(f"  WARNING: {DICT_FILE} not found, using heuristic feature selection")
        yes_cols = None

    # Build exclusion set: all tp* meta, strategy, datetime, date
    tp_meta = []
    for pct in range(10, 55, 5):
        k = f"tp{pct}"
        tp_meta += [f"{k}_target", f"{k}_exit_reason", f"{k}_exit_time",
                    f"{k}_exit_debit", f"{k}_pnl"]
    meta_cols = set(tp_meta + ["decision_datetime", "date", "strategy"])

    if yes_cols is not None:
        # Use dictionary: In Model=Yes minus meta/target columns
        feature_cols = [c for c in df.columns if c in yes_cols and c not in meta_cols]
    else:
        # Fallback: all numeric minus meta
        feature_cols = [c for c in df.columns if c not in meta_cols]

    # Only keep numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols


# ── Model builders ───────────────────────────────────────────────────────────

def build_xgb(params):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        min_child_weight=params.get("min_child_weight", 5),
        gamma=params.get("gamma", 0.1),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        random_state=42,
        eval_metric="auc",
        verbosity=0,
        early_stopping_rounds=30,
    )


def build_lgbm(params):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        min_child_weight=params.get("min_child_weight", 5),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        random_state=42,
        verbosity=-1,
    )


MODEL_BUILDERS = {"xgb": build_xgb, "lgbm": build_lgbm}


# ── Optuna tuning with per-trial printing ────────────────────────────────────

def optuna_tune(model_type, X_train, y_train, X_test, y_test, pos_weight, n_trials=50):
    import optuna
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample for speed
    if len(X_train) > TUNE_SAMPLE:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=TUNE_SAMPLE, random_state=42)
        sample_idx, _ = next(sss.split(X_train, y_train))
        X_tune = X_train.iloc[sample_idx]
        y_tune = y_train.iloc[sample_idx]
        print(f"    Tuning on {len(X_tune):,} subsample (of {len(X_train):,})")
    else:
        X_tune = X_train
        y_tune = y_train

    best_so_far = [0.0]

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "gamma": trial.suggest_float("gamma", 0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                "scale_pos_weight": pos_weight,
            }
            model = build_xgb(params)
            model.fit(X_tune, y_tune, eval_set=[(X_test, y_test)], verbose=False)
        elif model_type == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                "scale_pos_weight": pos_weight,
            }
            model = build_lgbm(params)
            model.fit(X_tune, y_tune)
        else:
            return 0.0

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        # Print every trial
        star = " ★" if auc > best_so_far[0] else ""
        best_so_far[0] = max(best_so_far[0], auc)
        print(f"    Trial {trial.number + 1:>3}/{n_trials}  AUC={auc:.4f}  best={best_so_far[0]:.4f}"
              f"  depth={params.get('max_depth', '?')}  lr={params.get('learning_rate', 0):.4f}"
              f"  n_est={params.get('n_estimators', '?')}{star}")
        sys.stdout.flush()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"    ── Best Optuna AUC: {study.best_value:.4f}")
    return study.best_params


# ── AUC helpers ──────────────────────────────────────────────────────────────

def safe_auc(y_true, probs):
    from sklearn.metrics import roc_auc_score
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, probs))


def strategy_auc_table(df, probs, split_name):
    rows = []
    for strategy, grp in df.groupby("strategy", sort=True):
        auc = safe_auc(grp[TARGET].astype(int), probs[grp.index])
        rows.append({
            "split": split_name,
            "strategy": strategy,
            "rows": len(grp),
            "positives": int(grp[TARGET].sum()),
            "win_rate": round(float(grp[TARGET].mean()), 4),
            "auc": auc,
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZDOM V1.2 TP10 Trainer")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (default 50)")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna, use defaults")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ZDOM V1.2 — TP10 Model Training")
    print(f"{'='*70}")
    print(f"  Target:  {TARGET}")
    print(f"  Split:   50% train / 25% test / 25% holdout")
    print(f"  Gap:     {GAP_DAYS} days between splits")
    print(f"  Tuning:  {'OFF' if args.no_tune else f'Optuna ({args.n_trials} trials)'}")
    print(f"  Models:  XGBoost + LightGBM")

    # Load data
    print(f"\nLoading {INPUT_FILE.name}...")
    df = pd.read_parquet(INPUT_FILE)
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "decision_datetime", "strategy"]).reset_index(drop=True)
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(f"  {df['date'].nunique()} trading days, {df['strategy'].nunique()} strategies")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Features
    feature_cols = get_predictor_cols(df)
    print(f"  Predictors: {len(feature_cols)}")

    # Split
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    train_end_idx = int(n_dates * TRAIN_PCT)
    remaining = n_dates - train_end_idx - 2 * GAP_DAYS
    test_days = remaining // 2
    holdout_days = remaining - test_days

    test_start_idx = train_end_idx + GAP_DAYS
    test_end_idx = test_start_idx + test_days
    holdout_start_idx = test_end_idx + GAP_DAYS

    train_dates = set(dates[:train_end_idx])
    test_dates = set(dates[test_start_idx:test_end_idx])
    holdout_dates = set(dates[holdout_start_idx:])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    holdout_df = df[df["date"].isin(holdout_dates)].reset_index(drop=True)

    split_info = {
        "train_days": len(train_dates),
        "test_days": len(test_dates),
        "holdout_days": len(holdout_dates),
        "gap_days": GAP_DAYS,
        "train_range": f"{min(train_dates).date()} -> {max(train_dates).date()}",
        "test_range": f"{min(test_dates).date()} -> {max(test_dates).date()}",
        "holdout_range": f"{min(holdout_dates).date()} -> {max(holdout_dates).date()}",
    }

    print(f"\n  Train:   {len(train_df):>10,} rows  ({len(train_dates):>3d} days)  "
          f"{min(train_dates).date()} -> {max(train_dates).date()}")
    print(f"  [gap:    {GAP_DAYS} days]")
    print(f"  Test:    {len(test_df):>10,} rows  ({len(test_dates):>3d} days)  "
          f"{min(test_dates).date()} -> {max(test_dates).date()}")
    print(f"  [gap:    {GAP_DAYS} days]")
    print(f"  Holdout: {len(holdout_df):>10,} rows  ({len(holdout_dates):>3d} days)  "
          f"{min(holdout_dates).date()} -> {max(holdout_dates).date()}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    X_holdout = holdout_df[feature_cols]
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    y_holdout = holdout_df[TARGET].astype(int)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print(f"\n  Target distribution:")
    print(f"    Train:   {y_train.mean():.1%} positive | pos_weight: {pos_weight:.2f}")
    print(f"    Test:    {y_test.mean():.1%} positive")
    print(f"    Holdout: {y_holdout.mean():.1%} positive")

    t0 = time.time()
    all_strategy_aucs = []

    for model_type in ["xgb", "lgbm"]:
        print(f"\n{'='*70}")
        print(f"  {model_type.upper()}")
        print(f"{'='*70}")

        # Tune
        params = {"scale_pos_weight": pos_weight}
        if not args.no_tune:
            print(f"\n  Optuna tuning ({args.n_trials} trials)...")
            best_params = optuna_tune(
                model_type, X_train, y_train, X_test, y_test,
                pos_weight, n_trials=args.n_trials,
            )
            if best_params:
                params.update(best_params)
            print(f"\n  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}, indent=4)}")

        # Train final model
        print(f"\n  Training final model on full train set...")
        builder = MODEL_BUILDERS[model_type]
        final_model = builder(params)
        if model_type == "xgb":
            final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            final_model.fit(X_train, y_train)

        # Score
        test_probs = final_model.predict_proba(X_test)[:, 1]
        holdout_probs = final_model.predict_proba(X_holdout)[:, 1]

        test_auc = safe_auc(y_test, test_probs)
        holdout_auc = safe_auc(y_holdout, holdout_probs)

        print(f"\n  OVERALL AUC:")
        print(f"    Test:    {test_auc:.4f}")
        print(f"    Holdout: {holdout_auc:.4f}")

        # Per-strategy AUC
        test_strat = strategy_auc_table(test_df, test_probs, "test")
        holdout_strat = strategy_auc_table(holdout_df, holdout_probs, "holdout")
        combined = pd.concat([test_strat, holdout_strat], ignore_index=True)
        combined["model"] = model_type
        all_strategy_aucs.append(combined)

        print(f"\n  PER-STRATEGY AUC:")
        print(f"  {'Strategy':<18} {'Test AUC':>10} {'Holdout AUC':>12} {'Win Rate':>10} {'Rows':>8}")
        print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
        for strategy in sorted(test_strat["strategy"].unique()):
            t_row = test_strat[test_strat["strategy"] == strategy].iloc[0]
            h_row = holdout_strat[holdout_strat["strategy"] == strategy].iloc[0]
            print(f"  {strategy:<18} {t_row['auc']:>10.4f} {h_row['auc']:>12.4f} "
                  f"{h_row['win_rate']:>9.1%} {h_row['rows']:>8,}")

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"\n  TOP 20 FEATURES:")
        for _, r in importance.head(20).iterrows():
            print(f"    {r['feature']:<50} {r['importance']:.4f}")

        # Save model
        model_path = MODELS_DIR / f"tp10_target_{model_type}_v1_2.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "model_type": model_type,
                "target": TARGET,
                "feature_cols": feature_cols,
                "test_auc": test_auc,
                "holdout_auc": holdout_auc,
                "params": params,
                "split_info": split_info,
                "importance": importance,
                "trained_at": datetime.now().isoformat(),
                "n_train_rows": len(train_df),
                "n_train_days": len(train_dates),
            }, f)
        print(f"\n  Saved: {model_path}")

        # Save strategy AUC
        auc_path = RESULTS_DIR / f"tp10_{model_type}_strategy_auc_v1_2.csv"
        combined.to_csv(auc_path, index=False)

        # Save importance
        imp_path = RESULTS_DIR / f"tp10_{model_type}_importance_v1_2.csv"
        importance.to_csv(imp_path, index=False)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Save combined results
    all_aucs = pd.concat(all_strategy_aucs, ignore_index=True)
    all_aucs.to_csv(RESULTS_DIR / "tp10_all_strategy_auc_v1_2.csv", index=False)

    with open(RESULTS_DIR / "tp10_split_report_v1_2.json", "w") as f:
        json.dump({"target": TARGET, "feature_count": len(feature_cols),
                    "split_info": split_info,
                    "rows": {"train": len(train_df), "test": len(test_df), "holdout": len(holdout_df)}},
                  f, indent=2)


if __name__ == "__main__":
    main()
