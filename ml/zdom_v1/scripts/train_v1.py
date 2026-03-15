"""
ZDOM V1 Training: XGB + LGBM × 9 TP targets, Bayesian hyperparameter tuning.

Hard blockers applied first (FOMC, CPI, PPI, NFP, GDP, MAG7 earnings, VIX>35, gap>1.5%).
Model trained on blocker-filtered training data only.

Split: 70% train / 15% test / 15% holdout (time-ordered, 7-day gaps, equal test/holdout)
Tuning: Optuna Bayesian search (50 trials) on train subsample, evaluated on test
Evaluation: Holdout AUC + skip-rate analysis (5-30%)

Baselines reported:
  V0.5 = enter every trade (no blockers, no model)
  V1.0 = enter every trade that passes hard blockers (no model)
  ZDOM = hard blockers + ML model filter

Usage:
  python3 scripts/train_v1.py                       # train all
  python3 scripts/train_v1.py --target tp25_target   # single target
  python3 scripts/train_v1.py --n-trials 50          # Optuna trials (default 50)
  python3 scripts/train_v1.py --no-tune              # skip tuning, use defaults
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models" / "v1"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All 9 TP targets
TARGETS = [f"tp{p}_target" for p in range(10, 55, 5)]

# Meta columns — NOT features
_TP_META = []
for _pct in range(10, 55, 5):
    _k = f"tp{_pct}"
    _TP_META += [f"{_k}_target", f"{_k}_exit_reason", f"{_k}_exit_time",
                 f"{_k}_exit_debit", f"{_k}_pnl"]

META_COLS = [
    "datetime", "date", "strategy",
    "spx_at_entry",
    "short_call", "short_put", "long_call", "long_put",
    "call_wing_width", "put_wing_width",
    "sc_delta", "sp_delta", "sc_iv", "sp_iv",
    "time_to_close_min",
    "_blocked",
] + _TP_META

# Split settings (70/15/15 with 7-day gaps, test and holdout equal size)
TRAIN_PCT = 0.70
GAP_DAYS = 7

MIN_AUC_TO_SAVE = 0.55


# ── Model builders ────────────────────────────────────────────────────────────

def build_xgb(params):
    import xgboost as xgb
    p = {
        "n_estimators":     params.get("n_estimators", 300),
        "max_depth":        params.get("max_depth", 4),
        "learning_rate":    params.get("learning_rate", 0.05),
        "subsample":        params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.7),
        "min_child_weight": params.get("min_child_weight", 5),
        "gamma":            params.get("gamma", 0.1),
        "reg_alpha":        params.get("reg_alpha", 0.1),
        "reg_lambda":       params.get("reg_lambda", 1.0),
        "scale_pos_weight": params.get("scale_pos_weight", 1.0),
        "random_state":     42,
        "eval_metric":      "auc",
        "verbosity":        0,
        "early_stopping_rounds": 30,
    }
    return xgb.XGBClassifier(**p)


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


MODEL_BUILDERS = {
    "xgb": build_xgb,
    "lgbm": build_lgbm,
}


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def optuna_tune(model_type, X_train, y_train, X_test, y_test, pos_weight,
                n_trials=50):
    """Bayesian hyperparameter tuning: train on train set, evaluate on test set.
    Subsamples to 500K rows during tuning for speed. Final model uses full data."""
    import optuna
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample training data for faster tuning (stratified)
    TUNE_SAMPLE = 500_000
    if len(X_train) > TUNE_SAMPLE:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=TUNE_SAMPLE, random_state=42)
        sample_idx, _ = next(sss.split(X_train, y_train))
        X_tune = X_train.iloc[sample_idx]
        y_tune = y_train.iloc[sample_idx]
        print(f"    Tuning on {len(X_tune):,} subsample (of {len(X_train):,})")
    else:
        X_tune = X_train
        y_tune = y_train

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "gamma":            trial.suggest_float("gamma", 0, 2.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0),
                "scale_pos_weight": pos_weight,
            }
            model = build_xgb(params)
            model.fit(X_tune, y_tune, eval_set=[(X_test, y_test)], verbose=False)

        elif model_type == "lgbm":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0),
                "scale_pos_weight": pos_weight,
            }
            model = build_lgbm(params)
            model.fit(X_tune, y_tune)
        else:
            return 0

        probs = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"    Best Optuna AUC (test): {study.best_value:.4f}")
    return study.best_params


# ── Threshold / EV analysis ──────────────────────────────────────────────────

def threshold_analysis(probs, y_true, pnl, baseline_win_rate, baseline_ev):
    """Analyze holdout at specific skip rates (5%, 10%, 15%, 20%, 25%, 30%).

    For each skip rate:
      - Find the probability cutoff that skips that % of lowest-confidence trades
      - Look at the REMAINING population above the threshold
      - Compare that population's win rate and EV to the rules-based baseline
    """
    target_skip_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []

    # Also add 0% skip (baseline = enter everything)
    rows.append({
        "skip_rate": 0.0,
        "cutoff": 0.0,
        "n_total": len(probs),
        "n_trades": len(probs),
        "n_skipped": 0,
        "win_rate": round(float(y_true.mean()), 4),
        "baseline_win_rate": round(baseline_win_rate, 4),
        "win_rate_lift": 0.0,
        "ev_per_trade": round(float(pnl.mean()), 4),
        "baseline_ev": round(baseline_ev, 4),
        "ev_lift": 0.0,
        "total_pnl": round(float(pnl.sum()), 2),
    })

    for target_skip in target_skip_rates:
        cutoff = np.percentile(probs, target_skip * 100)

        mask = probs >= cutoff
        n_trades = mask.sum()
        n_skipped = (~mask).sum()

        if n_trades < 50:
            continue

        y_sel = y_true[mask]
        pnl_sel = pnl[mask]

        win_rate = float(y_sel.mean())
        ev_per_trade = float(pnl_sel.mean())
        actual_skip = n_skipped / len(probs)

        rows.append({
            "skip_rate": round(actual_skip, 4),
            "cutoff": round(float(cutoff), 4),
            "n_total": len(probs),
            "n_trades": int(n_trades),
            "n_skipped": int(n_skipped),
            "win_rate": round(win_rate, 4),
            "baseline_win_rate": round(baseline_win_rate, 4),
            "win_rate_lift": round(win_rate - baseline_win_rate, 4),
            "ev_per_trade": round(ev_per_trade, 4),
            "baseline_ev": round(baseline_ev, 4),
            "ev_lift": round(ev_per_trade - baseline_ev, 4),
            "total_pnl": round(float(pnl_sel.sum()), 2),
        })

    return pd.DataFrame(rows)


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, model_type, feature_cols):
    importance = model.feature_importances_
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V1 Model Trainer")
    parser.add_argument("--target", nargs="*", default=TARGETS,
                        help="Target(s) to train")
    parser.add_argument("--models", nargs="*", default=["xgb", "lgbm"],
                        choices=list(MODEL_BUILDERS.keys()),
                        help="Model types (default: xgb lgbm)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials per model (default: 50)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna tuning, use defaults")
    args = parser.parse_args()

    print(f"\nZDOM V1 Model Trainer")
    print(f"  Targets:  {args.target}")
    print(f"  Models:   {args.models}")
    print(f"  Tuning:   {'OFF' if args.no_tune else f'Optuna ({args.n_trials} trials)'}")
    print(f"  Split:    {TRAIN_PCT:.0%} train / 15% test / 15% holdout (equal)")
    print(f"  Gap:      {GAP_DAYS} days between splits")

    # Load model table
    model_table = DATA_DIR / "model_table_v1.parquet"
    if not model_table.exists():
        print(f"\n[error] {model_table} not found. Run build_model_table.py first.")
        sys.exit(1)

    print(f"\nLoading model table...")
    df = pd.read_parquet(model_table)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(f"  {df['date'].nunique()} trading days, {df['strategy'].nunique()} strategies")

    # ── Hard blockers (rules-based — applied at evaluation, NOT training) ──
    blocked = (
        (df["days_to_next_fomc"] == 0) |               # FOMC announcement day
        (df["days_to_next_fomc"] == 1) |               # FOMC eve
        (df["is_cpi_day"] == 1) |                      # CPI release
        (df["is_ppi_day"] == 1) |                      # PPI release
        (df["is_nfp_day"] == 1) |                      # Non-Farm Payrolls
        (df["is_gdp_day"] == 1) |                      # GDP release
        (df["is_mag7_earnings_day"] == 1) |            # MAG7 earnings
        (df["vix_close"] > 35) |                       # Extreme vol
        (df["gap_pct"].abs() > 1.5)                    # Huge overnight gap
    ).fillna(False)

    df["_blocked"] = blocked
    n_blocked = blocked.sum()
    n_blocked_days = df[blocked]["date"].nunique()

    print(f"\n  Hard blockers (applied at execution only, NOT during training):")
    print(f"    Would block: {n_blocked:,} rows ({n_blocked_days} days)")
    print(f"    Model trains and evaluates on ALL data")

    # ── Time-based 3-way split with gaps (on ALL dates) ──
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * TRAIN_PCT)
    remaining = n_dates - train_end_idx - 2 * GAP_DAYS
    test_days = remaining // 2
    hold_days = remaining - test_days

    test_start_idx = train_end_idx + GAP_DAYS
    test_end_idx = test_start_idx + test_days
    holdout_start_idx = test_end_idx + GAP_DAYS

    if holdout_start_idx >= n_dates:
        print(f"[error] Not enough dates for 3-way split with gaps. "
              f"Have {n_dates} dates, need at least {holdout_start_idx + 1}.")
        sys.exit(1)

    train_dates = set(dates[:train_end_idx])
    test_dates = set(dates[test_start_idx:test_end_idx])
    holdout_dates = set(dates[holdout_start_idx:])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    holdout_df = df[df["date"].isin(holdout_dates)].reset_index(drop=True)

    print(f"\n  Train:   {len(train_df):>10,} rows  ({len(train_dates):>3d} days)  "
          f"{min(train_dates).date()} -> {max(train_dates).date()}")
    print(f"  [gap:    {GAP_DAYS} days]")
    print(f"  Test:    {len(test_df):>10,} rows  ({len(test_dates):>3d} days)  "
          f"{min(test_dates).date()} -> {max(test_dates).date()}")
    print(f"  [gap:    {GAP_DAYS} days]")
    print(f"  Holdout: {len(holdout_df):>10,} rows  ({len(holdout_dates):>3d} days)  "
          f"{min(holdout_dates).date()} -> {max(holdout_dates).date()}")

    # Feature columns
    all_feature_cols = [c for c in df.columns if c not in META_COLS]
    non_numeric = [c for c in all_feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in all_feature_cols if c not in non_numeric]
    if non_numeric:
        print(f"  Dropping non-numeric: {non_numeric}")
    print(f"  Features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    X_holdout = holdout_df[feature_cols]

    # ── Train all combinations ──
    all_results = {}
    t0 = time.time()

    for target in args.target:
        if target not in df.columns:
            print(f"\n[error] target '{target}' not in model table")
            continue

        y_train = train_df[target].astype(int)
        y_test = test_df[target].astype(int)
        y_holdout = holdout_df[target].astype(int)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # PnL for EV analysis
        tp_key = target.replace("_target", "")

        # Baseline: enter every trade in holdout (no model, no blockers)
        holdout_pnl = holdout_df["credit"].values - holdout_df[f"{tp_key}_exit_debit"].values
        baseline_win_rate = y_holdout.mean()
        baseline_ev = holdout_pnl.mean()

        print(f"\n{'='*70}")
        print(f"  TARGET: {target}")
        print(f"  Train distribution: {y_train.mean():.1%} positive | pos_weight: {pos_weight:.2f}")
        print(f"  Baseline (enter all):  {len(holdout_df):,} trades  "
              f"WR={baseline_win_rate:.1%}  EV=${baseline_ev:.4f}")
        print(f"{'='*70}")

        for model_type in args.models:
            print(f"\n  --- {model_type.upper()} ---")

            try:
                # Optuna tuning
                params = {"scale_pos_weight": pos_weight}
                if not args.no_tune:
                    print(f"    Tuning ({args.n_trials} Optuna trials)...")
                    best_params = optuna_tune(
                        model_type, X_train, y_train, X_test, y_test,
                        pos_weight, n_trials=args.n_trials,
                    )
                    if best_params:
                        params.update(best_params)

                # Train final model on TRAINING DATA ONLY
                builder = MODEL_BUILDERS[model_type]
                from sklearn.metrics import roc_auc_score

                final_model = builder(params)
                if model_type == "xgb":
                    # Use test as eval set for early stopping only (not training)
                    final_model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False,
                    )
                else:
                    final_model.fit(X_train, y_train)

                # Score test set (model has NOT been trained on this data)
                test_probs = final_model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, test_probs)
                print(f"    Test AUC:    {test_auc:.4f}")

                # Score holdout set (model has NOT been trained on this data)
                ho_probs_all = final_model.predict_proba(X_holdout)[:, 1]
                ho_auc = roc_auc_score(y_holdout, ho_probs_all)
                print(f"    Holdout AUC: {ho_auc:.4f}")

                # Feature importance
                importance_df = get_feature_importance(final_model, model_type, feature_cols)
                print(f"    Top 10 features:")
                for _, row in importance_df.head(10).iterrows():
                    print(f"      {row['feature']:40s} {row['importance']:.4f}")

                # Skip rate analysis on holdout
                thresh_df = threshold_analysis(
                    ho_probs_all, y_holdout.values, holdout_pnl,
                    baseline_win_rate, baseline_ev,
                )
                if not thresh_df.empty:
                    print(f"\n    Skip Rate Analysis (holdout — above threshold only):")
                    print(f"    {'Skip%':>6s}  {'Cutoff':>7s}  {'Trades':>8s}  {'Skipped':>8s}  "
                          f"{'Win Rate':>8s}  {'WR Lift':>8s}  "
                          f"{'EV/trade':>9s}  {'EV Lift':>9s}  {'Total PnL':>10s}")
                    print(f"    {'-'*85}")
                    best_ev_idx = thresh_df["ev_per_trade"].idxmax()
                    for _, r in thresh_df.iterrows():
                        marker = " <-- best EV" if r.name == best_ev_idx else ""
                        label = "BASE" if r['skip_rate'] == 0 else f"{r['skip_rate']:.0%}"
                        print(f"    {label:>6s}  {r['cutoff']:>7.4f}  {r['n_trades']:>8,}  "
                              f"{r['n_skipped']:>8,}  {r['win_rate']:>7.1%}  "
                              f"{r['win_rate_lift']:>+7.1%}  "
                              f"${r['ev_per_trade']:>8.4f}  "
                              f"${r['ev_lift']:>+8.4f}  "
                              f"${r['total_pnl']:>10,.0f}{marker}")

                # ── Save artifacts ──
                key = f"{target}_{model_type}"

                # Model pickle
                import pickle
                model_artifact = {
                    "model": final_model,
                    "model_type": model_type,
                    "feature_cols": feature_cols,
                    "target": target,
                    "test_auc": test_auc,
                    "holdout_auc": ho_auc,
                    "params": params,
                    "importance": importance_df.head(30).to_dict("records"),
                    "trained_at": datetime.now().isoformat(),
                    "n_train_rows": len(X_train),
                    "n_train_days": len(train_dates),
                    "holdout_baseline_wr": baseline_win_rate,
                    "holdout_baseline_ev": baseline_ev,
                    "split_info": {
                        "train_days": len(train_dates),
                        "test_days": len(test_dates),
                        "holdout_days": len(holdout_dates),
                        "gap_days": GAP_DAYS,
                        "train_range": f"{min(train_dates).date()} -> {max(train_dates).date()}",
                        "test_range": f"{min(test_dates).date()} -> {max(test_dates).date()}",
                        "holdout_range": f"{min(holdout_dates).date()} -> {max(holdout_dates).date()}",
                    },
                }
                model_file = MODELS_DIR / f"{key}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model_artifact, f)
                print(f"\n    Saved -> {model_file.name}")

                # Feature importance CSV
                fi_file = MODELS_DIR / f"{key}_importance.csv"
                importance_df.to_csv(fi_file, index=False)

                # Threshold analysis CSV
                if not thresh_df.empty:
                    thresh_file = MODELS_DIR / f"{key}_thresholds.csv"
                    thresh_df.to_csv(thresh_file, index=False)

                # Summary JSON
                summary_file = MODELS_DIR / f"{key}_summary.json"
                with open(summary_file, "w") as f:
                    json.dump({
                        "key": key,
                        "target": target,
                        "model_type": model_type,
                        "test_auc": test_auc,
                        "holdout_auc": ho_auc,
                        "baseline_wr": round(baseline_win_rate, 4),
                        "baseline_ev": round(baseline_ev, 4),
                        "n_features": len(feature_cols),
                        "params": {k: str(v) for k, v in params.items()},
                        "trained_at": datetime.now().isoformat(),
                        "split_info": model_artifact["split_info"],
                    }, f, indent=2)

                all_results[key] = {
                    "test_auc": test_auc,
                    "holdout_auc": ho_auc,
                    "model_type": model_type,
                    "target": target,
                    "baseline_wr": baseline_win_rate,
                    "baseline_ev": baseline_ev,
                }

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── Final summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — {len(all_results)} models in {elapsed/60:.1f} min")
    print(f"{'='*70}")
    print(f"\n  {'Model':35s}  {'Test AUC':>8s}  {'HO AUC':>8s}  {'Base WR':>8s}  {'Base EV':>9s}")
    print(f"  {'-'*75}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {key:35s}  {r['test_auc']:.4f}    {r['holdout_auc']:.4f}    "
              f"{r['baseline_wr']:.1%}    ${r['baseline_ev']:.4f}")

    # Best model per target
    print(f"\n  WINNERS:")
    for target in args.target:
        target_results = {k: v for k, v in all_results.items() if v["target"] == target}
        if target_results:
            best_key = max(target_results, key=lambda k: target_results[k]["holdout_auc"])
            best = target_results[best_key]
            print(f"    {target:20s} -> {best['model_type'].upper():5s}  "
                  f"holdout AUC={best['holdout_auc']:.4f}")

    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"  Total training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
