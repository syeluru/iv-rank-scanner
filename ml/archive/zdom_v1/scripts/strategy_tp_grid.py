"""
Strategy × TP Grid Analysis for ZDOM V1.

Builds a 2D table (9 delta strategies × 9 TP levels) showing model performance
for each combination. Uses trained models to score the holdout set, then slices
predictions by strategy to produce per-cell metrics.

Metrics per cell:
  - Holdout AUC (model discrimination for that strategy+TP)
  - Baseline win rate (enter everything)
  - Model win rate at 20% skip
  - EV per trade (baseline vs model)
  - EV lift (model improvement over baseline)

Usage:
  python3 scripts/strategy_tp_grid.py                    # full grid, default skip=20%
  python3 scripts/strategy_tp_grid.py --skip-rate 0.10   # 10% skip rate
  python3 scripts/strategy_tp_grid.py --metric ev_lift    # which metric to display
  python3 scripts/strategy_tp_grid.py --csv output.csv    # save full results to CSV
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models" / "v1"

STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]
TP_LEVELS = [f"tp{p}" for p in range(10, 55, 5)]
TARGETS = [f"{tp}_target" for tp in TP_LEVELS]

# Same META_COLS as train_v1.py
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

TRAIN_PCT = 0.70
GAP_DAYS = 7


def load_model(target, algo="best"):
    """Load a trained model for a given target. If algo='best', try xgb then lgbm."""
    if algo == "best":
        for a in ["xgb", "lgbm"]:
            path = MODELS_DIR / f"{target}_{a}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f), a
        return None, None
    else:
        path = MODELS_DIR / f"{target}_{algo}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f), algo
        return None, None


def compute_cell(probs, y_true, pnl, skip_rate):
    """Compute metrics for one (strategy, TP) cell at a given skip rate."""
    n = len(probs)
    if n < 30:
        return None

    # Baseline (enter everything)
    base_wr = float(y_true.mean())
    base_ev = float(pnl.mean())

    # AUC (need both classes)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = np.nan

    # Apply skip rate
    if skip_rate > 0:
        cutoff = np.percentile(probs, skip_rate * 100)
        mask = probs >= cutoff
    else:
        mask = np.ones(n, dtype=bool)

    n_trades = mask.sum()
    if n_trades < 10:
        return None

    model_wr = float(y_true[mask].mean())
    model_ev = float(pnl[mask].mean())

    return {
        "n_total": n,
        "n_trades": int(n_trades),
        "auc": round(auc, 4),
        "base_wr": round(base_wr, 4),
        "model_wr": round(model_wr, 4),
        "wr_lift": round(model_wr - base_wr, 4),
        "base_ev": round(base_ev, 4),
        "model_ev": round(model_ev, 4),
        "ev_lift": round(model_ev - base_ev, 4),
        "total_pnl": round(float(pnl[mask].sum()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Strategy × TP Grid Analysis")
    parser.add_argument("--skip-rate", type=float, default=0.20,
                        help="Skip rate for model filtering (default: 0.20)")
    parser.add_argument("--metric", default="ev_lift",
                        choices=["auc", "base_wr", "model_wr", "wr_lift",
                                 "base_ev", "model_ev", "ev_lift", "total_pnl"],
                        help="Metric to display in the grid (default: ev_lift)")
    parser.add_argument("--algo", default="best", choices=["best", "xgb", "lgbm"],
                        help="Algorithm to use (default: best available)")
    parser.add_argument("--csv", default=None,
                        help="Save full results to CSV")
    args = parser.parse_args()

    # ── Load model table ──
    model_table_path = DATA_DIR / "model_table_v1.parquet"
    if not model_table_path.exists():
        print(f"[error] {model_table_path} not found.")
        print(f"  Run the pipeline first: make target && make model_table")
        sys.exit(1)

    print(f"Loading model table...")
    df = pd.read_parquet(model_table_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    # ── Reproduce the same holdout split as train_v1.py ──
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    train_end_idx = int(n_dates * TRAIN_PCT)
    remaining = n_dates - train_end_idx - 2 * GAP_DAYS
    test_days = remaining // 2
    test_end_idx = train_end_idx + GAP_DAYS + test_days
    holdout_start_idx = test_end_idx + GAP_DAYS

    holdout_dates = set(dates[holdout_start_idx:])
    holdout_df = df[df["date"].isin(holdout_dates)].reset_index(drop=True)
    print(f"  Holdout: {len(holdout_df):,} rows, {len(holdout_dates)} days "
          f"({min(holdout_dates).date()} → {max(holdout_dates).date()})")

    # ── Feature columns ──
    all_feature_cols = [c for c in df.columns if c not in META_COLS]
    feature_cols = [c for c in all_feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    # ── Score holdout with each TP model, slice by strategy ──
    all_rows = []

    for tp in TP_LEVELS:
        target = f"{tp}_target"
        if target not in df.columns:
            print(f"  [skip] {target} not in data")
            continue

        artifact, algo_used = load_model(target, args.algo)
        if artifact is None:
            print(f"  [skip] No model found for {target}")
            continue

        model = artifact["model"]
        model_features = artifact.get("feature_cols", feature_cols)

        # Ensure we use the same features the model was trained on
        missing = [f for f in model_features if f not in holdout_df.columns]
        if missing:
            print(f"  [warn] {target}: {len(missing)} missing features, filling with NaN")

        X_holdout = holdout_df.reindex(columns=model_features, fill_value=np.nan)
        probs = model.predict_proba(X_holdout)[:, 1]

        print(f"  Scored {target} ({algo_used}): {len(probs):,} predictions")

        # PnL column
        pnl_col = f"{tp}_pnl"
        if pnl_col not in holdout_df.columns:
            # Fallback: compute from credit - exit_debit
            exit_debit_col = f"{tp}_exit_debit"
            holdout_pnl = holdout_df["credit"].values - holdout_df[exit_debit_col].values
        else:
            holdout_pnl = holdout_df[pnl_col].values

        y_holdout = holdout_df[target].values.astype(int)

        # Score per strategy
        for strat in STRATEGIES:
            mask = holdout_df["strategy"].values == strat
            if mask.sum() < 30:
                continue

            cell = compute_cell(
                probs[mask], y_holdout[mask], holdout_pnl[mask], args.skip_rate,
            )
            if cell is None:
                continue

            cell["strategy"] = strat
            cell["tp_level"] = tp
            cell["algo"] = algo_used
            all_rows.append(cell)

    if not all_rows:
        print("\n[error] No results generated. Ensure models are trained.")
        sys.exit(1)

    results = pd.DataFrame(all_rows)

    # ── Save full results ──
    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"\nFull results saved to {args.csv}")

    # ── Display grid ──
    metric = args.metric
    metric_labels = {
        "auc": "AUC",
        "base_wr": "Base Win Rate",
        "model_wr": f"Model WR ({args.skip_rate:.0%} skip)",
        "wr_lift": f"WR Lift ({args.skip_rate:.0%} skip)",
        "base_ev": "Base EV/trade",
        "model_ev": f"Model EV ({args.skip_rate:.0%} skip)",
        "ev_lift": f"EV Lift ({args.skip_rate:.0%} skip)",
        "total_pnl": f"Total PnL ({args.skip_rate:.0%} skip)",
    }

    pivot = results.pivot(index="strategy", columns="tp_level", values=metric)
    # Reorder columns and rows
    pivot = pivot.reindex(columns=TP_LEVELS, index=STRATEGIES)

    print(f"\n{'='*90}")
    print(f"  ZDOM V1 — Strategy × TP Grid")
    print(f"  Metric: {metric_labels.get(metric, metric)}")
    print(f"  Skip rate: {args.skip_rate:.0%}")
    print(f"  Holdout: {len(holdout_dates)} days "
          f"({min(holdout_dates).date()} → {max(holdout_dates).date()})")
    print(f"{'='*90}\n")

    # Format based on metric type
    if metric in ("auc", "base_wr", "model_wr", "wr_lift"):
        fmt = lambda x: f"{x:.1%}" if pd.notna(x) else "  —  "
    elif metric in ("base_ev", "model_ev", "ev_lift"):
        fmt = lambda x: f"${x:+.3f}" if pd.notna(x) else "   —  "
    elif metric == "total_pnl":
        fmt = lambda x: f"${x:+,.0f}" if pd.notna(x) else "   —  "
    else:
        fmt = lambda x: f"{x:.4f}" if pd.notna(x) else "  —  "

    # Header
    col_width = 9
    header = f"{'Strategy':>14s}"
    for tp in TP_LEVELS:
        header += f"  {tp.upper():>{col_width}s}"
    print(header)
    print(f"{'─' * len(header)}")

    # Find best cell per row for highlighting
    for strat in STRATEGIES:
        row_str = f"{strat:>14s}"
        if strat in pivot.index:
            row_vals = pivot.loc[strat]
            for tp in TP_LEVELS:
                val = row_vals.get(tp, np.nan)
                row_str += f"  {fmt(val):>{col_width}s}"
        else:
            for _ in TP_LEVELS:
                row_str += f"  {'  —  ':>{col_width}s}"
        print(row_str)

    # ── Summary: best cells ──
    print(f"\n{'─' * 60}")
    print(f"  Top 5 (strategy, TP) by {metric_labels.get(metric, metric)}:")
    top = results.nlargest(5, metric)
    for _, r in top.iterrows():
        print(f"    {r['strategy']} × {r['tp_level'].upper():>4s}  "
              f"→  {fmt(r[metric])}  "
              f"(AUC={r['auc']:.3f}, n={r['n_trades']:,})")

    print(f"\n  Bottom 5:")
    bottom = results.nsmallest(5, metric)
    for _, r in bottom.iterrows():
        print(f"    {r['strategy']} × {r['tp_level'].upper():>4s}  "
              f"→  {fmt(r[metric])}  "
              f"(AUC={r['auc']:.3f}, n={r['n_trades']:,})")

    # ── Additional grids ──
    print(f"\n{'='*90}")
    print(f"  AUC Grid (model discrimination)")
    print(f"{'='*90}\n")
    auc_pivot = results.pivot(index="strategy", columns="tp_level", values="auc")
    auc_pivot = auc_pivot.reindex(columns=TP_LEVELS, index=STRATEGIES)

    auc_fmt = lambda x: f"{x:.3f}" if pd.notna(x) else "  —  "
    header = f"{'Strategy':>14s}"
    for tp in TP_LEVELS:
        header += f"  {tp.upper():>{col_width}s}"
    print(header)
    print(f"{'─' * len(header)}")
    for strat in STRATEGIES:
        row_str = f"{strat:>14s}"
        if strat in auc_pivot.index:
            for tp in TP_LEVELS:
                val = auc_pivot.loc[strat].get(tp, np.nan)
                row_str += f"  {auc_fmt(val):>{col_width}s}"
        print(row_str)

    print(f"\n{'='*90}")
    print(f"  Baseline Win Rate Grid (enter everything, no model)")
    print(f"{'='*90}\n")
    wr_pivot = results.pivot(index="strategy", columns="tp_level", values="base_wr")
    wr_pivot = wr_pivot.reindex(columns=TP_LEVELS, index=STRATEGIES)

    wr_fmt = lambda x: f"{x:.1%}" if pd.notna(x) else "  —  "
    header = f"{'Strategy':>14s}"
    for tp in TP_LEVELS:
        header += f"  {tp.upper():>{col_width}s}"
    print(header)
    print(f"{'─' * len(header)}")
    for strat in STRATEGIES:
        row_str = f"{strat:>14s}"
        if strat in wr_pivot.index:
            for tp in TP_LEVELS:
                val = wr_pivot.loc[strat].get(tp, np.nan)
                row_str += f"  {wr_fmt(val):>{col_width}s}"
        print(row_str)

    print()


if __name__ == "__main__":
    main()
