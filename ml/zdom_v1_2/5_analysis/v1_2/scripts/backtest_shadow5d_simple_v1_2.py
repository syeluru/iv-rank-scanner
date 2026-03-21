"""
ZDOM V1.2 Backtest: Shadow 5-Delta, Simple Version

Holdout only. TP10 XGB model. $10K portfolio. 1 contract. 1 trade at a time.
$0.10 exit slippage. Skip rates 20%-40%.

Ranking: credit × probability (highest wins).
Shadow 5d: if IC_05d_25w is the optimal choice at a given minute, skip that
minute entirely. Do not trade anything. Move to next minute.

After TP/close_win → re-enter at exit time.
After SL/close_loss → done for the day.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_DIR / "4_models" / "v1_2" / "tp10_target_xgb_v1_2.pkl"
DATA_PATH = PROJECT_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final.parquet"
RESULTS_DIR = PROJECT_DIR / "5_analysis" / "v1_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STARTING_PORTFOLIO = 10_000.0
BP_PER_CONTRACT = 2_500.0
QTY = 1
FEES_PER_SHARE = 0.052  # $5.20 RT / 100 shares
EXIT_SLIP = 0.10
SHADOW_STRATEGY = "IC_05d_25w"
SKIP_RATES = [round(x, 2) for x in np.arange(0.20, 0.41, 0.01)]


def main():
    t0 = time.time()

    print("=" * 70)
    print("ZDOM V1.2 BACKTEST — Shadow 5d, credit×prob ranking")
    print("=" * 70)

    # Load model
    with open(MODEL_PATH, "rb") as f:
        pkg = pickle.load(f)
    model = pkg["model"]
    feature_cols = pkg["feature_cols"]
    split_info = pkg["split_info"]
    print(f"Model: {MODEL_PATH.name}")
    print(f"  Features: {len(feature_cols)}")

    # Load data
    print(f"\nLoading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "decision_datetime", "strategy"]).reset_index(drop=True)

    # Get holdout
    holdout_start = pd.Timestamp(split_info["holdout_range"].split(" -> ")[0])
    holdout_end = pd.Timestamp(split_info["holdout_range"].split(" -> ")[1])
    holdout = df[(df["date"] >= holdout_start) & (df["date"] <= holdout_end)].copy().reset_index(drop=True)
    print(f"  Holdout: {len(holdout):,} rows, {holdout['date'].nunique()} days")
    print(f"  Range: {holdout['date'].min().date()} -> {holdout['date'].max().date()}")
    print(f"  Strategies: {sorted(holdout['strategy'].unique())}")

    # Score
    print("\nScoring...")
    X = holdout[feature_cols]
    holdout["prob"] = model.predict_proba(X)[:, 1]

    # Compute ranking score: credit × probability
    holdout["rank_score"] = holdout["credit"] * holdout["prob"]

    # Per-strategy skip cutoffs on rank_score
    print("Computing per-strategy skip cutoffs on rank_score...")

    results = []
    all_trade_logs = {}

    for skip_rate in SKIP_RATES:
        # Compute cutoffs
        cutoffs = {}
        for strat in holdout["strategy"].unique():
            vals = holdout.loc[holdout["strategy"] == strat, "rank_score"]
            cutoffs[strat] = float(vals.quantile(skip_rate))

        # Simulate
        portfolio = STARTING_PORTFOLIO
        peak = portfolio
        max_dd = 0.0
        trades = []
        daily_returns = []

        for day in sorted(holdout["date"].unique()):
            day_df = holdout[holdout["date"] == day]
            minutes = sorted(day_df["decision_datetime"].unique())
            day_start = portfolio
            in_trade = False
            done_for_day = False
            trade_exit_time = None

            for minute in minutes:
                if done_for_day:
                    break

                if in_trade:
                    if minute < trade_exit_time:
                        continue
                    in_trade = False
                    trade_exit_time = None

                # Get all candidates at this minute
                cands = day_df[day_df["decision_datetime"] == minute].copy()
                if cands.empty:
                    continue

                # Apply per-strategy skip cutoff
                keep = []
                for idx, row in cands.iterrows():
                    if row["rank_score"] >= cutoffs.get(row["strategy"], 0):
                        keep.append(idx)
                cands = cands.loc[keep]
                if cands.empty:
                    continue

                # Pick highest rank_score
                best_idx = cands["rank_score"].idxmax()
                best = cands.loc[best_idx]

                # Shadow check: if IC_05d wins, skip this minute entirely
                if best["strategy"] == SHADOW_STRATEGY:
                    trades.append({
                        "date": str(day.date()),
                        "entry_time": str(minute),
                        "strategy": SHADOW_STRATEGY,
                        "action": "SHADOW_SKIP",
                        "credit": best["credit"],
                        "prob": best["prob"],
                        "rank_score": best["rank_score"],
                        "pnl_total": 0.0,
                        "portfolio_after": portfolio,
                    })
                    continue  # Move to next minute, don't lock BP

                # Real trade
                credit = best["credit"]
                exit_debit = best["tp10_exit_debit"]
                exit_reason = best["tp10_exit_reason"]
                exit_time = pd.Timestamp(best["tp10_exit_time"])

                pnl_per_share = credit - exit_debit - EXIT_SLIP - FEES_PER_SHARE
                pnl_total = pnl_per_share * 100 * QTY

                portfolio += pnl_total
                peak = max(peak, portfolio)
                dd = (peak - portfolio) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

                trades.append({
                    "date": str(day.date()),
                    "entry_time": str(minute),
                    "exit_time": str(exit_time),
                    "strategy": best["strategy"],
                    "action": "TRADE",
                    "credit": credit,
                    "exit_debit": exit_debit,
                    "exit_reason": exit_reason,
                    "prob": best["prob"],
                    "rank_score": best["rank_score"],
                    "pnl_per_share": pnl_per_share,
                    "pnl_total": pnl_total,
                    "portfolio_after": portfolio,
                })

                in_trade = True
                trade_exit_time = exit_time

                if exit_reason in ("sl", "close_loss"):
                    done_for_day = True

            daily_returns.append((portfolio - day_start) / day_start if day_start > 0 else 0.0)

        # Summarize
        trade_df = pd.DataFrame(trades)
        real = trade_df[trade_df["action"] == "TRADE"] if len(trade_df) > 0 else pd.DataFrame()
        shadow = trade_df[trade_df["action"] == "SHADOW_SKIP"] if len(trade_df) > 0 else pd.DataFrame()

        n_trades = len(real)
        n_shadow = len(shadow)
        n_wins = int((real["pnl_total"] > 0).sum()) if n_trades > 0 else 0
        wr = n_wins / n_trades if n_trades > 0 else 0
        total_ret = (portfolio - STARTING_PORTFOLIO) / STARTING_PORTFOLIO
        avg_pnl = real["pnl_total"].mean() if n_trades > 0 else 0

        dr = np.array(daily_returns)
        sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0

        results.append({
            "skip_rate": skip_rate,
            "trades": n_trades,
            "shadow_skips": n_shadow,
            "wins": n_wins,
            "win_rate": round(wr, 4),
            "total_return_pct": round(total_ret * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "final_portfolio": round(portfolio, 2),
            "avg_pnl": round(avg_pnl, 2),
        })

        all_trade_logs[skip_rate] = trade_df

        print(f"  skip={skip_rate:.0%}  trades={n_trades:>4}  shadow={n_shadow:>5}  "
              f"wins={n_wins:>4}  wr={wr:.1%}  ret={total_ret:>+7.1%}  "
              f"dd={max_dd:.1%}  sharpe={sharpe:>5.2f}  final=${portfolio:>9.0f}")
        sys.stdout.flush()

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = RESULTS_DIR / "backtest_shadow5d_simple_v1_2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Save best trade log (by Sharpe)
    best_row = summary_df.loc[summary_df["sharpe"].idxmax()]
    best_sr = best_row["skip_rate"]
    best_log = all_trade_logs[best_sr]
    trades_path = RESULTS_DIR / "backtest_shadow5d_simple_v1_2_trades.csv"
    best_log.to_csv(trades_path, index=False)
    print(f"Saved best trade log (skip={best_sr:.0%}): {trades_path}")

    print(f"\nBest config: skip={best_sr:.0%}, sharpe={best_row['sharpe']}, "
          f"ret={best_row['total_return_pct']}%, trades={int(best_row['trades'])}")

    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
