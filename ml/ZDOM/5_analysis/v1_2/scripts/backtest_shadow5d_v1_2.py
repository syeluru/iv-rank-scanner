"""
ZDOM V1.2 Backtest: Shadow 5-Delta Strategy

Runs on HOLDOUT data only (2025-06-12 -> 2026-02-27).
Uses the trained TP10 XGB model to score all 9 strategies at each minute.
Picks the highest-probability candidate. If it is IC_05d_25w, shadow it
(log it, lock buying power through its exit lifecycle, $0 PnL). Otherwise
trade it normally.

Rules:
  - $10,000 starting portfolio
  - 1 contract at a time, 1 trade at a time
  - $2,500 buying power per contract (25-wide wings)
  - Fees: $5.20 round-trip per contract ($0.052/share)
  - Exit slippage: $0.10, $0.20, $0.30 per share
  - Skip rates: 20% to 40% in 1% increments (21 rates)
  - Skip cutoffs computed PER STRATEGY from holdout probabilities
  - Strategy 1: highest-prob candidate wins. After TP/close_win -> re-enter
    at exit time. After SL/close_loss -> done for the day.
  - PnL/share = credit - exit_debit - exit_slip - fees_per_share
  - PnL total = pnl_per_share * 100 * qty

Output:
  - Summary table printed to stdout
  - Summary CSV: 5_analysis/v1_2/results/backtest_shadow5d_v1_2_summary.csv
  - Trade log CSV (best config): 5_analysis/v1_2/results/backtest_shadow5d_v1_2_trades.csv
"""

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_DIR / "4_models" / "v1_2" / "tp10_target_xgb_v1_2.pkl"
DATA_PATH = (
    PROJECT_DIR
    / "3_feature_engineering"
    / "v1_2"
    / "outputs"
    / "master_data_v1_2_final.parquet"
)
RESULTS_DIR = PROJECT_DIR / "5_analysis" / "v1_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

STARTING_PORTFOLIO = 10_000.0
BP_PER_CONTRACT = 2_500.0  # 25-wide wings
QTY = 1
FEES_RT = 5.20  # round-trip per contract (4 legs * $1.30)
FEES_PER_SHARE = FEES_RT / 100.0  # $0.052
SLIPPAGES = [0.10, 0.20, 0.30]  # per share exit slippage
SKIP_RATES = [round(x, 2) for x in np.arange(0.20, 0.41, 0.01)]
SHADOW_STRATEGY = "IC_05d_25w"


# ── Load model ───────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model: {MODEL_PATH.name}")
    with open(MODEL_PATH, "rb") as f:
        pkg = pickle.load(f)
    print(f"  Features: {len(pkg['feature_cols'])}")
    print(f"  Split: {pkg['split_info']}")
    return pkg


# ── Load & filter holdout data ───────────────────────────────────────────────

def load_holdout(pkg):
    print(f"\nLoading data: {DATA_PATH.name}")
    df = pd.read_parquet(DATA_PATH)
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "decision_datetime", "strategy"]).reset_index(
        drop=True
    )

    # Parse holdout range from split_info
    holdout_range = pkg["split_info"]["holdout_range"]
    start_str, end_str = holdout_range.split(" -> ")
    holdout_start = pd.Timestamp(start_str)
    holdout_end = pd.Timestamp(end_str)

    holdout = df[(df["date"] >= holdout_start) & (df["date"] <= holdout_end)].copy()
    holdout = holdout.reset_index(drop=True)

    print(f"  Full data: {len(df):,} rows")
    print(f"  Holdout:   {len(holdout):,} rows")
    print(f"  Holdout range: {holdout['date'].min().date()} -> {holdout['date'].max().date()}")
    print(f"  Holdout days: {holdout['date'].nunique()}")
    print(f"  Strategies: {sorted(holdout['strategy'].unique())}")

    return holdout


# ── Score holdout ────────────────────────────────────────────────────────────

def score_holdout(holdout, pkg):
    print("\nScoring holdout with model...")
    model = pkg["model"]
    feature_cols = pkg["feature_cols"]
    X = holdout[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    holdout = holdout.copy()
    holdout["prob"] = probs
    print(f"  Prob stats: mean={probs.mean():.4f}, std={probs.std():.4f}, "
          f"min={probs.min():.4f}, max={probs.max():.4f}")
    return holdout


# ── Compute per-strategy skip cutoffs ────────────────────────────────────────

def compute_skip_cutoffs(holdout, skip_rates):
    """For each strategy, compute the probability threshold at each skip rate.

    At skip_rate=0.30, the bottom 30% of that strategy's probability
    distribution are skipped. The cutoff is the quantile value.
    """
    cutoffs = {}
    strategies = sorted(holdout["strategy"].unique())
    for strat in strategies:
        strat_probs = holdout.loc[holdout["strategy"] == strat, "prob"]
        for sr in skip_rates:
            cutoffs[(strat, sr)] = float(strat_probs.quantile(sr))
    return cutoffs


# ── Run single backtest config ───────────────────────────────────────────────

def run_backtest(holdout, skip_cutoffs, slip, skip_rate):
    """Run one backtest configuration.

    Returns (summary_dict, trade_log_list).
    """
    portfolio = STARTING_PORTFOLIO
    peak = portfolio
    max_dd = 0.0
    daily_returns = []

    trades = []
    dates = sorted(holdout["date"].unique())

    for day in dates:
        day_df = holdout[holdout["date"] == day].copy()
        minutes = sorted(day_df["decision_datetime"].unique())

        day_start_portfolio = portfolio
        in_trade = False
        done_for_day = False
        trade_exit_time = None

        for minute in minutes:
            if done_for_day:
                break

            # If we're in a trade, wait for exit
            if in_trade:
                if minute < trade_exit_time:
                    continue
                # Trade has exited, we can look for next entry
                in_trade = False
                trade_exit_time = None
                # After a loss, done for the day (rule 9)
                # (This was already handled when the trade was opened)

            # Get all candidates at this minute
            candidates = day_df[day_df["decision_datetime"] == minute].copy()
            if candidates.empty:
                continue

            # Apply per-strategy skip cutoff: remove candidates below their
            # strategy-specific probability threshold
            mask = pd.Series(True, index=candidates.index)
            for idx, row in candidates.iterrows():
                cutoff = skip_cutoffs.get((row["strategy"], skip_rate), 0.0)
                if row["prob"] < cutoff:
                    mask[idx] = False
            candidates = candidates[mask]

            if candidates.empty:
                continue

            # Pick highest probability candidate
            best_idx = candidates["prob"].idxmax()
            best = candidates.loc[best_idx]

            # Shadow 5-delta check
            is_shadow = best["strategy"] == SHADOW_STRATEGY

            # Parse exit time
            exit_time = pd.Timestamp(best["tp10_exit_time"])
            exit_reason = best["tp10_exit_reason"]
            exit_debit = best["tp10_exit_debit"]
            credit = best["credit"]

            if is_shadow:
                # Shadow trade: $0 PnL, but lock BP through exit lifecycle
                pnl_per_share = 0.0
                pnl_total = 0.0
                trades.append({
                    "date": day,
                    "entry_time": minute,
                    "exit_time": exit_time,
                    "strategy": best["strategy"],
                    "credit": credit,
                    "exit_debit": exit_debit,
                    "exit_reason": exit_reason,
                    "prob": best["prob"],
                    "slip": slip,
                    "skip_rate": skip_rate,
                    "pnl_per_share": pnl_per_share,
                    "pnl_total": pnl_total,
                    "portfolio_after": portfolio,
                    "is_shadow": True,
                })
                # Lock through exit lifecycle
                in_trade = True
                trade_exit_time = exit_time
                # Shadow trades follow same exit lifecycle rules
                if exit_reason in ("sl", "close_loss"):
                    done_for_day = True
            else:
                # Real trade
                pnl_per_share = credit - exit_debit - slip - FEES_PER_SHARE
                pnl_total = pnl_per_share * 100 * QTY

                portfolio += pnl_total
                peak = max(peak, portfolio)
                dd = (peak - portfolio) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

                trades.append({
                    "date": day,
                    "entry_time": minute,
                    "exit_time": exit_time,
                    "strategy": best["strategy"],
                    "credit": credit,
                    "exit_debit": exit_debit,
                    "exit_reason": exit_reason,
                    "prob": best["prob"],
                    "slip": slip,
                    "skip_rate": skip_rate,
                    "pnl_per_share": pnl_per_share,
                    "pnl_total": pnl_total,
                    "portfolio_after": portfolio,
                    "is_shadow": False,
                })

                in_trade = True
                trade_exit_time = exit_time

                # After loss -> done for day
                if exit_reason in ("sl", "close_loss"):
                    done_for_day = True

        # Daily return
        day_return = (portfolio - day_start_portfolio) / day_start_portfolio if day_start_portfolio > 0 else 0.0
        daily_returns.append(day_return)

    # Compute summary
    trade_df = pd.DataFrame(trades)
    real_trades = trade_df[~trade_df["is_shadow"]] if len(trade_df) > 0 else trade_df
    shadow_trades = trade_df[trade_df["is_shadow"]] if len(trade_df) > 0 else trade_df

    n_trades = len(real_trades)
    n_shadow = len(shadow_trades)
    n_wins = int((real_trades["pnl_total"] > 0).sum()) if n_trades > 0 else 0
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    total_return = (portfolio - STARTING_PORTFOLIO) / STARTING_PORTFOLIO
    avg_pnl = real_trades["pnl_total"].mean() if n_trades > 0 else 0.0

    # Sharpe from daily returns
    daily_returns = np.array(daily_returns)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    summary = {
        "slip": slip,
        "skip_rate": skip_rate,
        "trades": n_trades,
        "shadow_trades": n_shadow,
        "wins": n_wins,
        "win_rate": round(win_rate, 4),
        "total_return_pct": round(total_return * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "final_portfolio": round(portfolio, 2),
        "avg_pnl": round(avg_pnl, 2),
    }

    return summary, trades


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 80)
    print("ZDOM V1.2 BACKTEST — Shadow 5-Delta Strategy")
    print("=" * 80)

    # Load
    pkg = load_model()
    holdout = load_holdout(pkg)
    holdout = score_holdout(holdout, pkg)

    # Skip cutoffs
    print("\nComputing per-strategy skip cutoffs...")
    skip_cutoffs = compute_skip_cutoffs(holdout, SKIP_RATES)
    strategies = sorted(holdout["strategy"].unique())
    print(f"  Sample cutoffs at skip_rate=0.30:")
    for s in strategies:
        print(f"    {s}: {skip_cutoffs[(s, 0.30)]:.4f}")

    # Run all configurations
    n_configs = len(SLIPPAGES) * len(SKIP_RATES)
    print(f"\nRunning {n_configs} configurations ({len(SLIPPAGES)} slips x {len(SKIP_RATES)} skip rates)...")
    print()

    results = []
    best_config = None
    best_sharpe = -999
    best_trades = None
    config_num = 0

    for slip in SLIPPAGES:
        for skip_rate in SKIP_RATES:
            config_num += 1
            summary, trade_log = run_backtest(holdout, skip_cutoffs, slip, skip_rate)
            results.append(summary)

            # Track best by Sharpe
            if summary["sharpe"] > best_sharpe:
                best_sharpe = summary["sharpe"]
                best_config = summary
                best_trades = trade_log

            # Progress
            if config_num % 7 == 0 or config_num == n_configs:
                elapsed = time.time() - t0
                print(
                    f"  [{config_num:>3}/{n_configs}] "
                    f"slip={slip:.2f} skip={skip_rate:.2f} -> "
                    f"trades={summary['trades']:>3} shadow={summary['shadow_trades']:>3} "
                    f"wr={summary['win_rate']:.1%} "
                    f"ret={summary['total_return_pct']:>+7.1f}% "
                    f"dd={summary['max_dd_pct']:>5.1f}% "
                    f"sharpe={summary['sharpe']:>5.2f} "
                    f"({elapsed:.0f}s)"
                )
                sys.stdout.flush()

    # Summary table
    summary_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")

    # Print grouped by slippage
    for slip in SLIPPAGES:
        print(f"\n--- Slippage: ${slip:.2f}/share ---")
        print(
            f"{'Skip%':>6} {'Trades':>6} {'Shadow':>6} {'Wins':>5} "
            f"{'WinR':>6} {'TotRet%':>8} {'MaxDD%':>7} {'Sharpe':>6} "
            f"{'Final$':>9} {'AvgPnL':>8}"
        )
        print("-" * 80)
        slip_rows = summary_df[summary_df["slip"] == slip].sort_values("skip_rate")
        for _, r in slip_rows.iterrows():
            print(
                f"{r['skip_rate']:>5.0%} {r['trades']:>6} {r['shadow_trades']:>6} "
                f"{r['wins']:>5} {r['win_rate']:>5.1%} {r['total_return_pct']:>+7.1f}% "
                f"{r['max_dd_pct']:>6.1f}% {r['sharpe']:>6.2f} "
                f"${r['final_portfolio']:>8.0f} ${r['avg_pnl']:>7.2f}"
            )

    # Best config
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION (by Sharpe)")
    print(f"{'='*80}")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    # Save summary CSV
    summary_path = RESULTS_DIR / "backtest_shadow5d_v1_2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Save best trade log
    if best_trades:
        trades_df = pd.DataFrame(best_trades)
        trades_path = RESULTS_DIR / "backtest_shadow5d_v1_2_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Saved trade log ({len(trades_df)} rows): {trades_path}")

        # Quick trade log stats
        real = trades_df[~trades_df["is_shadow"]]
        shadow = trades_df[trades_df["is_shadow"]]
        print(f"\n  Trade log stats (best config):")
        print(f"    Real trades: {len(real)}")
        print(f"    Shadow trades: {len(shadow)}")
        if len(real) > 0:
            print(f"    Strategy mix:")
            for s, cnt in real["strategy"].value_counts().items():
                print(f"      {s}: {cnt}")
            print(f"    Exit reasons:")
            for r, cnt in real["exit_reason"].value_counts().items():
                print(f"      {r}: {cnt}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
