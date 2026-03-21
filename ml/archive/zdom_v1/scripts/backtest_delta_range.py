"""
ZDOM V1 Walk-Forward Backtest — Delta-Filtered (10d-30d only).

Runs the same simulation grid as backtest_walkforward.py but restricted to
IC_10d_25w through IC_30d_25w (5 strategies). Adds reward/risk ratio analysis.

Grid: 2 strategies × 4 slippage levels × 5 skip rates = 40 simulations

Usage:
  python3 scripts/backtest_delta_range.py                  # full sim
  python3 scripts/backtest_delta_range.py --no-tune        # skip Optuna
"""

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models" / "v1"
OUTPUT_DIR = PROJECT_DIR / "output" / "backtest_delta_range"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

# Only 10-30 delta strategies
DELTA_RANGE = [10, 15, 20, 25, 30]
STRATEGIES = [f"IC_{d:02d}d_25w" for d in DELTA_RANGE]
TP_LEVELS = [f"tp{p}" for p in range(10, 55, 5)]
TARGETS = [f"{tp}_target" for tp in TP_LEVELS]

TRAIN_PCT = 0.70
GAP_DAYS = 7
BUYING_POWER_PER_CONTRACT = 2500
FEES_PER_SHARE = 0.052
START_PORTFOLIO = 10_000
WING_WIDTH = 25.0  # $25 wings
SL_MULT = 2.0

EXIT_SLIPS = [0.00, 0.10, 0.20, 0.30]
SKIP_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]

# Meta columns (not features)
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

COLORS = {
    1: "#2196F3",
    2: "#FF9800",
}

SLIP_LABELS = {
    0.00: ("S1", "Perfect Mid Fills"),
    0.10: ("S2", "Small Exit Slip"),
    0.20: ("S3", "Moderate Exit Slip"),
    0.30: ("S4", "Worst Case"),
}


# ── Import simulation engine from existing backtest ─────────────────────────

# We reuse the core simulation functions but with our filtered STRATEGIES
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from backtest_walkforward import (
    TradeRecord,
    Position,
    simulate_strategy1 as _sim_s1_orig,
    simulate_strategy2 as _sim_s2_orig,
    _get_candidates,
    compute_ev,
    _compute_summary,
)


# ── Data Loading & Splitting ────────────────────────────────────────────────

def load_model_table():
    """Load model_table parts, filter to 10-30 delta strategies."""
    parts = sorted(DATA_DIR.glob("model_table_v1_part*.parquet"))
    if not parts:
        single = DATA_DIR / "model_table_v1.parquet"
        if single.exists():
            df = pd.read_parquet(single)
        else:
            print("[error] No model_table files found.")
            sys.exit(1)
    else:
        print(f"Loading {len(parts)} model table parts...")
        dfs = [pd.read_parquet(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
    print(f"  Total: {len(df):,} rows × {df.shape[1]} cols, {df['date'].nunique()} days")

    # Filter to 10-30 delta only
    df = df[df["strategy"].isin(STRATEGIES)].reset_index(drop=True)
    print(f"  After delta filter (10-30): {len(df):,} rows, "
          f"{df['strategy'].nunique()} strategies: {sorted(df['strategy'].unique())}")

    return df


def apply_blockers(df):
    """Apply hard blockers."""
    blocked = (
        (df["days_to_next_fomc"] == 0) |
        (df["days_to_next_fomc"] == 1) |
        (df["is_cpi_day"] == 1) |
        (df["is_ppi_day"] == 1) |
        (df["is_nfp_day"] == 1) |
        (df["is_gdp_day"] == 1) |
        (df["is_mag7_earnings_day"] == 1) |
        (df["vix_close"] > 35) |
        (df["gap_pct"].abs() > 1.5)
    ).fillna(False)
    df["_blocked"] = blocked
    n_blocked = blocked.sum()
    print(f"  Hard blockers: {n_blocked:,} rows blocked ({df[blocked]['date'].nunique()} days)")
    return df


def split_data(df):
    """Train/test/holdout split."""
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * TRAIN_PCT)
    remaining = n_dates - train_end_idx - 2 * GAP_DAYS
    test_days = remaining // 2
    test_end_idx = train_end_idx + GAP_DAYS + test_days
    holdout_start_idx = test_end_idx + GAP_DAYS

    train_dates = set(dates[:train_end_idx])
    test_dates = set(dates[train_end_idx + GAP_DAYS:test_end_idx])
    holdout_dates = set(dates[holdout_start_idx:])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    holdout_df = df[df["date"].isin(holdout_dates)].reset_index(drop=True)

    print(f"  Train:   {len(train_df):>10,} rows ({len(train_dates):>3d} days)  "
          f"{min(train_dates).date()} -> {max(train_dates).date()}")
    print(f"  Test:    {len(test_df):>10,} rows ({len(test_dates):>3d} days)")
    print(f"  Holdout: {len(holdout_df):>10,} rows ({len(holdout_dates):>3d} days)  "
          f"{min(holdout_dates).date()} -> {max(holdout_dates).date()}")

    return train_df, test_df, holdout_df


def get_feature_cols(df):
    """Get numeric feature columns (exclude meta)."""
    all_feat = [c for c in df.columns if c not in META_COLS]
    return [c for c in all_feat if pd.api.types.is_numeric_dtype(df[c])]


# ── EV & Cutoff Pre-computation ─────────────────────────────────────────────

def precompute_ev_lookup(train_df):
    """Compute avg win/loss PnL per (strategy, TP) from training data."""
    ev_lookup = {}
    for strat in STRATEGIES:
        strat_df = train_df[train_df["strategy"] == strat]
        if len(strat_df) == 0:
            continue
        for tp in TP_LEVELS:
            target = f"{tp}_target"
            pnl_col = f"{tp}_pnl"
            if target not in strat_df.columns:
                continue
            wins = strat_df[strat_df[target] == 1]
            losses = strat_df[strat_df[target] == 0]
            if len(wins) == 0 or len(losses) == 0:
                continue
            ev_lookup[(strat, tp)] = (wins[pnl_col].mean(), losses[pnl_col].mean(),
                                       len(wins) / len(strat_df))
    print(f"  EV lookup: {len(ev_lookup)} (strategy, TP) combos")
    return ev_lookup


def precompute_skip_cutoffs(holdout_scored, skip_rates):
    """Compute probability cutoffs per (strategy, tp, skip_rate)."""
    cutoffs = {}
    for strat in STRATEGIES:
        strat_df = holdout_scored[holdout_scored["strategy"] == strat]
        if len(strat_df) == 0:
            continue
        for tp in TP_LEVELS:
            prob_col = f"prob_{tp}"
            if prob_col not in strat_df.columns:
                continue
            probs = strat_df[prob_col].values
            for sr in skip_rates:
                cutoffs[(strat, tp, sr)] = np.percentile(probs, sr * 100)
    return cutoffs


# ── Holdout Scoring ─────────────────────────────────────────────────────────

def score_holdout(holdout_df, models, feature_cols):
    """Score all holdout rows."""
    X = holdout_df[feature_cols]
    scored = holdout_df.copy()
    for tp, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        scored[f"prob_{tp}"] = probs
        print(f"  Scored {tp}: mean={probs.mean():.3f}, std={probs.std():.3f}")
    return scored


# ── Simulation Engine ───────────────────────────────────────────────────────

def simulate(holdout_scored, strategy_type, exit_slip, skip_rate,
             ev_lookup, cutoffs, start_portfolio=START_PORTFOLIO):
    """Run full simulation across all holdout days."""
    portfolio = start_portfolio
    all_trades = []
    equity_curve = [{"date": "start", "portfolio": portfolio}]

    holdout_days = sorted(holdout_scored["date"].unique())

    for day in holdout_days:
        day_df = holdout_scored[holdout_scored["date"] == day]

        if strategy_type == 1:
            day_trades, portfolio = _sim_s1_orig(
                day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                portfolio, "joint",
            )
        else:
            day_trades, portfolio = _sim_s2_orig(
                day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                portfolio, "joint",
            )

        all_trades.extend(day_trades)
        equity_curve.append({"date": str(day.date()), "portfolio": portfolio})

    summary = _compute_summary(
        equity_curve, all_trades, strategy_type, exit_slip,
        skip_rate, "joint", start_portfolio,
    )

    return equity_curve, all_trades, summary


# ── Reward/Risk Analysis ────────────────────────────────────────────────────

def compute_reward_risk(train_df):
    """Compute reward/risk metrics per strategy (delta level).

    Returns DataFrame with:
      - avg_credit: average credit received per share
      - avg_max_loss: average worst-case loss (SL hit = 2x credit)
      - credit_vs_max_loss: ratio of credit to max loss
      - credit_vs_width: credit as % of wing width ($25)
      - avg_short_call_delta, avg_short_put_delta
      - win_rates per TP level
    """
    rows = []
    for strat in STRATEGIES:
        strat_df = train_df[train_df["strategy"] == strat]
        if len(strat_df) == 0:
            continue

        delta_num = int(strat.split("_")[1].replace("d", ""))
        credit = strat_df["credit"].mean()
        max_loss_per_share = credit * SL_MULT  # SL at 2x credit
        actual_max_loss = WING_WIDTH - credit   # true max loss = width - credit

        row = {
            "strategy": strat,
            "delta": delta_num,
            "n_samples": len(strat_df),
            "avg_credit": round(credit, 3),
            "median_credit": round(strat_df["credit"].median(), 3),
            "std_credit": round(strat_df["credit"].std(), 3),
            "avg_sl_loss": round(max_loss_per_share, 3),
            "avg_actual_max_loss": round(actual_max_loss, 3),
            "credit_vs_sl_loss": round(credit / max_loss_per_share, 4) if max_loss_per_share > 0 else 0,
            "credit_vs_width_pct": round(credit / WING_WIDTH * 100, 2),
            "credit_vs_actual_max": round(credit / actual_max_loss, 4) if actual_max_loss > 0 else 0,
        }

        # Win rates per TP level
        for tp in TP_LEVELS:
            target = f"{tp}_target"
            if target in strat_df.columns:
                wr = strat_df[target].mean() * 100
                row[f"{tp}_win_rate"] = round(wr, 1)

                # Avg PnL per TP level
                pnl_col = f"{tp}_pnl"
                if pnl_col in strat_df.columns:
                    row[f"{tp}_avg_pnl"] = round(strat_df[pnl_col].mean(), 3)

        # Average deltas
        if "sc_delta" in strat_df.columns:
            row["avg_sc_delta"] = round(strat_df["sc_delta"].mean(), 4)
        if "sp_delta" in strat_df.columns:
            row["avg_sp_delta"] = round(strat_df["sp_delta"].mean(), 4)

        rows.append(row)

    rr_df = pd.DataFrame(rows)
    return rr_df


# ── PDF Generation ──────────────────────────────────────────────────────────

def render_summary_page(fig, summary_df, slip, slip_label, slip_desc, holdout_info):
    """Page 1: Summary comparison table."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    title = f"ZDOM V1 Backtest (10-30 Delta) -- Scenario {slip_label}: {slip_desc}"
    subtitle = (f"Exit Slippage: ${slip:.2f}/share  |  Fees: $0.052/share  |  "
                f"Starting Portfolio: $10,000  |  Strategies: {', '.join(STRATEGIES)}")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.93, subtitle, transform=ax.transAxes, fontsize=8,
            ha="center", va="top", color="#666666")
    ax.text(0.5, 0.905, holdout_info, transform=ax.transAxes, fontsize=8,
            ha="center", va="top", color="#999999")

    headers = ["Skip Rate", "Return %", "Max DD %", "Sharpe", "Win Rate",
               "# Trades", "T/Day", "Final $", "Avg PnL"]

    for strat_idx, strat in enumerate([1, 2]):
        s_data = summary_df[summary_df["strategy"] == strat].sort_values("skip_rate")
        strat_name = ("Strategy 1: Best EV, Max Contracts" if strat == 1
                      else "Strategy 2: Diversified, Max BP")
        y_title = 0.86 if strat == 1 else 0.48
        y_table = [0.52, 0.32] if strat == 1 else [0.14, 0.32]

        ax.text(0.5, y_title, strat_name, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="center", va="top", color=COLORS[strat])

        rows = []
        for _, r in s_data.iterrows():
            rows.append([
                f"{r['skip_rate']:.0%}",
                f"{r['total_return_pct']:+.1f}%",
                f"{r['max_drawdown_pct']:.1f}%",
                f"{r['sharpe']:.2f}",
                f"{r['win_rate']:.1f}%",
                f"{r['n_trades']:,.0f}",
                f"{r['trades_per_day']:.1f}",
                f"${r['final_portfolio']:,.0f}",
                f"${r['avg_pnl']:+,.1f}",
            ])

        if rows:
            t = ax.table(cellText=rows, colLabels=headers,
                         loc="center", bbox=[0.02, y_table[0], 0.96, y_table[1]])
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            header_color = "#E3F2FD" if strat == 1 else "#FFF3E0"
            highlight_color = "#BBDEFB" if strat == 1 else "#FFE0B2"
            for (i, j), cell in t.get_celld().items():
                if i == 0:
                    cell.set_facecolor(header_color)
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#CCCCCC")
            # Highlight best Sharpe
            if len(rows) > 0:
                best_idx = s_data["sharpe"].idxmax()
                best_row = s_data.index.get_loc(best_idx) + 1
                for j in range(len(headers)):
                    t.get_celld()[(best_row, j)].set_facecolor(highlight_color)

    ax.text(0.5, 0.08, "Highlighted rows = best Sharpe ratio per strategy",
            transform=ax.transAxes, fontsize=8, ha="center", va="top", color="#999999")


def render_reward_risk_page(fig, rr_df):
    """Page 2: Reward/Risk ratio analysis per delta level."""
    # 2 rows, 3 columns of charts
    axes = fig.subplots(2, 3)
    fig.suptitle("Reward / Risk Analysis by Delta Level (Training Data)",
                 fontsize=14, fontweight="bold")

    deltas = rr_df["delta"].values
    x = np.arange(len(deltas))
    labels = [f"{d}d" for d in deltas]
    bar_color = "#4CAF50"

    # Chart 1: Average credit received
    ax = axes[0][0]
    ax.bar(x, rr_df["avg_credit"], color=bar_color, alpha=0.8)
    ax.set_title("Avg Credit Received ($/share)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for i, v in enumerate(rr_df["avg_credit"]):
        ax.text(i, v + 0.05, f"${v:.2f}", ha="center", fontsize=8)

    # Chart 2: Credit vs Width (%)
    ax = axes[0][1]
    ax.bar(x, rr_df["credit_vs_width_pct"], color="#FF9800", alpha=0.8)
    ax.set_title("Credit as % of $25 Width", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    for i, v in enumerate(rr_df["credit_vs_width_pct"]):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)

    # Chart 3: Reward/Risk ratio (credit / actual max loss)
    ax = axes[0][2]
    ax.bar(x, rr_df["credit_vs_actual_max"], color="#2196F3", alpha=0.8)
    ax.set_title("Credit / Max Loss Ratio", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for i, v in enumerate(rr_df["credit_vs_actual_max"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    # Chart 4: Win rates across TP levels (grouped bar)
    ax = axes[1][0]
    tp_subset = ["tp10", "tp25", "tp50"]
    tp_colors = ["#4CAF50", "#FF9800", "#F44336"]
    width = 0.25
    for j, tp in enumerate(tp_subset):
        col = f"{tp}_win_rate"
        if col in rr_df.columns:
            vals = rr_df[col].values
            ax.bar(x + (j - 1) * width, vals, width, label=tp.upper(),
                   color=tp_colors[j], alpha=0.8)
    ax.set_title("Win Rate by TP Level (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.legend(fontsize=7)

    # Chart 5: Avg PnL per TP level
    ax = axes[1][1]
    for j, tp in enumerate(tp_subset):
        col = f"{tp}_avg_pnl"
        if col in rr_df.columns:
            vals = rr_df[col].values
            ax.bar(x + (j - 1) * width, vals, width, label=tp.upper(),
                   color=tp_colors[j], alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("Avg PnL/Share by TP Level ($)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$/share")
    ax.legend(fontsize=7)

    # Chart 6: Summary table
    ax = axes[1][2]
    ax.axis("off")
    table_data = []
    for _, r in rr_df.iterrows():
        table_data.append([
            f"IC_{r['delta']:02d}d",
            f"${r['avg_credit']:.2f}",
            f"${r['avg_actual_max_loss']:.2f}",
            f"{r['credit_vs_width_pct']:.1f}%",
            f"{r['credit_vs_actual_max']:.3f}",
            f"{r['n_samples']:,}",
        ])
    col_labels = ["Strategy", "Avg Credit", "Max Loss", "Credit/Width", "R/R Ratio", "N"]
    t = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                 bbox=[0.0, 0.1, 1.0, 0.85])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    for (i, j), cell in t.get_celld().items():
        if i == 0:
            cell.set_facecolor("#E8F5E9")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Reward/Risk Summary", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_detailed_stats_page(fig, trades_df, summary_df, slip):
    """Page 3: Detailed trade statistics."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.97, f"Detailed Trade Statistics -- Exit Slip ${slip:.2f}",
            transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")

    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
        if s_summary.empty:
            continue
        best_skip = s_summary.loc[s_summary["sharpe"].idxmax(), "skip_rate"]

        key = f"S{strat}_slip{slip:.2f}_skip{best_skip:.2f}"
        strat_trades = trades_df[trades_df["sim_key"] == key]
        stats = _compute_detailed_stats(strat_trades)
        if not stats:
            continue

        x_offset = 0.02 + strat_idx * 0.50
        color = COLORS[strat]
        strat_name = "S1: Best EV, Max Contracts" if strat == 1 else "S2: Diversified, Max BP"

        ax.text(x_offset + 0.22, 0.92, f"{strat_name}\n(skip={best_skip:.0%})",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="center", va="top", color=color)

        metrics = [
            ("Trade Count", ""),
            ("  Total trades", f"{stats['n_trades']:,}"),
            ("  Wins", f"{stats['n_wins']:,}"),
            ("  Losses", f"{stats['n_losses']:,}"),
            ("  Win rate", f"{stats['win_rate']:.1f}%"),
            ("", ""),
            ("P&L Metrics", ""),
            ("  Total P&L", f"${stats['total_pnl']:+,.0f}"),
            ("  Avg P&L/trade", f"${stats['avg_pnl']:+,.1f}"),
            ("  Median P&L/trade", f"${stats['median_pnl']:+,.1f}"),
            ("  Avg win", f"${stats['avg_win']:+,.1f}"),
            ("  Avg loss", f"${stats['avg_loss']:+,.1f}"),
            ("  Max win", f"${stats['max_win']:+,.0f}"),
            ("  Max loss", f"${stats['max_loss']:+,.0f}"),
            ("  Profit factor", f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "INF"),
            ("", ""),
            ("Streaks & Daily", ""),
            ("  Max win streak", f"{stats['max_win_streak']}"),
            ("  Max loss streak", f"{stats['max_loss_streak']}"),
            ("  Winning days", f"{stats['winning_days']}"),
            ("  Losing days", f"{stats['losing_days']}"),
            ("  Daily win rate", f"{stats['daily_win_rate']:.1f}%"),
            ("  Avg daily P&L", f"${stats['avg_daily_pnl']:+,.1f}"),
            ("  Avg trades/day", f"{stats['avg_trades_per_day']:.1f}"),
            ("", ""),
            ("Exit Reasons", ""),
            ("  TP hit", f"{stats['tp_exits']:,}"),
            ("  SL hit", f"{stats['sl_exits']:,}"),
            ("  Close (win)", f"{stats['close_win_exits']:,}"),
            ("  Close (loss)", f"{stats['close_loss_exits']:,}"),
            ("", ""),
            ("Position Size", ""),
            ("  Avg contracts", f"{stats['avg_qty']:.1f}"),
        ]

        y = 0.85
        for label, value in metrics:
            if label == "" and value == "":
                y -= 0.008
                continue
            if value == "":
                ax.text(x_offset + 0.02, y, label, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="top")
            else:
                ax.text(x_offset + 0.02, y, label, transform=ax.transAxes,
                        fontsize=8, va="top", color="#333333")
                ax.text(x_offset + 0.40, y, value, transform=ax.transAxes,
                        fontsize=8, va="top", ha="right", fontfamily="monospace")
            y -= 0.022

    ax.plot([0.50, 0.50], [0.05, 0.90], transform=ax.transAxes,
            color="#CCCCCC", linewidth=1, clip_on=False)


def _compute_detailed_stats(trades_df):
    """Compute detailed statistics from trade log."""
    if trades_df.empty:
        return {}

    wins = trades_df[trades_df["pnl_total"] > 0]
    losses = trades_df[trades_df["pnl_total"] <= 0]

    streaks = []
    current = 0
    for pnl in trades_df["pnl_total"]:
        if pnl > 0:
            current = max(current, 0) + 1
        else:
            current = min(current, 0) - 1
        streaks.append(current)

    win_streaks = [s for s in streaks if s > 0]
    loss_streaks = [-s for s in streaks if s < 0]

    daily = trades_df.groupby("day").agg(
        daily_pnl=("pnl_total", "sum"),
        n_trades=("pnl_total", "count"),
    )
    winning_days = (daily["daily_pnl"] > 0).sum()
    losing_days = (daily["daily_pnl"] <= 0).sum()

    return {
        "n_trades": len(trades_df),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "total_pnl": trades_df["pnl_total"].sum(),
        "avg_pnl": trades_df["pnl_total"].mean(),
        "median_pnl": trades_df["pnl_total"].median(),
        "avg_win": wins["pnl_total"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl_total"].mean() if len(losses) > 0 else 0,
        "max_win": trades_df["pnl_total"].max(),
        "max_loss": trades_df["pnl_total"].min(),
        "profit_factor": (wins["pnl_total"].sum() / abs(losses["pnl_total"].sum())
                          if len(losses) > 0 and losses["pnl_total"].sum() != 0 else float("inf")),
        "avg_qty": trades_df["qty"].mean(),
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "daily_win_rate": winning_days / (winning_days + losing_days) * 100 if (winning_days + losing_days) > 0 else 0,
        "avg_daily_pnl": daily["daily_pnl"].mean(),
        "avg_trades_per_day": daily["n_trades"].mean(),
        "tp_exits": (trades_df["exit_reason"] == "tp").sum(),
        "sl_exits": (trades_df["exit_reason"] == "sl").sum(),
        "close_win_exits": (trades_df["exit_reason"] == "close_win").sum(),
        "close_loss_exits": (trades_df["exit_reason"] == "close_loss").sum(),
    }


def render_equity_curves_page(fig, equity_df, slip):
    """Page 4: Equity curves."""
    axes = fig.subplots(2, 1, sharex=True)

    for strat_idx, strat in enumerate([1, 2]):
        ax = axes[strat_idx]
        strat_name = ("Strategy 1: Best EV, Max Contracts" if strat == 1
                      else "Strategy 2: Diversified, Max BP")

        for sr in SKIP_RATES:
            key = f"S{strat}_slip{slip:.2f}_skip{sr:.2f}"
            eq = equity_df[equity_df["sim_key"] == key].copy()
            if eq.empty:
                continue
            eq = eq[eq["date"] != "start"].copy()
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.sort_values("date")

            alpha = 0.4 + (sr / 0.40) * 0.6
            ax.plot(eq["date"], eq["portfolio"], label=f"Skip {sr:.0%}",
                    alpha=alpha, linewidth=1.2, color=COLORS[strat])

        ax.set_title(strat_name, fontsize=11, fontweight="bold", color=COLORS[strat])
        ax.set_ylabel("Portfolio ($)")
        ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    axes[1].set_xlabel("Date")
    fig.suptitle(f"Equity Curves (10-30 Delta) -- Exit Slip ${slip:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def render_distribution_page(fig, trades_df, summary_df, slip):
    """Page 5: PnL distribution and exit reasons."""
    axes = fig.subplots(2, 2)

    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
        if s_summary.empty:
            continue
        best_skip = s_summary.loc[s_summary["sharpe"].idxmax(), "skip_rate"]
        key = f"S{strat}_slip{slip:.2f}_skip{best_skip:.2f}"
        strat_trades = trades_df[trades_df["sim_key"] == key]

        if strat_trades.empty:
            continue

        color = COLORS[strat]
        strat_label = f"S{strat} (skip={best_skip:.0%})"

        # PnL histogram
        ax_hist = axes[0][strat_idx]
        pnls = strat_trades["pnl_total"].values
        bins = np.linspace(max(pnls.min(), -5000), min(pnls.max(), 10000), 50)
        ax_hist.hist(pnls, bins=bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax_hist.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_hist.axvline(x=np.mean(pnls), color="black", linestyle="-", alpha=0.5, linewidth=0.8)
        ax_hist.set_title(f"{strat_label} -- P&L Distribution", fontsize=9, fontweight="bold")
        ax_hist.set_xlabel("P&L ($)")
        ax_hist.set_ylabel("Count")
        ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Exit reason pie
        ax_pie = axes[1][strat_idx]
        reasons = strat_trades["exit_reason"].value_counts()
        reason_colors = {
            "tp": "#4CAF50", "sl": "#F44336",
            "close_win": "#8BC34A", "close_loss": "#FF9800",
        }
        pie_colors = [reason_colors.get(r, "#9E9E9E") for r in reasons.index]
        labels = [f"{r}\n({v:,})" for r, v in reasons.items()]
        ax_pie.pie(reasons.values, labels=labels, colors=pie_colors,
                   autopct="%1.0f%%", textprops={"fontsize": 8})
        ax_pie.set_title(f"{strat_label} -- Exit Reasons", fontsize=9, fontweight="bold")

    fig.suptitle(f"Trade Analysis (10-30 Delta) -- Exit Slip ${slip:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_composition_page(fig, trades_df, summary_df, slip):
    """Page 6: Which IC deltas and TP levels were selected."""
    axes = fig.subplots(2, 2)

    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
        if s_summary.empty:
            continue
        best_skip = s_summary.loc[s_summary["sharpe"].idxmax(), "skip_rate"]
        key = f"S{strat}_slip{slip:.2f}_skip{best_skip:.2f}"
        strat_trades = trades_df[trades_df["sim_key"] == key]

        if strat_trades.empty:
            continue

        color = COLORS[strat]
        strat_label = f"S{strat} (skip={best_skip:.0%})"

        # IC strategy breakdown
        ax_strat = axes[0][strat_idx]
        strat_counts = strat_trades["strategy"].value_counts().reindex(STRATEGIES).fillna(0).astype(int)
        bars = ax_strat.barh(range(len(strat_counts)), strat_counts.values, color=color, alpha=0.7)
        ax_strat.set_yticks(range(len(strat_counts)))
        ax_strat.set_yticklabels(strat_counts.index, fontsize=7)
        ax_strat.set_xlabel("# Trades")
        ax_strat.set_title(f"{strat_label} -- IC Delta Distribution", fontsize=9, fontweight="bold")
        for i, v in enumerate(strat_counts.values):
            if v > 0:
                ax_strat.text(v + 1, i, str(v), va="center", fontsize=7)

        # TP level breakdown
        ax_tp = axes[1][strat_idx]
        tp_order = [f"tp{p}" for p in range(10, 55, 5)]
        tp_counts = strat_trades["tp_level"].value_counts().reindex(tp_order).fillna(0).astype(int)
        bars = ax_tp.barh(range(len(tp_counts)), tp_counts.values, color=color, alpha=0.7)
        ax_tp.set_yticks(range(len(tp_counts)))
        ax_tp.set_yticklabels(tp_counts.index, fontsize=7)
        ax_tp.set_xlabel("# Trades")
        ax_tp.set_title(f"{strat_label} -- TP Level Distribution", fontsize=9, fontweight="bold")
        for i, v in enumerate(tp_counts.values):
            if v > 0:
                ax_tp.text(v + 1, i, str(v), va="center", fontsize=7)

    fig.suptitle(f"Trade Composition (10-30 Delta) -- Exit Slip ${slip:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_skip_rate_comparison_page(fig, summary_df, slip):
    """Page 7: Skip rate comparison bar charts."""
    axes = fig.subplots(2, 3)
    metrics = [
        ("total_return_pct", "Total Return (%)", True),
        ("sharpe", "Sharpe Ratio", False),
        ("max_drawdown_pct", "Max Drawdown (%)", False),
        ("win_rate", "Win Rate (%)", False),
        ("n_trades", "# Trades", False),
        ("avg_pnl", "Avg P&L ($)", True),
    ]

    x = np.arange(len(SKIP_RATES))
    width = 0.35

    for idx, (metric, label, show_zero) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        for strat in [1, 2]:
            s_data = summary_df[summary_df["strategy"] == strat].sort_values("skip_rate")
            vals = s_data[metric].values
            offset = -width / 2 if strat == 1 else width / 2
            ax.bar(x + offset, vals, width, color=COLORS[strat], alpha=0.8, label=f"S{strat}")
        if show_zero:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sr:.0%}" for sr in SKIP_RATES], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Skip Rate Comparison (10-30 Delta) -- Exit Slip ${slip:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def generate_pdfs(summary_df, trades_df, equity_df, rr_df, holdout_info):
    """Generate 4 PDFs (one per slippage scenario)."""
    slips = sorted(summary_df["exit_slip"].unique())

    for slip in slips:
        slip_label, slip_desc = SLIP_LABELS.get(slip, (f"${slip:.2f}", "Custom"))
        pdf_path = OUTPUT_DIR / f"backtest_{slip_label.lower()}_slip{slip:.2f}.pdf"

        print(f"\nGenerating {pdf_path.name}...")

        slip_summary = summary_df[summary_df["exit_slip"] == slip].copy()
        slip_trades = trades_df[trades_df["exit_slip"] == slip].copy()
        slip_equity = equity_df[equity_df["sim_key"].str.contains(f"slip{slip:.2f}")].copy()

        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary table
            fig = plt.figure(figsize=(11, 8.5))
            render_summary_page(fig, slip_summary, slip, slip_label, slip_desc, holdout_info)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 1: Summary table")

            # Page 2: Reward/Risk analysis
            fig = plt.figure(figsize=(11, 8.5))
            render_reward_risk_page(fig, rr_df)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 2: Reward/Risk analysis")

            # Page 3: Detailed stats
            fig = plt.figure(figsize=(11, 8.5))
            render_detailed_stats_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 3: Detailed statistics")

            # Page 4: Equity curves
            fig = plt.figure(figsize=(11, 8.5))
            render_equity_curves_page(fig, equity_df, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 4: Equity curves")

            # Page 5: Distribution + exit reasons
            fig = plt.figure(figsize=(11, 8.5))
            render_distribution_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 5: P&L distribution & exit reasons")

            # Page 6: Trade composition
            fig = plt.figure(figsize=(11, 8.5))
            render_composition_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 6: Trade composition")

            # Page 7: Skip rate comparison
            fig = plt.figure(figsize=(11, 8.5))
            render_skip_rate_comparison_page(fig, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 7: Skip rate comparison")

        print(f"  Saved: {pdf_path}")


# ── Output ──────────────────────────────────────────────────────────────────

def save_results(summaries, all_trades, equity_curves, rr_df):
    """Save results to CSV files."""
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\n  Saved summary: {OUTPUT_DIR / 'summary.csv'}")

    if all_trades:
        trade_records = []
        for key, trades in all_trades.items():
            for t in trades:
                trade_records.append({
                    "sim_key": key,
                    "day": t.day,
                    "strategy": t.strategy,
                    "tp_level": t.tp_level,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_reason": t.exit_reason,
                    "credit": t.credit,
                    "exit_debit": t.exit_debit,
                    "exit_slip": t.exit_slip,
                    "fees": t.fees,
                    "pnl_per_share": t.pnl_per_share,
                    "qty": t.qty,
                    "pnl_total": t.pnl_total,
                    "portfolio_after": t.portfolio_after,
                })
        trade_df = pd.DataFrame(trade_records)
        trade_df.to_csv(OUTPUT_DIR / "trade_log.csv", index=False)
        print(f"  Saved trades: {OUTPUT_DIR / 'trade_log.csv'}")

    if equity_curves:
        eq_records = []
        for key, curve in equity_curves.items():
            for point in curve:
                eq_records.append({"sim_key": key, **point})
        eq_df = pd.DataFrame(eq_records)
        eq_df.to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)
        print(f"  Saved equity: {OUTPUT_DIR / 'equity_curves.csv'}")

    # Reward/Risk table
    rr_df.to_csv(OUTPUT_DIR / "reward_risk.csv", index=False)
    print(f"  Saved R/R: {OUTPUT_DIR / 'reward_risk.csv'}")

    # Per-simulation trade logs
    if all_trades:
        logs_dir = OUTPUT_DIR / "trade_logs"
        logs_dir.mkdir(exist_ok=True)
        for key, trades in all_trades.items():
            if trades:
                records = [{
                    "day": t.day, "strategy": t.strategy, "tp_level": t.tp_level,
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                    "exit_reason": t.exit_reason, "credit": t.credit,
                    "exit_debit": t.exit_debit, "exit_slip": t.exit_slip,
                    "fees": t.fees, "pnl_per_share": t.pnl_per_share,
                    "qty": t.qty, "pnl_total": t.pnl_total,
                    "portfolio_after": t.portfolio_after,
                } for t in trades]
                pd.DataFrame(records).to_csv(logs_dir / f"{key}.csv", index=False)


def print_summary_table(summaries):
    """Print formatted summary table."""
    print(f"\n{'='*110}")
    print(f"  BACKTEST RESULTS (10-30 DELTA ONLY)")
    print(f"{'='*110}\n")

    print(f"  {'Strat':>5s}  {'Slip':>5s}  {'Skip':>5s}  "
          f"{'Return%':>8s}  {'MaxDD%':>7s}  {'Sharpe':>7s}  "
          f"{'WinRate':>7s}  {'Trades':>7s}  {'T/Day':>6s}  "
          f"{'Final$':>10s}  {'AvgPnL':>8s}")
    print(f"  {'-'*100}")

    for s in sorted(summaries, key=lambda x: (x["strategy"], x["exit_slip"],
                                               x["skip_rate"])):
        print(f"  S{s['strategy']:>4d}  ${s['exit_slip']:.2f}  "
              f"{s['skip_rate']:>4.0%}  "
              f"{s['total_return_pct']:>+7.1f}%  {s['max_drawdown_pct']:>6.1f}%  "
              f"{s['sharpe']:>7.2f}  {s['win_rate']:>6.1f}%  "
              f"{s['n_trades']:>7,}  {s['trades_per_day']:>5.1f}  "
              f"${s['final_portfolio']:>9,.0f}  ${s.get('avg_pnl', 0):>+7.1f}")


def print_reward_risk_table(rr_df):
    """Print reward/risk summary."""
    print(f"\n{'='*90}")
    print(f"  REWARD / RISK ANALYSIS (Training Data)")
    print(f"{'='*90}\n")

    print(f"  {'Strategy':>14s}  {'Avg Credit':>10s}  {'Max Loss':>10s}  "
          f"{'Credit/Width':>12s}  {'R/R Ratio':>10s}  {'Samples':>8s}")
    print(f"  {'-'*70}")

    for _, r in rr_df.iterrows():
        print(f"  {r['strategy']:>14s}  ${r['avg_credit']:>8.3f}  "
              f"${r['avg_actual_max_loss']:>8.3f}  "
              f"{r['credit_vs_width_pct']:>10.1f}%  "
              f"{r['credit_vs_actual_max']:>10.4f}  "
              f"{r['n_samples']:>8,}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZDOM V1 Backtest — Delta 10-30 Only")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--strategy", type=int, choices=[1, 2], default=None)
    parser.add_argument("--skip-rates", nargs="*", type=float, default=None)
    parser.add_argument("--slips", nargs="*", type=float, default=None)
    parser.add_argument("--portfolio", type=float, default=START_PORTFOLIO)
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF generation")
    args = parser.parse_args()

    skip_rates = args.skip_rates or SKIP_RATES
    exit_slips = args.slips or EXIT_SLIPS
    strategies = [args.strategy] if args.strategy else [1, 2]

    print(f"\n{'='*80}")
    print(f"  ZDOM V1 Walk-Forward Backtest — Delta 10-30 Only")
    print(f"{'='*80}")
    print(f"  Strategies:  {STRATEGIES}")
    print(f"  Sim types:   {strategies}")
    print(f"  Slippage:    {exit_slips}")
    print(f"  Skip rates:  {skip_rates}")
    print(f"  Portfolio:   ${args.portfolio:,.2f}")
    print(f"  Output:      {OUTPUT_DIR}")

    # Phase 1: Load data (filtered to 10-30 delta)
    print(f"\n-- Phase 1: Data Loading --")
    df = load_model_table()
    df = apply_blockers(df)
    train_df, test_df, holdout_df = split_data(df)
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    holdout_info = (f"{holdout_df['date'].nunique()} holdout days "
                    f"({holdout_df['date'].min().date()} -> {holdout_df['date'].max().date()})")

    # Phase 2: Load pre-trained models
    print(f"\n-- Phase 2: Loading Models --")
    t0 = time.time()

    models = {}
    for tp in TP_LEVELS:
        target = f"{tp}_target"
        best_model, best_auc, best_algo = None, -1, None

        for algo in ["xgb", "lgbm"]:
            model_file = MODELS_DIR / f"{target}_{algo}.pkl"
            summary_file = MODELS_DIR / f"{target}_{algo}_summary.json"
            if not model_file.exists():
                continue

            ho_auc = 0
            if summary_file.exists():
                with open(summary_file) as f:
                    ho_auc = json.load(f).get("holdout_auc", 0)

            if ho_auc > best_auc:
                with open(model_file, "rb") as f:
                    artifact = pickle.load(f)
                best_model = artifact["model"]
                best_auc = ho_auc
                best_algo = algo

        if best_model is not None:
            models[tp] = best_model
            print(f"  {tp}: loaded {best_algo.upper()} (holdout AUC={best_auc:.4f})")

    if len(models) < len(TP_LEVELS):
        missing = set(TP_LEVELS) - set(models.keys())
        print(f"  [warn] Missing models for: {missing}")
        if not models:
            print("  [error] No models found. Run train_v1.py first.")
            sys.exit(1)

    print(f"  Model loading: {time.time() - t0:.0f}s")

    # Phase 3: Pre-compute EV & cutoffs
    print(f"\n-- Phase 3: Pre-computing EV & Cutoffs --")
    ev_lookup = precompute_ev_lookup(train_df)

    # Phase 4: Score holdout
    print(f"\n-- Phase 4: Scoring Holdout --")
    holdout_scored = score_holdout(holdout_df, models, feature_cols)
    cutoffs = precompute_skip_cutoffs(holdout_scored, skip_rates)
    print(f"  Cutoffs: {len(cutoffs)} (strategy, TP, skip_rate) combos")

    # Phase 5: Reward/Risk analysis
    print(f"\n-- Phase 5: Reward/Risk Analysis --")
    rr_df = compute_reward_risk(train_df)
    print_reward_risk_table(rr_df)

    # Phase 6: Run simulations
    print(f"\n-- Phase 6: Running Simulations --")
    total_sims = len(strategies) * len(exit_slips) * len(skip_rates)
    print(f"  {total_sims} simulation configs")

    all_summaries = []
    all_trade_logs = {}
    all_equity_curves = {}

    sim_count = 0
    t0 = time.time()

    for strat in strategies:
        for slip in exit_slips:
            for sr in skip_rates:
                sim_count += 1
                key = f"S{strat}_slip{slip:.2f}_skip{sr:.2f}"

                eq, trades, summary = simulate(
                    holdout_scored, strat, slip, sr,
                    ev_lookup, cutoffs,
                    start_portfolio=args.portfolio,
                )

                all_summaries.append(summary)
                all_trade_logs[key] = trades
                all_equity_curves[key] = eq

                if sim_count % 5 == 0 or sim_count == total_sims:
                    elapsed = time.time() - t0
                    print(f"  [{sim_count}/{total_sims}] {key}: "
                          f"return={summary['total_return_pct']:+.1f}%, "
                          f"trades={summary['n_trades']}, "
                          f"sharpe={summary['sharpe']:.2f}  "
                          f"({elapsed:.0f}s)")

    sim_time = time.time() - t0
    print(f"\n  Simulations complete: {sim_time:.0f}s")

    # Output
    print_summary_table(all_summaries)
    save_results(all_summaries, all_trade_logs, all_equity_curves, rr_df)

    # Phase 7: Generate PDFs
    if not args.skip_pdf:
        print(f"\n-- Phase 7: Generating PDFs --")
        summary_df = pd.DataFrame(all_summaries)
        trades_df = pd.read_csv(OUTPUT_DIR / "trade_log.csv")
        equity_df = pd.read_csv(OUTPUT_DIR / "equity_curves.csv")
        generate_pdfs(summary_df, trades_df, equity_df, rr_df, holdout_info)

    print(f"\n  Done. Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
