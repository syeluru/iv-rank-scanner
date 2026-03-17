"""
Generate 7 PDF reports (one per slippage scenario) for ZDOM V1 backtest results.

Each PDF contains:
  - Page 1: Summary heatmap (skip rate vs metric) for both strategies
  - Page 2: Detailed trade statistics for best skip rate per strategy
  - Page 3: Equity curves (subset of skip rates)
  - Page 4: Win/loss distribution and exit reason breakdown
  - Page 5: Strategy-specific trade composition (IC deltas + TP levels)
  - Page 6: Line charts comparing key metrics across all skip rates
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # ml/zdom_v1/
OUTPUT_DIR = PROJECT_DIR / "output" / "backtest"  # default; overridable via --dir

# Show subset of skip rates for equity curves (auto-detected from data)
CURVE_SKIP_RATES = [0.20, 0.25, 0.30, 0.35, 0.40]  # default; overridden in main()

SLIP_LABELS = {
    0.00: ("S1", "Perfect Mid Fills"),
    0.05: ("S2", "$0.05 Exit Slip"),
    0.10: ("S3", "$0.10 Slip/Side"),
    0.15: ("S4", "$0.15 Exit Slip"),
    0.20: ("S5", "$0.20 Slip/Side"),
    0.25: ("S6", "$0.25 Exit Slip"),
    0.30: ("S7", "$0.30 Slip/Side"),
    0.40: ("S8", "$0.40 Slip/Side"),
}

COLORS = {
    1: "#2196F3",  # Blue for Strategy 1
    2: "#FF9800",  # Orange for Strategy 2
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_pct(v):
    return f"{v:+.1f}%" if v != 0 else "0.0%"

def _fmt_dollar(v):
    return f"${v:,.0f}"

def _fmt_num(v):
    return f"{v:,.0f}" if abs(v) >= 1 else f"{v:.2f}"


def compute_detailed_stats(trades_df):
    """Compute detailed statistics from trade log."""
    if trades_df.empty:
        return {}

    wins = trades_df[trades_df["pnl_total"] > 0]
    losses = trades_df[trades_df["pnl_total"] <= 0]

    # Streak analysis
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

    # Daily stats
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


# ── Page Renderers ───────────────────────────────────────────────────────────

def render_summary_page(fig, summary_df, slip, slip_label, slip_desc):
    """Page 1: Summary table with all 21 skip rates for both strategies."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    title = f"ZDOM V1 Backtest — Scenario {slip_label}: {slip_desc}"
    subtitle = (f"Exit Slippage: ${slip:.2f}/share  |  Fees: $0.052/share  |  "
                f"Starting Portfolio: $10,000")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.935, subtitle, transform=ax.transAxes, fontsize=8,
            ha="center", va="top", color="#666666")

    headers = ["Skip %", "Return", "MaxDD", "Sharpe", "WR", "Trades", "T/D", "Final$"]

    for strat_idx, strat in enumerate([1, 2]):
        s_data = summary_df[summary_df["strategy"] == strat].sort_values("skip_rate")
        color = COLORS[strat]
        strat_name = "S1: Best EV, Max Contracts" if strat == 1 else "S2: Diversified, Max BP"

        y_title = 0.90 if strat_idx == 0 else 0.46
        y_table_top = 0.56 if strat_idx == 0 else 0.12
        table_height = 0.32

        ax.text(0.5, y_title, strat_name,
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="center", va="top", color=color)

        rows = []
        for _, r in s_data.iterrows():
            rows.append([
                f"{r['skip_rate']:.0%}",
                f"{r['total_return_pct']:+.0f}%",
                f"{r['max_drawdown_pct']:.0f}%",
                f"{r['sharpe']:.2f}",
                f"{r['win_rate']:.0f}%",
                f"{int(r['n_trades'])}",
                f"{r['trades_per_day']:.1f}",
                f"${r['final_portfolio']:,.0f}",
            ])

        if rows:
            t = ax.table(cellText=rows, colLabels=headers,
                         loc="center", bbox=[0.02, y_table_top, 0.96, table_height])
            t.auto_set_font_size(False)
            t.set_fontsize(6.5)
            for (i, j), cell in t.get_celld().items():
                if i == 0:
                    bg = "#E3F2FD" if strat == 1 else "#FFF3E0"
                    cell.set_facecolor(bg)
                    cell.set_text_props(fontweight="bold", fontsize=7)
                cell.set_edgecolor("#DDDDDD")
                cell.set_height(cell.get_height() * 0.9)

            # Highlight best Sharpe row
            if not s_data.empty:
                best_idx = s_data["sharpe"].idxmax()
                best_row = s_data.index.get_loc(best_idx) + 1
                highlight = "#BBDEFB" if strat == 1 else "#FFE0B2"
                for j in range(len(headers)):
                    t.get_celld()[(best_row, j)].set_facecolor(highlight)

    ax.text(0.5, 0.06, "Highlighted = best Sharpe  |  "
            "106 holdout days (2025-09-26 → 2026-02-27)  |  "
            "IC_10d through IC_45d, $25 wings",
            transform=ax.transAxes, fontsize=7, ha="center", va="top", color="#999999")


def render_detailed_stats_page(fig, trades_df, summary_df, slip):
    """Page 2: Detailed trade statistics side-by-side for best skip rate per strategy."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.97, f"Detailed Trade Statistics — Exit Slip ${slip:.2f}",
            transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")

    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
        if s_summary.empty:
            continue
        best_skip = s_summary.loc[s_summary["sharpe"].idxmax(), "skip_rate"]

        key = f"S{strat}_slip{slip:.2f}_skip{best_skip:.2f}"
        strat_trades = trades_df[trades_df["sim_key"] == key]
        stats = compute_detailed_stats(strat_trades)

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

    # Divider
    ax.plot([0.50, 0.50], [0.05, 0.90], transform=ax.transAxes,
            color="#CCCCCC", linewidth=1, clip_on=False)


def render_equity_curves_page(fig, equity_df, slip):
    """Page 3: Equity curves for subset of skip rates, both strategies."""
    axes = fig.subplots(2, 1, sharex=True)

    for strat_idx, strat in enumerate([1, 2]):
        ax = axes[strat_idx]
        strat_name = "Strategy 1: Best EV, Max Contracts" if strat == 1 else "Strategy 2: Diversified, Max BP"

        cmap = plt.cm.Blues if strat == 1 else plt.cm.Oranges
        for i, sr in enumerate(CURVE_SKIP_RATES):
            key = f"S{strat}_slip{slip:.2f}_skip{sr:.2f}"
            eq = equity_df[equity_df["sim_key"] == key].copy()
            if eq.empty:
                continue

            eq = eq[eq["date"] != "start"].copy()
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.sort_values("date")

            color_val = 0.4 + (i / max(len(CURVE_SKIP_RATES) - 1, 1)) * 0.5
            ax.plot(eq["date"], eq["portfolio"], label=f"Skip {sr:.0%}",
                    linewidth=1.2, color=cmap(color_val))

        ax.set_title(strat_name, fontsize=11, fontweight="bold", color=COLORS[strat])
        ax.set_ylabel("Portfolio ($)")
        ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    axes[1].set_xlabel("Date")
    fig.suptitle(f"Equity Curves — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def render_distribution_page(fig, trades_df, summary_df, slip):
    """Page 4: PnL distribution and exit reason breakdown."""
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
        ax_hist.set_title(f"{strat_label} — P&L Distribution", fontsize=9, fontweight="bold")
        ax_hist.set_xlabel("P&L ($)")
        ax_hist.set_ylabel("Count")
        ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Exit reason pie
        ax_pie = axes[1][strat_idx]
        reasons = strat_trades["exit_reason"].value_counts()
        reason_colors = {
            "tp": "#4CAF50",
            "sl": "#F44336",
            "close_win": "#8BC34A",
            "close_loss": "#FF9800",
        }
        pie_colors = [reason_colors.get(r, "#9E9E9E") for r in reasons.index]
        labels = [f"{r}\n({v:,})" for r, v in reasons.items()]
        ax_pie.pie(reasons.values, labels=labels, colors=pie_colors,
                   autopct="%1.0f%%", textprops={"fontsize": 8})
        ax_pie.set_title(f"{strat_label} — Exit Reasons", fontsize=9, fontweight="bold")

    fig.suptitle(f"Trade Analysis — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_composition_page(fig, trades_df, summary_df, slip):
    """Page 5: Which strategies and TP levels were selected."""
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
        strat_counts = strat_trades["strategy"].value_counts().sort_index()
        bars = ax_strat.barh(range(len(strat_counts)), strat_counts.values,
                              color=color, alpha=0.7)
        ax_strat.set_yticks(range(len(strat_counts)))
        ax_strat.set_yticklabels(strat_counts.index, fontsize=7)
        ax_strat.set_xlabel("# Trades")
        ax_strat.set_title(f"{strat_label} — IC Delta Distribution", fontsize=9, fontweight="bold")
        for i, v in enumerate(strat_counts.values):
            ax_strat.text(v + 1, i, str(v), va="center", fontsize=7)

        # TP level breakdown
        ax_tp = axes[1][strat_idx]
        tp_counts = strat_trades["tp_level"].value_counts()
        tp_order = [f"tp{p}" for p in range(10, 55, 5)]
        tp_counts = tp_counts.reindex(tp_order).dropna().astype(int)
        bars = ax_tp.barh(range(len(tp_counts)), tp_counts.values,
                           color=color, alpha=0.7)
        ax_tp.set_yticks(range(len(tp_counts)))
        ax_tp.set_yticklabels(tp_counts.index, fontsize=7)
        ax_tp.set_xlabel("# Trades")
        ax_tp.set_title(f"{strat_label} — TP Level Distribution", fontsize=9, fontweight="bold")
        for i, v in enumerate(tp_counts.values):
            ax_tp.text(v + 1, i, str(v), va="center", fontsize=7)

    fig.suptitle(f"Trade Composition — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_skip_rate_comparison_page(fig, summary_df, slip):
    """Page 6: Line charts comparing key metrics across all skip rates."""
    axes = fig.subplots(2, 3)
    metrics = [
        ("total_return_pct", "Total Return (%)", True),
        ("sharpe", "Sharpe Ratio", False),
        ("max_drawdown_pct", "Max Drawdown (%)", False),
        ("win_rate", "Win Rate (%)", False),
        ("n_trades", "# Trades", False),
        ("avg_pnl", "Avg P&L ($)", True),
    ]

    for idx, (metric, label, show_zero) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]

        for strat in [1, 2]:
            s_data = summary_df[summary_df["strategy"] == strat].sort_values("skip_rate")
            ax.plot(s_data["skip_rate"] * 100, s_data[metric],
                    color=COLORS[strat], marker=".", markersize=3,
                    linewidth=1.2, alpha=0.8, label=f"S{strat}")

        if show_zero:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Skip Rate (%)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.15)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Skip Rate Sensitivity — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate backtest PDF reports")
    parser.add_argument("--dir", type=str, default=None,
                        help="Input/output directory (default: OUTPUT_DIR)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for PDF filenames (e.g. '_shadow5d')")
    args = parser.parse_args()

    out_dir = Path(args.dir) if args.dir else OUTPUT_DIR
    suffix = args.suffix

    print(f"Loading backtest results from {out_dir}...")
    summary_df = pd.read_csv(out_dir / "summary.csv")

    # Auto-detect curve skip rates (pick ~5 evenly spaced from available)
    global CURVE_SKIP_RATES
    all_skips = sorted(summary_df["skip_rate"].unique())
    if len(all_skips) <= 5:
        CURVE_SKIP_RATES = all_skips
    else:
        step = max(1, len(all_skips) // 5)
        CURVE_SKIP_RATES = [all_skips[i] for i in range(0, len(all_skips), step)][:5]
        if all_skips[-1] not in CURVE_SKIP_RATES:
            CURVE_SKIP_RATES[-1] = all_skips[-1]
    trades_df = pd.read_csv(out_dir / "trade_log.csv")
    equity_df = pd.read_csv(out_dir / "equity_curves.csv")

    print(f"  {len(summary_df)} summary rows, {len(trades_df):,} trades, "
          f"{len(equity_df):,} equity points")

    slips = sorted(summary_df["exit_slip"].unique())

    for slip in slips:
        slip_label, slip_desc = SLIP_LABELS.get(slip, (f"${slip:.2f}", "Custom"))
        pdf_path = out_dir / f"backtest_{slip_label.lower()}_slip{slip:.2f}{suffix}.pdf"

        print(f"\nGenerating {pdf_path.name}...")

        slip_summary = summary_df[summary_df["exit_slip"] == slip].copy()
        # Filter trades by sim_key prefix (handles bilateral slip where
        # trade_log exit_slip is effective/doubled but summary uses per-side)
        slip_key_prefix = f"_slip{slip:.2f}_"
        slip_trades = trades_df[trades_df["sim_key"].str.contains(slip_key_prefix)].copy()

        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary table
            fig = plt.figure(figsize=(11, 8.5))
            render_summary_page(fig, slip_summary, slip, slip_label, slip_desc)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 1: Summary table")

            # Page 2: Detailed stats
            fig = plt.figure(figsize=(11, 8.5))
            render_detailed_stats_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 2: Detailed statistics")

            # Page 3: Equity curves
            fig = plt.figure(figsize=(11, 8.5))
            render_equity_curves_page(fig, equity_df, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 3: Equity curves")

            # Page 4: Distribution + exit reasons
            fig = plt.figure(figsize=(11, 8.5))
            render_distribution_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 4: P&L distribution & exit reasons")

            # Page 5: Trade composition
            fig = plt.figure(figsize=(11, 8.5))
            render_composition_page(fig, trades_df, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 5: Trade composition")

            # Page 6: Skip rate comparison
            fig = plt.figure(figsize=(11, 8.5))
            render_skip_rate_comparison_page(fig, slip_summary, slip)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page 6: Skip rate sensitivity")

        print(f"  Saved: {pdf_path}")

    print(f"\nDone. {len(slips)} PDFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
