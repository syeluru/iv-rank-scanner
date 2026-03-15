"""
Generate 4 PDF reports (one per slippage scenario) for ZDOM V1 backtest results.

Each PDF contains:
  - Page 1: Summary comparison table (S1 vs S2 across all skip rates)
  - Page 2: Detailed trade statistics breakdown
  - Page 3: Equity curves for both strategies
  - Page 4: Win/loss distribution and trade breakdown by exit reason
  - Page 5: Strategy-specific trade composition (which IC deltas + TP levels chosen)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "backtest"

SKIP_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]
SLIP_LABELS = {
    0.00: ("S1", "Perfect Mid Fills"),
    0.10: ("S2", "Small Exit Slip"),
    0.20: ("S3", "Moderate Exit Slip"),
    0.30: ("S4", "Worst Case"),
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
    """Page 1: Summary comparison table."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    title = f"ZDOM V1 Backtest — Scenario {slip_label}: {slip_desc}"
    subtitle = f"Exit Slippage: ${slip:.2f}/share  |  Fees: $0.052/share  |  Starting Portfolio: $10,000"
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.93, subtitle, transform=ax.transAxes, fontsize=10,
            ha="center", va="top", color="#666666")

    # Table data
    headers = ["Skip Rate", "Return %", "Max DD %", "Sharpe", "Win Rate",
               "# Trades", "T/Day", "Final $", "Avg PnL"]

    s1 = summary_df[summary_df["strategy"] == 1].sort_values("skip_rate")
    s2 = summary_df[summary_df["strategy"] == 2].sort_values("skip_rate")

    # Strategy 1 table
    ax.text(0.5, 0.86, "Strategy 1: Best EV, Max Contracts",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="center", va="top", color=COLORS[1])

    rows1 = []
    for _, r in s1.iterrows():
        rows1.append([
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

    if rows1:
        t1 = ax.table(cellText=rows1, colLabels=headers,
                       loc="center", bbox=[0.02, 0.52, 0.96, 0.32])
        t1.auto_set_font_size(False)
        t1.set_fontsize(9)
        for (i, j), cell in t1.get_celld().items():
            if i == 0:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
        # Highlight best row
        if len(rows1) > 0:
            best_idx = s1["sharpe"].idxmax()
            best_row = s1.index.get_loc(best_idx) + 1
            for j in range(len(headers)):
                t1.get_celld()[(best_row, j)].set_facecolor("#BBDEFB")

    # Strategy 2 table
    ax.text(0.5, 0.48, "Strategy 2: Diversified, Max Buying Power",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="center", va="top", color=COLORS[2])

    rows2 = []
    for _, r in s2.iterrows():
        rows2.append([
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

    if rows2:
        t2 = ax.table(cellText=rows2, colLabels=headers,
                       loc="center", bbox=[0.02, 0.14, 0.96, 0.32])
        t2.auto_set_font_size(False)
        t2.set_fontsize(9)
        for (i, j), cell in t2.get_celld().items():
            if i == 0:
                cell.set_facecolor("#FFF3E0")
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
        if len(rows2) > 0:
            best_idx = s2["sharpe"].idxmax()
            best_row = s2.index.get_loc(best_idx) + 1
            for j in range(len(headers)):
                t2.get_celld()[(best_row, j)].set_facecolor("#FFE0B2")

    # Footer
    ax.text(0.5, 0.08, "Highlighted rows = best Sharpe ratio per strategy  |  "
            "106 holdout days (2025-09-26 → 2026-02-27)",
            transform=ax.transAxes, fontsize=8, ha="center", va="top", color="#999999")


def render_detailed_stats_page(fig, trades_df, summary_df, slip):
    """Page 2: Detailed trade statistics side-by-side for best skip rate per strategy."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.97, f"Detailed Trade Statistics — Exit Slip ${slip:.2f}",
            transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")

    # Use best Sharpe skip rate per strategy for detailed comparison
    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
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

        # Metrics list
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
                # Section header
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
    """Page 3: Equity curves for all skip rates, both strategies."""
    axes = fig.subplots(2, 1, sharex=True)

    for strat_idx, strat in enumerate([1, 2]):
        ax = axes[strat_idx]
        strat_name = "Strategy 1: Best EV, Max Contracts" if strat == 1 else "Strategy 2: Diversified, Max BP"

        for sr in SKIP_RATES:
            key = f"S{strat}_slip{slip:.2f}_skip{sr:.2f}"
            eq = equity_df[equity_df["sim_key"] == key].copy()
            if eq.empty:
                continue

            eq = eq[eq["date"] != "start"].copy()
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.sort_values("date")

            alpha = 0.4 + (sr / 0.40) * 0.6  # Darker for higher skip
            ax.plot(eq["date"], eq["portfolio"], label=f"Skip {sr:.0%}",
                    alpha=alpha, linewidth=1.2, color=COLORS[strat])

        ax.set_title(strat_name, fontsize=11, fontweight="bold", color=COLORS[strat])
        ax.set_ylabel("Portfolio ($)")
        ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    axes[1].set_xlabel("Date")
    fig.suptitle(f"Equity Curves — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def render_distribution_page(fig, trades_df, summary_df, slip):
    """Page 4: PnL distribution and exit reason breakdown."""
    axes = fig.subplots(2, 2)

    for strat_idx, strat in enumerate([1, 2]):
        s_summary = summary_df[summary_df["strategy"] == strat]
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
        # Sort by TP number
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
    """Page 6: Bar charts comparing key metrics across skip rates."""
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
            bars = ax.bar(x + offset, vals, width, color=COLORS[strat],
                          alpha=0.8, label=f"S{strat}")

        if show_zero:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sr:.0%}" for sr in SKIP_RATES], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Skip Rate Comparison — Exit Slip ${slip:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading backtest results...")
    summary_df = pd.read_csv(OUTPUT_DIR / "summary.csv")
    trades_df = pd.read_csv(OUTPUT_DIR / "trade_log.csv")
    equity_df = pd.read_csv(OUTPUT_DIR / "equity_curves.csv")

    print(f"  {len(summary_df)} summary rows, {len(trades_df):,} trades, "
          f"{len(equity_df):,} equity points")

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
            print(f"  Page 6: Skip rate comparison")

        print(f"  Saved: {pdf_path}")

    print(f"\nDone. 4 PDFs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
