#!/usr/bin/env python3
"""Analyze Phase 2 results by entry time - Task 1 & Task 2."""

import pandas as pd
import numpy as np
import sys

RESULTS_CSV = "output/hedged_ic_backtest/results_phase2.csv"
BEST_COMBOS_CSV = "output/hedged_ic_backtest/best_combos_phase2.csv"

# ── Load best_combos to get top 25 label ordering ──────────────────────
best_combos = pd.read_csv(BEST_COMBOS_CSV)
top25_labels = best_combos["label"].head(25).tolist()
print(f"Top 25 labels from best_combos_phase2.csv:")
for i, lab in enumerate(top25_labels, 1):
    print(f"  {i:2d}. {lab}")
print()

# ── Chunked aggregation ────────────────────────────────────────────────
print("Reading results_phase2.csv in chunks (13M+ rows)...")
CHUNK_SIZE = 2_000_000

# We need group-level aggregates: (label, entry_time) -> count, sum_pnl, sum_win
agg = {}  # key=(label, entry_time) -> [count, sum_pnl, sum_win]

# For Task 2, collect rows for top 10 configs after we know them.
# First pass: just aggregate.
all_entry_times = set()

for i, chunk in enumerate(pd.read_csv(RESULTS_CSV, chunksize=CHUNK_SIZE)):
    print(f"  Chunk {i+1}: {len(chunk):,} rows")
    grouped = chunk.groupby(["label", "entry_time"]).agg(
        count=("pnl_dollar", "size"),
        sum_pnl=("pnl_dollar", "sum"),
        sum_win=("win", "sum"),
    ).reset_index()

    for _, row in grouped.iterrows():
        key = (row["label"], row["entry_time"])
        if key not in agg:
            agg[key] = [0, 0.0, 0]
        agg[key][0] += row["count"]
        agg[key][1] += row["sum_pnl"]
        agg[key][2] += row["sum_win"]

    all_entry_times.update(chunk["entry_time"].unique())

print(f"\nTotal unique (label, entry_time) groups: {len(agg):,}")
print(f"Unique entry times: {sorted(all_entry_times)}")

# Build DataFrame from aggregated data
rows = []
for (label, entry_time), (count, sum_pnl, sum_win) in agg.items():
    rows.append({
        "label": label,
        "entry_time": entry_time,
        "count": count,
        "total_pnl": sum_pnl,
        "avg_pnl_dollar": sum_pnl / count if count > 0 else 0,
        "win_rate": sum_win / count if count > 0 else 0,
    })
df_agg = pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════
# TASK 1: Heatmap-style table for top 25 labels by entry time
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TASK 1: TOP 25 CONFIGS — HEATMAP OF AVG P&L BY ENTRY TIME")
print("=" * 100)

entry_times_sorted = sorted(all_entry_times)

# Filter to top 25 labels
df_top25 = df_agg[df_agg["label"].isin(top25_labels)].copy()

# Pivot: rows=label, cols=entry_time, values=avg_pnl_dollar
pivot_pnl = df_top25.pivot_table(
    index="label", columns="entry_time", values="avg_pnl_dollar", aggfunc="first"
)
pivot_count = df_top25.pivot_table(
    index="label", columns="entry_time", values="count", aggfunc="first"
)
pivot_wr = df_top25.pivot_table(
    index="label", columns="entry_time", values="win_rate", aggfunc="first"
)

# Reorder rows by best_combos ranking
pivot_pnl = pivot_pnl.reindex(top25_labels)
pivot_count = pivot_count.reindex(top25_labels)
pivot_wr = pivot_wr.reindex(top25_labels)

print(f"\n{'Label':<30}", end="")
for et in entry_times_sorted:
    print(f"  {et:>12}", end="")
print(f"  {'Best ET':>12}  {'Best Avg':>10}")
print("-" * (30 + 14 * len(entry_times_sorted) + 26))

for label in top25_labels:
    if label not in pivot_pnl.index:
        continue
    print(f"{label:<30}", end="")
    best_et = None
    best_val = -np.inf
    for et in entry_times_sorted:
        val = pivot_pnl.loc[label, et] if et in pivot_pnl.columns and pd.notna(pivot_pnl.loc[label, et]) else np.nan
        cnt = pivot_count.loc[label, et] if et in pivot_count.columns and pd.notna(pivot_count.loc[label, et]) else 0
        if pd.notna(val):
            # Color coding: green if positive, red if negative
            marker = "+" if val > 0 else " "
            print(f"  {marker}${val:>9.2f}", end="")
            if val > best_val:
                best_val = val
                best_et = et
        else:
            print(f"  {'N/A':>12}", end="")
    print(f"  {best_et:>12}  ${best_val:>9.2f}")

# Detailed table with count and win_rate
print(f"\n\nDETAILED VIEW (avg_pnl | count | win_rate) for top 25:")
print("-" * 120)
for label in top25_labels:
    if label not in pivot_pnl.index:
        continue
    print(f"\n  {label}:")
    for et in entry_times_sorted:
        val = pivot_pnl.loc[label, et] if et in pivot_pnl.columns and pd.notna(pivot_pnl.loc[label, et]) else np.nan
        cnt = int(pivot_count.loc[label, et]) if et in pivot_count.columns and pd.notna(pivot_count.loc[label, et]) else 0
        wr = pivot_wr.loc[label, et] if et in pivot_wr.columns and pd.notna(pivot_wr.loc[label, et]) else np.nan
        if pd.notna(val):
            print(f"    {et}: avg_pnl=${val:>10.2f}  count={cnt:>6,}  win_rate={wr:.1%}")

# ── Best single (label, entry_time) combination ──────────────────────
print("\n\n" + "=" * 100)
print("BEST SINGLE (label, entry_time) COMBINATION")
print("=" * 100)
df_filtered = df_agg[df_agg["count"] >= 500].copy()
best_row = df_filtered.loc[df_filtered["avg_pnl_dollar"].idxmax()]
print(f"  Label:       {best_row['label']}")
print(f"  Entry Time:  {best_row['entry_time']}")
print(f"  Avg P&L:     ${best_row['avg_pnl_dollar']:.2f}")
print(f"  Win Rate:    {best_row['win_rate']:.1%}")
print(f"  Count:       {int(best_row['count']):,}")
print(f"  Total P&L:   ${best_row['total_pnl']:,.2f}")

# ── Top 25 (label, entry_time) combos ────────────────────────────────
print("\n\n" + "=" * 100)
print("TOP 25 (label, entry_time) COMBOS BY AVG P&L (min 500 trades)")
print("=" * 100)
top25_combos = df_filtered.nlargest(25, "avg_pnl_dollar")
print(f"\n{'Rank':<6}{'Label':<35}{'Entry':>8}{'Avg P&L':>12}{'Win Rate':>10}{'Count':>8}{'Total P&L':>14}")
print("-" * 93)
for i, (_, row) in enumerate(top25_combos.iterrows(), 1):
    print(f"{i:<6}{row['label']:<35}{row['entry_time']:>8}${row['avg_pnl_dollar']:>10.2f}{row['win_rate']:>9.1%}{int(row['count']):>8,}${row['total_pnl']:>13,.2f}")

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Example trades from top 10 configs
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("TASK 2: EXAMPLE TRADES FROM TOP 10 CONFIGS")
print("=" * 100)

top10_combos = top25_combos.head(10)
target_keys = set()
for _, row in top10_combos.iterrows():
    target_keys.add((row["label"], row["entry_time"]))

print(f"\nCollecting example trades for {len(target_keys)} configs...")
print("Re-reading CSV to extract specific trades...\n")

# Collect all trades for target configs
target_trades = {k: [] for k in target_keys}

for i, chunk in enumerate(pd.read_csv(RESULTS_CSV, chunksize=CHUNK_SIZE)):
    for key in target_keys:
        label, entry_time = key
        mask = (chunk["label"] == label) & (chunk["entry_time"] == entry_time)
        matching = chunk[mask]
        if len(matching) > 0:
            target_trades[key].append(matching)
    print(f"  Chunk {i+1} processed")

# Concatenate
for key in target_keys:
    if target_trades[key]:
        target_trades[key] = pd.concat(target_trades[key], ignore_index=True)
    else:
        target_trades[key] = pd.DataFrame()

# All columns we want to display
display_cols = [
    "date", "entry_time", "spx_entry", "spx_close", "spx_move", "spx_move_pct",
    "delta", "si_width", "gap", "li_width",
    "si_short_put", "si_long_put", "si_short_call", "si_long_call",
    "li_short_put", "li_long_put", "li_short_call", "li_long_call",
    "si_credit", "li_debit", "net_credit", "fees_per_share", "net_credit_after_fees",
    "num_legs", "si_pnl", "li_pnl", "total_pnl", "total_pnl_after_fees",
    "pnl_dollar", "pnl_pct", "win"
]

def print_trade(trade, trade_label):
    """Print all columns of a single trade row."""
    print(f"\n    --- {trade_label} ---")
    for col in display_cols:
        if col in trade.index:
            val = trade[col]
            if isinstance(val, float):
                if "pct" in col:
                    print(f"      {col:<28} {val:>12.4f}")
                elif "pnl" in col or "credit" in col or "debit" in col or "fees" in col:
                    print(f"      {col:<28} ${val:>11.2f}")
                else:
                    print(f"      {col:<28} {val:>12.2f}")
            else:
                print(f"      {col:<28} {val}")

rank = 0
for _, combo_row in top10_combos.iterrows():
    rank += 1
    key = (combo_row["label"], combo_row["entry_time"])
    df_trades = target_trades[key]

    print(f"\n{'─' * 90}")
    print(f"  CONFIG #{rank}: {key[0]} @ {key[1]}")
    print(f"  Avg P&L: ${combo_row['avg_pnl_dollar']:.2f} | Win Rate: {combo_row['win_rate']:.1%} | Trades: {int(combo_row['count']):,}")
    print(f"{'─' * 90}")

    if df_trades.empty:
        print("    No trades found!")
        continue

    wins = df_trades[df_trades["win"] == 1].copy()
    losses = df_trades[df_trades["win"] == 0].copy()

    # 1. Median win
    if not wins.empty:
        median_pnl = wins["pnl_dollar"].median()
        median_idx = (wins["pnl_dollar"] - median_pnl).abs().idxmin()
        print_trade(wins.loc[median_idx], f"MEDIAN WIN (pnl_dollar ~ ${median_pnl:.2f})")

    # 2. Best trade
    best_idx = df_trades["pnl_dollar"].idxmax()
    print_trade(df_trades.loc[best_idx], f"BEST TRADE (max pnl_dollar)")

    # 3. Median loss
    if not losses.empty:
        median_loss = losses["pnl_dollar"].median()
        median_loss_idx = (losses["pnl_dollar"] - median_loss).abs().idxmin()
        print_trade(losses.loc[median_loss_idx], f"MEDIAN LOSS (pnl_dollar ~ ${median_loss:.2f})")
    else:
        print("\n    --- NO LOSING TRADES ---")

print("\n\nDone.")
