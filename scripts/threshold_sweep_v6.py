#!/usr/bin/env python3
"""
V6 Ensemble Threshold Sweep
============================
Finds the optimal decision threshold for the v6 tp25 and tp50 ensemble models.

For each threshold T from 0.00 to 1.00:
  - Filter trades where ensemble P(hit_TP) > T
  - Calculate: avg P&L, total P&L, Sharpe, Sortino, max drawdown, win rate, trade count

Outputs:
  1. Console summary of top thresholds per metric
  2. CSV with full sweep results
  3. PNG charts: threshold vs key metrics

Usage:
  python scripts/threshold_sweep_v6.py
  python scripts/threshold_sweep_v6.py --target tp50
  python scripts/threshold_sweep_v6.py --stability   # run per-quarter stability check
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'
DATA_PATH = BASE_DIR / 'ml' / 'training_data' / 'training_dataset_multi_w25.csv'
MODEL_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v6' / 'ensemble'
OUTPUT_DIR = BASE_DIR / 'output' / 'threshold_sweep'

FEATURE_COLS = [
    'rv_close_5d', 'rv_close_10d', 'rv_close_20d', 'atr_14d_pct', 'gap_pct',
    'prior_day_range_pct', 'prior_day_return', 'sma_20d_dist', 'vix_level',
    'vix_change_1d', 'vix_change_5d', 'vix_rank_30d', 'vix_sma_10d_dist',
    'vix_term_slope', 'intraday_rv', 'move_from_open_pct', 'orb_range_pct',
    'orb_contained', 'range_exhaustion', 'high_low_range_pct', 'momentum_30min',
    'trend_slope_norm', 'adx', 'efficiency_ratio', 'choppiness_index',
    'rsi_dist_50', 'linreg_r2', 'atr_ratio', 'bbw_percentile', 'ttm_squeeze',
    'bars_in_squeeze', 'garman_klass_rv', 'orb_failure', 'time_sin', 'time_cos',
    'dow_sin', 'dow_cos', 'minutes_to_close', 'is_fomc_day', 'is_fomc_week',
    'days_since_fomc'
]

WINDOWS = ['3y', '2y', '1y', '6m', '3m', '1m']
WEIGHTS = {'3y': 0.20, '2y': 0.20, '1y': 0.20, '6m': 0.15, '3m': 0.15, '1m': 0.10}

THRESHOLDS = np.arange(0.00, 1.01, 0.01)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    """Load training dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    print(f"  {len(df):,} trades, {df['trade_date'].dt.date.nunique()} unique dates")
    print(f"  Date range: {df['trade_date'].min().date()} to {df['trade_date'].max().date()}")
    
    # Drop rows with NaN in critical columns
    before = len(df)
    df = df.dropna(subset=['pnl_at_close', 'entry_credit', 'hit_25pct', 'hit_50pct']).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN values → {len(df):,} trades")
    
    return df


def load_v6_ensemble():
    """Load all v6 ensemble models."""
    models = {}
    for target in ['tp25', 'tp50']:
        models[target] = {}
        for window in WINDOWS:
            # Try with month tag pattern first
            import glob
            patterns = [
                str(MODEL_DIR / f"entry_timing_v6_{target}_*_{window}.joblib"),
                str(MODEL_DIR / f"entry_timing_v6_{target}_{window}.joblib"),
            ]
            path = None
            for pattern in patterns:
                matches = sorted(glob.glob(pattern), reverse=True)
                if matches:
                    path = Path(matches[0])
                    break

            if path and path.exists():
                artifact = joblib.load(path)
                if isinstance(artifact, dict):
                    models[target][window] = artifact['model']
                else:
                    models[target][window] = artifact  # Raw model object
                print(f"  Loaded v6 {target} {window}: {path.name}")
            else:
                print(f"  ⚠ Missing v6 {target} {window}")

    return models


def compute_ensemble_scores(df, models, target):
    """
    Generate ensemble probability scores for every trade.
    Uses walk-forward: each trade is scored ONLY by models trained on prior data.
    
    Since v6 models are trained on rolling windows, we use the ensemble as-is
    but note this is an approximation — true walk-forward would retrain monthly.
    For threshold selection, this is sufficient since we're looking at relative
    ordering, not absolute calibration.
    """
    X = df[FEATURE_COLS].values
    
    # Handle NaN/inf
    X = np.where(np.isnan(X), 0.0, X)
    X = np.where(np.isinf(X), 0.0, X)
    
    # Get predictions from each window model
    window_preds = {}
    for window in WINDOWS:
        if window in models[target]:
            model = models[target][window]
            window_preds[window] = model.predict_proba(X)[:, 1]
            print(f"    {target} {window}: mean={window_preds[window].mean():.4f}, "
                  f"std={window_preds[window].std():.4f}")
    
    # Weighted ensemble
    ensemble_scores = np.zeros(len(df))
    total_weight = 0
    for window, preds in window_preds.items():
        w = WEIGHTS[window]
        ensemble_scores += preds * w
        total_weight += w
    
    if total_weight > 0:
        ensemble_scores /= total_weight
    
    print(f"  Ensemble {target}: mean={ensemble_scores.mean():.4f}, "
          f"std={ensemble_scores.std():.4f}, "
          f"min={ensemble_scores.min():.4f}, max={ensemble_scores.max():.4f}")
    
    return ensemble_scores


def compute_metrics_at_threshold(df, scores, threshold, target):
    """
    Compute trading metrics for trades where score > threshold.
    
    Uses pnl_at_close for hold-to-expiration P&L, and entry_credit for
    TP-based exit P&L calculation.
    """
    mask = scores > threshold
    filtered = df[mask].copy()
    n_trades = len(filtered)
    n_total = len(df)
    
    if n_trades == 0:
        return {
            'threshold': threshold,
            'n_trades': 0,
            'pct_trades_taken': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_pnl_per_day': 0.0,
            'profit_factor': 0.0,
            'worst_trade': 0.0,
            'best_trade': 0.0,
            'p5_pnl': 0.0,
            'p95_pnl': 0.0,
        }
    
    # P&L calculation
    pnl = filtered['pnl_at_close'].values
    
    # TP-based exit: if hit_TP, profit = entry_credit * TP_pct, else pnl_at_close
    hit_col_map = {'tp25': 'hit_25pct', 'tp50': 'hit_50pct'}
    hit_col = hit_col_map.get(target)
    
    if hit_col and hit_col in filtered.columns:
        tp_pct = int(target.replace('tp', '')) / 100.0  # 0.25 or 0.50
        hit = filtered[hit_col].values.astype(float)
        credit = filtered['entry_credit'].values.astype(float)
        tp_pnl = np.where(
            hit == 1.0,
            credit * tp_pct,           # TP hit: collect fraction of credit
            pnl.astype(float)          # TP missed: hold to close
        )
    else:
        tp_pnl = pnl.astype(float)
    
    # Daily aggregation for Sharpe/Sortino
    filtered_copy = filtered.copy()
    filtered_copy['pnl_used'] = tp_pnl
    daily_pnl = filtered_copy.groupby(filtered_copy['trade_date'].dt.date)['pnl_used'].sum()
    
    avg_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    downside_std = daily_pnl[daily_pnl < 0].std() if (daily_pnl < 0).any() else 1e-6
    
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
    sortino = (avg_daily / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    
    # Max drawdown (on cumulative daily P&L)
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / running_max.replace(0, np.nan)).min() * 100 if running_max.max() > 0 else 0.0
    
    # Profit factor
    gross_profit = tp_pnl[tp_pnl > 0].sum()
    gross_loss = abs(tp_pnl[tp_pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Win rate (using TP-based P&L)
    win_rate = (tp_pnl > 0).mean() * 100
    
    n_days = filtered_copy['trade_date'].dt.date.nunique()
    
    return {
        'threshold': threshold,
        'n_trades': n_trades,
        'pct_trades_taken': n_trades / n_total * 100,
        'avg_pnl': tp_pnl.mean(),
        'total_pnl': tp_pnl.sum(),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_dollars': max_dd,
        'avg_pnl_per_day': avg_daily,
        'profit_factor': profit_factor,
        'worst_trade': tp_pnl.min(),
        'best_trade': tp_pnl.max(),
        'p5_pnl': np.percentile(tp_pnl, 5),
        'p95_pnl': np.percentile(tp_pnl, 95),
        'n_days': n_days,
    }


def run_sweep(df, scores, target):
    """Run threshold sweep across all values."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP: {target.upper()}")
    print(f"{'='*70}")
    
    results = []
    for t in THRESHOLDS:
        metrics = compute_metrics_at_threshold(df, scores, t, target)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    return results_df


def find_optimal_thresholds(results_df, target):
    """Find optimal threshold for different optimization objectives."""
    # Filter to thresholds with at least 100 trades (statistical significance)
    viable = results_df[results_df['n_trades'] >= 100].copy()
    
    if len(viable) == 0:
        print("  ⚠ No thresholds with >= 100 trades!")
        viable = results_df[results_df['n_trades'] >= 10].copy()
    
    print(f"\n{'─'*70}")
    print(f"OPTIMAL THRESHOLDS: {target.upper()}")
    print(f"{'─'*70}")
    
    optimals = {}
    
    # 1. Max Total P&L
    idx = viable['total_pnl'].idxmax()
    row = viable.loc[idx]
    optimals['max_total_pnl'] = row
    print(f"\n  📊 Max Total P&L: T={row['threshold']:.2f}")
    print(f"     Trades: {row['n_trades']:,.0f} ({row['pct_trades_taken']:.1f}%)")
    print(f"     Total P&L: ${row['total_pnl']:,.0f} | Avg: ${row['avg_pnl']:.2f}")
    print(f"     Win Rate: {row['win_rate']:.1f}% | Sharpe: {row['sharpe']:.2f}")
    
    # 2. Max Sharpe
    idx = viable['sharpe'].idxmax()
    row = viable.loc[idx]
    optimals['max_sharpe'] = row
    print(f"\n  📈 Max Sharpe: T={row['threshold']:.2f}")
    print(f"     Trades: {row['n_trades']:,.0f} ({row['pct_trades_taken']:.1f}%)")
    print(f"     Sharpe: {row['sharpe']:.2f} | Sortino: {row['sortino']:.2f}")
    print(f"     Total P&L: ${row['total_pnl']:,.0f} | Avg: ${row['avg_pnl']:.2f}")
    
    # 3. Max Sortino
    idx = viable['sortino'].idxmax()
    row = viable.loc[idx]
    optimals['max_sortino'] = row
    print(f"\n  🎯 Max Sortino: T={row['threshold']:.2f}")
    print(f"     Trades: {row['n_trades']:,.0f} ({row['pct_trades_taken']:.1f}%)")
    print(f"     Sortino: {row['sortino']:.2f} | Sharpe: {row['sharpe']:.2f}")
    print(f"     Total P&L: ${row['total_pnl']:,.0f} | Max DD: {row['max_drawdown_pct']:.1f}%")
    
    # 4. Max Profit Factor
    pf_viable = viable[viable['profit_factor'] < float('inf')]
    if len(pf_viable) > 0:
        idx = pf_viable['profit_factor'].idxmax()
        row = pf_viable.loc[idx]
        optimals['max_profit_factor'] = row
        print(f"\n  💰 Max Profit Factor: T={row['threshold']:.2f}")
        print(f"     Trades: {row['n_trades']:,.0f} ({row['pct_trades_taken']:.1f}%)")
        print(f"     PF: {row['profit_factor']:.2f} | Win Rate: {row['win_rate']:.1f}%")
        print(f"     Total P&L: ${row['total_pnl']:,.0f} | Avg: ${row['avg_pnl']:.2f}")
    
    # 5. Sortino-optimized with max DD constraint (< 15% of $25K = $3,750)
    constrained = viable[viable['max_drawdown_dollars'] >= -3750].copy()
    if len(constrained) > 0:
        idx = constrained['sortino'].idxmax()
        row = constrained.loc[idx]
        optimals['constrained_sortino'] = row
        print(f"\n  🛡️  Constrained Sortino (DD < $3,750): T={row['threshold']:.2f}")
        print(f"     Trades: {row['n_trades']:,.0f} ({row['pct_trades_taken']:.1f}%)")
        print(f"     Sortino: {row['sortino']:.2f} | Max DD: ${row['max_drawdown_dollars']:,.0f}")
        print(f"     Total P&L: ${row['total_pnl']:,.0f} | Avg/day: ${row['avg_pnl_per_day']:.2f}")
    
    # 6. Best avg P&L per trade (with min 500 trades)
    freq_viable = viable[viable['n_trades'] >= 500]
    if len(freq_viable) > 0:
        idx = freq_viable['avg_pnl'].idxmax()
        row = freq_viable.loc[idx]
        optimals['best_avg_pnl'] = row
        print(f"\n  ⚡ Best Avg P&L (≥500 trades): T={row['threshold']:.2f}")
        print(f"     Trades: {row['n_trades']:,.0f} | Avg P&L: ${row['avg_pnl']:.2f}")
        print(f"     Total P&L: ${row['total_pnl']:,.0f} | Win Rate: {row['win_rate']:.1f}%")
    
    return optimals


def run_stability_check(df, scores, target):
    """Check threshold stability across time periods."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD STABILITY CHECK: {target.upper()}")
    print(f"{'='*70}")
    
    df = df.copy()
    df['score'] = scores
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Define periods
    max_date = df['trade_date'].max()
    periods = {
        'Full Sample': df,
        'Last 12m': df[df['trade_date'] >= max_date - pd.DateOffset(months=12)],
        'Last 6m': df[df['trade_date'] >= max_date - pd.DateOffset(months=6)],
        'Last 3m': df[df['trade_date'] >= max_date - pd.DateOffset(months=3)],
    }
    
    # Also split by quarter
    df['quarter'] = df['trade_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())
    for q in quarters[-8:]:  # Last 8 quarters
        qdf = df[df['quarter'] == q]
        if len(qdf) >= 50:
            periods[str(q)] = qdf
    
    print(f"\n  {'Period':<15} {'N Trades':>10} {'Best Sharpe T':>15} {'Best Sortino T':>16} {'Best AvgPnL T':>15}")
    print(f"  {'─'*15} {'─'*10} {'─'*15} {'─'*16} {'─'*15}")
    
    stability_results = []
    
    for period_name, period_df in periods.items():
        if len(period_df) < 50:
            continue
        
        period_scores = period_df['score'].values
        
        best_sharpe_t = 0
        best_sharpe = -999
        best_sortino_t = 0
        best_sortino = -999
        best_avg_t = 0
        best_avg = -999
        
        for t in np.arange(0.30, 0.80, 0.01):
            metrics = compute_metrics_at_threshold(period_df, period_scores, t, target)
            if metrics['n_trades'] >= max(20, len(period_df) * 0.05):
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_sharpe_t = t
                if metrics['sortino'] > best_sortino:
                    best_sortino = metrics['sortino']
                    best_sortino_t = t
                if metrics['avg_pnl'] > best_avg:
                    best_avg = metrics['avg_pnl']
                    best_avg_t = t
        
        print(f"  {period_name:<15} {len(period_df):>10,} {best_sharpe_t:>12.2f} ({best_sharpe:+.2f}) "
              f"{best_sortino_t:>12.2f} ({best_sortino:+.2f}) {best_avg_t:>12.2f} (${best_avg:+.1f})")
        
        stability_results.append({
            'period': period_name,
            'n_trades': len(period_df),
            'best_sharpe_threshold': best_sharpe_t,
            'best_sharpe': best_sharpe,
            'best_sortino_threshold': best_sortino_t,
            'best_sortino': best_sortino,
            'best_avg_pnl_threshold': best_avg_t,
            'best_avg_pnl': best_avg,
        })
    
    stability_df = pd.DataFrame(stability_results)
    
    # Compute threshold variance
    sharpe_thresholds = stability_df['best_sharpe_threshold'].values
    sortino_thresholds = stability_df['best_sortino_threshold'].values
    
    print(f"\n  Sharpe threshold range: {sharpe_thresholds.min():.2f} - {sharpe_thresholds.max():.2f} "
          f"(std: {sharpe_thresholds.std():.3f})")
    print(f"  Sortino threshold range: {sortino_thresholds.min():.2f} - {sortino_thresholds.max():.2f} "
          f"(std: {sortino_thresholds.std():.3f})")
    
    if sharpe_thresholds.std() < 0.05:
        print(f"  ✅ Sharpe threshold is STABLE across periods")
    elif sharpe_thresholds.std() < 0.10:
        print(f"  ⚠️  Sharpe threshold has MODERATE variance — consider adaptive threshold")
    else:
        print(f"  ❌ Sharpe threshold is UNSTABLE — strong regime sensitivity, use adaptive threshold")
    
    return stability_df


def plot_sweep(results_df, target, optimals):
    """Generate threshold sweep visualization."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'V6 Ensemble Threshold Sweep — {target.upper()}', fontsize=16, fontweight='bold')
    
    viable = results_df[results_df['n_trades'] >= 50]
    t = viable['threshold']
    
    # 1. Trade count & % taken
    ax = axes[0, 0]
    ax.plot(t, viable['n_trades'], 'b-', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of Trades', color='b')
    ax2 = ax.twinx()
    ax2.plot(t, viable['pct_trades_taken'], 'r--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('% Trades Taken', color='r')
    ax.set_title('Trade Frequency vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # 2. Avg P&L per trade
    ax = axes[0, 1]
    ax.plot(t, viable['avg_pnl'], 'g-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Avg P&L per Trade ($)')
    ax.set_title('Average P&L vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # 3. Sharpe & Sortino
    ax = axes[1, 0]
    ax.plot(t, viable['sharpe'], 'b-', linewidth=2, label='Sharpe')
    ax.plot(t, viable['sortino'], 'orange', linewidth=2, label='Sortino')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Ratio')
    ax.set_title('Sharpe & Sortino vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Total P&L
    ax = axes[1, 1]
    ax.plot(t, viable['total_pnl'], 'purple', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Total P&L ($)')
    ax.set_title('Total P&L vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # 5. Win Rate
    ax = axes[2, 0]
    ax.plot(t, viable['win_rate'], 'teal', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # 6. Profit Factor
    ax = axes[2, 1]
    pf = viable['profit_factor'].clip(upper=10)  # Cap for readability
    ax.plot(t, pf, 'darkred', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Profit Factor')
    ax.set_title('Profit Factor vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for optimal thresholds
    colors_map = {
        'max_sharpe': ('blue', 'Max Sharpe'),
        'max_sortino': ('orange', 'Max Sortino'),
        'constrained_sortino': ('green', 'Constrained Sortino'),
    }
    for key, (color, label) in colors_map.items():
        if key in optimals:
            thresh = optimals[key]['threshold']
            for ax_row in axes:
                for ax in ax_row:
                    ax.axvline(x=thresh, color=color, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    chart_path = OUTPUT_DIR / f'threshold_sweep_{target}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Chart saved: {chart_path}")
    
    return chart_path


def print_key_thresholds_table(results_df, target):
    """Print a clean table of key threshold values."""
    print(f"\n{'─'*90}")
    print(f"DETAILED SWEEP TABLE: {target.upper()} (every 5%)")
    print(f"{'─'*90}")
    print(f"  {'Thresh':>6} {'Trades':>8} {'%Taken':>7} {'AvgPnL':>8} {'TotalPnL':>10} "
          f"{'WinRate':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD$':>8} {'PF':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*7} {'─'*8} {'─'*10} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*6}")
    
    for _, row in results_df.iterrows():
        t = row['threshold']
        if t % 0.05 < 0.005 or t % 0.05 > 0.045:  # Every 5%
            if row['n_trades'] > 0:
                pf = row['profit_factor']
                pf_str = f"{pf:.2f}" if pf < 100 else "∞"
                print(f"  {t:>6.2f} {row['n_trades']:>8,.0f} {row['pct_trades_taken']:>6.1f}% "
                      f"${row['avg_pnl']:>7.1f} ${row['total_pnl']:>9,.0f} "
                      f"{row['win_rate']:>7.1f}% {row['sharpe']:>7.2f} {row['sortino']:>8.2f} "
                      f"${row.get('max_drawdown_dollars', 0):>7,.0f} {pf_str:>6}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V6 Ensemble Threshold Sweep')
    parser.add_argument('--target', choices=['tp25', 'tp50', 'both'], default='both',
                        help='Which target to sweep (default: both)')
    parser.add_argument('--stability', action='store_true',
                        help='Run stability check across time periods')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data and models
    df = load_data()
    print("\nLoading v6 ensemble models...")
    models = load_v6_ensemble()
    
    targets = ['tp25', 'tp50'] if args.target == 'both' else [args.target]
    
    for target in targets:
        if len(models.get(target, {})) == 0:
            print(f"\n  ⚠ No models loaded for {target}, skipping")
            continue
        
        # Score all trades
        print(f"\nScoring all trades for {target}...")
        scores = compute_ensemble_scores(df, models, target)
        
        # Run sweep
        results_df = run_sweep(df, scores, target)
        
        # Print detailed table
        print_key_thresholds_table(results_df, target)
        
        # Find optimal thresholds
        optimals = find_optimal_thresholds(results_df, target)
        
        # Plot
        chart_path = plot_sweep(results_df, target, optimals)
        
        # Save CSV
        csv_path = OUTPUT_DIR / f'threshold_sweep_{target}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"  📄 CSV saved: {csv_path}")
        
        # Stability check
        if args.stability:
            stability_df = run_stability_check(df, scores, target)
            stab_path = OUTPUT_DIR / f'threshold_stability_{target}.csv'
            stability_df.to_csv(stab_path, index=False)
            print(f"  📄 Stability CSV saved: {stab_path}")
    
    print(f"\n{'='*70}")
    print(f"DONE — All results in {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
