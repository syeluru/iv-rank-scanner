#!/usr/bin/env python3
"""
Analyze past month of IC trades: how many would have yielded full credit
(expired worthless) if held to expiration vs. where they were actually closed.

Combines data from:
1. JSONL bot logs (explicit strikes, recent trades)
2. trade_tracking.json (all historical trades, parsed from option symbols)

Outputs:
- CSV with per-trade analysis
- HTML chart showing results
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv


def parse_option_symbol(sym):
    """Parse SPXW option symbol to extract expiry, type, strike.

    Format: 'SPXW  260210C06980000'
    -> expiry=2026-02-10, type=CALL, strike=6980.0
    """
    sym = sym.strip()
    # Match pattern: root + whitespace + YYMMDD + C/P + strike
    m = re.match(r'(\w+)\s+(\d{6})([CP])(\d{8})', sym)
    if not m:
        return None
    root, exp_str, cp, strike_str = m.groups()

    year = 2000 + int(exp_str[:2])
    month = int(exp_str[2:4])
    day = int(exp_str[4:6])

    return {
        'root': root,
        'expiry': date(year, month, day),
        'option_type': 'CALL' if cp == 'C' else 'PUT',
        'strike': int(strike_str) / 1000.0,
    }


def extract_ic_from_trade(trade):
    """Extract IC strikes from a trade_tracking.json trade entry.

    Returns dict with long_put, short_put, short_call, long_call, expiry
    or None if not a valid 4-leg IC.
    """
    # Collect all leg symbols (from both current_legs and closed_legs)
    all_symbols = set()
    all_symbols.update(trade.get('current_legs', {}).keys())
    all_symbols.update(trade.get('closed_legs', {}).keys())

    # Parse each symbol
    parsed = []
    for sym in all_symbols:
        p = parse_option_symbol(sym)
        if p:
            parsed.append(p)

    # Separate puts and calls
    puts = sorted([p for p in parsed if p['option_type'] == 'PUT'], key=lambda x: x['strike'])
    calls = sorted([p for p in parsed if p['option_type'] == 'CALL'], key=lambda x: x['strike'])

    if len(puts) < 2 or len(calls) < 2:
        return None

    # For an IC with possible rolls (>4 legs), find the widest spread
    # Use the FINAL set of legs (the last roll)
    # Long put = lowest put, Short put = highest put
    # Short call = lowest call, Long call = highest call
    # But with rolls, we may have overlapping strikes.
    # Best approach: take the two most extreme puts and two most extreme calls
    # that form a valid IC structure

    # For 0DTE, all legs should share the same expiry
    # Group by expiry and take the latest expiry group (most recent roll)
    expiry_groups = defaultdict(lambda: {'puts': [], 'calls': []})
    for p in parsed:
        if p['option_type'] == 'PUT':
            expiry_groups[p['expiry']]['puts'].append(p['strike'])
        else:
            expiry_groups[p['expiry']]['calls'].append(p['strike'])

    # Find the expiry that has a valid IC (at least 2 puts + 2 calls)
    valid_ics = []
    for expiry, legs in expiry_groups.items():
        p_strikes = sorted(set(legs['puts']))
        c_strikes = sorted(set(legs['calls']))
        if len(p_strikes) >= 2 and len(c_strikes) >= 2:
            # Standard IC: long_put < short_put < short_call < long_call
            valid_ics.append({
                'expiry': expiry,
                'long_put': p_strikes[0],      # lowest put = long (protective)
                'short_put': p_strikes[-1],     # highest put = short (sold)
                'short_call': c_strikes[0],     # lowest call = short (sold)
                'long_call': c_strikes[-1],     # highest call = long (protective)
            })

    if not valid_ics:
        return None

    # If multiple valid ICs (due to rolls), take the one with latest expiry
    valid_ics.sort(key=lambda x: x['expiry'])
    return valid_ics[-1]


def load_jsonl_trades():
    """Load trades from JSONL bot logs."""
    logs_dir = PROJECT_ROOT / 'logs'
    trades = []

    for jsonl_file in sorted(logs_dir.glob('zero_dte_bot_*.jsonl')):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                strikes = entry.get('strikes', {})
                if not all(k in strikes for k in ['long_put', 'short_put', 'short_call', 'long_call']):
                    continue

                trade_date = datetime.fromisoformat(entry['timestamp']).date()

                trades.append({
                    'source': 'jsonl',
                    'trade_date': trade_date,
                    'expiry': trade_date,  # 0DTE
                    'timestamp': entry['timestamp'],
                    'long_put': strikes['long_put'],
                    'short_put': strikes['short_put'],
                    'short_call': strikes['short_call'],
                    'long_call': strikes['long_call'],
                    'quantity': entry.get('quantity', 1),
                    'credit': entry.get('credit', 0),
                    'exit_reason': entry.get('exit_reason', 'unknown'),
                    'mode': entry.get('mode', 'unknown'),
                    'label': entry.get('label', ''),
                    'portfolio_value': entry.get('portfolio_value', 0),
                })

    return trades


def load_tracking_trades(cutoff_date):
    """Load IC trades from trade_tracking.json since cutoff_date."""
    tracking_file = PROJECT_ROOT / 'data' / 'storage' / 'trade_tracking.json'

    with open(tracking_file) as f:
        data = json.load(f)

    trades = []
    seen_order_ids = set()

    for trade_id, trade in data['trades'].items():
        if trade['strategy'] != 'IRON_CONDOR':
            continue

        # Parse opened_at
        opened_str = trade['opened_at']
        try:
            opened = datetime.fromisoformat(opened_str.replace('+0000', '+00:00'))
        except ValueError:
            continue

        trade_date = opened.date()
        if trade_date < cutoff_date:
            continue

        # Deduplicate by opened_order_id
        oid = trade.get('opened_order_id')
        if oid in seen_order_ids:
            continue
        seen_order_ids.add(oid)

        # Extract IC structure
        ic = extract_ic_from_trade(trade)
        if ic is None:
            continue

        trades.append({
            'source': 'tracking',
            'trade_id': trade_id,
            'trade_date': trade_date,
            'expiry': ic['expiry'],
            'timestamp': opened_str,
            'long_put': ic['long_put'],
            'short_put': ic['short_put'],
            'short_call': ic['short_call'],
            'long_call': ic['long_call'],
            'quantity': trade.get('quantity', 1),
            'credit': 0,  # Not available in tracking data
            'exit_reason': 'closed' if trade['status'] == 'CLOSED' else 'open',
            'mode': 'live',
            'label': '',
            'status': trade['status'],
        })

    return trades


def fetch_spx_closes(dates):
    """Fetch SPX closing prices from parquet cache + ThetaData intraday cache."""
    if not dates:
        return {}

    closes = {}

    # Source 1: yfinance daily parquet cache
    parquet_path = PROJECT_ROOT / 'ml' / 'training_data' / 'cache' / 'yf_daily.parquet'
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        spx = df[df['_ticker'] == 'SPX']
        for d in dates:
            ts = pd.Timestamp(d)
            if ts in spx.index:
                closes[d] = float(spx.loc[ts, 'Close'])

    # Source 2: ThetaData intraday cache (for dates not in parquet)
    intraday_dir = PROJECT_ROOT / 'ml' / 'training_data' / 'cache' / 'spx_intraday'
    missing = [d for d in dates if d not in closes]
    for d in missing:
        intraday_file = intraday_dir / f'{d}.json'
        if intraday_file.exists():
            with open(intraday_file) as f:
                candles = json.load(f)
            if candles:
                # Last candle close = market close
                closes[d] = candles[-1]['close']

    # Source 3: yfinance live fetch for any remaining missing dates
    still_missing = [d for d in dates if d not in closes]
    if still_missing:
        try:
            import yfinance as yf
            min_d = min(still_missing) - timedelta(days=3)
            max_d = max(still_missing) + timedelta(days=3)
            df = yf.download('^GSPC', start=str(min_d), end=str(max_d), progress=False)
            for d in still_missing:
                ts = pd.Timestamp(d)
                if ts in df.index:
                    val = df.loc[ts, 'Close']
                    closes[d] = float(val.iloc[0] if hasattr(val, 'iloc') else val)
        except Exception as e:
            print(f"   Warning: yfinance fetch failed: {e}")

    return closes


def deduplicate_trades(jsonl_trades, tracking_trades):
    """Combine and deduplicate trades from both sources.

    JSONL trades take priority (they have more data like credit, exit_reason).
    Dedup by matching date + similar strikes.
    """
    # Index JSONL trades by (date, short_put, short_call)
    jsonl_keys = set()
    for t in jsonl_trades:
        key = (t['trade_date'], t['short_put'], t['short_call'])
        jsonl_keys.add(key)

    # Only add tracking trades that don't overlap with JSONL
    combined = list(jsonl_trades)
    for t in tracking_trades:
        key = (t['expiry'], t['short_put'], t['short_call'])  # Use expiry for 0DTE
        if key not in jsonl_keys:
            combined.append(t)

    # Sort by date then timestamp
    combined.sort(key=lambda x: x['timestamp'])
    return combined


def analyze_trades(trades, spx_closes):
    """Analyze each trade: would it have yielded full credit at expiration?"""
    results = []

    for t in trades:
        expiry = t['expiry']
        spx_close = spx_closes.get(expiry)

        if spx_close is None:
            max_profit = None
            within_bounds = None
            distance_to_short = None
        else:
            short_put = t['short_put']
            short_call = t['short_call']

            within_bounds = short_put <= spx_close <= short_call

            # How far from the nearest short strike?
            if spx_close < short_put:
                distance_to_short = spx_close - short_put  # negative = breached put
            elif spx_close > short_call:
                distance_to_short = spx_close - short_call  # positive = breached call
            else:
                distance_to_short = min(spx_close - short_put, short_call - spx_close)

            max_profit = within_bounds

        results.append({
            'trade_date': t['trade_date'].isoformat(),
            'expiry': expiry.isoformat(),
            'timestamp': t['timestamp'],
            'long_put': t['long_put'],
            'short_put': t['short_put'],
            'short_call': t['short_call'],
            'long_call': t['long_call'],
            'ic_width': t['short_call'] - t['short_put'],
            'quantity': t['quantity'],
            'credit': t.get('credit', 0),
            'exit_reason': t.get('exit_reason', ''),
            'mode': t.get('mode', ''),
            'spx_close': spx_close,
            'within_bounds': within_bounds,
            'max_profit_possible': 'YES' if within_bounds else ('NO' if within_bounds is not None else 'N/A'),
            'distance_to_nearest_short': round(distance_to_short, 2) if distance_to_short is not None else None,
            'label': t.get('label', ''),
        })

    return results


def write_csv(results, output_path):
    """Write results to CSV."""
    if not results:
        print("No results to write.")
        return

    fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV written to: {output_path}")


def create_chart(results, output_path):
    """Create an interactive chart showing max profit analysis."""
    # Filter to trades with SPX close data
    valid = [r for r in results if r['spx_close'] is not None]
    if not valid:
        print("No valid trades with SPX data for charting.")
        return

    # Group by expiry date for the daily view
    daily = defaultdict(list)
    for r in valid:
        daily[r['expiry']].append(r)

    dates_sorted = sorted(daily.keys())

    # --- Chart 1: Per-trade scatter showing SPX close vs IC bounds ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'SPX Close vs IC Bounds (Per Trade)',
            'Max Profit Rate by Day',
            'Distance to Nearest Short Strike',
            'Summary Statistics',
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Trade index for x-axis
    for i, r in enumerate(valid):
        color = '#00cc66' if r['within_bounds'] else '#ff4444'
        marker_symbol = 'diamond' if r['within_bounds'] else 'x'

        # SPX close point
        fig.add_trace(go.Scatter(
            x=[i], y=[r['spx_close']],
            mode='markers',
            marker=dict(color=color, size=10, symbol=marker_symbol),
            name='Max Profit' if r['within_bounds'] else 'Breached',
            showlegend=(i == 0 or (i > 0 and valid[i-1]['within_bounds'] != r['within_bounds'])),
            hovertext=f"{r['expiry']}<br>SPX: {r['spx_close']:.0f}<br>"
                      f"Short Put: {r['short_put']:.0f}<br>Short Call: {r['short_call']:.0f}<br>"
                      f"Exit: {r['exit_reason']}<br>Credit: ${r['credit']:.2f}",
            hoverinfo='text',
        ), row=1, col=1)

        # IC bounds as vertical range
        fig.add_trace(go.Scatter(
            x=[i, i], y=[r['short_put'], r['short_call']],
            mode='lines',
            line=dict(color='rgba(100,100,255,0.3)', width=8),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="SPX Price", row=1, col=1)

    # --- Chart 2: Daily max profit rate ---
    day_labels = []
    day_rates = []
    day_colors = []
    for d in dates_sorted:
        trades_on_day = daily[d]
        n_max = sum(1 for t in trades_on_day if t['within_bounds'])
        rate = n_max / len(trades_on_day) * 100
        day_labels.append(d)
        day_rates.append(rate)
        day_colors.append('#00cc66' if rate == 100 else ('#ffaa00' if rate > 0 else '#ff4444'))

    fig.add_trace(go.Bar(
        x=day_labels, y=day_rates,
        marker_color=day_colors,
        text=[f"{r:.0f}%" for r in day_rates],
        textposition='auto',
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="% Would Hit Max Profit", range=[0, 110], row=1, col=2)

    # --- Chart 3: Distance to nearest short strike ---
    distances = [r['distance_to_nearest_short'] for r in valid]
    colors_dist = ['#00cc66' if r['within_bounds'] else '#ff4444' for r in valid]

    fig.add_trace(go.Scatter(
        x=list(range(len(valid))),
        y=distances,
        mode='markers',
        marker=dict(color=colors_dist, size=8),
        showlegend=False,
        hovertext=[f"{r['expiry']}: {r['distance_to_nearest_short']:.1f} pts" for r in valid],
        hoverinfo='text',
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
    fig.update_xaxes(title_text="Trade #", row=2, col=1)
    fig.update_yaxes(title_text="Distance to Short Strike (pts)", row=2, col=1)

    # --- Chart 4: Summary table ---
    total = len(valid)
    max_profit_count = sum(1 for r in valid if r['within_bounds'])
    breached_count = total - max_profit_count

    # Breakdown by breach side
    put_breaches = sum(1 for r in valid if not r['within_bounds'] and r['spx_close'] < r['short_put'])
    call_breaches = sum(1 for r in valid if not r['within_bounds'] and r['spx_close'] > r['short_call'])

    avg_distance_winners = 0
    avg_distance_losers = 0
    winners = [r for r in valid if r['within_bounds']]
    losers = [r for r in valid if not r['within_bounds']]
    if winners:
        avg_distance_winners = sum(r['distance_to_nearest_short'] for r in winners) / len(winners)
    if losers:
        avg_distance_losers = sum(abs(r['distance_to_nearest_short']) for r in losers) / len(losers)

    # Total credit left on table (only for JSONL trades with credit data)
    credit_trades = [r for r in valid if r['within_bounds'] and r['credit'] > 0]

    fig.add_trace(go.Table(
        header=dict(
            values=['Metric', 'Value'],
            fill_color='#1e1e2e',
            font=dict(color='white', size=13),
            align='left',
        ),
        cells=dict(
            values=[
                [
                    'Total Trades Analyzed',
                    'Would Have Hit Max Profit',
                    'Would NOT Have Hit Max Profit',
                    'Max Profit Rate',
                    'Put Side Breaches',
                    'Call Side Breaches',
                    'Avg Cushion (Winners)',
                    'Avg Breach Distance (Losers)',
                    'Unique Trading Days',
                    'Days with 100% Max Profit',
                ],
                [
                    str(total),
                    str(max_profit_count),
                    str(breached_count),
                    f"{max_profit_count/total*100:.1f}%",
                    str(put_breaches),
                    str(call_breaches),
                    f"{avg_distance_winners:.1f} pts",
                    f"{avg_distance_losers:.1f} pts",
                    str(len(dates_sorted)),
                    str(sum(1 for d in dates_sorted if all(t['within_bounds'] for t in daily[d]))),
                ],
            ],
            fill_color='#2a2a3e',
            font=dict(color='white', size=12),
            align='left',
            height=28,
        ),
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text="Iron Condor Max Profit Analysis — Past Month",
            font=dict(size=20, color='white'),
        ),
        template='plotly_dark',
        height=900,
        width=1400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    fig.write_html(str(output_path))
    print(f"Chart written to: {output_path}")


def main():
    cutoff = date(2026, 2, 10)  # Past month

    print("=" * 60)
    print("Iron Condor Max Profit Analysis")
    print(f"Analyzing trades from {cutoff} to present")
    print("=" * 60)

    # Load from both sources
    print("\n1. Loading JSONL bot logs...")
    jsonl_trades = load_jsonl_trades()
    print(f"   Found {len(jsonl_trades)} trades in JSONL logs")

    print("\n2. Loading trade_tracking.json...")
    tracking_trades = load_tracking_trades(cutoff)
    print(f"   Found {len(tracking_trades)} IC trades in tracking data")

    print("\n3. Deduplicating...")
    all_trades = deduplicate_trades(jsonl_trades, tracking_trades)
    print(f"   {len(all_trades)} unique trades after dedup")

    # Filter to live trades only (skip dry_run)
    live_trades = [t for t in all_trades if t.get('mode') != 'dry_run']
    print(f"   {len(live_trades)} live trades (excluding dry_run)")

    # Get unique expiry dates
    expiry_dates = list(set(t['expiry'] for t in live_trades))
    print(f"\n4. Fetching SPX closes for {len(expiry_dates)} dates...")
    spx_closes = fetch_spx_closes(expiry_dates)
    print(f"   Got closes for {len(spx_closes)} dates")
    for d in sorted(spx_closes):
        print(f"   {d}: SPX close = {spx_closes[d]:.2f}")

    # Analyze
    print("\n5. Analyzing trades...")
    results = analyze_trades(live_trades, spx_closes)

    # Summary
    valid_results = [r for r in results if r['within_bounds'] is not None]
    max_profit_count = sum(1 for r in valid_results if r['within_bounds'])
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total trades analyzed: {len(valid_results)}")
    if valid_results:
        print(f"  Would have hit MAX PROFIT: {max_profit_count} ({max_profit_count/len(valid_results)*100:.1f}%)")
        print(f"  Would NOT have hit max profit: {len(valid_results) - max_profit_count}")
    else:
        print("  No trades with SPX close data available.")
    print(f"{'=' * 60}")

    # Output
    output_dir = PROJECT_ROOT / 'output' / 'max_profit_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'max_profit_by_trade.csv'
    chart_path = output_dir / 'max_profit_analysis.html'

    write_csv(results, csv_path)
    create_chart(results, chart_path)

    print(f"\nDone! Open {chart_path} in your browser to view the chart.")


if __name__ == '__main__':
    main()
