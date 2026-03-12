#!/usr/bin/env python3
"""
Export Schwab account balance history to CSV.

Two modes:
  snapshot  - Record today's balance (run daily via cron to build history)
  export    - Export accumulated snapshots to CSV
  txns      - Export last 60 days of transactions to CSV

Usage:
    venv/bin/python3 scripts/export_balance_history.py snapshot         # save today's balance
    venv/bin/python3 scripts/export_balance_history.py export           # export all snapshots
    venv/bin/python3 scripts/export_balance_history.py txns             # export transactions
    venv/bin/python3 scripts/export_balance_history.py txns --days 30   # last 30 days
"""

import sys
import os
import csv
import json
import asyncio
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from execution.broker_api.schwab_client import schwab_client

SNAPSHOT_FILE = Path("data_store/daily_balances.json")


# ============================================================================
# SNAPSHOT MODE — save today's EOD balance (run daily)
# ============================================================================

async def save_snapshot():
    """Fetch current balance from Schwab and append to snapshot file."""
    logger.info("Fetching current account balance...")
    account = await schwab_client.get_account()
    sec = account.get('securitiesAccount', {})

    current = sec.get('currentBalances', {})
    initial = sec.get('initialBalances', {})

    snapshot = {
        'date': date.today().isoformat(),
        'timestamp': datetime.now().isoformat(),
        'liquidation_value': current.get('liquidationValue', 0.0),
        'sod_liquidation_value': initial.get('liquidationValue', 0.0),
        'cash_balance': current.get('cashBalance', 0.0),
        'buying_power': current.get('buyingPower', 0.0),
        'long_market_value': current.get('longMarketValue', 0.0),
        'short_market_value': current.get('shortMarketValue', 0.0),
        'day_pnl': current.get('liquidationValue', 0.0) - initial.get('liquidationValue', 0.0),
    }

    # Load existing snapshots
    snapshots = _load_snapshots()

    # Replace if same date exists (idempotent)
    today_str = date.today().isoformat()
    snapshots = [s for s in snapshots if s['date'] != today_str]
    snapshots.append(snapshot)
    snapshots.sort(key=lambda s: s['date'])

    # Save
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, 'w') as f:
        json.dump(snapshots, f, indent=2)

    lv = snapshot['liquidation_value']
    pnl = snapshot['day_pnl']
    print(f"Saved snapshot for {today_str}: ${lv:,.2f} (day P&L: ${pnl:+,.2f})")
    logger.info(f"Snapshot saved to {SNAPSHOT_FILE} ({len(snapshots)} total days)")


def _load_snapshots() -> list[dict]:
    """Load existing snapshots from file."""
    if SNAPSHOT_FILE.exists():
        with open(SNAPSHOT_FILE, 'r') as f:
            return json.load(f)
    return []


# ============================================================================
# EXPORT MODE — export snapshots to CSV
# ============================================================================

def export_snapshots(output_path: str):
    """Export accumulated snapshots to CSV."""
    snapshots = _load_snapshots()

    if not snapshots:
        print("No snapshots found. Run 'snapshot' mode first to capture daily balances.")
        print(f"  venv/bin/python3 {__file__} snapshot")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'eod_balance', 'sod_balance', 'day_pnl', 'day_pnl_pct',
            'cash_balance', 'buying_power', 'long_market_value', 'short_market_value',
        ])

        for snap in snapshots:
            lv = snap['liquidation_value']
            sod = snap.get('sod_liquidation_value', lv)
            pnl = snap.get('day_pnl', 0.0)
            pct = (pnl / sod * 100) if sod else 0.0

            writer.writerow([
                snap['date'],
                f"{lv:.2f}",
                f"{sod:.2f}",
                f"{pnl:.2f}",
                f"{pct:.2f}",
                f"{snap.get('cash_balance', 0):.2f}",
                f"{snap.get('buying_power', 0):.2f}",
                f"{snap.get('long_market_value', 0):.2f}",
                f"{snap.get('short_market_value', 0):.2f}",
            ])

    print(f"Exported {len(snapshots)} days to {output_path}")

    # Summary
    if len(snapshots) >= 2:
        first_lv = snapshots[0]['liquidation_value']
        last_lv = snapshots[-1]['liquidation_value']
        total_change = last_lv - first_lv
        total_pct = (total_change / first_lv * 100) if first_lv else 0

        print(f"\n{'='*50}")
        print(f"  Period: {snapshots[0]['date']} -> {snapshots[-1]['date']}")
        print(f"  Start:  ${first_lv:,.2f}")
        print(f"  End:    ${last_lv:,.2f}")
        print(f"  Change: ${total_change:+,.2f} ({total_pct:+.2f}%)")
        print(f"  Days:   {len(snapshots)}")
        print(f"{'='*50}")


# ============================================================================
# TRANSACTIONS MODE — export raw transaction ledger
# ============================================================================

async def export_transactions(days: int, output_path: str):
    """Export transactions for the last N days to CSV."""
    end = date.today()
    start = end - timedelta(days=days)
    logger.info(f"Fetching transactions from {start} to {end}...")

    transactions = await schwab_client.get_transactions(
        start_date=start,
        end_date=end,
    )
    logger.info(f"Got {len(transactions)} transactions")

    if not transactions:
        print("No transactions found.")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'time', 'type', 'description', 'symbol',
            'net_amount', 'order_id', 'activity_id',
        ])

        for txn in sorted(transactions, key=lambda t: t.get('time', '')):
            txn_time = txn.get('time', '')
            txn_date = txn_time[:10] if txn_time else ''
            txn_clock = txn_time[11:19] if len(txn_time) > 19 else ''

            # Extract symbol from transferItems
            symbol = ''
            for item in txn.get('transferItems', []):
                inst = item.get('instrument', {})
                s = inst.get('symbol', '')
                if s and s != 'CURRENCY_USD':
                    symbol = s
                    break

            writer.writerow([
                txn_date,
                txn_clock,
                txn.get('type', ''),
                txn.get('description', ''),
                symbol,
                f"{txn.get('netAmount', 0):.2f}",
                txn.get('orderId', ''),
                txn.get('activityId', ''),
            ])

    print(f"Exported {len(transactions)} transactions to {output_path}")

    # Summary by type
    from collections import Counter
    types = Counter(t.get('type', 'UNKNOWN') for t in transactions)
    print(f"\nTransaction types:")
    for t, count in types.most_common():
        total = sum(txn.get('netAmount', 0) for txn in transactions if txn.get('type') == t)
        print(f"  {t:30s}  {count:4d} txns  ${total:+,.2f}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='Schwab balance history tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s snapshot                    Save today's balance
  %(prog)s export                      Export all saved balances to CSV
  %(prog)s export -o ~/Desktop/b.csv   Export to specific file
  %(prog)s txns                        Export 60 days of transactions
  %(prog)s txns --days 30              Export 30 days of transactions

Tip: Run 'snapshot' daily (e.g. via cron at 4pm ET) to build balance history.
     Schwab API has no historical balance endpoint, so snapshots are the only
     way to get accurate EOD balances over time.
        """
    )
    parser.add_argument('mode', choices=['snapshot', 'export', 'txns'],
                        help='snapshot=save today, export=CSV of saved balances, txns=transaction ledger')
    parser.add_argument('--days', type=int, default=60, help='Days for txns mode (max 60)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    if args.mode == 'snapshot':
        await save_snapshot()

    elif args.mode == 'export':
        output = args.output or os.path.expanduser('~/Desktop/balance_history.csv')
        export_snapshots(output)

    elif args.mode == 'txns':
        days = min(args.days, 60)
        output = args.output or os.path.expanduser('~/Desktop/schwab_transactions.csv')
        await export_transactions(days, output)


if __name__ == '__main__':
    asyncio.run(main())
