#!/usr/bin/env python3
"""
Populate Trade Matches Database

Fetches orders, matches them into trade lifecycles, and saves to database.
This creates the foundation for intraday P&L tracking.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.broker_api.schwab_client import SchwabClient
from monitoring.services.trade_matcher import TradeMatchingEngine
from monitoring.services.trade_match_persistence import TradeMatchPersistence
from data.storage.database import DatabaseManager
from data.storage.schemas import get_all_create_statements
from config.settings import settings
from loguru import logger


async def populate_trade_matches(lookback_days: int = 14, clear_existing: bool = False):
    """
    Populate trade matches database.

    Args:
        lookback_days: Number of days to fetch
        clear_existing: Whether to clear existing matches first
    """

    logger.info("=" * 80)
    logger.info("POPULATING TRADE MATCHES DATABASE")
    logger.info("=" * 80)

    # Ensure database tables exist
    logger.info("\nEnsuring database schema is up to date...")
    db = DatabaseManager(settings.trades_db_path)
    with db.get_connection() as conn:
        for statement in get_all_create_statements():
            conn.execute(statement)
    logger.info("✅ Database schema ready")

    # Initialize services
    persistence = TradeMatchPersistence()

    if clear_existing:
        logger.info("\nClearing existing trade matches...")
        persistence.clear_all_matches()
        logger.info("✅ Existing matches cleared")

    # Fetch orders
    logger.info(f"\nFetching orders (last {lookback_days} days)...")
    client = SchwabClient()

    from_date = datetime.now() - timedelta(days=lookback_days)
    to_date = datetime.now()

    orders = await client.get_orders_for_account(
        status='FILLED',
        from_entered_datetime=from_date,
        to_entered_datetime=to_date
    )

    logger.info(f"✅ Fetched {len(orders)} orders")

    # Match trades
    logger.info("\nMatching trades...")
    matcher = TradeMatchingEngine()
    matched_trades = matcher.match_orders(orders)

    logger.info(f"✅ Matched {len(matched_trades)} trades")
    logger.info(f"  - Open: {sum(1 for t in matched_trades if t.status == 'OPEN')}")
    logger.info(f"  - Partially Closed: {sum(1 for t in matched_trades if t.status == 'PARTIALLY_CLOSED')}")
    logger.info(f"  - Fully Closed: {sum(1 for t in matched_trades if t.status == 'FULLY_CLOSED')}")
    logger.info(f"  - Expired: {sum(1 for t in matched_trades if t.status == 'EXPIRED')}")

    # Save to database
    logger.info("\nSaving to database...")
    persistence.save_matches(matched_trades)

    logger.info("✅ Saved to database")

    # Verify
    logger.info("\nVerifying database...")
    with db.get_connection() as conn:
        match_count = conn.execute("SELECT COUNT(*) FROM trade_matches").fetchone()[0]
        leg_count = conn.execute("SELECT COUNT(*) FROM trade_legs").fetchone()[0]
        roll_count = conn.execute("SELECT COUNT(*) FROM trade_rolls").fetchone()[0]

    logger.info(f"✅ Verification complete:")
    logger.info(f"  - {match_count} matches in database")
    logger.info(f"  - {leg_count} legs in database")
    logger.info(f"  - {roll_count} rolls in database")

    # Display summary
    total_realized = sum(t.realized_pnl for t in matched_trades)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nTotal Realized P&L: ${total_realized:,.2f}")
    logger.info(f"Average per Trade: ${total_realized / len(matched_trades):,.2f}")

    closed_trades = [t for t in matched_trades if t.status in ['FULLY_CLOSED', 'EXPIRED']]
    if closed_trades:
        winning_trades = [t for t in closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in closed_trades if t.realized_pnl < 0]

        logger.info(f"\nWin Rate: {len(winning_trades)}/{len(closed_trades)} ({len(winning_trades)/len(closed_trades)*100:.1f}%)")
        if winning_trades:
            logger.info(f"  Best Trade: ${max(t.realized_pnl for t in winning_trades):,.2f}")
        if losing_trades:
            logger.info(f"  Worst Trade: ${min(t.realized_pnl for t in losing_trades):,.2f}")

    logger.info("\n✅ Trade matches database populated successfully!\n")
    logger.info("Next steps:")
    logger.info("  1. Integrate ThetaData for historical prices")
    logger.info("  2. Calculate intraday P&L snapshots")
    logger.info("  3. Build visualization dashboard")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Populate trade matches database')
    parser.add_argument('--days', type=int, default=14, help='Days of history to fetch')
    parser.add_argument('--clear', action='store_true', help='Clear existing matches first')
    args = parser.parse_args()

    await populate_trade_matches(
        lookback_days=args.days,
        clear_existing=args.clear
    )


if __name__ == '__main__':
    asyncio.run(main())
