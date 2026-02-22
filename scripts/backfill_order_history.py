#!/usr/bin/env python3
"""
Backfill Order History Cache

One-time script to populate order_pnl_cache and daily_pnl_summary tables
with historical order data from Schwab API.

Usage:
    python scripts/backfill_order_history.py [--days N]

Options:
    --days N    Number of days to backfill (default: 14)
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.services.order_history_analyzer import OrderHistoryAnalyzer
from loguru import logger


async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Backfill order history cache')
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='Number of days of history to backfill (default: 14)'
    )
    args = parser.parse_args()

    lookback_days = args.days

    logger.info("=" * 60)
    logger.info(f"Backfilling order history - last {lookback_days} days")
    logger.info("=" * 60)

    # Create analyzer
    analyzer = OrderHistoryAnalyzer(lookback_days=lookback_days)

    try:
        # Refresh cache (fetches from API and updates database)
        await analyzer.refresh_cache()

        # Display summary
        stats = analyzer.get_summary_stats()

        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKFILL COMPLETE - Summary:")
        logger.info("=" * 60)
        logger.info(f"Total P&L:      ${stats['total_pnl']:,.2f}")
        logger.info(f"Total Trades:   {stats['total_trades']}")
        logger.info(f"Win Rate:       {stats['win_rate']:.1f}%")
        logger.info(f"Best Day:       ${stats['best_day']:,.2f}")
        logger.info(f"Worst Day:      ${stats['worst_day']:,.2f}")
        logger.info(f"Avg Daily P&L:  ${stats['avg_daily_pnl']:,.2f}")
        logger.info(f"Trading Days:   {stats['num_days']}")
        logger.info("=" * 60)

        # Display daily breakdown
        daily_df = analyzer.get_daily_summary()

        if not daily_df.empty:
            logger.info("")
            logger.info("Daily Breakdown:")
            logger.info("-" * 60)

            for _, row in daily_df.sort_values('date', ascending=False).iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                pnl = row['total_pnl']
                num_trades = row['num_trades']
                wins = row['num_winning_trades']
                losses = row['num_losing_trades']

                pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"

                logger.info(
                    f"{date_str}: {pnl_str:>10}  "
                    f"({num_trades} trades: {wins}W/{losses}L)"
                )

            logger.info("-" * 60)

        logger.info("")
        logger.info("✅ Backfill successful!")
        logger.info("You can now start the dashboard to view the History tab.")

    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
