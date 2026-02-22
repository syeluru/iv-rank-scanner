#!/usr/bin/env python3
"""
Backfill Intraday P&L Snapshots

Calculates realized and unrealized P&L at 30-minute intervals using ThetaData.
Saves results to database for visualization in the History tab.
"""

import sys
import asyncio
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.services.intraday_pnl_calculator import IntradayPnLCalculator
from loguru import logger


async def backfill_intraday_pnl(lookback_days: int = 14):
    """
    Backfill intraday P&L snapshots.

    Args:
        lookback_days: Number of days to backfill
    """
    logger.info("=" * 80)
    logger.info("BACKFILLING INTRADAY P&L SNAPSHOTS")
    logger.info("=" * 80)

    # Initialize calculator
    calculator = IntradayPnLCalculator()

    # Check ThetaData connection first
    logger.info("\nChecking ThetaData connection...")
    try:
        # Try a simple test query to verify terminal is running
        import requests
        response = requests.get(
            "http://127.0.0.1:25503/v3/option/list/expirations",
            params={"symbol": "SPXW"},
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✅ ThetaData connection successful")
        else:
            raise Exception(f"Unexpected status code: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ ThetaData connection failed: {e}")
        logger.info("\nPlease ensure Theta Terminal is running:")
        logger.info("  cd ~/Downloads")
        logger.info("  java -jar ThetaTerminalv3.jar")
        logger.info("\nSee docs/THETADATA_SETUP.md for details")
        return

    # Backfill snapshots
    logger.info(f"\nBackfilling {lookback_days} days of intraday P&L...")
    num_snapshots = await calculator.backfill_snapshots(
        lookback_days=lookback_days,
        interval_minutes=30
    )

    logger.info(f"\n✅ Successfully backfilled {num_snapshots} snapshots!")

    # Display sample results
    logger.info("\nSample snapshots:")
    end_date = date.today()
    start_date = end_date - timedelta(days=3)
    recent_snapshots = calculator.get_snapshots_for_date_range(start_date, end_date)

    for snapshot in recent_snapshots[:10]:
        logger.info(
            f"  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}: "
            f"Realized=${snapshot.realized_pnl:,.2f}, "
            f"Unrealized=${snapshot.unrealized_pnl:,.2f}, "
            f"Total=${snapshot.total_pnl:,.2f}"
        )

    if len(recent_snapshots) > 10:
        logger.info(f"  ... and {len(recent_snapshots) - 10} more")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\n1. Start the dashboard:")
    logger.info("   python monitoring/dashboards/command_center.py")
    logger.info("\n2. Navigate to the History tab")
    logger.info("\n3. View your intraday P&L chart!")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Backfill intraday P&L snapshots')
    parser.add_argument('--days', type=int, default=14, help='Days to backfill')
    args = parser.parse_args()

    await backfill_intraday_pnl(lookback_days=args.days)


if __name__ == '__main__':
    asyncio.run(main())
