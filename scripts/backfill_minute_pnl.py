#!/usr/bin/env python3
"""
Backfill minute-level P&L data for all historical trades.

This script:
1. Fetches all trade matches and legs from database
2. For each leg, generates 1-minute timestamps (market hours only)
3. Fetches prices from ThetaData
4. Calculates P&L for each leg
5. Aggregates to match-level snapshots
6. Tracks progress to support resuming after interruptions

Usage:
    # Backfill all trades
    venv/bin/python3 scripts/backfill_minute_pnl.py

    # Backfill specific match
    venv/bin/python3 scripts/backfill_minute_pnl.py --match-id <match_id>

    # Resume interrupted backfill
    venv/bin/python3 scripts/backfill_minute_pnl.py --resume
"""

import argparse
import duckdb
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.collectors.minute_price_fetcher import MinutePriceFetcher
from monitoring.services.minute_pnl_calculator import MinutePnLCalculator
from monitoring.services.minute_pnl_utils import (
    generate_minute_timestamps,
    localize_eastern
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinutePnLBackfiller:
    """Orchestrates backfilling of minute-level P&L data."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize backfiller.

        Args:
            db_path: Path to DuckDB database
        """
        if db_path is None:
            db_path = project_root / "data_store" / "trades.duckdb"
        self.db_path = db_path
        self.price_fetcher = MinutePriceFetcher(db_path)
        self.pnl_calculator = MinutePnLCalculator(db_path)

    def get_all_matches(self) -> list[dict]:
        """Get all trade matches from database."""
        with duckdb.connect(str(self.db_path)) as conn:
            results = conn.execute("""
                SELECT match_id, underlying, opened_at, closed_at
                FROM trade_matches
                ORDER BY opened_at
            """).fetchall()

            matches = []
            for row in results:
                matches.append({
                    'match_id': row[0],
                    'underlying': row[1],
                    'opened_at': localize_eastern(row[2]),
                    'closed_at': localize_eastern(row[3])
                })
            return matches

    def get_legs_for_match(self, match_id: str) -> list[dict]:
        """Get all legs for a specific match."""
        with duckdb.connect(str(self.db_path)) as conn:
            results = conn.execute("""
                SELECT leg_id, symbol, side, entry_quantity, entry_price, exit_price,
                       entry_time, exit_time
                FROM trade_legs
                WHERE match_id = ?
                ORDER BY entry_time
            """, [match_id]).fetchall()

            legs = []
            for row in results:
                legs.append({
                    'leg_id': row[0],
                    'match_id': match_id,
                    'symbol': row[1],
                    'side': row[2],
                    'quantity': row[3],
                    'entry_price': row[4],
                    'exit_price': row[5],
                    'entry_time': localize_eastern(row[6]),
                    'exit_time': localize_eastern(row[7])
                })
            return legs

    def get_backfill_status(self, leg_id: str) -> Optional[str]:
        """
        Check backfill status for a leg.

        Returns:
            Status string ('pending', 'completed', 'failed') or None if not started
        """
        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute("""
                SELECT status
                FROM backfill_progress
                WHERE leg_id = ?
            """, [leg_id]).fetchone()

            return result[0] if result else None

    def mark_backfill_status(self, leg_id: str, status: str, error_msg: str = None):
        """Mark backfill status for a leg."""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backfill_progress (leg_id, status, last_updated, error_message)
                VALUES (?, ?, ?, ?)
            """, [leg_id, status, datetime.now(), error_msg])

    def backfill_leg(self, leg: dict) -> bool:
        """
        Backfill minute-level data for a single leg.

        Steps:
        1. Generate minute timestamps (market hours only)
        2. Fetch prices from ThetaData
        3. Save prices to database
        4. Calculate P&L
        5. Mark as completed

        Args:
            leg: Leg dictionary

        Returns:
            True if successful
        """
        leg_id = leg['leg_id']

        try:
            # Check if already completed
            status = self.get_backfill_status(leg_id)
            if status == 'completed':
                logger.info(f"Leg {leg_id} already completed, skipping")
                return True

            logger.info(f"Backfilling leg {leg_id}: {leg['symbol']}")

            # Mark as pending
            self.mark_backfill_status(leg_id, 'pending')

            # Generate timestamps
            timestamps = generate_minute_timestamps(
                leg['entry_time'],
                leg['exit_time'],
                market_hours_only=True
            )

            if not timestamps:
                logger.warning(f"No timestamps generated for leg {leg_id}")
                self.mark_backfill_status(leg_id, 'failed', 'No timestamps generated')
                return False

            logger.info(f"  Generated {len(timestamps)} minute timestamps")

            # Fetch prices
            price_records = self.price_fetcher.fetch_minute_prices_for_leg(leg, timestamps)

            if not price_records:
                logger.warning(f"No price records fetched for leg {leg_id}")
                self.mark_backfill_status(leg_id, 'failed', 'No price records fetched')
                return False

            logger.info(f"  Fetched {len(price_records)} price records")

            # Save prices
            if not self.price_fetcher.save_leg_prices(price_records):
                logger.error(f"Failed to save prices for leg {leg_id}")
                self.mark_backfill_status(leg_id, 'failed', 'Failed to save prices')
                return False

            # Calculate P&L
            if not self.pnl_calculator.calculate_leg_pnl(leg):
                logger.error(f"Failed to calculate P&L for leg {leg_id}")
                self.mark_backfill_status(leg_id, 'failed', 'Failed to calculate P&L')
                return False

            # Mark as completed
            self.mark_backfill_status(leg_id, 'completed')
            logger.info(f"✓ Completed leg {leg_id}")
            return True

        except Exception as e:
            logger.error(f"Error backfilling leg {leg_id}: {e}")
            import traceback
            traceback.print_exc()
            self.mark_backfill_status(leg_id, 'failed', str(e))
            return False

    def backfill_match(self, match_id: str) -> bool:
        """
        Backfill minute-level data for all legs in a match.

        Args:
            match_id: Trade match ID

        Returns:
            True if all legs successful
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Backfilling match: {match_id}")
        logger.info(f"{'='*60}")

        try:
            # Get all legs
            legs = self.get_legs_for_match(match_id)
            logger.info(f"Found {len(legs)} legs for match {match_id}")

            # Backfill each leg
            success_count = 0
            for i, leg in enumerate(legs, 1):
                logger.info(f"\nLeg {i}/{len(legs)}: {leg['leg_id']}")
                if self.backfill_leg(leg):
                    success_count += 1

            # Aggregate to match-level
            if success_count == len(legs):
                logger.info(f"\nAggregating P&L for match {match_id}")
                if self.pnl_calculator.aggregate_match_pnl(match_id):
                    # Mark roll events
                    self.pnl_calculator.mark_roll_events(match_id)
                    logger.info(f"✓ Completed match {match_id}")
                    return True
                else:
                    logger.error(f"Failed to aggregate P&L for match {match_id}")
                    return False
            else:
                logger.warning(f"Only {success_count}/{len(legs)} legs successful for match {match_id}")
                return False

        except Exception as e:
            logger.error(f"Error backfilling match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def backfill_all(self, resume: bool = False):
        """
        Backfill all matches.

        Args:
            resume: If True, skip completed legs
        """
        logger.info("Starting minute-level P&L backfill")

        # Connect to ThetaData
        if not self.price_fetcher.connect():
            logger.error("Failed to connect to ThetaData Terminal")
            logger.error("Make sure it's running: java -jar ThetaTerminalv3.jar")
            return

        # Get all matches
        matches = self.get_all_matches()
        logger.info(f"Found {len(matches)} matches to backfill")

        success_count = 0
        for i, match in enumerate(matches, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Match {i}/{len(matches)}: {match['match_id']}")
            logger.info(f"{'='*60}")

            if self.backfill_match(match['match_id']):
                success_count += 1

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Backfill complete!")
        logger.info(f"Successful: {success_count}/{len(matches)} matches")
        logger.info(f"{'='*60}")

        # Disconnect
        self.price_fetcher.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backfill minute-level P&L data')
    parser.add_argument('--match-id', help='Backfill specific match only')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted backfill')
    args = parser.parse_args()

    backfiller = MinutePnLBackfiller()

    if args.match_id:
        # Backfill specific match
        if not backfiller.price_fetcher.connect():
            logger.error("Failed to connect to ThetaData Terminal")
            return

        success = backfiller.backfill_match(args.match_id)
        backfiller.price_fetcher.disconnect()

        if success:
            logger.info(f"\n✓ Successfully backfilled match {args.match_id}")
        else:
            logger.error(f"\n✗ Failed to backfill match {args.match_id}")

    else:
        # Backfill all matches
        backfiller.backfill_all(resume=args.resume)


if __name__ == '__main__':
    main()
