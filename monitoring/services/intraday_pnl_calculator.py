#!/usr/bin/env python3
"""
Intraday P&L Calculator

Calculates realized and unrealized P&L at 30-minute intervals throughout the trading day.
Uses ThetaData to fetch historical option prices for mark-to-market calculations.
"""

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
import pytz

from data.collectors.thetadata_client import ThetaDataClient
from monitoring.services.trade_match_persistence import TradeMatchPersistence
from data.storage.database import DatabaseManager
from config.settings import settings


@dataclass
class IntradaySnapshot:
    """P&L snapshot at a specific time."""
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    num_open_positions: int
    num_closed_positions: int
    events: List[str]  # e.g., ["Opened SPX IC", "Closed NVDA IC"]


@dataclass
class PriceLookup:
    """Option price at specific time."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    underlying_price: float


class IntradayPnLCalculator:
    """
    Calculates intraday P&L at 30-minute intervals.

    Workflow:
    1. Generate time intervals (9:30-16:00 ET, 30-min increments)
    2. For each interval:
       - Fetch option prices from ThetaData
       - Calculate realized P&L from closed legs
       - Calculate unrealized P&L from open legs (mark-to-market)
       - Identify trade events (opens, closes, rolls)
    3. Save snapshots to database
    """

    def __init__(self):
        self.theta_client = ThetaDataClient()
        self.persistence = TradeMatchPersistence()
        self.db = DatabaseManager(settings.trades_db_path)
        self.eastern = pytz.timezone('US/Eastern')

    def generate_intervals(
        self,
        start_date: date,
        end_date: date,
        interval_minutes: int = 30
    ) -> List[datetime]:
        """
        Generate 30-minute intervals for market hours.

        Market hours: 9:30 AM - 4:00 PM ET
        Intervals: 9:30, 10:00, 10:30, ..., 15:30, 16:00

        Args:
            start_date: First date to include
            end_date: Last date to include
            interval_minutes: Minutes between snapshots (default: 30)

        Returns:
            List of datetime objects in ET timezone
        """
        intervals = []

        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Generate intervals for this trading day
            market_open = time(9, 30)
            market_close = time(16, 0)

            current_time = datetime.combine(current_date, market_open)
            end_time = datetime.combine(current_date, market_close)

            # Localize to ET
            current_time = self.eastern.localize(current_time)
            end_time = self.eastern.localize(end_time)

            while current_time <= end_time:
                intervals.append(current_time)
                current_time += timedelta(minutes=interval_minutes)

            current_date += timedelta(days=1)

        return intervals

    async def calculate_snapshot(
        self,
        snapshot_time: datetime
    ) -> IntradaySnapshot:
        """
        Calculate P&L snapshot at a specific time.

        Args:
            snapshot_time: Time to calculate P&L (ET timezone)

        Returns:
            IntradaySnapshot with realized/unrealized P&L
        """
        logger.info(f"Calculating P&L snapshot for {snapshot_time}")

        # Get all trades (use a wide date range to get everything)
        from datetime import date
        all_matches = self.persistence.get_matches_by_date_range(
            date(2020, 1, 1),  # Far back start date
            date(2030, 12, 31)  # Far future end date
        )

        realized_pnl = 0.0
        unrealized_pnl = 0.0
        num_open = 0
        num_closed = 0
        events = []

        for match in all_matches:
            # Check if trade was active at snapshot time
            opened_at = match.get('opened_at')
            closed_at = match.get('closed_at')

            if isinstance(opened_at, str):
                opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
            if isinstance(closed_at, str):
                closed_at = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))

            # Convert to ET for comparison
            if opened_at:
                if opened_at.tzinfo is None:
                    # Assume naive datetime is already in ET, localize it
                    opened_at = self.eastern.localize(opened_at)
                else:
                    opened_at = opened_at.astimezone(self.eastern)
            if closed_at:
                if closed_at.tzinfo is None:
                    # Assume naive datetime is already in ET, localize it
                    closed_at = self.eastern.localize(closed_at)
                else:
                    closed_at = closed_at.astimezone(self.eastern)

            # Trade events at this timestamp
            if opened_at and abs((opened_at - snapshot_time).total_seconds()) < 60:
                events.append(f"Opened {match['underlying']}")

            if closed_at and abs((closed_at - snapshot_time).total_seconds()) < 60:
                events.append(f"Closed {match['underlying']}")

            # Skip if trade hasn't opened yet
            if not opened_at or snapshot_time < opened_at:
                continue

            # Get legs for this trade
            legs = self.persistence.get_legs_for_match(match['match_id'])

            # Calculate realized P&L from closed legs up to this point
            for leg in legs:
                if leg['status'] == 'CLOSED':
                    closed_time = leg.get('exit_time')
                    if isinstance(closed_time, str):
                        closed_time = datetime.fromisoformat(closed_time.replace('Z', '+00:00'))
                    if closed_time:
                        if closed_time.tzinfo is None:
                            # Assume naive datetime is already in ET, localize it
                            closed_time = self.eastern.localize(closed_time)
                        else:
                            closed_time = closed_time.astimezone(self.eastern)

                    # Only count if closed before or at snapshot time
                    if closed_time and closed_time <= snapshot_time:
                        realized_pnl += leg.get('realized_pnl', 0.0)
                        num_closed += 1

            # Calculate unrealized P&L for legs still open at snapshot time
            if not closed_at or snapshot_time < closed_at:
                open_legs = [
                    leg for leg in legs
                    if leg['status'] == 'OPEN' or (
                        leg['status'] == 'CLOSED' and
                        self._leg_closed_after_snapshot(leg, snapshot_time)
                    )
                ]

                if open_legs:
                    num_open += 1
                    leg_unrealized = await self._calculate_unrealized_for_legs(
                        open_legs,
                        snapshot_time
                    )
                    unrealized_pnl += leg_unrealized

        total_pnl = realized_pnl + unrealized_pnl

        return IntradaySnapshot(
            timestamp=snapshot_time,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            num_open_positions=num_open,
            num_closed_positions=num_closed,
            events=events
        )

    def _leg_closed_after_snapshot(self, leg: dict, snapshot_time: datetime) -> bool:
        """Check if leg was closed after snapshot time."""
        exit_time = leg.get('exit_time')
        if not exit_time:
            return True

        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        if exit_time:
            if exit_time.tzinfo is None:
                # Assume naive datetime is already in ET, localize it
                exit_time = self.eastern.localize(exit_time)
            else:
                exit_time = exit_time.astimezone(self.eastern)

        return exit_time > snapshot_time

    async def _calculate_unrealized_for_legs(
        self,
        legs: List[dict],
        snapshot_time: datetime
    ) -> float:
        """
        Calculate unrealized P&L for open legs using ThetaData prices.

        Args:
            legs: List of open leg dictionaries
            snapshot_time: Time to fetch prices for

        Returns:
            Total unrealized P&L for these legs
        """
        if not legs:
            return 0.0

        # Extract unique expiration dates
        expirations = set()
        for leg in legs:
            exp = leg.get('expiration')
            if isinstance(exp, str):
                exp = datetime.fromisoformat(exp).date()
            elif isinstance(exp, datetime):
                exp = exp.date()
            if exp:
                expirations.add(exp)

        # Fetch option prices from ThetaData
        prices_by_symbol = {}

        for expiration in expirations:
            try:
                # Get options chain for this expiration
                chain = await self.theta_client.get_options_chain(
                    trade_date=snapshot_time.date(),
                    trade_time=snapshot_time.time(),
                    expiration=expiration
                )

                if chain:
                    for opt in chain:
                        symbol = opt.get('symbol', '')
                        bid = opt.get('bid', 0)
                        ask = opt.get('ask', 0)
                        mid = (bid + ask) / 2 if bid and ask else 0
                        underlying = opt.get('underlying_price', 0)

                        prices_by_symbol[symbol] = PriceLookup(
                            symbol=symbol,
                            timestamp=snapshot_time,
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            underlying_price=underlying
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to fetch prices for {expiration} at {snapshot_time}: {e}"
                )

        # Calculate unrealized P&L for each leg
        total_unrealized = 0.0

        for leg in legs:
            symbol = leg.get('symbol')
            entry_price = leg.get('entry_price', 0)
            quantity = leg.get('current_quantity', 0)
            side = leg.get('side', 'LONG')

            if not symbol or symbol not in prices_by_symbol:
                logger.warning(
                    f"No price data for {symbol} at {snapshot_time}, skipping"
                )
                continue

            price_lookup = prices_by_symbol[symbol]
            current_price = price_lookup.mid

            # Calculate P&L based on side
            if side == 'LONG':
                # Bought to open, would sell to close
                pnl = (current_price - entry_price) * quantity * 100
            else:  # SHORT
                # Sold to open, would buy to close
                pnl = (entry_price - current_price) * quantity * 100

            total_unrealized += pnl

        return total_unrealized

    async def backfill_snapshots(
        self,
        lookback_days: int = 14,
        interval_minutes: int = 30
    ) -> int:
        """
        Backfill intraday P&L snapshots for last N days.

        Args:
            lookback_days: Number of days to backfill
            interval_minutes: Minutes between snapshots

        Returns:
            Number of snapshots created
        """
        logger.info(f"Backfilling intraday P&L snapshots ({lookback_days} days)")

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Generate intervals
        intervals = self.generate_intervals(start_date, end_date, interval_minutes)
        logger.info(f"Generated {len(intervals)} intervals to process")

        # Calculate snapshots
        snapshots = []
        for i, interval in enumerate(intervals):
            try:
                snapshot = await self.calculate_snapshot(interval)
                snapshots.append(snapshot)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(intervals)} intervals")

            except Exception as e:
                logger.error(f"Failed to calculate snapshot for {interval}: {e}")

        # Save to database
        logger.info(f"Saving {len(snapshots)} snapshots to database...")
        self._save_snapshots(snapshots)

        logger.info(f"✅ Backfilled {len(snapshots)} snapshots")
        return len(snapshots)

    def _save_snapshots(self, snapshots: List[IntradaySnapshot]):
        """Save snapshots to database."""
        with self.db.get_connection() as conn:
            for snapshot in snapshots:
                conn.execute("""
                    INSERT OR REPLACE INTO intraday_pnl_snapshots (
                        snapshot_time,
                        realized_pnl,
                        unrealized_pnl,
                        total_pnl,
                        num_open_positions,
                        num_closed_positions,
                        events_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.timestamp.isoformat(),
                    snapshot.realized_pnl,
                    snapshot.unrealized_pnl,
                    snapshot.total_pnl,
                    snapshot.num_open_positions,
                    snapshot.num_closed_positions,
                    ','.join(snapshot.events) if snapshot.events else ''
                ))

    def get_snapshots_for_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[IntradaySnapshot]:
        """
        Retrieve cached snapshots from database.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of IntradaySnapshot objects
        """
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    snapshot_time,
                    realized_pnl,
                    unrealized_pnl,
                    total_pnl,
                    num_open_positions,
                    num_closed_positions,
                    events_json
                FROM intraday_pnl_snapshots
                WHERE DATE(snapshot_time) BETWEEN ? AND ?
                ORDER BY snapshot_time
            """, (start_date.isoformat(), end_date.isoformat())).fetchall()

        snapshots = []
        for row in rows:
            timestamp = row[0]  # DuckDB returns datetime object directly
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            events = row[6].split(',') if row[6] else []

            snapshots.append(IntradaySnapshot(
                timestamp=timestamp,
                realized_pnl=row[1],
                unrealized_pnl=row[2],
                total_pnl=row[3],
                num_open_positions=row[4],
                num_closed_positions=row[5],
                events=events
            ))

        return snapshots
