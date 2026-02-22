"""
Minute-level P&L calculator.

Calculates unrealized and realized P&L for each leg at 1-minute intervals
and aggregates to match-level snapshots.
"""

import duckdb
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from monitoring.services.minute_pnl_utils import (
    localize_eastern,
    format_timestamp_for_id
)

logger = logging.getLogger(__name__)


class MinutePnLCalculator:
    """Calculates and stores minute-level P&L data."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize P&L calculator.

        Args:
            db_path: Path to DuckDB database (defaults to data_store/trades.duckdb)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data_store" / "trades.duckdb"
        self.db_path = db_path

    def calculate_leg_pnl(self, leg: Dict) -> bool:
        """
        Calculate and save P&L for a single leg.

        Reads price data from minute_leg_prices and calculates P&L
        based on entry/exit prices and timestamps.

        Args:
            leg: Dictionary with leg data (leg_id, symbol, side, quantity, entry_price, exit_price, exit_time)

        Returns:
            True if successful
        """
        try:
            leg_id = leg['leg_id']
            side = leg['side']
            quantity = leg['quantity']
            entry_price = leg['entry_price']
            exit_price = leg.get('exit_price')
            exit_time = localize_eastern(leg.get('exit_time'))

            logger.info(f"Calculating P&L for leg {leg_id} ({side} {quantity}x @ ${entry_price})")

            with duckdb.connect(str(self.db_path)) as conn:
                # Fetch all price records for this leg
                price_records = conn.execute("""
                    SELECT price_id, timestamp, mid
                    FROM minute_leg_prices
                    WHERE leg_id = ?
                    ORDER BY timestamp
                """, [leg_id]).fetchall()

                if not price_records:
                    logger.warning(f"No price records found for leg {leg_id}")
                    return False

                pnl_records = []

                for price_id, timestamp, current_mid in price_records:
                    # Ensure timestamp is timezone-aware
                    timestamp = localize_eastern(timestamp)

                    # Convert Decimal to float (DuckDB returns Decimal for numeric columns)
                    current_mid = float(current_mid) if current_mid is not None else 0.0
                    entry_price_float = float(entry_price)
                    exit_price_float = float(exit_price) if exit_price is not None else 0.0

                    # Determine if this is realized or unrealized
                    is_realized = False
                    if exit_time and timestamp >= exit_time:
                        is_realized = True

                    # Calculate P&L based on side
                    if is_realized:
                        # Use actual exit price for realized P&L
                        price_diff = exit_price_float - entry_price_float if exit_price_float else 0
                        if side == 'SHORT':
                            price_diff = -price_diff  # Invert for short positions

                        realized_pnl = price_diff * abs(quantity) * 100
                        unrealized_pnl = 0
                        current_price = exit_price_float
                    else:
                        # Calculate unrealized P&L
                        price_diff = current_mid - entry_price_float
                        if side == 'SHORT':
                            price_diff = -price_diff  # Invert for short positions

                        unrealized_pnl = price_diff * abs(quantity) * 100
                        realized_pnl = 0
                        current_price = current_mid

                    pnl_record = {
                        'pnl_id': f"{leg_id}_{format_timestamp_for_id(timestamp)}",
                        'leg_id': leg_id,
                        'timestamp': timestamp,
                        'unrealized_pnl': round(unrealized_pnl, 2),
                        'realized_pnl': round(realized_pnl, 2),
                        'is_realized': is_realized,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'quantity': quantity,
                        'side': side
                    }
                    pnl_records.append(pnl_record)

                # Save all P&L records
                for record in pnl_records:
                    conn.execute("""
                        INSERT INTO minute_leg_pnl (
                            pnl_id, leg_id, timestamp, unrealized_pnl, realized_pnl,
                            is_realized, entry_price, current_price, quantity, side
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (pnl_id) DO NOTHING
                    """, [
                        record['pnl_id'],
                        record['leg_id'],
                        record['timestamp'],
                        record['unrealized_pnl'],
                        record['realized_pnl'],
                        record['is_realized'],
                        record['entry_price'],
                        record['current_price'],
                        record['quantity'],
                        record['side']
                    ])

                logger.info(f"Saved {len(pnl_records)} P&L records for leg {leg_id}")
                return True

        except Exception as e:
            logger.error(f"Error calculating P&L for leg {leg['leg_id']}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def aggregate_match_pnl(self, match_id: str) -> bool:
        """
        Aggregate leg-level P&L to match-level snapshots.

        Sums P&L across all legs for each timestamp and saves to minute_match_pnl.

        Args:
            match_id: Trade match ID

        Returns:
            True if successful
        """
        try:
            logger.info(f"Aggregating P&L for match {match_id}")

            with duckdb.connect(str(self.db_path)) as conn:
                # Get all legs for this match
                legs = conn.execute("""
                    SELECT leg_id
                    FROM trade_legs
                    WHERE match_id = ?
                """, [match_id]).fetchall()

                if not legs:
                    logger.warning(f"No legs found for match {match_id}")
                    return False

                leg_ids = [leg[0] for leg in legs]

                # Get all unique timestamps across all legs
                timestamps = conn.execute("""
                    SELECT DISTINCT timestamp
                    FROM minute_leg_pnl
                    WHERE leg_id IN ({})
                    ORDER BY timestamp
                """.format(','.join(['?' for _ in leg_ids])), leg_ids).fetchall()

                if not timestamps:
                    logger.warning(f"No P&L timestamps found for match {match_id}")
                    return False

                snapshot_records = []

                for (timestamp,) in timestamps:
                    # Ensure timestamp is timezone-aware
                    timestamp = localize_eastern(timestamp)

                    # Aggregate P&L for this timestamp
                    result = conn.execute("""
                        SELECT
                            SUM(unrealized_pnl) as total_unrealized,
                            SUM(realized_pnl) as total_realized,
                            SUM(CASE WHEN is_realized = FALSE THEN 1 ELSE 0 END) as num_open,
                            SUM(CASE WHEN is_realized = TRUE THEN 1 ELSE 0 END) as num_closed
                        FROM minute_leg_pnl
                        WHERE leg_id IN ({})
                        AND timestamp = ?
                    """.format(','.join(['?' for _ in leg_ids])), leg_ids + [timestamp]).fetchone()

                    total_unrealized = result[0] or 0
                    total_realized = result[1] or 0
                    num_open = result[2] or 0
                    num_closed = result[3] or 0
                    total_pnl = total_unrealized + total_realized

                    snapshot_record = {
                        'snapshot_id': f"{match_id}_{format_timestamp_for_id(timestamp)}",
                        'match_id': match_id,
                        'timestamp': timestamp,
                        'total_unrealized_pnl': round(total_unrealized, 2),
                        'total_realized_pnl': round(total_realized, 2),
                        'total_pnl': round(total_pnl, 2),
                        'num_open_legs': int(num_open),
                        'num_closed_legs': int(num_closed),
                        'events': ''  # Will be populated later with roll events
                    }
                    snapshot_records.append(snapshot_record)

                # Save all snapshot records
                for record in snapshot_records:
                    conn.execute("""
                        INSERT INTO minute_match_pnl (
                            snapshot_id, match_id, timestamp, total_unrealized_pnl,
                            total_realized_pnl, total_pnl, num_open_legs, num_closed_legs, events
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (snapshot_id) DO NOTHING
                    """, [
                        record['snapshot_id'],
                        record['match_id'],
                        record['timestamp'],
                        record['total_unrealized_pnl'],
                        record['total_realized_pnl'],
                        record['total_pnl'],
                        record['num_open_legs'],
                        record['num_closed_legs'],
                        record['events']
                    ])

                logger.info(f"Saved {len(snapshot_records)} snapshots for match {match_id}")
                return True

        except Exception as e:
            logger.error(f"Error aggregating P&L for match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def mark_roll_events(self, match_id: str) -> bool:
        """
        Mark roll events in match-level snapshots.

        Identifies timestamps where legs were rolled (closed + new opened)
        and updates the events field.

        Args:
            match_id: Trade match ID

        Returns:
            True if successful
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get roll data from trade_rolls table
                rolls = conn.execute("""
                    SELECT roll_time, closed_leg_symbols, opened_leg_symbols
                    FROM trade_rolls
                    WHERE match_id = ?
                    ORDER BY roll_time
                """, [match_id]).fetchall()

                if not rolls:
                    logger.debug(f"No rolls found for match {match_id}")
                    return True

                # Update events field for each roll
                for roll_time, closed_leg_symbols, opened_leg_symbols in rolls:
                    roll_time = localize_eastern(roll_time)

                    # Find snapshot at or near this roll time
                    snapshot = conn.execute("""
                        SELECT snapshot_id
                        FROM minute_match_pnl
                        WHERE match_id = ?
                        AND timestamp >= ?
                        ORDER BY timestamp
                        LIMIT 1
                    """, [match_id, roll_time]).fetchone()

                    if snapshot:
                        snapshot_id = snapshot[0]
                        event_text = f"ROLL: Closed {len(closed_leg_symbols.split(','))} legs, Opened {len(opened_leg_symbols.split(','))} legs"

                        conn.execute("""
                            UPDATE minute_match_pnl
                            SET events = ?
                            WHERE snapshot_id = ?
                        """, [event_text, snapshot_id])

                logger.info(f"Marked {len(rolls)} roll events for match {match_id}")
                return True

        except Exception as e:
            logger.error(f"Error marking roll events for match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
