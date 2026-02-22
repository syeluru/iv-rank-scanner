"""
Minute-level price fetcher for option legs.

Fetches bid/ask/mid prices and Greeks at 1-minute intervals from ThetaData
and saves to database.
"""

import duckdb
import logging
from datetime import datetime, time, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time as time_module
import pytz

from .thetadata_client import ThetaDataClient
from monitoring.services.minute_pnl_utils import (
    parse_occ_symbol,
    generate_minute_timestamps,
    format_timestamp_for_id,
    localize_eastern
)

logger = logging.getLogger(__name__)


class MinutePriceFetcher:
    """Fetches and stores minute-level price data for option legs."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize price fetcher.

        Args:
            db_path: Path to DuckDB database (defaults to data_store/trades.duckdb)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data_store" / "trades.duckdb"
        self.db_path = db_path
        self.theta_client = ThetaDataClient()

    def fetch_minute_prices_for_leg(
        self,
        leg: Dict,
        timestamps: List[datetime]
    ) -> List[Dict]:
        """
        Fetch minute-by-minute prices for a single leg.

        Groups timestamps by trading day and fetches each day in one API call.

        Args:
            leg: Dictionary with leg data (symbol, entry_time, exit_time, etc.)
            timestamps: List of minute timestamps to fetch

        Returns:
            List of price records
        """
        if not timestamps:
            logger.warning(f"No timestamps provided for leg {leg['leg_id']}")
            return []

        try:
            # Parse OCC symbol
            symbol_info = parse_occ_symbol(leg['symbol'])
            root = symbol_info['root']
            expiration = symbol_info['expiration']
            strike = symbol_info['strike']
            right = 'call' if symbol_info['right'] == 'C' else 'put'

            # Group timestamps by trading day
            timestamps_by_date = {}
            for ts in timestamps:
                ts = localize_eastern(ts)
                date_key = ts.date()
                if date_key not in timestamps_by_date:
                    timestamps_by_date[date_key] = []
                timestamps_by_date[date_key].append(ts)

            all_price_records = []

            # Fetch each trading day
            for trade_date, day_timestamps in timestamps_by_date.items():
                logger.info(f"Fetching {len(day_timestamps)} minutes for {leg['symbol']} on {trade_date}")

                # Get min/max times for this day
                day_timestamps.sort()
                start_time = day_timestamps[0].time()
                end_time = day_timestamps[-1].time()

                # Fetch Greeks data for the entire day range
                greeks_data = self.theta_client._get_first_order_greeks(
                    symbol=root,
                    expiration=expiration,
                    right=right,
                    trade_date=trade_date,
                    trade_time=start_time,
                    end_time=end_time
                )

                if not greeks_data:
                    logger.warning(f"No Greeks data for {leg['symbol']} on {trade_date}")
                    continue

                # Build a map of (strike, timestamp) -> data for quick lookup
                data_map = {}
                eastern = pytz.timezone('US/Eastern')

                for row in greeks_data:
                    row_strike = float(row.get('strike', 0))
                    row_timestamp = row.get('timestamp', '')

                    # Parse timestamp
                    if isinstance(row_timestamp, str) and 'T' in row_timestamp:
                        # ISO format: 2026-02-07T10:30:00
                        try:
                            dt = datetime.fromisoformat(row_timestamp.split('.')[0])
                            # ThetaData returns timestamps in Eastern Time as naive datetimes
                            dt_aware = eastern.localize(dt)
                            key = (row_strike, dt_aware)
                            data_map[key] = row
                        except Exception as e:
                            logger.debug(f"Could not parse timestamp {row_timestamp}: {e}")
                            continue

                # Create price records for requested timestamps
                for ts in day_timestamps:
                    # Find matching data point (strike and timestamp within 1 minute)
                    matching_data = None
                    for (data_strike, data_ts), data in data_map.items():
                        # Check if strike matches (within 0.01)
                        if abs(data_strike - strike) < 0.01:
                            # Check if timestamp within 1 minute
                            time_diff = abs((data_ts - ts).total_seconds())
                            if time_diff <= 60:
                                matching_data = data
                                break

                    if not matching_data:
                        logger.debug(f"No data for {leg['symbol']} at {ts}")
                        continue

                    # Extract price data
                    bid = float(matching_data.get('bid', 0) or 0)
                    ask = float(matching_data.get('ask', 0) or 0)
                    mid = (bid + ask) / 2 if bid and ask else 0

                    price_record = {
                        'price_id': f"{leg['leg_id']}_{format_timestamp_for_id(ts)}",
                        'leg_id': leg['leg_id'],
                        'timestamp': ts,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'underlying_price': float(matching_data.get('underlying_price', 0) or 0),
                        'delta': float(matching_data.get('delta', 0) or 0),
                        'gamma': float(matching_data.get('gamma', 0) or 0),
                        'theta': float(matching_data.get('theta', 0) or 0),
                        'vega': float(matching_data.get('vega', 0) or 0),
                        'implied_vol': float(matching_data.get('implied_vol', 0) or 0),
                    }
                    all_price_records.append(price_record)

                # Rate limiting: brief pause between days
                time_module.sleep(0.5)

            logger.info(f"Fetched {len(all_price_records)} price records for {leg['symbol']}")
            return all_price_records

        except Exception as e:
            logger.error(f"Error fetching prices for leg {leg['leg_id']}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def save_leg_prices(self, price_records: List[Dict]) -> bool:
        """
        Save price records to database.

        Args:
            price_records: List of price record dictionaries

        Returns:
            True if successful
        """
        if not price_records:
            return True

        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Use batch insert for efficiency
                for record in price_records:
                    conn.execute("""
                        INSERT INTO minute_leg_prices (
                            price_id, leg_id, timestamp, bid, ask, mid,
                            underlying_price, delta, gamma, theta, vega, implied_vol
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (price_id) DO NOTHING
                    """, [
                        record['price_id'],
                        record['leg_id'],
                        record['timestamp'],
                        record['bid'],
                        record['ask'],
                        record['mid'],
                        record['underlying_price'],
                        record['delta'],
                        record['gamma'],
                        record['theta'],
                        record['vega'],
                        record['implied_vol']
                    ])

                logger.info(f"Saved {len(price_records)} price records to database")
                return True

        except Exception as e:
            logger.error(f"Error saving price records: {e}")
            import traceback
            traceback.print_exc()
            return False

    def connect(self) -> bool:
        """Connect to ThetaData Terminal."""
        return self.theta_client.connect()

    def disconnect(self):
        """Disconnect from ThetaData Terminal."""
        self.theta_client.disconnect()
