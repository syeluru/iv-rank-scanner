"""
1-minute candle data loader for ML feature extraction.

Loads intraday candles from DuckDB for historical volatility and
intraday pattern analysis.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from data.storage.database import DatabaseManager
from config.settings import settings


class CandleLoader:
    """Load 1-minute candles from DuckDB."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize candle loader.

        Args:
            db_path: Path to ML database. Defaults to settings.DATA_STORE_PATH / trading_data.duckdb
        """
        if db_path is None:
            db_path = settings.DATA_STORE_PATH / "trading_data.duckdb"

        # Convert to Path if string
        if isinstance(db_path, str):
            db_path = Path(db_path)

        self.db = DatabaseManager(db_path)

    async def get_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Load intraday candles for symbol.

        Args:
            symbol: Symbol to load (e.g., 'SPX', 'VIX')
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM intraday_candles
        WHERE symbol = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """

        df = self.db.fetch_df(query, (symbol, start_time, end_time))
        return df

    async def get_latest_candles(
        self,
        symbol: str,
        num_candles: int = 60
    ) -> pd.DataFrame:
        """
        Get most recent N candles.

        Args:
            symbol: Symbol to load
            num_candles: Number of recent candles to retrieve

        Returns:
            DataFrame with most recent candles
        """
        end_time = datetime.now()
        # Add buffer for market hours, holidays, etc.
        start_time = end_time - timedelta(minutes=num_candles * 2)

        df = await self.get_candles(symbol, start_time, end_time)

        # Return only the requested number of most recent candles
        return df.tail(num_candles)

    async def get_today_candles(self, symbol: str) -> pd.DataFrame:
        """
        Get all candles for today.

        Args:
            symbol: Symbol to load

        Returns:
            DataFrame with today's candles
        """
        now = datetime.now()
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return await self.get_candles(symbol, start_time, now)

    async def has_data(self, symbol: str) -> bool:
        """
        Check if candle data exists for symbol.

        Args:
            symbol: Symbol to check

        Returns:
            True if data exists, False otherwise
        """
        query = """
        SELECT COUNT(*) as count
        FROM intraday_candles
        WHERE symbol = ?
        LIMIT 1
        """

        result = self.db.fetch_one(query, (symbol,))
        return result is not None and result.get('count', 0) > 0

    def insert_candles(self, symbol: str, candles_df: pd.DataFrame) -> int:
        """
        Insert candles into database.

        Args:
            symbol: Symbol these candles belong to
            candles_df: DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Number of candles inserted
        """
        from loguru import logger

        if candles_df.empty:
            logger.warning(f"No candles to insert for {symbol}")
            return 0

        try:
            # Add symbol column
            candles_df = candles_df.copy()
            candles_df['symbol'] = symbol

            # Ensure timestamp is datetime
            if 'timestamp' in candles_df.columns:
                candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])

            # Required columns
            required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in candles_df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Select only required columns
            candles_df = candles_df[required_cols]

            # Insert into database (use REPLACE to handle duplicates)
            insert_query = """
            INSERT OR REPLACE INTO intraday_candles
                (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            # Convert to list of tuples
            records = candles_df.to_records(index=False).tolist()

            # Batch insert
            self.db.executemany(insert_query, records)

            logger.info(f"✓ Inserted {len(records)} candles for {symbol}")

            return len(records)

        except Exception as e:
            logger.error(f"Failed to insert candles for {symbol}: {e}", exc_info=True)
            return 0

    def get_available_date_range(self, symbol: str) -> Optional[tuple[datetime, datetime]]:
        """
        Get date range of available data for symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (first_timestamp, last_timestamp) or None if no data
        """
        from loguru import logger

        query = """
        SELECT MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
        FROM intraday_candles
        WHERE symbol = ?
        """

        try:
            result = self.db.execute(query, (symbol,)).fetchone()

            if result and result[0] is not None:
                first_ts = pd.to_datetime(result[0])
                last_ts = pd.to_datetime(result[1])
                return (first_ts, last_ts)

            return None

        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return None

    def get_candle_count(self, symbol: str) -> int:
        """
        Get total number of candles stored for symbol.

        Args:
            symbol: Symbol to count

        Returns:
            Number of candles
        """
        from loguru import logger

        query = "SELECT COUNT(*) FROM intraday_candles WHERE symbol = ?"

        try:
            result = self.db.execute(query, (symbol,)).fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to count candles for {symbol}: {e}")
            return 0
