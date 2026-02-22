"""
ThetaData provider for historical market data.

Provides access to:
- 1-minute OHLC bars for stocks/indices
- Market internals (TICK, TRIN, VOLD)
- Historical options data
- Clean, professional API with unlimited calls
"""

from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
from loguru import logger

try:
    from thetadata import ThetaClient, StockReqType, DateRange
    THETADATA_AVAILABLE = True
except ImportError:
    THETADATA_AVAILABLE = False
    logger.warning("thetadata package not installed. Run: pip install thetadata")

from config.settings import settings


class ThetaDataProvider:
    """ThetaData API wrapper for historical market data."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize ThetaData provider.

        Args:
            username: ThetaData username (default: from settings)
            password: ThetaData password (default: from settings)
        """
        if not THETADATA_AVAILABLE:
            raise ImportError(
                "thetadata package required. Install with: pip install thetadata"
            )

        self.username = username or settings.THETADATA_USERNAME
        self.password = password or settings.THETADATA_PASSWORD

        if not self.username or not self.password:
            raise ValueError(
                "ThetaData credentials not configured. "
                "Set THETADATA_USERNAME and THETADATA_PASSWORD in settings or .env"
            )

        self.client = None
        self._connected = False

    def connect(self):
        """Connect to ThetaData terminal."""
        if self._connected:
            return

        try:
            logger.info("Connecting to ThetaData...")
            self.client = ThetaClient(username=self.username, passwd=self.password)
            self.client.connect()
            self._connected = True
            logger.info("✓ Connected to ThetaData")

        except Exception as e:
            logger.error(f"Failed to connect to ThetaData: {e}")
            raise

    def disconnect(self):
        """Disconnect from ThetaData terminal."""
        if self._connected and self.client:
            try:
                # ThetaClient doesn't have close(), connection closes automatically
                self._connected = False
                logger.info("Disconnected from ThetaData")
            except Exception as e:
                logger.warning(f"Error disconnecting from ThetaData: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def get_hist_stock_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_seconds: int = 60
    ) -> pd.DataFrame:
        """
        Fetch historical stock bars (OHLCV).

        Args:
            symbol: Stock/index symbol (e.g., 'SPX', 'VIX')
            start_date: Start date
            end_date: End date
            interval_seconds: Bar interval in seconds (default: 60 = 1-minute)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self._connected:
            self.connect()

        try:
            logger.info(
                f"Fetching {symbol} bars from {start_date.date()} to {end_date.date()}"
            )

            # ThetaData uses milliseconds for interval
            interval_ms = interval_seconds * 1000

            # Create DateRange object
            date_range = DateRange(start_date, end_date)

            # Fetch data
            data = self.client.get_hist_stock(
                req=StockReqType.OHLC,
                root=symbol,
                date_range=date_range,
                interval_size=interval_ms,
                use_rth=True,  # Only regular trading hours
                progress_bar=False
            )

            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Rename columns to standard format
            df = data.copy()

            # ThetaData returns: ms_of_day, open, high, low, close, volume, count, date
            # We need: timestamp, open, high, low, close, volume

            if 'date' in df.columns and 'ms_of_day' in df.columns:
                # Combine date and ms_of_day to create timestamp
                df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(
                    df['ms_of_day'], unit='ms'
                )
            elif 'timestamp' not in df.columns:
                # Fallback: use index if it's a DatetimeIndex
                if isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = df.index
                else:
                    raise ValueError("Cannot determine timestamp column")

            # Select and order columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"✓ Fetched {len(df)} bars for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_hist_bars_batch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_seconds: int = 60,
        batch_days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical bars in batches to handle large date ranges.

        Args:
            symbol: Stock/index symbol
            start_date: Start date
            end_date: End date
            interval_seconds: Bar interval in seconds
            batch_days: Days per batch (default: 30)

        Returns:
            Combined DataFrame with all bars
        """
        if not self._connected:
            self.connect()

        all_data = []
        current_start = start_date

        while current_start < end_date:
            # Calculate batch end (ensure it doesn't exceed end_date)
            current_end = min(
                current_start + timedelta(days=batch_days),
                end_date
            )

            logger.info(
                f"Fetching {symbol} batch: "
                f"{current_start.date()} to {current_end.date()}"
            )

            # Fetch batch
            batch_df = self.get_hist_stock_bars(
                symbol=symbol,
                start_date=current_start,
                end_date=current_end,
                interval_seconds=interval_seconds
            )

            if not batch_df.empty:
                all_data.append(batch_df)

            # Move to next batch
            current_start = current_end + timedelta(days=1)

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # Combine all batches
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (may occur at batch boundaries)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        logger.info(
            f"✓ Total {len(combined_df)} bars for {symbol} "
            f"({start_date.date()} to {end_date.date()})"
        )

        return combined_df

    def get_market_internals(self) -> dict:
        """
        Get current market internals (TICK, TRIN).

        Returns:
            Dict with 'tick' and 'trin' values
        """
        if not self._connected:
            self.connect()

        try:
            # Fetch NYSE TICK
            tick_quote = self.client.get_last_quote(root="$TICK")
            tick_value = tick_quote.bid if tick_quote else None

            # Fetch TRIN
            trin_quote = self.client.get_last_quote(root="$TRIN")
            trin_value = trin_quote.bid if trin_quote else None

            return {
                'tick': float(tick_value) if tick_value is not None else None,
                'trin': float(trin_value) if trin_value is not None else None
            }

        except Exception as e:
            logger.error(f"Failed to fetch market internals: {e}")
            return {'tick': None, 'trin': None}
