"""
Intraday pattern features with range exhaustion (11 features).

Extracts intraday price and volume patterns:
- Opening range metrics
- Range exhaustion indicators
- VWAP relationships
- Volume patterns
"""

from datetime import datetime, time, timedelta
from typing import Dict
import numpy as np
import pandas as pd
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class IntradayFeaturesExtractor(BaseFeatureExtractor):
    """Extract intraday pattern and range exhaustion features."""

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(self, candle_loader):
        """
        Initialize intraday features extractor.

        Args:
            candle_loader: CandleLoader instance
        """
        self.candle_loader = candle_loader

    async def extract(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Extract intraday features.

        Args:
            symbol: Underlying symbol
            timestamp: Current timestamp

        Returns:
            Dict with 11 intraday features
        """
        features = {}

        try:
            # Get today's candles up to current time
            today_start = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            candles = await self.candle_loader.get_candles(symbol, today_start, timestamp)

            if candles.empty:
                logger.warning(f"No candles found for {symbol}")
                return self._get_default_features(timestamp)

            # Get historical candles (last 20 days) for comparison
            hist_end = today_start
            hist_start = hist_end - timedelta(days=30)  # Buffer for weekends
            hist_candles = await self.candle_loader.get_candles(symbol, hist_start, hist_end)

            # Calculate all features
            features.update(self._extract_time_features(timestamp))
            features.update(self._extract_range_exhaustion_features(candles, hist_candles))
            features.update(self._extract_vwap_features(candles))
            features.update(self._extract_trend_features(candles))

            logger.debug(f"Extracted {len(features)} intraday features")

        except Exception as e:
            logger.error(f"Intraday feature extraction failed: {e}", exc_info=True)
            features = self._get_default_features(timestamp)

        return features

    def _extract_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract time-based features."""
        # Minutes since market open
        market_open_dt = timestamp.replace(hour=9, minute=30, second=0)
        minutes_since_open = (timestamp - market_open_dt).seconds / 60

        return {
            'time_of_day_hour': timestamp.hour,
            'minutes_since_open': minutes_since_open,
        }

    def _extract_range_exhaustion_features(
        self,
        today_candles: pd.DataFrame,
        hist_candles: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract range exhaustion features.

        These are HIGH IMPACT features based on backtest results.
        """
        if today_candles.empty:
            return {
                'morning_range_percentile_20d': 50.0,
                'range_exhaustion_ratio': 0.5,
                'range_rate': 0.0,
            }

        # Current range (high - low so far today)
        current_range = today_candles['high'].max() - today_candles['low'].min()

        # Opening price
        open_price = today_candles.iloc[0]['open']

        # Current range as % of open
        range_pct = (current_range / open_price) * 100 if open_price > 0 else 0

        # Range percentile compared to last 20 days
        if not hist_candles.empty:
            # Group by day and get daily ranges
            hist_candles['date'] = pd.to_datetime(hist_candles['timestamp']).dt.date
            daily_ranges = hist_candles.groupby('date').apply(
                lambda x: (x['high'].max() - x['low'].min()) / x.iloc[0]['open'] * 100
            )

            # Get last 20 days
            daily_ranges = daily_ranges.tail(20)

            # Calculate percentile
            if len(daily_ranges) > 0:
                morning_range_percentile = (
                    (daily_ranges < range_pct).sum() / len(daily_ranges) * 100
                )
            else:
                morning_range_percentile = 50.0
        else:
            morning_range_percentile = 50.0

        # Expected range at this time of day (based on historical data)
        minutes_elapsed = len(today_candles)
        if minutes_elapsed > 0 and not hist_candles.empty:
            # Average range after N minutes
            # This is a simplified calculation
            expected_range_pct = range_pct * 0.7  # Placeholder
        else:
            expected_range_pct = range_pct

        # Range exhaustion ratio
        range_exhaustion_ratio = (
            range_pct / expected_range_pct if expected_range_pct > 0 else 1.0
        )

        # Range expansion rate (% per minute)
        range_rate = range_pct / minutes_elapsed if minutes_elapsed > 0 else 0.0

        return {
            'morning_range_percentile_20d': morning_range_percentile,
            'range_exhaustion_ratio': min(range_exhaustion_ratio, 3.0),  # Cap at 3x
            'range_rate': range_rate,
        }

    def _extract_vwap_features(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Extract VWAP-based features."""
        if candles.empty or len(candles) < 2:
            return {
                'price_vs_vwap': 0.0,
                'volume_vs_avg': 1.0,
            }

        # Calculate VWAP
        candles['typical_price'] = (candles['high'] + candles['low'] + candles['close']) / 3
        candles['vwap_cumsum'] = (candles['typical_price'] * candles['volume']).cumsum()
        candles['volume_cumsum'] = candles['volume'].cumsum()
        vwap = candles['vwap_cumsum'].iloc[-1] / candles['volume_cumsum'].iloc[-1]

        # Current price vs VWAP
        current_price = candles.iloc[-1]['close']
        price_vs_vwap = (current_price - vwap) / vwap if vwap > 0 else 0.0

        # Current volume vs average
        avg_volume = candles['volume'].mean()
        current_volume = candles.iloc[-1]['volume']
        volume_vs_avg = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            'price_vs_vwap': price_vs_vwap,
            'volume_vs_avg': min(volume_vs_avg, 5.0),  # Cap at 5x
        }

    def _extract_trend_features(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Extract trend and momentum features."""
        if candles.empty:
            return {
                'high_low_range_pct': 0.0,
                'close_vs_open_pct': 0.0,
                'intraday_trend': 0.0,
                'orb_range_pct': 0.0,
            }

        # High-low range %
        high = candles['high'].max()
        low = candles['low'].min()
        open_price = candles.iloc[0]['open']
        high_low_range_pct = ((high - low) / open_price) * 100 if open_price > 0 else 0

        # Close vs open %
        close_price = candles.iloc[-1]['close']
        close_vs_open_pct = ((close_price - open_price) / open_price) * 100 if open_price > 0 else 0

        # Intraday trend (linear regression slope)
        if len(candles) >= 10:
            x = np.arange(len(candles))
            y = candles['close'].values
            slope, _ = np.polyfit(x, y, 1)
            intraday_trend = slope
        else:
            intraday_trend = 0.0

        # Opening range breakout (first 30 min range)
        first_30min = candles.head(30)
        if not first_30min.empty:
            orb_high = first_30min['high'].max()
            orb_low = first_30min['low'].min()
            orb_range_pct = ((orb_high - orb_low) / open_price) * 100 if open_price > 0 else 0
        else:
            orb_range_pct = 0.0

        return {
            'high_low_range_pct': high_low_range_pct,
            'close_vs_open_pct': close_vs_open_pct,
            'intraday_trend': intraday_trend,
            'orb_range_pct': orb_range_pct,
        }

    def _get_default_features(self, timestamp: datetime) -> Dict[str, float]:
        """Return default feature values."""
        return {
            'time_of_day_hour': timestamp.hour,
            'minutes_since_open': 0.0,
            'morning_range_percentile_20d': 50.0,
            'range_exhaustion_ratio': 1.0,
            'range_rate': 0.0,
            'price_vs_vwap': 0.0,
            'volume_vs_avg': 1.0,
            'high_low_range_pct': 0.0,
            'close_vs_open_pct': 0.0,
            'intraday_trend': 0.0,
            'orb_range_pct': 0.0,
        }
