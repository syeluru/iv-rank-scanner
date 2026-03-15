"""
Regime classification features (3 features).

Classifies current market regime:
- Volatility regime (low/medium/high VIX)
- Trend regime (up/down/sideways based on SPX)
- Macro event proximity (days until FOMC/CPI/NFP)
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class RegimeFeaturesExtractor(BaseFeatureExtractor):
    """Extract market regime features."""

    # VIX regime thresholds
    VIX_LOW_THRESHOLD = 15.0
    VIX_HIGH_THRESHOLD = 25.0

    # Trend detection lookback
    TREND_LOOKBACK_DAYS = 20

    def __init__(self, candle_loader=None, macro_event_loader=None):
        """
        Initialize regime features extractor.

        Args:
            candle_loader: Optional CandleLoader instance for VIX/SPX data
            macro_event_loader: Optional loader for macro event calendar
        """
        self.candle_loader = candle_loader
        self.macro_event_loader = macro_event_loader

    async def extract(
        self,
        timestamp: datetime,
        vix_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract regime features.

        Args:
            timestamp: Current timestamp
            vix_level: Current VIX level (optional, will fetch if not provided)

        Returns:
            Dict with 3 regime features
        """
        features = {}

        # Strip timezone info — DB stores tz-naive timestamps
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        try:
            # Volatility regime
            features['vol_regime'] = await self._classify_vol_regime(timestamp, vix_level)

            # Trend regime
            features['trend_regime'] = await self._classify_trend_regime(timestamp)

            # Macro event proximity
            features['macro_event_proximity'] = await self._get_event_proximity(timestamp)

            logger.debug(f"Extracted {len(features)} regime features")

        except Exception as e:
            logger.error(f"Regime feature extraction failed: {e}", exc_info=True)
            features = self._get_default_features()

        return features

    async def _classify_vol_regime(
        self,
        timestamp: datetime,
        vix_level: Optional[float] = None
    ) -> float:
        """
        Classify volatility regime based on VIX.

        Returns:
            0.0 = low vol (VIX < 15)
            1.0 = medium vol (VIX 15-25)
            2.0 = high vol (VIX > 25)
        """
        # Get VIX level if not provided
        if vix_level is None and self.candle_loader:
            vix_candles = await self.candle_loader.get_latest_candles('VIX', 1)
            if not vix_candles.empty:
                vix_level = vix_candles.iloc[-1]['close']

        if vix_level is None:
            return 1.0  # Default to medium

        # Classify
        if vix_level < self.VIX_LOW_THRESHOLD:
            return 0.0  # Low vol
        elif vix_level < self.VIX_HIGH_THRESHOLD:
            return 1.0  # Medium vol
        else:
            return 2.0  # High vol

    async def _classify_trend_regime(self, timestamp: datetime) -> float:
        """
        Classify trend regime based on SPX price action.

        Returns:
            -1.0 = downtrend
             0.0 = sideways/choppy
             1.0 = uptrend
        """
        if not self.candle_loader:
            return 0.0  # Default to sideways

        try:
            # Get last N days of SPX data
            end_time = timestamp
            start_time = end_time - timedelta(days=self.TREND_LOOKBACK_DAYS + 5)
            spx_candles = await self.candle_loader.get_candles('SPX', start_time, end_time)

            if spx_candles.empty or len(spx_candles) < 10:
                return 0.0

            # Group by day and get daily closes
            spx_candles['date'] = pd.to_datetime(spx_candles['timestamp']).dt.date
            daily = spx_candles.groupby('date')['close'].last()

            if len(daily) < 10:
                return 0.0

            # Calculate linear regression slope
            x = np.arange(len(daily))
            y = daily.values

            # Fit line
            slope, _ = np.polyfit(x, y, 1)

            # Normalize slope to trend strength
            # Slope is in points per day; normalize by price
            avg_price = y.mean()
            normalized_slope = (slope / avg_price) * 100 * len(daily)  # % over period

            # Classify
            if normalized_slope < -2.0:  # Down > 2% over period
                return -1.0
            elif normalized_slope > 2.0:  # Up > 2% over period
                return 1.0
            else:
                return 0.0  # Sideways

        except Exception as e:
            logger.warning(f"Trend regime classification failed: {e}")
            return 0.0

    async def _get_event_proximity(self, timestamp: datetime) -> float:
        """
        Get days until next major macro event (FOMC, CPI, NFP).

        Returns:
            Float representing days until next event (0-30)
            Returns 30.0 if no events in next 30 days
        """
        if not self.macro_event_loader:
            # If no event loader, use heuristics
            return self._estimate_event_proximity(timestamp)

        try:
            # Query macro_events table for next event
            # This would require DatabaseManager integration
            # For now, use heuristic
            return self._estimate_event_proximity(timestamp)

        except Exception as e:
            logger.warning(f"Event proximity lookup failed: {e}")
            return 30.0

    def _estimate_event_proximity(self, timestamp: datetime) -> float:
        """
        Estimate days until next major event using calendar heuristics.

        FOMC: ~8 meetings per year (every 6 weeks)
        CPI: Monthly, typically 2nd week
        NFP: First Friday of month

        Returns:
            Days until next likely event
        """
        current_date = timestamp.date()

        # Check for NFP (first Friday)
        first_friday = self._get_first_friday(current_date.year, current_date.month)
        days_to_nfp = (first_friday - current_date).days

        if days_to_nfp < 0:
            # Already passed this month, check next month
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year if current_date.month < 12 else current_date.year + 1
            first_friday = self._get_first_friday(next_year, next_month)
            days_to_nfp = (first_friday - current_date).days

        # Check for CPI (assume 10th-15th of month)
        cpi_date = current_date.replace(day=13)  # Approximate
        days_to_cpi = (cpi_date - current_date).days

        if days_to_cpi < 0:
            # Next month
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year if current_date.month < 12 else current_date.year + 1
            cpi_date = current_date.replace(year=next_year, month=next_month, day=13)
            days_to_cpi = (cpi_date - current_date).days

        # Return minimum (nearest event)
        nearest_event_days = min(days_to_nfp, days_to_cpi)

        # Cap at 30
        return float(min(nearest_event_days, 30))

    def _get_first_friday(self, year: int, month: int) -> datetime:
        """Get first Friday of given month."""
        # Start with first day of month
        first_day = datetime(year, month, 1).date()

        # Find first Friday (weekday 4 = Friday)
        days_ahead = 4 - first_day.weekday()
        if days_ahead < 0:
            days_ahead += 7

        return first_day + timedelta(days=days_ahead)

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values (neutral regime)."""
        return {
            'vol_regime': 1.0,  # Medium vol
            'trend_regime': 0.0,  # Sideways
            'macro_event_proximity': 15.0,  # Mid-month
        }
