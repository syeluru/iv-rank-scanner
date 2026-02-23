"""
Realized volatility features (6 features).

Extracts historical volatility metrics:
- Parkinson volatility (high/low range-based)
- Garman-Klass volatility (OHLC-based, most efficient)
- Close-to-close volatility (simple returns)
- Multi-period analysis (5d, 10d, 20d)
- Volatility acceleration ratios
"""

from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class RealizedVolExtractor(BaseFeatureExtractor):
    """Extract realized volatility features."""

    def __init__(self, candle_loader):
        """
        Initialize realized vol extractor.

        Args:
            candle_loader: CandleLoader instance
        """
        self.candle_loader = candle_loader

    async def extract(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Extract realized volatility features.

        Args:
            symbol: Underlying symbol (e.g., 'SPX')
            timestamp: Current timestamp

        Returns:
            Dict with 6 realized vol features (annualized %)
        """
        features = {}

        try:
            # Get historical candles (25 trading days for 20-day vol)
            end_time = timestamp
            start_time = end_time - timedelta(days=45)  # Buffer for weekends + holidays
            candles = await self.candle_loader.get_candles(symbol, start_time, end_time)

            if candles.empty or len(candles) < 10:
                logger.warning(f"Insufficient data for {symbol} realized vol")
                return self._get_default_features()

            # Group by day to get daily OHLC (filter weekends — ThetaData forward-fills them)
            candles['date'] = pd.to_datetime(candles['timestamp']).dt.date
            daily = candles.groupby('date').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).reset_index()
            daily['date'] = pd.to_datetime(daily['date'])
            daily = daily[
                (daily['date'].dt.weekday < 5) & (daily['close'] > 0)
            ].reset_index(drop=True)

            if len(daily) < 5:
                logger.warning(f"Insufficient daily data for {symbol}")
                return self._get_default_features()

            # Calculate features
            features['rv_5d'] = self._close_to_close_vol(daily, 5)
            features['rv_10d'] = self._close_to_close_vol(daily, 10)
            features['rv_20d'] = self._close_to_close_vol(daily, 20)
            features['parkinson_vol_10d'] = self._parkinson_vol(daily, 10)
            features['garman_klass_vol_10d'] = self._garman_klass_vol(daily, 10)

            # Volatility ratio (acceleration/deceleration)
            if features['rv_20d'] > 0:
                features['rv_ratio_5_20'] = features['rv_5d'] / features['rv_20d']
            else:
                features['rv_ratio_5_20'] = 1.0

            logger.debug(f"Extracted {len(features)} realized vol features")

        except Exception as e:
            logger.error(f"Realized vol extraction failed: {e}", exc_info=True)
            features = self._get_default_features()

        return features

    def _close_to_close_vol(self, daily_df: pd.DataFrame, lookback: int) -> float:
        """
        Calculate close-to-close realized volatility.

        Args:
            daily_df: DataFrame with daily OHLC
            lookback: Number of days to look back

        Returns:
            Annualized volatility in percentage
        """
        if len(daily_df) < lookback:
            return 15.0  # Default ~15% vol

        # Get last N days
        recent = daily_df.tail(lookback).copy()

        # Calculate log returns
        recent['returns'] = np.log(recent['close'] / recent['close'].shift(1))

        # Drop NaN
        returns = recent['returns'].dropna()

        if len(returns) < 2:
            return 15.0

        # Calculate volatility (annualized)
        vol = returns.std() * np.sqrt(252) * 100  # As percentage

        return float(vol)

    def _parkinson_vol(self, daily_df: pd.DataFrame, lookback: int) -> float:
        """
        Calculate Parkinson volatility estimator.

        Uses high-low range, more efficient than close-to-close.
        Formula: sqrt(1/(4*ln(2)) * mean((ln(H/L))^2))

        Args:
            daily_df: DataFrame with daily OHLC
            lookback: Number of days

        Returns:
            Annualized volatility in percentage
        """
        if len(daily_df) < lookback:
            return 15.0

        recent = daily_df.tail(lookback).copy()

        # Calculate log(high/low)^2
        recent['hl_ratio'] = np.log(recent['high'] / recent['low']) ** 2

        # Parkinson estimator
        mean_hl2 = recent['hl_ratio'].mean()
        vol = np.sqrt(mean_hl2 / (4 * np.log(2))) * np.sqrt(252) * 100

        return float(vol)

    def _garman_klass_vol(self, daily_df: pd.DataFrame, lookback: int) -> float:
        """
        Calculate Garman-Klass volatility estimator.

        Most efficient OHLC-based estimator.
        Formula: sqrt(mean(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))

        Args:
            daily_df: DataFrame with daily OHLC
            lookback: Number of days

        Returns:
            Annualized volatility in percentage
        """
        if len(daily_df) < lookback:
            return 15.0

        recent = daily_df.tail(lookback).copy()

        # Calculate components
        hl = np.log(recent['high'] / recent['low']) ** 2
        co = np.log(recent['close'] / recent['open']) ** 2

        # Garman-Klass estimator
        gk = 0.5 * hl - (2 * np.log(2) - 1) * co
        vol = np.sqrt(gk.mean()) * np.sqrt(252) * 100

        return float(vol)

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values (neutral volatility)."""
        return {
            'rv_5d': 15.0,
            'rv_10d': 15.0,
            'rv_20d': 15.0,
            'parkinson_vol_10d': 15.0,
            'garman_klass_vol_10d': 15.0,
            'rv_ratio_5_20': 1.0,
        }
