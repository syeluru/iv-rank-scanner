"""
Market-based features (10 features).

Extracts broad market condition features:
- VIX levels and changes
- Market internals
- Term structure
- Momentum divergence (SPX/VIX conflict detection)
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class MarketFeaturesExtractor(BaseFeatureExtractor):
    """Extract market-based features."""

    def __init__(self, candle_loader):
        """
        Initialize market features extractor.

        Args:
            candle_loader: CandleLoader instance for SPX/VIX data
        """
        self.candle_loader = candle_loader

    async def extract(self, timestamp: datetime) -> Dict[str, float]:
        """
        Extract market features.

        Args:
            timestamp: Current timestamp

        Returns:
            Dict with 10 market features
        """
        features = {}

        try:
            # Get VIX data (last 30 days for changes, last 30 min for correlation)
            vix_end = timestamp
            vix_start_30d = vix_end - timedelta(days=35)  # Buffer for weekends
            vix_candles = await self.candle_loader.get_candles('VIX', vix_start_30d, vix_end)

            # Get SPX data (for correlation)
            spx_candles = await self.candle_loader.get_candles('SPX', vix_start_30d, vix_end)

            # Extract VIX features
            features.update(self._extract_vix_features(vix_candles, timestamp))

            # Extract momentum divergence features (HIGH IMPACT)
            features.update(self._extract_momentum_divergence(spx_candles, vix_candles, timestamp))

            logger.debug(f"Extracted {len(features)} market features")

        except Exception as e:
            logger.error(f"Market feature extraction failed: {e}", exc_info=True)
            features = self._get_default_features()

        return features

    def _extract_vix_features(self, vix_candles: pd.DataFrame, timestamp: datetime) -> Dict[str, float]:
        """Extract VIX-based features."""
        if vix_candles.empty:
            return {
                'vix_level': 18.5,  # Default neutral level
                'vix_change_1d': 0.0,
                'vix_change_5d': 0.0,
                'vix9d_vix': 1.0,
                'vvix_level': 100.0,
                'term_structure_slope': 0.0,
                'market_breadth': 0.5,
            }

        # Current VIX level
        current_vix = vix_candles.iloc[-1]['close']

        # VIX change over 1 day
        one_day_ago = timestamp - timedelta(days=1)
        vix_1d_candles = vix_candles[vix_candles['timestamp'] <= one_day_ago]
        if not vix_1d_candles.empty:
            vix_1d_ago = vix_1d_candles.iloc[-1]['close']
            vix_change_1d = ((current_vix - vix_1d_ago) / vix_1d_ago) * 100
        else:
            vix_change_1d = 0.0

        # VIX change over 5 days
        five_days_ago = timestamp - timedelta(days=5)
        vix_5d_candles = vix_candles[vix_candles['timestamp'] <= five_days_ago]
        if not vix_5d_candles.empty:
            vix_5d_ago = vix_5d_candles.iloc[-1]['close']
            vix_change_5d = ((current_vix - vix_5d_ago) / vix_5d_ago) * 100
        else:
            vix_change_5d = 0.0

        # VIX9D/VIX ratio (short-term vs current volatility)
        # Note: VIX9D would need separate data source; using approximation
        # In production, fetch actual VIX9D from market data
        vix9d_vix = 1.0  # Placeholder - 1.0 = neutral

        # VVIX (volatility of VIX)
        # Calculate from VIX price swings over last 20 days
        last_20d = vix_candles.tail(20 * 390)  # ~20 trading days of 1-min candles
        if len(last_20d) >= 10:
            vix_returns = last_20d['close'].pct_change().dropna()
            vvix_level = vix_returns.std() * np.sqrt(252 * 390) * 100  # Annualized vol
        else:
            vvix_level = 100.0  # Default

        # Term structure slope (VIX futures curve slope)
        # Placeholder - would need VIX futures data
        term_structure_slope = 0.0  # Positive = contango, negative = backwardation

        # Market breadth (advance/decline)
        # Placeholder - would need NYSE breadth data
        market_breadth = 0.5  # 0-1 scale, 0.5 = neutral

        return {
            'vix_level': float(current_vix),
            'vix_change_1d': float(vix_change_1d),
            'vix_change_5d': float(vix_change_5d),
            'vix9d_vix': float(vix9d_vix),
            'vvix_level': float(vvix_level),
            'term_structure_slope': float(term_structure_slope),
            'market_breadth': float(market_breadth),
        }

    def _extract_momentum_divergence(
        self,
        spx_candles: pd.DataFrame,
        vix_candles: pd.DataFrame,
        timestamp: datetime
    ) -> Dict[str, float]:
        """
        Extract momentum divergence features (HIGH IMPACT).

        These detect SPX/VIX conflicts - early warning signals that market
        sentiment is confused or positioning for reversal.

        Normally: SPX ↑ → VIX ↓ (negative correlation ~-0.7)
        Warning: SPX ↑ → VIX ↑ (positive correlation = danger)
        """
        if spx_candles.empty or vix_candles.empty:
            return {
                'spx_vix_correlation_30min': -0.7,  # Normal negative correlation
                'spx_up_vix_up': 0.0,
                'divergence_score': 0.0,
            }

        # Get last 30 minutes of data
        start_30min = timestamp - timedelta(minutes=30)
        spx_30min = spx_candles[spx_candles['timestamp'] >= start_30min].copy()
        vix_30min = vix_candles[vix_candles['timestamp'] >= start_30min].copy()

        if len(spx_30min) < 10 or len(vix_30min) < 10:
            # Not enough data
            return {
                'spx_vix_correlation_30min': -0.7,
                'spx_up_vix_up': 0.0,
                'divergence_score': 0.0,
            }

        # Align timestamps (merge on timestamp)
        spx_30min = spx_30min.set_index('timestamp')
        vix_30min = vix_30min.set_index('timestamp')
        combined = pd.DataFrame({
            'spx_close': spx_30min['close'],
            'vix_close': vix_30min['close']
        }).dropna()

        if len(combined) < 10:
            return {
                'spx_vix_correlation_30min': -0.7,
                'spx_up_vix_up': 0.0,
                'divergence_score': 0.0,
            }

        # Calculate returns
        combined['spx_return'] = combined['spx_close'].pct_change()
        combined['vix_return'] = combined['vix_close'].pct_change()
        combined = combined.dropna()

        # 30-minute rolling correlation
        if len(combined) >= 5:
            correlation = combined['spx_return'].corr(combined['vix_return'])
        else:
            correlation = -0.7

        # Check for SPX up + VIX up (danger signal)
        spx_move = (combined['spx_close'].iloc[-1] - combined['spx_close'].iloc[0]) / combined['spx_close'].iloc[0]
        vix_move = (combined['vix_close'].iloc[-1] - combined['vix_close'].iloc[0]) / combined['vix_close'].iloc[0]

        spx_up_vix_up = 1.0 if (spx_move > 0 and vix_move > 0) else 0.0

        # Divergence score (composite metric)
        # Combines correlation anomaly + directional conflict
        # Score > 0.5 = warning, > 0.7 = strong warning
        correlation_anomaly = max(0, correlation + 0.7) / 1.7  # Normalize to 0-1 (0.7 = neutral)
        divergence_score = (correlation_anomaly * 0.6 + spx_up_vix_up * 0.4)

        return {
            'spx_vix_correlation_30min': float(correlation),
            'spx_up_vix_up': float(spx_up_vix_up),
            'divergence_score': float(divergence_score),
        }

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values."""
        return {
            'vix_level': 18.5,
            'vix_change_1d': 0.0,
            'vix_change_5d': 0.0,
            'vix9d_vix': 1.0,
            'vvix_level': 100.0,
            'term_structure_slope': 0.0,
            'market_breadth': 0.5,
            'spx_vix_correlation_30min': -0.7,
            'spx_up_vix_up': 0.0,
            'divergence_score': 0.0,
        }
