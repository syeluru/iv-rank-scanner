"""
Feature extraction pipeline for ML predictions.

Orchestrates extraction of all 41 features across 6 categories:
1. Option Features (8 features) - IV, skew, deltas
2. Market Features (10 features) - VIX, momentum divergence
3. Realized Vol Features (6 features) - Historical vol metrics
4. Intraday Features (11 features) - Range exhaustion, VWAP, trends
5. Regime Features (3 features) - Vol regime, trend, event proximity
6. External Features (3 features) - HY OAS, yield curve, stress index
"""

from datetime import datetime
from typing import Dict, Optional
from loguru import logger


class FeaturePipeline:
    """Orchestrates feature extraction for ML predictions."""

    EXPECTED_FEATURE_COUNT = 41

    def __init__(self):
        """Initialize feature pipeline."""
        self._feature_cache = {}
        self._cache_timestamps = {}

    async def extract_features_realtime(
        self,
        underlying_symbol: str,
        timestamp: datetime,
        candle_loader,
        option_chain: dict,
        underlying_price: Optional[float] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract all 41 features for real-time prediction.

        Args:
            underlying_symbol: Underlying symbol (e.g., 'SPX')
            timestamp: Current timestamp
            candle_loader: CandleLoader instance for historical data
            option_chain: Option chain data from Schwab API
            underlying_price: Current underlying price (optional)
            cache_key: Optional cache key (default: f"{symbol}_{timestamp.minute}")

        Returns:
            Dict of 41 feature values
        """
        # Check cache
        if cache_key is None:
            cache_key = f"{underlying_symbol}_{timestamp.replace(second=0, microsecond=0)}"

        if cache_key in self._feature_cache:
            cache_age = (datetime.now() - self._cache_timestamps[cache_key]).seconds
            if cache_age < 300:  # 5 minute cache
                logger.debug(f"Using cached features for {cache_key}")
                return self._feature_cache[cache_key]

        features = {}

        try:
            # Import feature extractors (lazy import to avoid circular dependencies)
            from ml.features.option_features import OptionFeaturesExtractor
            from ml.features.market_features import MarketFeaturesExtractor
            from ml.features.realized_vol_features import RealizedVolExtractor
            from ml.features.intraday_features import IntradayFeaturesExtractor
            from ml.features.regime_features import RegimeFeaturesExtractor
            from ml.features.external_features import ExternalFeaturesExtractor

            # 1. Option Features (8 features)
            logger.debug("Extracting option features...")
            opt_extractor = OptionFeaturesExtractor()
            opt_features = await opt_extractor.extract(option_chain, underlying_price)
            features.update(opt_features)

            # 2. Market Features (10 features) - includes momentum divergence
            logger.debug("Extracting market features...")
            mkt_extractor = MarketFeaturesExtractor(candle_loader)
            mkt_features = await mkt_extractor.extract(timestamp)
            features.update(mkt_features)

            # 3. Realized Vol Features (6 features)
            logger.debug("Extracting realized volatility features...")
            vol_extractor = RealizedVolExtractor(candle_loader)
            vol_features = await vol_extractor.extract(underlying_symbol, timestamp)
            features.update(vol_features)

            # 4. Intraday Features (11 features) - includes range exhaustion
            logger.debug("Extracting intraday features...")
            intraday_extractor = IntradayFeaturesExtractor(candle_loader)
            intraday_features = await intraday_extractor.extract(underlying_symbol, timestamp)
            features.update(intraday_features)

            # 5. Regime Features (3 features)
            logger.debug("Extracting regime features...")
            regime_extractor = RegimeFeaturesExtractor(candle_loader)
            regime_features = await regime_extractor.extract(timestamp)
            features.update(regime_features)

            # 6. External Features (3 features)
            logger.debug("Extracting external features...")
            ext_extractor = ExternalFeaturesExtractor()
            ext_features = await ext_extractor.extract(timestamp)
            features.update(ext_features)

            # Validate feature count
            if len(features) != self.EXPECTED_FEATURE_COUNT:
                logger.warning(
                    f"Expected {self.EXPECTED_FEATURE_COUNT} features, got {len(features)}. "
                    f"Difference: {len(features) - self.EXPECTED_FEATURE_COUNT}"
                )

            # Cache the features
            self._feature_cache[cache_key] = features
            self._cache_timestamps[cache_key] = datetime.now()

            logger.info(f"✓ Extracted {len(features)} features for {underlying_symbol}")

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            raise

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear feature cache.

        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        if cache_key:
            self._feature_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
        else:
            self._feature_cache.clear()
            self._cache_timestamps.clear()


# Global instance
feature_pipeline = FeaturePipeline()
