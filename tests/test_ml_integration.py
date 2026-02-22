"""
Unit tests for ML integration.

Tests ML predictor, feature pipeline, and strategy integration.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import tempfile
from pathlib import Path


@pytest.fixture
def mock_candle_loader():
    """Mock candle loader."""
    loader = Mock()
    loader.get_candles = AsyncMock(return_value=Mock())
    loader.get_latest_candles = AsyncMock(return_value=Mock())
    return loader


@pytest.fixture
def mock_option_chain():
    """Mock option chain data."""
    return {
        'calls': [],
        'puts': [],
        'underlying_price': 5800.0
    }


@pytest.fixture
def mock_features():
    """Mock feature dict with all 35 features."""
    return {
        # Option features (8)
        'iv_call_atm': 0.20,
        'iv_put_atm': 0.21,
        'iv_call_otm_10': 0.19,
        'iv_put_otm_10': 0.22,
        'iv_skew': 0.02,
        'delta_call_atm': 0.50,
        'delta_put_atm': -0.50,
        'put_call_ratio': 1.1,

        # Market features (7)
        'vix_level': 18.5,
        'vix_change_1d': 0.5,
        'vix_change_5d': -1.2,
        'vix9d_vix': 0.95,
        'vvix_level': 85.0,
        'term_structure_slope': 0.02,
        'market_breadth': 0.6,

        # Realized vol features (6)
        'rv_5d': 0.15,
        'rv_10d': 0.16,
        'rv_20d': 0.14,
        'parkinson_vol_10d': 0.17,
        'garman_klass_vol_10d': 0.16,
        'rv_ratio_5_20': 1.07,

        # Intraday features (8)
        'time_of_day_hour': 10,
        'minutes_since_open': 60,
        'price_vs_vwap': 0.001,
        'orb_range_pct': 0.3,
        'volume_vs_avg': 1.2,
        'high_low_range_pct': 0.4,
        'close_vs_open_pct': 0.1,
        'intraday_trend': 0.5,

        # Regime features (3)
        'vol_regime': 1.0,
        'trend_regime': 0.0,
        'macro_event_proximity': 7.0,

        # External features (3)
        'hy_oas': 350.0,
        'yield_curve_10y_2y': 0.5,
        'market_stress_index': 0.3,
    }


class TestMLPredictor:
    """Test ML predictor class."""

    def test_predictor_initialization_without_model(self):
        """Test predictor initialization when model doesn't exist."""
        from ml.models.predictor import MLPredictor

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=True) as tmp:
            # Path doesn't exist
            predictor = MLPredictor(model_path=Path(tmp.name + '_nonexistent'))
            assert not predictor.is_ready()

    def test_predictor_requires_model_for_prediction(self, mock_features):
        """Test that predictor requires loaded model."""
        from ml.models.predictor import MLPredictor

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=True) as tmp:
            predictor = MLPredictor(model_path=Path(tmp.name + '_nonexistent'))

            with pytest.raises(ValueError, match="ML model not loaded"):
                predictor.predict(mock_features)


class TestFeaturePipeline:
    """Test feature extraction pipeline."""

    @pytest.mark.asyncio
    async def test_feature_pipeline_extracts_35_features(
        self,
        mock_candle_loader,
        mock_option_chain
    ):
        """Test that pipeline extracts exactly 35 features."""
        from ml.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()

        # Extract features
        features = await pipeline.extract_features_realtime(
            underlying_symbol='SPX',
            timestamp=datetime.now(),
            candle_loader=mock_candle_loader,
            option_chain=mock_option_chain
        )

        # Should have 35 features (though they may be zeros in stub implementation)
        assert len(features) == 35
        assert all(isinstance(v, (int, float)) for v in features.values())

    @pytest.mark.asyncio
    async def test_feature_pipeline_caching(
        self,
        mock_candle_loader,
        mock_option_chain
    ):
        """Test that pipeline caches features."""
        from ml.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()
        cache_key = "test_key"

        # First extraction
        features1 = await pipeline.extract_features_realtime(
            underlying_symbol='SPX',
            timestamp=datetime.now(),
            candle_loader=mock_candle_loader,
            option_chain=mock_option_chain,
            cache_key=cache_key
        )

        # Second extraction with same key should hit cache
        features2 = await pipeline.extract_features_realtime(
            underlying_symbol='SPX',
            timestamp=datetime.now(),
            candle_loader=mock_candle_loader,
            option_chain=mock_option_chain,
            cache_key=cache_key
        )

        assert features1 == features2

        # Clear cache
        pipeline.clear_cache(cache_key)


class TestIronCondorStrategy:
    """Test iron condor strategy with ML integration."""

    @pytest.mark.asyncio
    async def test_strategy_without_ml(self):
        """Test that strategy works without ML components."""
        from strategies.algo.zero_dte_iron_condor_strategy import ZeroDTEIronCondorStrategy
        from strategies.indicators.technical_indicators import TechnicalIndicators

        indicators = TechnicalIndicators()
        strategy = ZeroDTEIronCondorStrategy(
            indicators=indicators,
            ml_predictor=None,
            candle_loader=None
        )

        assert not strategy.ml_enabled

    @pytest.mark.asyncio
    async def test_strategy_with_ml_but_disabled(self):
        """Test strategy when ML is disabled in settings."""
        from strategies.algo.zero_dte_iron_condor_strategy import ZeroDTEIronCondorStrategy
        from strategies.indicators.technical_indicators import TechnicalIndicators
        from config.settings import settings

        # Temporarily disable ML
        original_ml_enabled = settings.ML_ENABLED
        settings.ML_ENABLED = False

        try:
            indicators = TechnicalIndicators()
            mock_predictor = Mock()

            strategy = ZeroDTEIronCondorStrategy(
                indicators=indicators,
                ml_predictor=mock_predictor,
                candle_loader=Mock()
            )

            assert not strategy.ml_enabled

        finally:
            settings.ML_ENABLED = original_ml_enabled


class TestCandleLoader:
    """Test candle loader."""

    @pytest.mark.asyncio
    async def test_candle_loader_initialization(self):
        """Test candle loader can be initialized."""
        from ml.data.candle_loader import CandleLoader

        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
            loader = CandleLoader(db_path=tmp.name)
            assert loader is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
