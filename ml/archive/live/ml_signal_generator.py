"""
ML-based signal generator for live trading.

Generates iron condor entry signals using the trained ML model
with confidence scoring and real-time feature extraction.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from loguru import logger

from ml.models.predictor import MLPredictor
from ml.features.pipeline import feature_pipeline
from ml.data.candle_loader import CandleLoader
from strategies.base_strategy import Signal, SignalType, SignalDirection
from config.settings import settings


class MLSignalGenerator:
    """
    Generates ML-powered iron condor signals at scheduled times.

    Uses the trained LightGBM model to predict trade success probability
    and only generates signals when confidence exceeds threshold.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_confidence: float = 0.80,
        candle_loader: Optional[CandleLoader] = None
    ):
        """
        Initialize ML signal generator.

        Args:
            model_path: Path to trained model (default: from settings)
            min_confidence: Minimum confidence threshold (0-1)
            candle_loader: CandleLoader instance (created if None)
        """
        self.model_path = model_path or settings.ML_MODEL_PATH
        self.min_confidence = min_confidence

        # Initialize ML predictor
        self.predictor = MLPredictor(model_path=self.model_path)

        # Initialize candle loader
        self.candle_loader = candle_loader or CandleLoader()

        # Track last prediction for debugging
        self._last_prediction: Optional[float] = None
        self._last_features: Optional[Dict[str, float]] = None

        if self.predictor.is_ready():
            logger.info(
                f"✓ MLSignalGenerator ready "
                f"(threshold: {self.min_confidence:.1%}, "
                f"features: {self.predictor.num_features})"
            )
        else:
            logger.warning("⚠️  ML model not loaded - signal generation will fail")

    async def generate_signal(
        self,
        symbol: str,
        provider: Optional[Any] = None
    ) -> Optional[Signal]:
        """
        Generate iron condor signal with ML confidence check.

        Args:
            symbol: Underlying symbol (e.g., 'SPX')
            provider: Market data provider for option chains

        Returns:
            Signal if ML confidence >= threshold, None otherwise
        """
        if not self.predictor.is_ready():
            logger.error("ML model not loaded - cannot generate signal")
            return None

        try:
            # Get current market data
            timestamp = datetime.now()

            # Get option chain from provider
            option_chain = {}
            if provider:
                try:
                    option_chain = await provider.get_option_chain(symbol)
                except Exception as e:
                    logger.warning(f"Failed to get option chain: {e}")

            # Extract features
            logger.debug(f"Extracting features for {symbol}...")
            features = await feature_pipeline.extract_features_realtime(
                underlying_symbol=symbol,
                timestamp=timestamp,
                candle_loader=self.candle_loader,
                option_chain=option_chain
            )

            # Get ML prediction
            prediction = self.predictor.predict(features)

            # Store for debugging
            self._last_prediction = prediction
            self._last_features = features

            logger.info(
                f"ML Prediction for {symbol}: {prediction:.1%} confidence "
                f"(threshold: {self.min_confidence:.1%})"
            )

            # Check threshold
            if prediction < self.min_confidence:
                logger.debug(
                    f"Prediction below threshold - no signal generated "
                    f"({prediction:.1%} < {self.min_confidence:.1%})"
                )
                return None

            # Generate signal
            signal = await self._create_iron_condor_signal(
                symbol=symbol,
                confidence=prediction,
                features=features,
                option_chain=option_chain
            )

            logger.info(
                f"✓ Signal generated for {symbol}: "
                f"confidence={prediction:.1%}, "
                f"estimated_credit=${signal.entry_price:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            return None

    async def _create_iron_condor_signal(
        self,
        symbol: str,
        confidence: float,
        features: Dict[str, float],
        option_chain: dict
    ) -> Signal:
        """
        Create iron condor signal with strike selection.

        Args:
            symbol: Underlying symbol
            confidence: ML confidence score
            features: Extracted features
            option_chain: Option chain data

        Returns:
            Signal object with IC parameters
        """
        # Get current price from features or option chain
        current_price = option_chain.get('underlying_price', 5800.0)

        # Strike selection (simplified - could use delta targeting)
        # For SPX, use ~10-delta strikes with 50-point wings
        wing_width = 50
        short_put_strike = round((current_price * 0.98) / 5) * 5  # ~2% OTM
        short_call_strike = round((current_price * 1.02) / 5) * 5  # ~2% OTM
        long_put_strike = short_put_strike - wing_width
        long_call_strike = short_call_strike + wing_width

        # Estimate credit (simplified - would use actual option prices)
        estimated_credit = wing_width * 0.15  # ~15% of width

        # Exit levels (50% profit target, 200% loss)
        take_profit_price = round(estimated_credit * 0.50, 2)
        stop_loss_price = round(estimated_credit * 2.00, 2)

        return Signal(
            signal_id=str(uuid.uuid4()),
            strategy_name='ml_iron_condor',
            signal_type=SignalType.ENTRY,
            direction=SignalDirection.NEUTRAL,
            symbol=symbol,
            entry_price=estimated_credit,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            quantity=1,  # Will be multiplied by position scaler
            position_size_pct=0.03,
            confidence=confidence,
            priority=90,  # High priority (ML-based)
            option_type='IRON_CONDOR',
            notes=(
                f"ML IC @ {datetime.now().strftime('%H:%M')} | "
                f"Confidence: {confidence:.1%} | "
                f"{long_put_strike}/{short_put_strike}/"
                f"{short_call_strike}/{long_call_strike} | "
                f"Credit: ${estimated_credit:.2f}"
            ),
            metadata={
                'strategy_type': 'iron_condor',
                'underlying_price': current_price,
                'long_put': long_put_strike,
                'short_put': short_put_strike,
                'short_call': short_call_strike,
                'long_call': long_call_strike,
                'wing_width': wing_width,
                'estimated_credit': estimated_credit,
                'ml_confidence': confidence,
                'ml_features': features,
                'entry_timestamp': datetime.now().isoformat(),
            }
        )

    def get_last_prediction(self) -> Optional[float]:
        """Get the most recent prediction score."""
        return self._last_prediction

    def get_last_features(self) -> Optional[Dict[str, float]]:
        """Get the most recent feature values."""
        return self._last_features

    def get_feature_importance_summary(self) -> str:
        """Get a summary of which features drove the prediction."""
        if not self._last_features or not self.predictor.is_ready():
            return "No prediction available"

        # Get top 5 features by importance
        importances = self.predictor.model.feature_importances_
        feature_names = self.predictor.feature_names

        top_features = sorted(
            zip(feature_names, importances, [self._last_features.get(f, 0) for f in feature_names]),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        summary = "Top 5 feature drivers:\n"
        for name, importance, value in top_features:
            summary += f"  - {name}: {value:.3f} (importance: {importance:.1f})\n"

        return summary
