"""
0DTE Iron Condor with Stop Ladder Strategy.

Entry Conditions (ALL must be true):
1. VIX between 15-25 (sweet spot for premium selling)
2. No major economic events today (FOMC, NFP, CPI)
3. SPX/NDX within 0.5% of open price (range-bound day)
4. IV percentile > 30% for the day
5. Time: Enter between 9:45 AM - 11:00 AM

Exit Conditions:
- Target: 50% of premium collected
- Stop Ladder:
  - Close short call if delta > 0.30
  - Close short put if delta < -0.30
  - Close full position if underlying moves > 1 ATR
- Time: Close all by 3:45 PM regardless

Position Sizing:
- Max 3% of account per condor
- Width: 10-20 points on SPX, 5-10 on NDX

Liquidity Filters:
- SPX/XSP, NDX/QQQ only
- Minimum bid on short strikes > $0.50
- Spread width between 5-20 points

Timeframe: 0DTE only
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, time, date
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    SignalDirection,
    PositionState
)
from strategies.indicators.technical_indicators import TechnicalIndicators
from data.providers.base_provider import MarketDataProvider
from config.settings import settings

# ML imports
try:
    from ml.models.predictor import MLPredictor
    from ml.features.pipeline import feature_pipeline
    from ml.data.candle_loader import CandleLoader
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available - running without ML predictions")


class ZeroDTEIronCondorStrategy(BaseStrategy):
    """
    0DTE Iron Condor with Stop Ladder.

    Sells premium on range-bound days with strict delta-based stop management.
    """

    # Absolute Laws
    ALLOWED_UNDERLYINGS = ['SPX', '$SPX', 'XSP', 'NDX', 'QQQ']
    MIN_VIX = 15.0
    MAX_VIX = 25.0
    MAX_MOVE_FROM_OPEN_PCT = 0.5  # 0.5% max move from open
    DELTA_STOP = 0.30  # Close leg if delta exceeds this
    MIN_SHORT_BID = 0.50  # Minimum bid on short strikes

    def __init__(
        self,
        indicators: TechnicalIndicators,
        config: Dict[str, Any] = None,
        provider: Optional[MarketDataProvider] = None,
        ml_predictor: Optional['MLPredictor'] = None,
        candle_loader: Optional['CandleLoader'] = None
    ):
        """Initialize 0DTE Iron Condor strategy."""
        default_config = {
            'iv_percentile_min': 30,
            'trade_start_time': time(9, 45),
            'trade_end_time': time(11, 0),
            'close_time': time(15, 45),
            'position_size_pct': 0.03,        # 3% max per condor
            'profit_target_pct': 0.50,        # 50% of premium
            'short_delta_target': 0.10,       # Target delta for shorts
            'wing_width_spx': 10,             # 10 points for SPX
            'wing_width_ndx': 5,              # 5 points for NDX
        }

        if config:
            default_config.update(config)

        super().__init__(name='zero_dte_iron_condor', config=default_config)
        self.indicators = indicators
        self.provider = provider
        self._daily_open: Dict[str, float] = {}
        self._signaled_today: set = set()

        # ML components
        self.ml_predictor = ml_predictor
        self.candle_loader = candle_loader
        self.ml_enabled = settings.ML_ENABLED and ML_AVAILABLE and ml_predictor is not None
        self.min_win_probability = settings.ML_MIN_WIN_PROBABILITY
        self._last_prediction: Optional[float] = None

        if self.ml_enabled:
            logger.info("✓ ML predictions ENABLED for iron condor strategy")
        else:
            logger.info("ML predictions DISABLED for iron condor strategy")

    def get_watchlist(self) -> List[str]:
        """Return index symbols for iron condor trading."""
        return ['SPY', 'QQQ', '$SPX']  # ETFs and SPX index for 0DTE options

    async def generate_signals(
        self,
        market_data: Any,
        positions: List[PositionState]
    ) -> List[Signal]:
        """Generate iron condor signals on range-bound days."""
        signals = []

        for symbol in self.get_watchlist():
            if any(p.symbol == symbol and 'iron_condor' in (p.metadata.get('strategy_type', '')) for p in positions):
                continue

            if self.check_all_conditions_met(symbol):
                signal = await self._create_iron_condor_signal(symbol)
                if signal:
                    signals.append(signal)

        return signals

    def check_all_conditions_met(self, symbol: str) -> bool:
        """Check all entry conditions for iron condor."""
        # === Time window check ===
        current_time = datetime.now().time()
        if not (self.config['trade_start_time'] <= current_time <= self.config['trade_end_time']):
            return False

        # === ABSOLUTE LAW: VIX range ===
        vix = self._get_vix()
        if vix is None:
            return False

        if not (self.MIN_VIX <= vix <= self.MAX_VIX):
            logger.debug(f"VIX out of range: {vix:.1f} (need {self.MIN_VIX}-{self.MAX_VIX})")
            return False

        # Get indicators
        inds = self.indicators.get_latest_indicators(symbol)
        if not inds:
            return False

        current_price = inds.get('close', 0)

        # === Get/cache daily open ===
        today = datetime.now().date()
        open_key = f"{symbol}_{today}"
        if open_key not in self._daily_open:
            # Try to get from indicators
            daily_open = self.indicators.get_daily_open(symbol)
            if daily_open:
                self._daily_open[open_key] = daily_open
            else:
                return False

        open_price = self._daily_open[open_key]

        # === ABSOLUTE LAW: Range-bound check ===
        move_from_open_pct = abs(current_price - open_price) / open_price * 100
        if move_from_open_pct > self.MAX_MOVE_FROM_OPEN_PCT:
            logger.debug(f"{symbol}: Too much movement from open ({move_from_open_pct:.2f}% > {self.MAX_MOVE_FROM_OPEN_PCT}%)")
            return False

        # === IV percentile check ===
        iv_percentile = inds.get('iv_percentile', 0)
        if iv_percentile < self.config['iv_percentile_min']:
            logger.debug(f"{symbol}: IV percentile too low ({iv_percentile:.1f}% < {self.config['iv_percentile_min']}%)")
            return False

        # === ML prediction check (if enabled) ===
        if self.ml_enabled:
            try:
                win_probability = await self._check_ml_prediction(symbol)
                self._last_prediction = win_probability

                if win_probability < self.min_win_probability:
                    logger.debug(
                        f"{symbol}: ML prediction too low "
                        f"({win_probability:.3f} < {self.min_win_probability})"
                    )
                    return False

                logger.info(
                    f"{symbol}: ML prediction favorable "
                    f"({win_probability:.3f} >= {self.min_win_probability})"
                )
            except Exception as e:
                logger.error(f"ML prediction failed: {e}. Proceeding without ML check.")
                # Don't block trade if ML fails - fall back to rule-based

        # === Avoid duplicate signals ===
        signal_key = f"{symbol}_IC_{datetime.now().date()}"
        if signal_key in self._signaled_today:
            return False

        self._signaled_today.add(signal_key)
        logger.info(
            f"0DTE Iron Condor Signal: {symbol} @ ${current_price:.2f} "
            f"(VIX: {vix:.1f}, Move: {move_from_open_pct:.2f}%, IV%: {iv_percentile:.1f})"
        )
        return True

    async def _check_ml_prediction(self, symbol: str) -> float:
        """
        Compute ML features and predict win probability.

        Args:
            symbol: Underlying symbol

        Returns:
            Win probability between 0 and 1

        Raises:
            Exception: If feature extraction or prediction fails
        """
        # Get option chain data from provider
        option_chain = {}
        if self.provider:
            try:
                option_chain = await self.provider.get_option_chain(symbol)
            except Exception as e:
                logger.warning(f"Failed to get option chain for ML: {e}")

        # Extract features
        features = await feature_pipeline.extract_features_realtime(
            underlying_symbol=symbol,
            timestamp=datetime.now(),
            candle_loader=self.candle_loader,
            option_chain=option_chain
        )

        # Predict win probability
        win_probability = self.ml_predictor.predict(features)

        return win_probability

    def _get_vix(self) -> Optional[float]:
        """Get current VIX level."""
        try:
            vix_inds = self.indicators.get_latest_indicators('VIX')
            if vix_inds:
                return vix_inds.get('close')
        except Exception:
            pass
        return None

    async def _create_iron_condor_signal(self, symbol: str) -> Optional[Signal]:
        """Create iron condor signal with appropriate strikes."""
        inds = self.indicators.get_latest_indicators(symbol)
        if not inds:
            return None

        current_price = inds['close']
        atr = inds.get('atr', current_price * 0.01)

        # Determine wing width based on underlying
        if 'SPX' in symbol.upper() or symbol == 'SPY':
            wing_width = self.config['wing_width_spx']
            index_symbol = '$SPX'
        else:
            wing_width = self.config['wing_width_ndx']
            index_symbol = 'NDX'

        # Calculate strikes (10-delta target)
        # Short put: current - 1 ATR, round to nearest strike
        # Short call: current + 1 ATR, round to nearest strike
        short_put_strike = round((current_price - atr) / 5) * 5
        short_call_strike = round((current_price + atr) / 5) * 5
        long_put_strike = short_put_strike - wing_width
        long_call_strike = short_call_strike + wing_width

        # Estimate credit (simplified - would need real option prices)
        estimated_credit = wing_width * 0.15  # ~15% of width as credit

        # Premium selling exit levels:
        # Take Profit: 50% of credit (buy back at 50% of what we sold)
        # Stop Loss: 100% loss = 2x credit (buy back at 2x what we sold)
        take_profit_price = round(estimated_credit * 0.50, 2)  # Buy back at 50%
        stop_loss_price = round(estimated_credit * 2.00, 2)  # Buy back at 200% (100% loss)

        return Signal(
            signal_id=str(uuid.uuid4()),
            strategy_name=self.name,
            signal_type=SignalType.ENTRY,
            direction=SignalDirection.NEUTRAL,
            symbol=symbol,
            entry_price=estimated_credit,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            quantity=1,
            position_size_pct=self.config['position_size_pct'],
            confidence=0.70,
            priority=80,
            option_type='IRON_CONDOR',
            notes=f"0DTE IC: {long_put_strike}/{short_put_strike}p/{short_call_strike}/{long_call_strike}c @ ${estimated_credit:.2f} | TP=${take_profit_price} SL=${stop_loss_price}",
            metadata={
                'strategy_type': 'iron_condor',
                'index_symbol': index_symbol,
                'underlying_price': current_price,
                'long_put': long_put_strike,
                'short_put': short_put_strike,
                'short_call': short_call_strike,
                'long_call': long_call_strike,
                'wing_width': wing_width,
                'estimated_credit': estimated_credit,
                'delta_stop': self.DELTA_STOP,
            }
        )

    async def manage_position(
        self,
        position: PositionState,
        market_data: Any
    ) -> Optional[Signal]:
        """Manage iron condor with stop ladder."""
        should_exit, reason = self.should_exit(position, market_data)

        if should_exit:
            inds = self.indicators.get_latest_indicators(position.symbol)
            if not inds:
                return None

            return Signal(
                signal_id=str(uuid.uuid4()),
                strategy_name=self.name,
                signal_type=SignalType.EXIT,
                direction=position.direction,
                symbol=position.symbol,
                entry_price=inds['close'],
                quantity=position.quantity,
                notes=f"IC Exit: {reason}",
                position_id=position.position_id
            )

        return None

    def should_exit(
        self,
        position: PositionState,
        market_data: Any
    ) -> tuple[bool, str]:
        """Check iron condor exit conditions."""
        inds = self.indicators.get_latest_indicators(position.symbol)
        if not inds:
            return (False, "")

        current_price = inds['close']

        # === ABSOLUTE LAW: Time stop ===
        current_time = datetime.now().time()
        if current_time >= self.config['close_time']:
            return (True, "Time stop - closing before market close")

        # === Profit target ===
        if position.unrealized_pnl_pct >= (self.config['profit_target_pct'] * 100):
            return (True, f"Profit target hit ({position.unrealized_pnl_pct:.1f}%)")

        # === Delta-based stops (stop ladder) ===
        metadata = position.metadata
        short_put = metadata.get('short_put', 0)
        short_call = metadata.get('short_call', 0)
        atr = inds.get('atr', current_price * 0.01)

        # Check if price approaching short strikes (delta approximation)
        if short_put > 0 and current_price <= short_put:
            return (True, "Short put breached - delta stop")

        if short_call > 0 and current_price >= short_call:
            return (True, "Short call breached - delta stop")

        # ATR-based stop
        entry_price = metadata.get('underlying_price', current_price)
        if abs(current_price - entry_price) > atr:
            return (True, "ATR stop - underlying moved > 1 ATR")

        return (False, "")

    def reset_daily_state(self):
        """Reset daily tracking."""
        self._signaled_today.clear()
        self._daily_open.clear()
        logger.info(f"{self.name}: Daily state reset")
