"""
Position scaling manager for progressive entries.

Handles scaling from initial position to maximum position size
over time with safety checks and re-validation.
"""

from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from loguru import logger

from strategies.base_strategy import Signal
from config.settings import settings


@dataclass
class ScaleEntry:
    """Record of a single scale-in entry."""
    timestamp: datetime
    contracts: int
    confidence: float
    total_contracts: int
    price: float


class PositionScaler:
    """
    Manages progressive position scaling with safety checks.

    Example timeline:
    - 11:30 AM: Enter 10 contracts (if ML > 80%)
    - 12:00 PM: Add 5 contracts (if ML still > 80%)
    - 12:30 PM: Add 5 contracts (if ML still > 80%)
    - 1:00 PM: Add 5 contracts (if ML still > 80%)
    - 1:30 PM: Stop (max 25 reached or after 2 PM cutoff)
    """

    def __init__(
        self,
        initial_size: int = 10,
        scale_size: int = 5,
        max_total: int = 25,
        scale_interval_minutes: int = 30,
        cutoff_time: time = time(14, 0),  # 2 PM
        min_confidence: float = 0.80
    ):
        """
        Initialize position scaler.

        Args:
            initial_size: Initial position size (contracts)
            scale_size: Size of each scale-in (contracts)
            max_total: Maximum total position size (contracts)
            scale_interval_minutes: Minutes between scale-ins
            cutoff_time: No new entries after this time
            min_confidence: Minimum ML confidence for scaling
        """
        self.initial_size = initial_size
        self.scale_size = scale_size
        self.max_total = max_total
        self.scale_interval = timedelta(minutes=scale_interval_minutes)
        self.cutoff_time = cutoff_time
        self.min_confidence = min_confidence

        # Position tracking
        self.current_position = 0
        self.entry_time: Optional[datetime] = None
        self.last_scale_time: Optional[datetime] = None
        self.scale_history: List[ScaleEntry] = []

        # Signal tracking
        self._initial_signal: Optional[Signal] = None

        logger.info(
            f"PositionScaler initialized: "
            f"{initial_size}→{max_total} contracts, "
            f"scale every {scale_interval_minutes}min, "
            f"cutoff {cutoff_time.strftime('%H:%M')}"
        )

    def has_position(self) -> bool:
        """Check if we currently have a position."""
        return self.current_position > 0

    def is_maxed_out(self) -> bool:
        """Check if position is at maximum size."""
        return self.current_position >= self.max_total

    def should_scale(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if we should attempt to scale up.

        Args:
            current_time: Current timestamp (default: now)

        Returns:
            True if scaling conditions met, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()

        # No position yet
        if not self.has_position():
            logger.debug("No position to scale")
            return False

        # Already maxed out
        if self.is_maxed_out():
            logger.debug(f"Already at max position ({self.max_total})")
            return False

        # Past cutoff time
        if current_time.time() >= self.cutoff_time:
            logger.debug(f"Past cutoff time ({self.cutoff_time.strftime('%H:%M')})")
            return False

        # Not enough time elapsed since last scale
        if self.last_scale_time:
            time_since_last = current_time - self.last_scale_time
            if time_since_last < self.scale_interval:
                minutes_remaining = (self.scale_interval - time_since_last).seconds / 60
                logger.debug(
                    f"Too soon to scale (wait {minutes_remaining:.1f} more minutes)"
                )
                return False

        return True

    async def enter_initial_position(
        self,
        signal: Signal,
        order_placer: Optional[Any] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Place initial position entry.

        Args:
            signal: Signal object with trade parameters
            order_placer: Order execution interface (None = dry run)
            dry_run: If True, don't actually place order

        Returns:
            True if entry successful, False otherwise
        """
        if self.has_position():
            logger.warning("Already have position - skipping initial entry")
            return False

        try:
            # Adjust signal quantity for initial size
            signal.quantity = self.initial_size

            # Place order
            if dry_run or order_placer is None:
                logger.info(
                    f"[DRY RUN] Would enter {self.initial_size} contracts "
                    f"@ ${signal.entry_price:.2f} credit "
                    f"(confidence: {signal.confidence:.1%})"
                )
                success = True
            else:
                success = await self._execute_order(signal, order_placer)

            if success:
                # Update tracking
                self.current_position = self.initial_size
                self.entry_time = datetime.now()
                self.last_scale_time = self.entry_time
                self._initial_signal = signal

                # Record entry
                self.scale_history.append(ScaleEntry(
                    timestamp=self.entry_time,
                    contracts=self.initial_size,
                    confidence=signal.confidence,
                    total_contracts=self.current_position,
                    price=signal.entry_price
                ))

                logger.info(
                    f"✓ Initial entry: {self.initial_size} contracts "
                    f"@ ${signal.entry_price:.2f} "
                    f"(confidence: {signal.confidence:.1%})"
                )

                return True
            else:
                logger.error("Initial entry order failed")
                return False

        except Exception as e:
            logger.error(f"Initial entry failed: {e}", exc_info=True)
            return False

    async def scale_position(
        self,
        signal: Signal,
        order_placer: Optional[Any] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Add contracts to existing position.

        Args:
            signal: Updated signal with current conditions
            order_placer: Order execution interface
            dry_run: If True, don't actually place order

        Returns:
            True if scale successful, False otherwise
        """
        if not self.should_scale():
            return False

        # Check ML confidence
        if signal.confidence < self.min_confidence:
            logger.warning(
                f"ML confidence too low for scaling: "
                f"{signal.confidence:.1%} < {self.min_confidence:.1%}"
            )
            return False

        try:
            # Calculate contracts to add
            contracts_to_add = min(
                self.scale_size,
                self.max_total - self.current_position
            )

            # Adjust signal quantity
            signal.quantity = contracts_to_add

            # Place order
            if dry_run or order_placer is None:
                logger.info(
                    f"[DRY RUN] Would add {contracts_to_add} contracts "
                    f"@ ${signal.entry_price:.2f} "
                    f"(total: {self.current_position + contracts_to_add}/{self.max_total})"
                )
                success = True
            else:
                success = await self._execute_order(signal, order_placer)

            if success:
                # Update tracking
                self.current_position += contracts_to_add
                self.last_scale_time = datetime.now()

                # Record scale
                self.scale_history.append(ScaleEntry(
                    timestamp=self.last_scale_time,
                    contracts=contracts_to_add,
                    confidence=signal.confidence,
                    total_contracts=self.current_position,
                    price=signal.entry_price
                ))

                logger.info(
                    f"✓ Scaled up: +{contracts_to_add} contracts "
                    f"(total: {self.current_position}/{self.max_total}) "
                    f"@ ${signal.entry_price:.2f} "
                    f"(confidence: {signal.confidence:.1%})"
                )

                return True
            else:
                logger.error("Scale order failed")
                return False

        except Exception as e:
            logger.error(f"Position scaling failed: {e}", exc_info=True)
            return False

    async def _execute_order(self, signal: Signal, order_placer: Any) -> bool:
        """
        Execute order through order placer.

        Args:
            signal: Signal with order details
            order_placer: Order execution interface (OrderRouter instance)

        Returns:
            True if order placed successfully
        """
        try:
            # Execute through order router
            if order_placer:
                logger.info(f"Submitting order: {signal.quantity} {signal.symbol} contracts")

                # Submit signal to order router
                result = await order_placer.submit_signal(signal)

                if result.get('success'):
                    order_id = result.get('order_id')
                    logger.info(f"✓ Order placed successfully: {order_id}")
                    return True
                else:
                    error = result.get('error', 'Unknown error')
                    logger.error(f"✗ Order placement failed: {error}")
                    return False
            else:
                logger.warning("No order placer provided - cannot execute order")
                return False

        except Exception as e:
            logger.error(f"Order execution failed: {e}", exc_info=True)
            return False

    def get_scaling_summary(self) -> Dict[str, Any]:
        """
        Get summary of scaling history.

        Returns:
            Dict with scaling statistics
        """
        if not self.scale_history:
            return {
                'total_entries': 0,
                'total_contracts': 0,
                'average_confidence': 0.0,
                'average_price': 0.0,
            }

        return {
            'total_entries': len(self.scale_history),
            'total_contracts': self.current_position,
            'max_position': self.max_total,
            'average_confidence': sum(e.confidence for e in self.scale_history) / len(self.scale_history),
            'average_price': sum(e.price for e in self.scale_history) / len(self.scale_history),
            'first_entry': self.scale_history[0].timestamp.strftime('%H:%M:%S'),
            'last_entry': self.scale_history[-1].timestamp.strftime('%H:%M:%S'),
            'entries': [
                {
                    'time': e.timestamp.strftime('%H:%M:%S'),
                    'contracts': e.contracts,
                    'total': e.total_contracts,
                    'confidence': f"{e.confidence:.1%}",
                    'price': f"${e.price:.2f}"
                }
                for e in self.scale_history
            ]
        }

    def reset(self):
        """Reset scaler for new trading day."""
        self.current_position = 0
        self.entry_time = None
        self.last_scale_time = None
        self.scale_history = []
        self._initial_signal = None
        logger.info("PositionScaler reset for new day")
