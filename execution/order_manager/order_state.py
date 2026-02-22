"""
Order state machine for tracking order lifecycle.

This module implements a state machine for order management, tracking orders
from creation through submission, execution, and completion/cancellation.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from uuid import uuid4
from loguru import logger


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "PENDING"           # Created but not yet submitted
    VALIDATING = "VALIDATING"     # Pre-trade checks in progress
    VALIDATED = "VALIDATED"       # Passed pre-trade checks
    SUBMITTING = "SUBMITTING"     # Being submitted to broker
    SUBMITTED = "SUBMITTED"       # Submitted to broker, awaiting acceptance
    ACCEPTED = "ACCEPTED"         # Accepted by broker
    WORKING = "WORKING"           # Active in market
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Some quantity filled
    FILLED = "FILLED"             # Completely filled
    CANCELLING = "CANCELLING"     # Cancel request sent
    CANCELLED = "CANCELLED"       # Cancelled successfully
    REJECTED = "REJECTED"         # Rejected by broker or pre-trade checks
    EXPIRED = "EXPIRED"           # Order expired
    FAILED = "FAILED"             # Technical failure


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side/action."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class OrderDuration(Enum):
    """Order time in force."""
    DAY = "DAY"
    GTC = "GTC"  # Good til cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


@dataclass
class OrderFill:
    """Represents a single fill event for an order."""
    fill_id: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: float = 0.0


@dataclass
class Order:
    """
    Represents a trading order with state tracking.

    This class tracks the complete lifecycle of an order from creation
    through execution and completion.
    """

    # Identity
    order_id: str = field(default_factory=lambda: str(uuid4()))
    broker_order_id: Optional[str] = None

    # Order details
    symbol: str = ""
    order_type: OrderType = OrderType.LIMIT
    side: OrderSide = OrderSide.BUY
    quantity: int = 0

    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Option details (for options orders)
    underlying_symbol: Optional[str] = None  # e.g., "DIS" for underlying
    option_type: Optional[str] = None  # 'CALL' or 'PUT'
    strike: Optional[float] = None
    expiration: Optional[date] = None

    # Duration
    duration: OrderDuration = OrderDuration.DAY

    # State tracking
    status: OrderStatus = OrderStatus.PENDING
    status_message: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution details
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fills: list[OrderFill] = field(default_factory=list)

    # Strategy context
    strategy_name: Optional[str] = None
    position_id: Optional[str] = None
    notes: Optional[str] = None

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None

    def __post_init__(self):
        """Validate order after initialization."""
        if not self.symbol:
            raise ValueError("Symbol is required")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop price required for STOP orders")

    @property
    def is_option(self) -> bool:
        """Check if this is an options order."""
        return self.option_type is not None and self.strike is not None and self.expiration is not None

    @property
    def occ_symbol(self) -> Optional[str]:
        """
        Build the OCC option symbol.

        Format: SYMBOL + YYMMDD + C/P + 00000000 (strike * 1000, 8 digits)
        Example: DIS   260306C00109000 for DIS $109 Call expiring 2026-03-06

        Returns:
            OCC symbol string or None if not an option
        """
        if not self.is_option:
            return None

        # Use underlying_symbol if available, otherwise use symbol
        underlying = self.underlying_symbol or self.symbol

        # Pad symbol to 6 characters (OCC standard)
        padded_symbol = underlying.upper().ljust(6)

        # Format expiration as YYMMDD
        exp_str = self.expiration.strftime('%y%m%d')

        # Option type: C for Call, P for Put
        opt_type = 'C' if self.option_type.upper() == 'CALL' else 'P'

        # Strike price: multiply by 1000, pad to 8 digits
        strike_int = int(self.strike * 1000)
        strike_str = str(strike_int).zfill(8)

        return f"{padded_symbol}{exp_str}{opt_type}{strike_str}"

    @property
    def is_terminal_state(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED
        }

    @property
    def is_active(self) -> bool:
        """Check if order is actively working in the market."""
        return self.status in {
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED
        }

    @property
    def remaining_quantity(self) -> int:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Get percentage of order filled."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100.0

    def transition_to(
        self,
        new_status: OrderStatus,
        message: Optional[str] = None
    ) -> bool:
        """
        Transition order to a new status.

        Args:
            new_status: Target status
            message: Optional status message

        Returns:
            True if transition was valid and completed
        """
        if not self._is_valid_transition(self.status, new_status):
            logger.error(
                f"Invalid state transition: {self.status} -> {new_status} "
                f"for order {self.order_id}"
            )
            return False

        old_status = self.status
        self.status = new_status
        self.status_message = message

        # Update timestamps
        now = datetime.now()
        if new_status == OrderStatus.VALIDATED:
            self.validated_at = now
        elif new_status == OrderStatus.SUBMITTED:
            self.submitted_at = now
        elif new_status == OrderStatus.ACCEPTED:
            self.accepted_at = now
        elif new_status in {OrderStatus.FILLED, OrderStatus.CANCELLED,
                           OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED}:
            self.completed_at = now

        logger.info(
            f"Order {self.order_id} transitioned: {old_status.value} -> {new_status.value}"
        )

        return True

    def add_fill(
        self,
        quantity: int,
        price: float,
        commission: float = 0.0,
        fees: float = 0.0
    ) -> None:
        """
        Add a fill to the order.

        Args:
            quantity: Number of shares/contracts filled
            price: Fill price
            commission: Commission paid
            fees: Other fees paid
        """
        fill = OrderFill(
            fill_id=str(uuid4()),
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            commission=commission,
            fees=fees
        )

        self.fills.append(fill)
        self.filled_quantity += quantity

        # Recalculate average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.avg_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0

        # Update status based on fill
        if self.filled_quantity >= self.quantity:
            self.transition_to(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.transition_to(OrderStatus.PARTIALLY_FILLED)

        logger.info(
            f"Order {self.order_id} filled: {quantity} @ ${price:.2f} "
            f"(total: {self.filled_quantity}/{self.quantity})"
        )

    def record_error(self, error_message: str) -> None:
        """
        Record an error for this order.

        Args:
            error_message: Description of the error
        """
        self.error_count += 1
        self.last_error = error_message
        logger.error(f"Order {self.order_id} error #{self.error_count}: {error_message}")

    def _is_valid_transition(
        self,
        from_status: OrderStatus,
        to_status: OrderStatus
    ) -> bool:
        """
        Check if a state transition is valid.

        Args:
            from_status: Current status
            to_status: Target status

        Returns:
            True if transition is allowed
        """
        # Define valid state transitions
        valid_transitions = {
            OrderStatus.PENDING: {
                OrderStatus.VALIDATING,
                OrderStatus.REJECTED,
                OrderStatus.FAILED
            },
            OrderStatus.VALIDATING: {
                OrderStatus.VALIDATED,
                OrderStatus.REJECTED,
                OrderStatus.FAILED
            },
            OrderStatus.VALIDATED: {
                OrderStatus.SUBMITTING,
                OrderStatus.CANCELLED,
                OrderStatus.FAILED
            },
            OrderStatus.SUBMITTING: {
                OrderStatus.SUBMITTED,
                OrderStatus.REJECTED,
                OrderStatus.FAILED
            },
            OrderStatus.SUBMITTED: {
                OrderStatus.ACCEPTED,
                OrderStatus.WORKING,
                OrderStatus.REJECTED,
                OrderStatus.CANCELLING,
                OrderStatus.FAILED
            },
            OrderStatus.ACCEPTED: {
                OrderStatus.WORKING,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLING,
                OrderStatus.EXPIRED,
                OrderStatus.FAILED
            },
            OrderStatus.WORKING: {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLING,
                OrderStatus.EXPIRED,
                OrderStatus.FAILED
            },
            OrderStatus.PARTIALLY_FILLED: {
                OrderStatus.FILLED,
                OrderStatus.CANCELLING,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
                OrderStatus.FAILED
            },
            OrderStatus.CANCELLING: {
                OrderStatus.CANCELLED,
                OrderStatus.FILLED,  # Can fill while cancel is pending
                OrderStatus.FAILED
            },
            # Terminal states (no transitions out)
            OrderStatus.FILLED: set(),
            OrderStatus.CANCELLED: set(),
            OrderStatus.REJECTED: set(),
            OrderStatus.EXPIRED: set(),
            OrderStatus.FAILED: set()
        }

        allowed_transitions = valid_transitions.get(from_status, set())
        return to_status in allowed_transitions

    def to_dict(self) -> dict:
        """Convert order to dictionary for storage."""
        return {
            'order_id': self.order_id,
            'broker_order_id': self.broker_order_id,
            'symbol': self.symbol,
            'underlying_symbol': self.underlying_symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': self.quantity,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'option_type': self.option_type,
            'strike': self.strike,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'duration': self.duration.value,
            'status': self.status.value,
            'status_message': self.status_message,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'strategy_name': self.strategy_name,
            'position_id': self.position_id,
            'notes': self.notes,
            'error_count': self.error_count,
            'last_error': self.last_error
        }

    def __repr__(self) -> str:
        """String representation of order."""
        return (
            f"Order({self.order_id[:8]}, {self.symbol}, "
            f"{self.side.value}, {self.filled_quantity}/{self.quantity}, "
            f"${self.limit_price or self.stop_price or 'MKT'}, "
            f"{self.status.value})"
        )
