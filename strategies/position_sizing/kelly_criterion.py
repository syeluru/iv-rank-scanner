"""
Kelly Criterion position sizing.

The Kelly Criterion calculates optimal position size based on:
- Win rate
- Average win size
- Average loss size

Formula: f* = (p * b - q) / b
Where:
- f* = fraction of capital to risk
- p = probability of win
- q = probability of loss (1 - p)
- b = ratio of win to loss (avg_win / avg_loss)

For conservative trading, we use fractional Kelly (typically 25-50%).
"""

from typing import Optional
from loguru import logger


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.

    Implements full Kelly and fractional Kelly for conservative sizing.
    """

    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize Kelly calculator.

        Args:
            kelly_fraction: Fraction of full Kelly to use (default: 0.25 = 25%)
                          Lower values are more conservative
        """
        self.kelly_fraction = kelly_fraction
        logger.info(f"KellyCriterion initialized with {kelly_fraction:.0%} Kelly")

    def calculate_position_size(
        self,
        portfolio_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_size: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            portfolio_value: Total portfolio value
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (positive number)
            max_position_size: Optional maximum position size limit

        Returns:
            Position size in dollars
        """
        # Input validation
        if not (0 < win_rate < 1):
            logger.warning(f"Invalid win rate: {win_rate}. Using 50%")
            win_rate = 0.50

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid avg win/loss: {avg_win}/{avg_loss}. Using defaults")
            avg_win = 1.0
            avg_loss = 1.0

        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss

        # Calculate full Kelly
        loss_rate = 1 - win_rate
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Apply fractional Kelly
        kelly_pct = full_kelly * self.kelly_fraction

        # Don't risk negative or > 100%
        kelly_pct = max(0.0, min(1.0, kelly_pct))

        # Calculate position size
        position_size = portfolio_value * kelly_pct

        # Apply maximum position size if specified
        if max_position_size is not None:
            position_size = min(position_size, max_position_size)

        logger.debug(
            f"Kelly sizing: WR={win_rate:.1%}, W/L={win_loss_ratio:.2f}, "
            f"Kelly={full_kelly:.1%}, Fractional={kelly_pct:.1%}, "
            f"Size=${position_size:,.0f}"
        )

        return position_size

    def calculate_quantity(
        self,
        portfolio_value: float,
        price_per_unit: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_quantity: Optional[int] = None
    ) -> int:
        """
        Calculate quantity to trade using Kelly Criterion.

        Args:
            portfolio_value: Total portfolio value
            price_per_unit: Price per share/contract
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            max_quantity: Optional maximum quantity

        Returns:
            Number of units to trade
        """
        if price_per_unit <= 0:
            logger.error(f"Invalid price: {price_per_unit}")
            return 0

        # Calculate position size in dollars
        position_size = self.calculate_position_size(
            portfolio_value, win_rate, avg_win, avg_loss
        )

        # Convert to quantity
        quantity = int(position_size / price_per_unit)

        # Apply max quantity if specified
        if max_quantity is not None:
            quantity = min(quantity, max_quantity)

        # Must be at least 1
        quantity = max(1, quantity)

        return quantity

    def estimate_from_confidence(
        self,
        portfolio_value: float,
        price_per_unit: float,
        confidence: float,
        risk_reward_ratio: float = 2.0,
        max_position_pct: float = 0.10
    ) -> int:
        """
        Estimate position size from signal confidence.

        Useful when we don't have enough historical data for
        accurate win rate calculation.

        Args:
            portfolio_value: Total portfolio value
            price_per_unit: Price per share/contract
            confidence: Signal confidence (0.0 to 1.0)
            risk_reward_ratio: Expected reward/risk ratio
            max_position_pct: Maximum position as % of portfolio

        Returns:
            Quantity to trade
        """
        # Use confidence as proxy for win rate
        estimated_win_rate = 0.45 + (confidence * 0.15)  # Range: 45-60%

        # Estimate win/loss from risk/reward ratio
        estimated_avg_win = risk_reward_ratio
        estimated_avg_loss = 1.0

        # Calculate position size
        quantity = self.calculate_quantity(
            portfolio_value=portfolio_value,
            price_per_unit=price_per_unit,
            win_rate=estimated_win_rate,
            avg_win=estimated_avg_win,
            avg_loss=estimated_avg_loss
        )

        # Respect maximum position size
        max_quantity = int((portfolio_value * max_position_pct) / price_per_unit)
        quantity = min(quantity, max_quantity)

        return max(1, quantity)


class FixedFractionalSizer:
    """
    Simple fixed fractional position sizing.

    Always risks a fixed percentage of portfolio regardless of
    win rate or other factors. More conservative than Kelly.
    """

    def __init__(self, risk_per_trade: float = 0.02):
        """
        Initialize fixed fractional sizer.

        Args:
            risk_per_trade: Percentage of portfolio to risk per trade (default: 2%)
        """
        self.risk_per_trade = risk_per_trade
        logger.info(f"FixedFractionalSizer initialized with {risk_per_trade:.1%} risk")

    def calculate_position_size(
        self,
        portfolio_value: float,
        max_position_size: Optional[float] = None
    ) -> float:
        """
        Calculate position size using fixed fraction.

        Args:
            portfolio_value: Total portfolio value
            max_position_size: Optional maximum position size

        Returns:
            Position size in dollars
        """
        position_size = portfolio_value * self.risk_per_trade

        if max_position_size is not None:
            position_size = min(position_size, max_position_size)

        return position_size

    def calculate_quantity(
        self,
        portfolio_value: float,
        price_per_unit: float,
        max_quantity: Optional[int] = None
    ) -> int:
        """
        Calculate quantity using fixed fraction.

        Args:
            portfolio_value: Total portfolio value
            price_per_unit: Price per unit
            max_quantity: Optional maximum quantity

        Returns:
            Quantity to trade
        """
        position_size = self.calculate_position_size(portfolio_value)
        quantity = int(position_size / price_per_unit)

        if max_quantity is not None:
            quantity = min(quantity, max_quantity)

        return max(1, quantity)


# Global instances for convenience
kelly_sizer = KellyCriterion(kelly_fraction=0.25)  # Conservative 25% Kelly
fixed_sizer = FixedFractionalSizer(risk_per_trade=0.02)  # Fixed 2% risk
