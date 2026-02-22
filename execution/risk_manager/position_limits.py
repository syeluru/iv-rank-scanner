"""
Position limits and risk management.

This module enforces position-level risk limits including maximum position size,
concentration limits, and portfolio-level constraints.
"""

from dataclasses import dataclass
from typing import Optional
from loguru import logger

from config.settings import settings


@dataclass
class PositionLimits:
    """Position size and risk limits."""

    max_position_size_pct: float = 0.10  # 10% of portfolio
    max_positions: int = 10
    max_sector_concentration_pct: float = 0.30  # 30% in single sector
    max_symbol_concentration_pct: float = 0.15  # 15% in single symbol
    max_leverage: float = 1.0  # No leverage by default


class PositionLimitManager:
    """
    Manages position limits and risk constraints.

    Enforces portfolio-level risk management including:
    - Maximum position size
    - Position count limits
    - Sector concentration
    - Symbol concentration
    - Leverage limits
    """

    def __init__(
        self,
        limits: Optional[PositionLimits] = None
    ):
        """
        Initialize position limit manager.

        Args:
            limits: Position limits (defaults to settings-based limits)
        """
        self.limits = limits or PositionLimits(
            max_position_size_pct=settings.MAX_POSITION_SIZE,
            max_positions=settings.MAX_POSITIONS,
            max_leverage=settings.MAX_PORTFOLIO_LEVERAGE
        )

        logger.info(f"PositionLimitManager initialized: {self.limits}")

    def check_position_size(
        self,
        position_value: float,
        portfolio_value: float
    ) -> tuple[bool, str]:
        """
        Check if position size is within limits.

        Args:
            position_value: Value of the position
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (is_valid, message)
        """
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"

        position_pct = position_value / portfolio_value

        if position_pct > self.limits.max_position_size_pct:
            return False, (
                f"Position size {position_pct:.1%} exceeds limit "
                f"{self.limits.max_position_size_pct:.1%}"
            )

        return True, f"Position size OK: {position_pct:.1%}"

    def check_position_count(
        self,
        current_positions: int
    ) -> tuple[bool, str]:
        """
        Check if adding another position would exceed limit.

        Args:
            current_positions: Number of current open positions

        Returns:
            Tuple of (is_valid, message)
        """
        if current_positions >= self.limits.max_positions:
            return False, (
                f"Already at max positions: "
                f"{current_positions}/{self.limits.max_positions}"
            )

        return True, f"Position count OK: {current_positions}/{self.limits.max_positions}"

    def check_symbol_concentration(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        existing_positions: list[dict]
    ) -> tuple[bool, str]:
        """
        Check if adding position would create excessive symbol concentration.

        Args:
            symbol: Symbol to check
            position_value: Value of new position
            portfolio_value: Total portfolio value
            existing_positions: List of existing positions

        Returns:
            Tuple of (is_valid, message)
        """
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"

        # Calculate current exposure to this symbol
        current_exposure = sum(
            p.get('current_value', 0)
            for p in existing_positions
            if p.get('symbol') == symbol and p.get('status') == 'OPEN'
        )

        # Add new position value
        total_exposure = current_exposure + position_value
        exposure_pct = total_exposure / portfolio_value

        if exposure_pct > self.limits.max_symbol_concentration_pct:
            return False, (
                f"Symbol concentration {exposure_pct:.1%} exceeds limit "
                f"{self.limits.max_symbol_concentration_pct:.1%} for {symbol}"
            )

        return True, f"Symbol concentration OK: {exposure_pct:.1%} for {symbol}"

    def check_leverage(
        self,
        total_position_value: float,
        portfolio_value: float
    ) -> tuple[bool, str]:
        """
        Check if portfolio leverage is within limits.

        Args:
            total_position_value: Sum of all position values
            portfolio_value: Account equity

        Returns:
            Tuple of (is_valid, message)
        """
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"

        leverage = total_position_value / portfolio_value

        if leverage > self.limits.max_leverage:
            return False, (
                f"Leverage {leverage:.2f}x exceeds limit "
                f"{self.limits.max_leverage:.2f}x"
            )

        return True, f"Leverage OK: {leverage:.2f}x"

    def calculate_max_position_value(
        self,
        portfolio_value: float,
        symbol: str,
        existing_positions: list[dict]
    ) -> float:
        """
        Calculate maximum allowed position value.

        Considers both position size limit and symbol concentration limit.

        Args:
            portfolio_value: Total portfolio value
            symbol: Symbol for the position
            existing_positions: List of existing positions

        Returns:
            Maximum position value allowed
        """
        if portfolio_value <= 0:
            return 0.0

        # Max based on position size limit
        max_by_position_size = portfolio_value * self.limits.max_position_size_pct

        # Max based on symbol concentration
        current_exposure = sum(
            p.get('current_value', 0)
            for p in existing_positions
            if p.get('symbol') == symbol and p.get('status') == 'OPEN'
        )

        max_symbol_exposure = portfolio_value * self.limits.max_symbol_concentration_pct
        max_by_concentration = max_symbol_exposure - current_exposure

        # Return the smaller of the two
        max_allowed = min(max_by_position_size, max_by_concentration)

        return max(0.0, max_allowed)  # Can't be negative

    def get_portfolio_summary(
        self,
        positions: list[dict],
        portfolio_value: float
    ) -> dict:
        """
        Get portfolio risk metrics summary.

        Args:
            positions: List of all positions
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with risk metrics
        """
        open_positions = [p for p in positions if p.get('status') == 'OPEN']

        # Calculate total position value
        total_position_value = sum(p.get('current_value', 0) for p in open_positions)

        # Calculate leverage
        leverage = total_position_value / portfolio_value if portfolio_value > 0 else 0

        # Find largest position
        largest_position = max(
            open_positions,
            key=lambda p: p.get('current_value', 0),
            default=None
        )

        largest_position_pct = 0
        if largest_position and portfolio_value > 0:
            largest_position_pct = largest_position.get('current_value', 0) / portfolio_value

        # Count positions by symbol
        symbol_counts = {}
        for position in open_positions:
            symbol = position.get('symbol')
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            'total_positions': len(open_positions),
            'max_positions': self.limits.max_positions,
            'positions_available': max(0, self.limits.max_positions - len(open_positions)),
            'total_position_value': total_position_value,
            'portfolio_value': portfolio_value,
            'leverage': leverage,
            'max_leverage': self.limits.max_leverage,
            'largest_position_pct': largest_position_pct,
            'max_position_size_pct': self.limits.max_position_size_pct,
            'unique_symbols': len(symbol_counts),
            'symbols_with_multiple_positions': sum(1 for count in symbol_counts.values() if count > 1)
        }

    def validate_new_position(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        existing_positions: list[dict]
    ) -> tuple[bool, list[str]]:
        """
        Validate if a new position can be added.

        Runs all limit checks and returns combined result.

        Args:
            symbol: Symbol for new position
            position_value: Value of new position
            portfolio_value: Total portfolio value
            existing_positions: List of existing positions

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check position count
        open_positions = [p for p in existing_positions if p.get('status') == 'OPEN']
        valid, msg = self.check_position_count(len(open_positions))
        if not valid:
            errors.append(msg)

        # Check position size
        valid, msg = self.check_position_size(position_value, portfolio_value)
        if not valid:
            errors.append(msg)

        # Check symbol concentration
        valid, msg = self.check_symbol_concentration(
            symbol, position_value, portfolio_value, existing_positions
        )
        if not valid:
            errors.append(msg)

        # Check leverage
        total_value = sum(p.get('current_value', 0) for p in open_positions)
        total_value += position_value
        valid, msg = self.check_leverage(total_value, portfolio_value)
        if not valid:
            errors.append(msg)

        if errors:
            logger.warning(
                f"Position validation failed for {symbol}: {'; '.join(errors)}"
            )
            return False, errors

        logger.info(f"Position validation passed for {symbol}")
        return True, []


# Global position limit manager instance
position_limit_manager = PositionLimitManager()
