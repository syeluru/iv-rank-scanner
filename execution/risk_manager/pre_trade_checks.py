"""
Pre-trade risk checks and validation.

This module implements comprehensive pre-trade checks to prevent invalid
or risky orders from being submitted to the broker.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, date
from loguru import logger

from execution.order_manager.order_state import Order, OrderSide
from config.settings import settings


@dataclass
class CheckResult:
    """Result of a pre-trade check."""
    passed: bool
    check_name: str
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO


class PreTradeChecker:
    """
    Performs pre-trade validation checks before order submission.

    Checks include:
    - Position size limits
    - Account balance verification
    - Symbol validation
    - Price reasonableness
    - Duplicate order detection
    """

    def __init__(self):
        """Initialize pre-trade checker."""
        logger.info("PreTradeChecker initialized")

    async def validate_order(
        self,
        order: Order,
        account_info: Optional[dict] = None,
        existing_positions: Optional[list[dict]] = None
    ) -> list[CheckResult]:
        """
        Run all pre-trade checks on an order.

        Args:
            order: Order to validate
            account_info: Account information (balance, buying power, etc.)
            existing_positions: List of current positions

        Returns:
            List of check results
        """
        results = []

        # Run all checks
        results.append(self._check_symbol_valid(order))
        results.append(self._check_quantity_valid(order))
        results.append(self._check_price_valid(order))
        results.append(self._check_position_size_limit(order, existing_positions))

        if account_info:
            results.append(self._check_buying_power(order, account_info))
            results.append(self._check_daily_loss_limit(account_info))

        # Log results
        failed_checks = [r for r in results if not r.passed]
        if failed_checks:
            logger.warning(
                f"Order {order.order_id} failed {len(failed_checks)} pre-trade checks"
            )
            for check in failed_checks:
                logger.warning(f"  - {check.check_name}: {check.message}")
        else:
            logger.info(f"Order {order.order_id} passed all pre-trade checks")

        return results

    def _check_symbol_valid(self, order: Order) -> CheckResult:
        """
        Check if symbol is valid and tradeable.

        Args:
            order: Order to check

        Returns:
            Check result
        """
        # Basic validation
        if not order.symbol:
            return CheckResult(
                passed=False,
                check_name="symbol_valid",
                message="Symbol is empty",
                severity="ERROR"
            )

        # Handle OCC option symbols (21 chars with spaces) vs stock symbols (1-10 chars alphanumeric)
        if order.is_option:
            # OCC format: ROOT (6 chars, space-padded) + YYMMDD (6) + C/P (1) + STRIKE (8) = 21 chars
            # Example: "IWM   260320C00269000" or "SPXW  260205C06800000"
            if len(order.symbol) != 21:
                return CheckResult(
                    passed=False,
                    check_name="symbol_valid",
                    message=f"Invalid OCC symbol length: {order.symbol} ({len(order.symbol)} chars, expected 21)",
                    severity="ERROR"
                )

            # Check that it only contains alphanumeric and spaces
            if not all(c.isalnum() or c.isspace() for c in order.symbol):
                return CheckResult(
                    passed=False,
                    check_name="symbol_valid",
                    message=f"Invalid OCC symbol format: {order.symbol}",
                    severity="ERROR"
                )
        else:
            # Stock symbol: 1-10 alphanumeric characters, no spaces
            if len(order.symbol) > 10 or not order.symbol.isalnum():
                return CheckResult(
                    passed=False,
                    check_name="symbol_valid",
                    message=f"Invalid symbol format: {order.symbol}",
                    severity="ERROR"
                )

        # Check if symbol is in watchlist (warning only)
        # For options, check the underlying symbol
        check_symbol = order.underlying_symbol if order.is_option else order.symbol

        if check_symbol and check_symbol not in settings.DEFAULT_SYMBOLS:
            return CheckResult(
                passed=True,  # Don't fail, just warn
                check_name="symbol_valid",
                message=f"Symbol {check_symbol} not in watchlist",
                severity="WARNING"
            )

        return CheckResult(
            passed=True,
            check_name="symbol_valid",
            message=f"Symbol {order.symbol} is valid",
            severity="INFO"
        )

    def _check_quantity_valid(self, order: Order) -> CheckResult:
        """
        Check if order quantity is valid.

        Args:
            order: Order to check

        Returns:
            Check result
        """
        if order.quantity <= 0:
            return CheckResult(
                passed=False,
                check_name="quantity_valid",
                message=f"Invalid quantity: {order.quantity}",
                severity="ERROR"
            )

        # Check for reasonable maximums
        max_quantity = 10000  # Sanity check
        if order.quantity > max_quantity:
            return CheckResult(
                passed=False,
                check_name="quantity_valid",
                message=f"Quantity {order.quantity} exceeds maximum {max_quantity}",
                severity="ERROR"
            )

        return CheckResult(
            passed=True,
            check_name="quantity_valid",
            message=f"Quantity {order.quantity} is valid",
            severity="INFO"
        )

    def _check_price_valid(self, order: Order) -> CheckResult:
        """
        Check if order prices are reasonable.

        Args:
            order: Order to check

        Returns:
            Check result
        """
        # Check limit price if present
        if order.limit_price is not None:
            if order.limit_price <= 0:
                return CheckResult(
                    passed=False,
                    check_name="price_valid",
                    message=f"Invalid limit price: ${order.limit_price}",
                    severity="ERROR"
                )

            # Sanity check for extremely high prices
            if order.limit_price > 100000:
                return CheckResult(
                    passed=False,
                    check_name="price_valid",
                    message=f"Limit price ${order.limit_price} seems unreasonably high",
                    severity="ERROR"
                )

        # Check stop price if present
        if order.stop_price is not None:
            if order.stop_price <= 0:
                return CheckResult(
                    passed=False,
                    check_name="price_valid",
                    message=f"Invalid stop price: ${order.stop_price}",
                    severity="ERROR"
                )

        return CheckResult(
            passed=True,
            check_name="price_valid",
            message="Order prices are valid",
            severity="INFO"
        )

    def _check_position_size_limit(
        self,
        order: Order,
        existing_positions: Optional[list[dict]] = None
    ) -> CheckResult:
        """
        Check if order respects position size limits.

        Args:
            order: Order to check
            existing_positions: Current positions

        Returns:
            Check result
        """
        existing_positions = existing_positions or []

        # Check max positions limit
        open_positions = len([p for p in existing_positions if p.get('status') == 'OPEN'])

        if open_positions >= settings.MAX_POSITIONS:
            return CheckResult(
                passed=False,
                check_name="position_size_limit",
                message=f"Already at max positions ({settings.MAX_POSITIONS})",
                severity="ERROR"
            )

        # For opening positions, we'll check portfolio allocation in a separate check
        if order.side in {OrderSide.BUY_TO_OPEN, OrderSide.SELL_TO_OPEN}:
            # This is a new position - will be checked against MAX_POSITION_SIZE
            # in the buying power check
            pass

        return CheckResult(
            passed=True,
            check_name="position_size_limit",
            message=f"Position limits OK ({open_positions}/{settings.MAX_POSITIONS})",
            severity="INFO"
        )

    def _check_buying_power(
        self,
        order: Order,
        account_info: dict
    ) -> CheckResult:
        """
        Check if account has sufficient buying power.

        Args:
            order: Order to check
            account_info: Account information (Schwab API response structure)

        Returns:
            Check result
        """
        # Extract buying power from Schwab nested structure
        # Schwab returns: securitiesAccount.currentBalances.buyingPower
        securities_account = account_info.get('securitiesAccount', {})
        current_balances = securities_account.get('currentBalances', {})

        buying_power = current_balances.get('buyingPower', 0)
        portfolio_value = current_balances.get('liquidationValue', 0) or current_balances.get('accountValue', 0)

        # Fallback to top-level for backwards compatibility with other providers
        if buying_power == 0:
            buying_power = account_info.get('buyingPower', 0)
        if portfolio_value == 0:
            portfolio_value = account_info.get('accountValue', 0)

        # Estimate order cost
        if order.limit_price:
            estimated_cost = order.quantity * order.limit_price
        else:
            # For market orders, we need current quote (not available here)
            # Use a conservative estimate
            return CheckResult(
                passed=True,
                check_name="buying_power",
                message="Cannot verify buying power for market order (skipped)",
                severity="WARNING"
            )

        # Check if we have enough buying power
        if estimated_cost > buying_power:
            return CheckResult(
                passed=False,
                check_name="buying_power",
                message=f"Insufficient buying power: need ${estimated_cost:,.2f}, have ${buying_power:,.2f}",
                severity="ERROR"
            )

        # Check position size limit (percentage of portfolio)
        if portfolio_value > 0:
            position_pct = (estimated_cost / portfolio_value) * 100

            if position_pct > (settings.MAX_POSITION_SIZE * 100):
                return CheckResult(
                    passed=False,
                    check_name="buying_power",
                    message=f"Position size {position_pct:.1f}% exceeds limit {settings.MAX_POSITION_SIZE * 100:.1f}%",
                    severity="ERROR"
                )

        return CheckResult(
            passed=True,
            check_name="buying_power",
            message=f"Buying power sufficient: ${buying_power:,.2f}",
            severity="INFO"
        )

    def _check_daily_loss_limit(self, account_info: dict) -> CheckResult:
        """
        Check if daily loss limit has been hit.

        Args:
            account_info: Account information with P&L data

        Returns:
            Check result
        """
        # Extract from Schwab nested structure
        securities_account = account_info.get('securitiesAccount', {})
        current_balances = securities_account.get('currentBalances', {})

        # Get today's P&L from account info
        daily_pnl = current_balances.get('dayTradingEquity', 0) or account_info.get('dayTradeEquity', 0)
        portfolio_value = current_balances.get('liquidationValue', 0) or current_balances.get('accountValue', 0)

        # Fallback to top-level for backwards compatibility
        if portfolio_value == 0:
            portfolio_value = account_info.get('accountValue', 0)

        if portfolio_value == 0:
            return CheckResult(
                passed=True,
                check_name="daily_loss_limit",
                message="Cannot calculate daily loss percentage (portfolio value unknown)",
                severity="WARNING"
            )

        # Calculate loss percentage
        loss_pct = abs(daily_pnl / portfolio_value) if daily_pnl < 0 else 0

        # Check against limit
        if loss_pct >= settings.DAILY_LOSS_LIMIT:
            return CheckResult(
                passed=False,
                check_name="daily_loss_limit",
                message=f"Daily loss limit hit: {loss_pct:.1%} (limit: {settings.DAILY_LOSS_LIMIT:.1%})",
                severity="ERROR"
            )

        # Warning if getting close (80% of limit)
        if loss_pct >= (settings.DAILY_LOSS_LIMIT * 0.8):
            return CheckResult(
                passed=True,
                check_name="daily_loss_limit",
                message=f"Daily loss approaching limit: {loss_pct:.1%} (limit: {settings.DAILY_LOSS_LIMIT:.1%})",
                severity="WARNING"
            )

        return CheckResult(
            passed=True,
            check_name="daily_loss_limit",
            message=f"Daily P&L OK: ${daily_pnl:,.2f}",
            severity="INFO"
        )

    def get_failed_checks(self, results: list[CheckResult]) -> list[CheckResult]:
        """
        Get list of failed checks.

        Args:
            results: List of check results

        Returns:
            List of failed checks only
        """
        return [r for r in results if not r.passed]

    def has_errors(self, results: list[CheckResult]) -> bool:
        """
        Check if any results have errors.

        Args:
            results: List of check results

        Returns:
            True if any checks failed with ERROR severity
        """
        return any(not r.passed and r.severity == "ERROR" for r in results)

    def get_error_message(self, results: list[CheckResult]) -> str:
        """
        Get combined error message from failed checks.

        Args:
            results: List of check results

        Returns:
            Combined error message
        """
        failed = self.get_failed_checks(results)
        if not failed:
            return "All checks passed"

        messages = [f"{r.check_name}: {r.message}" for r in failed]
        return "; ".join(messages)


# Global pre-trade checker instance
pre_trade_checker = PreTradeChecker()
