"""
Safety checks for live trading.

Pre-trade validations to prevent dangerous entries:
- VIX range checks
- Time window validation
- Position limit enforcement
- Maximum loss checks
- Market condition filters
"""

from datetime import datetime, time
from typing import Optional, Dict, Any, Tuple

from loguru import logger

from strategies.base_strategy import Signal
from config.settings import settings


class SafetyChecks:
    """
    Pre-trade safety validation system.

    Performs comprehensive checks before allowing trade entry:
    1. VIX range (15-25 sweet spot)
    2. Time windows (no trading too early/late)
    3. Position limits (max contracts per day)
    4. Market gap limits (no huge overnight gaps)
    5. Maximum loss limits
    """

    # Hard limits (cannot be overridden)
    MIN_VIX = 12.0
    MAX_VIX = 35.0
    MAX_GAP_PERCENT = 1.0  # 1% max gap
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EARLIEST_ENTRY = time(10, 0)  # Don't trade first 30 min
    LATEST_ENTRY = time(14, 0)    # Don't trade after 2 PM

    def __init__(
        self,
        max_contracts_per_day: int = 25,
        max_loss_per_day: float = 5000.0,
        preferred_vix_min: float = 15.0,
        preferred_vix_max: float = 25.0
    ):
        """
        Initialize safety checks.

        Args:
            max_contracts_per_day: Maximum total contracts per day
            max_loss_per_day: Maximum loss allowed per day ($)
            preferred_vix_min: Preferred minimum VIX level
            preferred_vix_max: Preferred maximum VIX level
        """
        self.max_contracts_per_day = max_contracts_per_day
        self.max_loss_per_day = max_loss_per_day
        self.preferred_vix_min = preferred_vix_min
        self.preferred_vix_max = preferred_vix_max

        # Daily tracking
        self.contracts_today = 0
        self.pnl_today = 0.0
        self.trades_today = 0
        self.last_reset_date = datetime.now().date()

        logger.info(
            f"SafetyChecks initialized: "
            f"VIX range [{preferred_vix_min}-{preferred_vix_max}], "
            f"max {max_contracts_per_day} contracts/day, "
            f"max ${max_loss_per_day:,.0f} loss/day"
        )

    def check_all(
        self,
        signal: Signal,
        vix_level: Optional[float] = None,
        current_time: Optional[datetime] = None,
        spx_gap: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Run all safety checks.

        Args:
            signal: Signal to validate
            vix_level: Current VIX level (if available)
            current_time: Current timestamp (default: now)
            spx_gap: SPX gap from previous close (%)

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        if current_time is None:
            current_time = datetime.now()

        # Reset daily tracking if new day
        self._check_and_reset_daily(current_time)

        # Run individual checks
        checks = [
            self._check_time_window(current_time),
            self._check_vix_range(vix_level),
            self._check_position_limits(signal.quantity),
            self._check_loss_limits(),
            self._check_gap_size(spx_gap),
        ]

        # Find first failure
        for passed, reason in checks:
            if not passed:
                logger.warning(f"Safety check FAILED: {reason}")
                return False, reason

        logger.info("✓ All safety checks passed")
        return True, "All checks passed"

    def _check_time_window(self, current_time: datetime) -> Tuple[bool, str]:
        """Validate trading time window."""
        current = current_time.time()

        # Before market open
        if current < self.MARKET_OPEN:
            return False, f"Before market open ({self.MARKET_OPEN.strftime('%H:%M')})"

        # After market close
        if current >= self.MARKET_CLOSE:
            return False, f"After market close ({self.MARKET_CLOSE.strftime('%H:%M')})"

        # Too early (first 30 min volatility)
        if current < self.EARLIEST_ENTRY:
            return False, f"Too early - wait until {self.EARLIEST_ENTRY.strftime('%H:%M')}"

        # Too late (avoid end-of-day risk)
        if current >= self.LATEST_ENTRY:
            return False, f"Too late - no entries after {self.LATEST_ENTRY.strftime('%H:%M')}"

        return True, "Time window OK"

    def _check_vix_range(self, vix_level: Optional[float]) -> Tuple[bool, str]:
        """Validate VIX is in acceptable range."""
        if vix_level is None:
            logger.warning("VIX level not provided - skipping VIX check")
            return True, "VIX check skipped (no data)"

        # Hard limits
        if vix_level < self.MIN_VIX:
            return False, f"VIX too low: {vix_level:.1f} < {self.MIN_VIX:.1f} (complacency risk)"

        if vix_level > self.MAX_VIX:
            return False, f"VIX too high: {vix_level:.1f} > {self.MAX_VIX:.1f} (volatility risk)"

        # Preferred range (warning only)
        if vix_level < self.preferred_vix_min:
            logger.warning(
                f"VIX below preferred range: {vix_level:.1f} < {self.preferred_vix_min:.1f}"
            )
        elif vix_level > self.preferred_vix_max:
            logger.warning(
                f"VIX above preferred range: {vix_level:.1f} > {self.preferred_vix_max:.1f}"
            )

        return True, f"VIX OK ({vix_level:.1f})"

    def _check_position_limits(self, contracts: int) -> Tuple[bool, str]:
        """Validate position size limits."""
        new_total = self.contracts_today + contracts

        if new_total > self.max_contracts_per_day:
            return False, (
                f"Would exceed daily limit: "
                f"{new_total} > {self.max_contracts_per_day} contracts"
            )

        return True, f"Position limit OK ({new_total}/{self.max_contracts_per_day})"

    def _check_loss_limits(self) -> Tuple[bool, str]:
        """Validate we haven't exceeded daily loss limit."""
        if self.pnl_today < -self.max_loss_per_day:
            return False, (
                f"Daily loss limit exceeded: "
                f"${self.pnl_today:,.0f} < -${self.max_loss_per_day:,.0f}"
            )

        return True, f"Loss limit OK (${self.pnl_today:,.0f})"

    def _check_gap_size(self, gap_pct: Optional[float]) -> Tuple[bool, str]:
        """Validate overnight gap is not too large."""
        if gap_pct is None:
            logger.warning("Gap size not provided - skipping gap check")
            return True, "Gap check skipped (no data)"

        abs_gap = abs(gap_pct)

        if abs_gap > self.MAX_GAP_PERCENT:
            return False, (
                f"Gap too large: {gap_pct:+.2f}% > {self.MAX_GAP_PERCENT:.2f}% "
                f"(directional risk)"
            )

        return True, f"Gap OK ({gap_pct:+.2f}%)"

    def record_entry(self, contracts: int):
        """Record a new entry."""
        self.contracts_today += contracts
        self.trades_today += 1
        logger.info(
            f"Entry recorded: {contracts} contracts "
            f"(total today: {self.contracts_today}/{self.max_contracts_per_day})"
        )

    def update_pnl(self, pnl_delta: float):
        """Update daily P&L."""
        self.pnl_today += pnl_delta
        logger.info(f"P&L updated: ${pnl_delta:+,.2f} (total today: ${self.pnl_today:,.2f})")

    def _check_and_reset_daily(self, current_time: datetime):
        """Reset daily counters if new trading day."""
        current_date = current_time.date()

        if current_date > self.last_reset_date:
            logger.info(
                f"New trading day - resetting counters "
                f"(previous: {self.trades_today} trades, "
                f"{self.contracts_today} contracts, "
                f"${self.pnl_today:,.2f} P&L)"
            )

            self.contracts_today = 0
            self.pnl_today = 0.0
            self.trades_today = 0
            self.last_reset_date = current_date

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's trading."""
        return {
            'date': self.last_reset_date.isoformat(),
            'trades': self.trades_today,
            'contracts': self.contracts_today,
            'max_contracts': self.max_contracts_per_day,
            'pnl': self.pnl_today,
            'max_loss': -self.max_loss_per_day,
            'can_trade': (
                self.contracts_today < self.max_contracts_per_day and
                self.pnl_today > -self.max_loss_per_day
            )
        }

    def can_trade_now(
        self,
        vix_level: Optional[float] = None,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Quick check if trading is allowed right now.

        Args:
            vix_level: Current VIX level
            current_time: Current timestamp

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if current_time is None:
            current_time = datetime.now()

        # Check time window
        passed, reason = self._check_time_window(current_time)
        if not passed:
            return False, reason

        # Check VIX
        if vix_level is not None:
            passed, reason = self._check_vix_range(vix_level)
            if not passed:
                return False, reason

        # Check daily limits
        if self.contracts_today >= self.max_contracts_per_day:
            return False, "Daily contract limit reached"

        if self.pnl_today < -self.max_loss_per_day:
            return False, "Daily loss limit reached"

        return True, "Trading allowed"
