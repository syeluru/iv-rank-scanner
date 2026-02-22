"""
Implied Volatility Calculator using Black-Scholes model.

This module provides real-time IV calculations from option prices
using the mibian library for Black-Scholes calculations.

IV Percentile = (Current IV rank in historical data) / (Total observations) * 100
IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
"""

import mibian
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date, timedelta
from collections import deque
from loguru import logger
import numpy as np


class IVCalculator:
    """
    Calculates Implied Volatility from option prices using Black-Scholes.

    Maintains historical IV data for percentile calculations.
    """

    # Default risk-free rate (approximate 1-year Treasury yield)
    DEFAULT_RISK_FREE_RATE = 4.5  # 4.5% as of 2024

    # Historical lookback for IV percentile
    IV_HISTORY_DAYS = 252  # ~1 year of trading days

    def __init__(self, risk_free_rate: float = None):
        """
        Initialize IV Calculator.

        Args:
            risk_free_rate: Annual risk-free rate in percent (e.g., 4.5 for 4.5%)
        """
        self.risk_free_rate = risk_free_rate or self.DEFAULT_RISK_FREE_RATE

        # Historical IV storage: symbol -> deque of (timestamp, iv) tuples
        self._iv_history: Dict[str, deque] = {}

        # Cache for recent IV calculations to avoid redundant API calls
        self._iv_cache: Dict[str, Tuple[datetime, float]] = {}
        self._cache_ttl_seconds = 60  # Cache IV for 60 seconds

        logger.info(f"IVCalculator initialized with risk-free rate: {self.risk_free_rate}%")

    def calculate_iv_from_option(
        self,
        underlying_price: float,
        strike: float,
        days_to_expiration: int,
        option_price: float,
        option_type: str = 'CALL'
    ) -> Optional[float]:
        """
        Calculate implied volatility from option price using Black-Scholes.

        Args:
            underlying_price: Current price of underlying asset
            strike: Option strike price
            days_to_expiration: Days until expiration
            option_price: Current market price of the option
            option_type: 'CALL' or 'PUT'

        Returns:
            Implied volatility as a percentage (e.g., 25.5 for 25.5%), or None if calculation fails
        """
        if days_to_expiration <= 0:
            logger.warning("Cannot calculate IV for expired option")
            return None

        if option_price <= 0:
            logger.warning("Option price must be positive")
            return None

        if underlying_price <= 0 or strike <= 0:
            logger.warning("Prices must be positive")
            return None

        try:
            # mibian.BS takes [underlying, strike, rate, days]
            # Then you pass callPrice or putPrice to get implied volatility

            if option_type.upper() == 'CALL':
                bs = mibian.BS(
                    [underlying_price, strike, self.risk_free_rate, days_to_expiration],
                    callPrice=option_price
                )
            else:
                bs = mibian.BS(
                    [underlying_price, strike, self.risk_free_rate, days_to_expiration],
                    putPrice=option_price
                )

            iv = bs.impliedVolatility

            # Sanity check - IV should be between 1% and 500%
            if iv is None or iv < 1 or iv > 500:
                logger.debug(f"IV calculation returned unusual value: {iv}")
                return None

            return float(iv)

        except Exception as e:
            logger.debug(f"IV calculation failed: {e}")
            return None

    def calculate_iv_from_straddle(
        self,
        underlying_price: float,
        strike: float,
        days_to_expiration: int,
        call_price: float,
        put_price: float
    ) -> Optional[float]:
        """
        Calculate IV from ATM straddle prices (more accurate for ATM).

        Uses the average of call and put IVs.

        Args:
            underlying_price: Current underlying price
            strike: ATM strike price
            days_to_expiration: Days to expiration
            call_price: ATM call price
            put_price: ATM put price

        Returns:
            Average implied volatility
        """
        call_iv = self.calculate_iv_from_option(
            underlying_price, strike, days_to_expiration, call_price, 'CALL'
        )
        put_iv = self.calculate_iv_from_option(
            underlying_price, strike, days_to_expiration, put_price, 'PUT'
        )

        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        elif call_iv:
            return call_iv
        elif put_iv:
            return put_iv
        else:
            return None

    def update_iv_history(self, symbol: str, iv: float, timestamp: datetime = None):
        """
        Add IV reading to historical data for percentile calculations.

        Args:
            symbol: Stock/ETF symbol
            iv: Implied volatility value
            timestamp: When the IV was calculated (default: now)
        """
        if symbol not in self._iv_history:
            self._iv_history[symbol] = deque(maxlen=self.IV_HISTORY_DAYS * 78)  # ~78 readings per day

        ts = timestamp or datetime.now()
        self._iv_history[symbol].append((ts, iv))

        logger.debug(f"Updated IV history for {symbol}: {iv:.2f}% (total: {len(self._iv_history[symbol])} readings)")

    def get_iv_percentile(self, symbol: str, current_iv: float = None) -> Optional[float]:
        """
        Calculate IV percentile (0-100) based on historical data.

        IV Percentile = % of past readings that are BELOW current IV

        Args:
            symbol: Stock symbol
            current_iv: Current IV (if None, uses latest from history)

        Returns:
            IV percentile (0-100) or None if insufficient data
        """
        if symbol not in self._iv_history or len(self._iv_history[symbol]) < 20:
            logger.debug(f"Insufficient IV history for {symbol}")
            return None

        history = self._iv_history[symbol]

        # Use current IV or get the latest from history
        if current_iv is None:
            current_iv = history[-1][1]

        # Extract just the IV values
        iv_values = [iv for _, iv in history]

        # Calculate percentile: % of values below current
        below_count = sum(1 for iv in iv_values if iv < current_iv)
        percentile = (below_count / len(iv_values)) * 100

        return percentile

    def get_iv_rank(self, symbol: str, current_iv: float = None) -> Optional[float]:
        """
        Calculate IV Rank (0-100) based on 52-week high/low.

        IV Rank = (Current IV - 52wk Low) / (52wk High - 52wk Low) * 100

        Args:
            symbol: Stock symbol
            current_iv: Current IV (if None, uses latest from history)

        Returns:
            IV Rank (0-100) or None if insufficient data
        """
        if symbol not in self._iv_history or len(self._iv_history[symbol]) < 20:
            return None

        history = self._iv_history[symbol]

        if current_iv is None:
            current_iv = history[-1][1]

        iv_values = [iv for _, iv in history]

        iv_high = max(iv_values)
        iv_low = min(iv_values)

        if iv_high == iv_low:
            return 50.0  # No range, return middle

        iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100

        return max(0, min(100, iv_rank))  # Clamp to 0-100

    def get_iv_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive IV statistics for a symbol.

        Returns:
            Dict with current_iv, percentile, rank, high, low, avg
        """
        if symbol not in self._iv_history or len(self._iv_history[symbol]) < 5:
            return None

        history = self._iv_history[symbol]
        iv_values = [iv for _, iv in history]

        current_iv = iv_values[-1]

        return {
            'current_iv': current_iv,
            'iv_percentile': self.get_iv_percentile(symbol, current_iv),
            'iv_rank': self.get_iv_rank(symbol, current_iv),
            'iv_high': max(iv_values),
            'iv_low': min(iv_values),
            'iv_avg': sum(iv_values) / len(iv_values),
            'iv_readings': len(iv_values)
        }

    def get_cached_iv(self, symbol: str) -> Optional[float]:
        """
        Get cached IV if still valid.

        Returns:
            Cached IV or None if expired/not found
        """
        if symbol not in self._iv_cache:
            return None

        cached_time, cached_iv = self._iv_cache[symbol]

        if (datetime.now() - cached_time).total_seconds() < self._cache_ttl_seconds:
            return cached_iv

        return None

    def set_cached_iv(self, symbol: str, iv: float):
        """
        Cache an IV calculation.

        Args:
            symbol: Stock symbol
            iv: Calculated IV
        """
        self._iv_cache[symbol] = (datetime.now(), iv)

    def calculate_expected_move(
        self,
        underlying_price: float,
        iv: float,
        days: int
    ) -> float:
        """
        Calculate expected price move based on IV.

        Expected Move = Price × IV × sqrt(Days/365)

        Args:
            underlying_price: Current stock price
            iv: Implied volatility (as percentage, e.g., 25 for 25%)
            days: Number of days

        Returns:
            Expected move in dollars
        """
        iv_decimal = iv / 100
        return underlying_price * iv_decimal * np.sqrt(days / 365)

    def calculate_straddle_expected_move(
        self,
        call_price: float,
        put_price: float
    ) -> float:
        """
        Calculate expected move from ATM straddle price.

        Market expects price to move roughly the straddle price.

        Args:
            call_price: ATM call price
            put_price: ATM put price

        Returns:
            Expected move in dollars
        """
        return call_price + put_price

    def clear_history(self, symbol: str = None):
        """
        Clear IV history for a symbol or all symbols.

        Args:
            symbol: Symbol to clear, or None for all
        """
        if symbol:
            if symbol in self._iv_history:
                del self._iv_history[symbol]
                logger.info(f"Cleared IV history for {symbol}")
        else:
            self._iv_history.clear()
            logger.info("Cleared all IV history")


# Global IV calculator instance
iv_calculator = IVCalculator()
