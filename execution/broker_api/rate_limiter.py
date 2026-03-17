"""
Token bucket rate limiter for API calls.

This module implements a token bucket algorithm to enforce rate limits
on Schwab API calls (120 calls per minute).
"""

import asyncio
import time
from typing import Optional
from loguru import logger


class RateLimiter:
    """
    Token bucket rate limiter for async API calls.

    The token bucket algorithm allows bursts up to the bucket capacity,
    then enforces a steady rate of token replenishment.
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds (e.g., 60 for per-minute)
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        logger.info(f"RateLimiter initialized: {max_calls} calls per {period}s")

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens before making API call.

        This method blocks until enough tokens are available.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.max_calls,
                    self.tokens + (elapsed * self.max_calls / self.period)
                )
                self.last_refill = now

                # If we have enough tokens, consume them and return
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(
                        f"Acquired {tokens} token(s). "
                        f"Remaining: {self.tokens:.2f}/{self.max_calls}"
                    )
                    return

                # Not enough tokens - calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed * self.period / self.max_calls

                logger.debug(
                    f"Rate limit reached. Waiting {wait_time:.2f}s "
                    f"for {tokens_needed:.2f} token(s)"
                )

                # Release lock while waiting
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()

    def get_available_tokens(self) -> float:
        """
        Get current number of available tokens (non-blocking).

        Returns:
            Number of available tokens
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        available = min(
            self.max_calls,
            self.tokens + (elapsed * self.max_calls / self.period)
        )
        return available

    async def wait_if_needed(self, tokens: int = 1) -> bool:
        """
        Check if we need to wait, and wait if necessary.

        Args:
            tokens: Number of tokens needed

        Returns:
            True if had to wait, False if tokens were immediately available
        """
        available = self.get_available_tokens()
        if available >= tokens:
            await self.acquire(tokens)
            return False
        else:
            await self.acquire(tokens)
            return True


class MultiRateLimiter:
    """
    Manages multiple rate limiters for different API endpoints.

    Some APIs have different rate limits for different endpoint categories.
    """

    def __init__(self):
        """Initialize multi-rate limiter."""
        self.limiters: dict[str, RateLimiter] = {}
        logger.info("MultiRateLimiter initialized")

    def add_limiter(self, name: str, max_calls: int, period: float) -> None:
        """
        Add a named rate limiter.

        Args:
            name: Identifier for this limiter
            max_calls: Maximum calls per period
            period: Period in seconds
        """
        self.limiters[name] = RateLimiter(max_calls, period)
        logger.info(f"Added rate limiter '{name}': {max_calls} calls per {period}s")

    async def acquire(self, limiter_name: str, tokens: int = 1) -> None:
        """
        Acquire tokens from a named limiter.

        Args:
            limiter_name: Name of the limiter to use
            tokens: Number of tokens to acquire

        Raises:
            KeyError: If limiter_name doesn't exist
        """
        if limiter_name not in self.limiters:
            raise KeyError(f"Rate limiter '{limiter_name}' not found")

        await self.limiters[limiter_name].acquire(tokens)

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """
        Get a rate limiter by name.

        Args:
            name: Limiter name

        Returns:
            RateLimiter instance or None if not found
        """
        return self.limiters.get(name)
