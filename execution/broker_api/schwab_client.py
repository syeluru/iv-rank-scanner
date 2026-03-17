"""
Schwab API client wrapper with async support and rate limiting.

This module provides a high-level interface to the Schwab API with
automatic rate limiting, error handling, and retries.
"""

import asyncio
from typing import Optional, Any
from datetime import datetime, date
from loguru import logger

try:
    import schwab
    from schwab.client import Client as SchwabAPIClient
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    SchwabAPIClient = None
    logger.warning("schwab-py not installed")

from execution.broker_api.auth_manager import auth_manager
from execution.broker_api.rate_limiter import RateLimiter
from config.settings import settings


class SchwabClient:
    """
    Async wrapper around Schwab API client with rate limiting.

    Provides methods for fetching market data, placing orders,
    and managing positions.
    """

    def __init__(self):
        """Initialize Schwab client."""
        if not SCHWAB_AVAILABLE:
            raise ImportError("schwab-py is required. Install with: pip install schwab-py")

        self._client: Optional[SchwabAPIClient] = None
        self.rate_limiter = RateLimiter(
            max_calls=settings.SCHWAB_RATE_LIMIT_CALLS,
            period=settings.SCHWAB_RATE_LIMIT_PERIOD
        )
        self.account_id = settings.SCHWAB_ACCOUNT_ID
        self._account_hash_cache: Optional[str] = None

        logger.info("SchwabClient initialized")

    def _get_client(self) -> SchwabAPIClient:
        """
        Get or create Schwab client instance.

        Returns:
            Authenticated Schwab client
        """
        if self._client is None:
            self._client = auth_manager.get_client()
            logger.info("Schwab client connection established")
        return self._client

    def re_authenticate(self) -> bool:
        """
        Run interactive OAuth flow to get fresh tokens when refresh token expires.

        Returns:
            True if re-authentication succeeded
        """
        logger.warning("Running interactive re-authentication (browser will open)...")
        try:
            client = auth_manager.authenticate_interactive()
            if client:
                self._client = client
                self._account_hash_cache = None
                logger.info("Re-authentication successful — client refreshed")
                return True
        except Exception as e:
            logger.error(f"Re-authentication failed: {e}")
        return False

    async def _rate_limited_call(self, func: callable, *args, **kwargs) -> Any:
        """
        Execute function with rate limiting.

        Args:
            func: Function to call (can be a callable builder object)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        await self.rate_limiter.acquire()

        # Run synchronous Schwab API call in executor
        loop = asyncio.get_event_loop()

        # run_in_executor doesn't support kwargs, so use functools.partial
        if kwargs:
            from functools import partial
            func = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, func)
        elif args:
            from functools import partial
            func = partial(func, *args)
            return await loop.run_in_executor(None, func)
        else:
            # No args or kwargs, call directly
            return await loop.run_in_executor(None, func)

    # ===== Market Data Methods =====

    async def get_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data dictionary
        """
        client = self._get_client()
        response = await self._rate_limited_call(
            client.get_quote,
            symbol
        )

        if response.status_code != 200:
            logger.error(f"Failed to get quote for {symbol}: {response.status_code}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got quote for {symbol}")
        return data

    async def get_quotes(self, symbols: list[str]) -> dict:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to quote data
        """
        client = self._get_client()
        response = await self._rate_limited_call(
            client.get_quotes,
            symbols
        )

        if response.status_code != 200:
            logger.error(f"Failed to get quotes: {response.status_code}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got quotes for {len(symbols)} symbols")
        return data

    async def get_option_chain(
        self,
        symbol: str,
        contract_type: Optional[str] = None,
        strike_count: Optional[int] = None,
        include_underlying_quote: bool = True,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        strike_range: Optional[str] = None
    ) -> dict:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol
            contract_type: 'CALL', 'PUT', or None for both
            strike_count: Number of strikes to return
            include_underlying_quote: Include underlying quote
            from_date: Start date for expirations
            to_date: End date for expirations

        Returns:
            Option chain data
        """
        client = self._get_client()

        # Build kwargs for the API call using correct parameter names
        kwargs = {}

        if contract_type:
            # Import the enum type
            from schwab.client import Client
            if contract_type.upper() == 'CALL':
                kwargs['contract_type'] = Client.Options.ContractType.CALL
            elif contract_type.upper() == 'PUT':
                kwargs['contract_type'] = Client.Options.ContractType.PUT
            else:
                kwargs['contract_type'] = Client.Options.ContractType.ALL

        if strike_count is not None:
            kwargs['strike_count'] = strike_count

        if include_underlying_quote:
            kwargs['include_underlying_quote'] = True  # Correct parameter name

        if from_date is not None:
            kwargs['from_date'] = from_date

        if to_date is not None:
            kwargs['to_date'] = to_date

        if strike_range is not None:
            from schwab.client import Client
            strike_range_map = {
                'ITM': Client.Options.StrikeRange.IN_THE_MONEY,
                'NTM': Client.Options.StrikeRange.NEAR_THE_MONEY,
                'OTM': Client.Options.StrikeRange.OUT_OF_THE_MONEY,
                'SAM': Client.Options.StrikeRange.STRIKES_ABOVE_MARKET,
                'SBM': Client.Options.StrikeRange.STRIKES_BELOW_MARKET,
                'SNM': Client.Options.StrikeRange.STRIKES_NEAR_MARKET,
                'ALL': Client.Options.StrikeRange.ALL,
            }
            if strike_range.upper() in strike_range_map:
                kwargs['strike_range'] = strike_range_map[strike_range.upper()]

        # Call the API with all parameters
        response = await self._rate_limited_call(
            client.get_option_chain,
            symbol,
            **kwargs
        )

        if response.status_code != 200:
            logger.error(f"Failed to get option chain for {symbol}: {response.status_code}")
            logger.error(f"Response: {response.text if response.text else 'No response body'}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got option chain for {symbol}")
        return data

    async def get_price_history(
        self,
        symbol: str,
        period_type: str = 'year',
        period: int = 1,
        frequency_type: str = 'daily',
        frequency: int = 1
    ) -> dict:
        """
        Get historical price data.

        Args:
            symbol: Stock symbol
            period_type: 'day', 'month', 'year', 'ytd'
            period: Number of periods
            frequency_type: 'minute', 'daily', 'weekly', 'monthly'
            frequency: Frequency count

        Returns:
            Historical price data
        """
        client = self._get_client()

        # Import enums
        from schwab.client import Client

        # Build kwargs with enum values
        kwargs = {
            'period_type': getattr(Client.PriceHistory.PeriodType, period_type.upper()),
            'period': period,
            'frequency_type': getattr(Client.PriceHistory.FrequencyType, frequency_type.upper()),
            'frequency': frequency
        }

        # Call the API with all parameters
        response = await self._rate_limited_call(
            client.get_price_history,
            symbol,
            **kwargs
        )

        if response.status_code != 200:
            logger.error(f"Failed to get price history for {symbol}: {response.status_code}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got price history for {symbol}")
        return data

    # ===== Account Methods =====

    async def _get_account_hash(self) -> str:
        """
        Get the encrypted account hash for API calls.

        This retrieves the hash value from Schwab's account list endpoint
        and caches it for subsequent calls.

        Returns:
            Account hash string

        Raises:
            ValueError: If account cannot be found or API error
        """
        if self._account_hash_cache:
            return self._account_hash_cache

        client = self._get_client()

        # Get account numbers and hashes
        response = await self._rate_limited_call(client.get_account_numbers)

        if response.status_code != 200:
            logger.error(f"Failed to get account numbers: {response.status_code}")
            raise ValueError(f"API error: {response.status_code}")

        accounts = response.json()

        if not accounts or not isinstance(accounts, list):
            raise ValueError("No accounts found")

        # Find matching account by account number
        if self.account_id:
            for account in accounts:
                if account.get('accountNumber') == self.account_id:
                    hash_value = account.get('hashValue')
                    if not hash_value:
                        raise ValueError(f"No hash value for account {self.account_id}")
                    self._account_hash_cache = hash_value
                    logger.info(f"Found account hash for {self.account_id}")
                    return hash_value

            # Account ID not found in list
            logger.warning(f"Account {self.account_id} not found in account list")
            logger.info(f"Available accounts: {[a.get('accountNumber') for a in accounts]}")

        # Use first account if no match or no account_id configured
        first_account = accounts[0]
        hash_value = first_account.get('hashValue')

        if not hash_value:
            raise ValueError("No hash value in account data")

        self._account_hash_cache = hash_value
        logger.info(f"Using first available account: {first_account.get('accountNumber')}")
        return hash_value

    async def get_account(self, account_id: Optional[str] = None, include_positions: bool = True) -> dict:
        """
        Get account information.

        Args:
            account_id: Account hash or account number (defaults to configured account)
                       If account number is provided, it will be converted to hash.
            include_positions: Whether to include positions in the response (default True)

        Returns:
            Account data
        """
        client = self._get_client()

        # If account_id provided, use it directly (assume it's already a hash)
        # Otherwise, get the hash for the configured account
        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        # Build kwargs for the API call
        kwargs = {}
        if include_positions:
            # Use the proper enum from the client class
            kwargs['fields'] = [client.Account.Fields.POSITIONS]

        response = await self._rate_limited_call(
            client.get_account,
            acc_hash,
            **kwargs
        )

        if response.status_code != 200:
            logger.error(f"Failed to get account: {response.status_code}")
            logger.error(f"Response: {response.text if response.text else 'No response body'}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got account data")
        return data

    async def get_account_positions(self, account_id: Optional[str] = None) -> list[dict]:
        """
        Get current positions for account.

        Args:
            account_id: Account ID (defaults to configured account)

        Returns:
            List of position dictionaries
        """
        account_data = await self.get_account(account_id)

        # Extract positions from account data
        positions = account_data.get('securitiesAccount', {}).get('positions', [])
        logger.debug(f"Found {len(positions)} positions")
        return positions

    async def get_buying_power(self, account_id: Optional[str] = None) -> float:
        """
        Get available buying power for options trading.

        Prefers availableFundsNonMarginableTrade (options can't use margin leverage),
        falls back to general buyingPower if unavailable.

        Args:
            account_id: Account ID (defaults to configured account)

        Returns:
            Buying power as float
        """
        account_data = await self.get_account(account_id)

        balances = (
            account_data
            .get('securitiesAccount', {})
            .get('currentBalances', {})
        )

        # Options require non-marginable funds — equity buyingPower includes
        # margin leverage (2x/4x) and will overestimate what's available.
        option_bp = balances.get('availableFundsNonMarginableTrade', 0.0)
        equity_bp = balances.get('buyingPower', 0.0)
        buying_power = option_bp if option_bp > 0 else equity_bp

        logger.info(
            f"Buying power: option_bp=${option_bp:,.2f}, "
            f"equity_bp=${equity_bp:,.2f}, using=${buying_power:,.2f}"
        )
        return float(buying_power)

    # ===== Order Methods =====

    async def place_order(
        self,
        order: dict,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Place an order.

        Args:
            order: Order specification dictionary
            account_id: Account hash (defaults to configured account)

        Returns:
            Order response
        """
        if settings.PAPER_TRADING:
            logger.warning("Paper trading mode - order not actually placed")
            return {
                'status': 'PAPER_TRADING',
                'order': order,
                'message': 'Paper trading mode - order simulation only'
            }

        client = self._get_client()

        # Get account hash if not provided
        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        response = await self._rate_limited_call(
            client.place_order,
            acc_hash,
            order
        )

        if response.status_code not in [200, 201]:
            logger.error(f"Failed to place order: {response.status_code}")
            logger.error(f"Response: {response.text if response.text else 'No response body'}")
            raise ValueError(f"API error: {response.status_code}")

        logger.info(f"Order placed successfully: {order.get('orderType')}")

        # Extract order ID from Location header (Schwab returns it there, not in body)
        # Location header format: https://api.schwabapi.com/trader/v1/accounts/{hash}/orders/{orderId}
        result = response.json() if response.text else {}

        location_header = response.headers.get('Location', '')
        if location_header:
            # Extract order ID from the end of the Location URL
            order_id = location_header.rstrip('/').split('/')[-1]
            if order_id:
                result['orderId'] = order_id
                logger.info(f"Extracted order ID from Location header: {order_id}")

        # Fallback: check if orderId is in the response body
        if 'orderId' not in result:
            logger.warning("No order ID found in Location header or response body")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response body: {response.text[:500] if response.text else 'empty'}")

        return result if result else {'status': 'success'}

    async def get_order(
        self,
        order_id: str,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Get order status.

        Args:
            order_id: Order ID
            account_id: Account hash (defaults to configured account)

        Returns:
            Order data
        """
        client = self._get_client()

        # Get account hash if not provided
        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        response = await self._rate_limited_call(
            client.get_order,
            order_id,
            acc_hash
        )

        if response.status_code != 200:
            logger.error(f"Failed to get order {order_id}: {response.status_code}")
            raise ValueError(f"API error: {response.status_code}")

        data = response.json()
        logger.debug(f"Got order data for {order_id}")
        return data

    async def get_orders_for_account(
        self,
        account_id: Optional[str] = None,
        status: Optional[str] = None,
        from_entered_datetime: Optional[datetime] = None,
        to_entered_datetime: Optional[datetime] = None,
        max_results: Optional[int] = None
    ) -> list[dict]:
        """
        Get orders for account.

        Args:
            account_id: Account hash (defaults to configured account)
            status: Filter by status (e.g., 'FILLED', 'WORKING', 'PENDING_ACTIVATION')
            from_entered_datetime: Start of date range
            to_entered_datetime: End of date range
            max_results: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        client = self._get_client()

        # Get account hash if not provided
        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        # Build kwargs
        kwargs = {}
        if status:
            kwargs['status'] = client.Order.Status[status]
        if from_entered_datetime:
            kwargs['from_entered_datetime'] = from_entered_datetime
        if to_entered_datetime:
            kwargs['to_entered_datetime'] = to_entered_datetime
        if max_results:
            kwargs['max_results'] = max_results

        response = await self._rate_limited_call(
            client.get_orders_for_account,
            acc_hash,
            **kwargs
        )

        if response.status_code != 200:
            logger.error(f"Failed to get orders: {response.status_code}")
            logger.error(f"Response: {response.text if response.text else 'No response body'}")
            return []

        data = response.json()
        logger.debug(f"Got {len(data)} orders")
        return data

    async def get_transactions(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_types: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> list[dict]:
        """
        Get transactions for account.

        Args:
            start_date: Only transactions after this date (defaults to 60 days prior)
            end_date: Only transactions before this date (defaults to now)
            transaction_types: Filter by type (e.g. 'TRADE', 'DIVIDEND_OR_INTEREST')
            account_id: Account hash (defaults to configured account)

        Returns:
            List of transaction dictionaries
        """
        client = self._get_client()

        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        kwargs = {}
        if start_date:
            kwargs['start_date'] = start_date
        if end_date:
            kwargs['end_date'] = end_date
        if transaction_types:
            kwargs['transaction_types'] = transaction_types

        response = await self._rate_limited_call(
            client.get_transactions,
            acc_hash,
            **kwargs
        )

        if response.status_code != 200:
            logger.error(f"Failed to get transactions: {response.status_code}")
            logger.error(f"Response: {response.text if response.text else 'No response body'}")
            return []

        data = response.json()
        logger.info(f"Got {len(data)} transactions")
        return data

    async def cancel_order(
        self,
        order_id: str,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            account_id: Account hash (defaults to configured account)

        Returns:
            True if cancelled successfully
        """
        if settings.PAPER_TRADING:
            logger.warning("Paper trading mode - cancel simulation")
            return True

        client = self._get_client()

        # Get account hash if not provided
        if account_id:
            acc_hash = account_id
        else:
            acc_hash = await self._get_account_hash()

        response = await self._rate_limited_call(
            client.cancel_order,
            order_id,
            acc_hash
        )

        success = response.status_code in [200, 204]
        if success:
            logger.info(f"Order {order_id} cancelled successfully")
        else:
            logger.error(f"Failed to cancel order {order_id}: {response.status_code}")

        return success


# Global Schwab client instance
schwab_client = SchwabClient()
