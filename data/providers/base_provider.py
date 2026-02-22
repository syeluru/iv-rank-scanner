"""
Abstract base class for market data providers.

This module defines interfaces for broker-agnostic market data access.
Any broker (Schwab, Interactive Brokers, TD Ameritrade, etc.) can implement
these interfaces to provide data to the signal generation engine.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from dataclasses import dataclass


@dataclass
class Quote:
    """Standardized quote data across all providers."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime

    # Additional fields
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class OptionContract:
    """Standardized option contract data."""
    symbol: str
    underlying_symbol: str
    strike: float
    expiration: date
    option_type: str  # 'CALL' or 'PUT'

    # Pricing
    bid: float
    ask: float
    last: float

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Volume and interest
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    # IV
    implied_volatility: Optional[float] = None

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration."""
        return (self.expiration - date.today()).days


@dataclass
class OptionChain:
    """Standardized option chain data."""
    underlying_symbol: str
    underlying_price: float
    timestamp: datetime

    calls: List[OptionContract]
    puts: List[OptionContract]

    def get_puts_by_expiration(self, expiration: date) -> List[OptionContract]:
        """Get all puts for a specific expiration."""
        return [p for p in self.puts if p.expiration == expiration]

    def get_calls_by_expiration(self, expiration: date) -> List[OptionContract]:
        """Get all calls for a specific expiration."""
        return [c for c in self.calls if c.expiration == expiration]

    def get_puts_by_strike(self, strike: float) -> List[OptionContract]:
        """Get all puts for a specific strike."""
        return [p for p in self.puts if p.strike == strike]

    def get_calls_by_strike(self, strike: float) -> List[OptionContract]:
        """Get all calls for a specific strike."""
        return [c for c in self.calls if c.strike == strike]


@dataclass
class AccountInfo:
    """Standardized account information."""
    account_id: str
    account_value: float
    cash_balance: float
    buying_power: float

    # Optional fields
    day_trading_buying_power: Optional[float] = None
    maintenance_requirement: Optional[float] = None
    positions_value: Optional[float] = None


@dataclass
class Position:
    """Standardized position data."""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    # Optional fields
    instrument_type: Optional[str] = None  # 'EQUITY', 'OPTION', etc.
    asset_type: Optional[str] = None


class MarketDataProvider(ABC):
    """
    Abstract interface for market data providers.

    Any broker implementation must provide these methods to supply
    data to the signal generation engine.
    """

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote object or None if not available
        """
        pass

    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to Quote
        """
        pass

    @abstractmethod
    async def get_option_chain(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        contract_type: Optional[str] = None,
        strike_count: Optional[int] = None
    ) -> Optional[OptionChain]:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol
            from_date: Start date for expirations
            to_date: End date for expirations
            contract_type: 'CALL', 'PUT', or None for both
            strike_count: Number of strikes around ATM

        Returns:
            OptionChain object or None if not available
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.

        Returns:
            AccountInfo object or None if not available
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get current account positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if provider is connected and ready.

        Returns:
            True if connected and authenticated
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.

        Returns:
            Provider name (e.g., 'Schwab', 'IBKR', 'TDAmeritrade')
        """
        pass


class MockMarketDataProvider(MarketDataProvider):
    """
    Mock market data provider for testing and paper trading.

    Provides simulated market data without requiring broker connection.
    Can be used to run signal generation engine offline.
    """

    def __init__(self):
        """Initialize mock provider."""
        self._connected = True

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Return mock quote."""
        # Generate realistic-looking mock data
        base_price = hash(symbol) % 500 + 50  # $50-$550

        return Quote(
            symbol=symbol,
            bid=base_price - 0.05,
            ask=base_price + 0.05,
            last=base_price,
            volume=1000000,
            timestamp=datetime.now(),
            open=base_price * 0.99,
            high=base_price * 1.02,
            low=base_price * 0.98,
            close=base_price
        )

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Return mock quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quote = await self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes

    async def get_option_chain(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        contract_type: Optional[str] = None,
        strike_count: Optional[int] = None
    ) -> Optional[OptionChain]:
        """Return mock option chain."""
        # Get mock stock price
        quote = await self.get_quote(symbol)
        if not quote:
            return None

        stock_price = quote.last

        # Generate mock option contracts
        from datetime import timedelta

        expiration = date.today() + timedelta(days=42)
        strikes = [stock_price * (0.95 + i * 0.02) for i in range(10)]

        puts = []
        calls = []

        for strike in strikes:
            # Mock put
            put_delta = -0.30 if strike < stock_price else -0.15
            puts.append(OptionContract(
                symbol=f"{symbol}_{expiration.strftime('%y%m%d')}P{int(strike)}",
                underlying_symbol=symbol,
                strike=strike,
                expiration=expiration,
                option_type='PUT',
                bid=max(0.1, (strike - stock_price) * 0.5 + 2.0),
                ask=max(0.2, (strike - stock_price) * 0.5 + 2.2),
                last=max(0.15, (strike - stock_price) * 0.5 + 2.1),
                delta=put_delta,
                volume=100,
                open_interest=500,
                implied_volatility=0.30
            ))

            # Mock call
            call_delta = 0.30 if strike > stock_price else 0.70
            calls.append(OptionContract(
                symbol=f"{symbol}_{expiration.strftime('%y%m%d')}C{int(strike)}",
                underlying_symbol=symbol,
                strike=strike,
                expiration=expiration,
                option_type='CALL',
                bid=max(0.1, (stock_price - strike) * 0.5 + 2.0),
                ask=max(0.2, (stock_price - strike) * 0.5 + 2.2),
                last=max(0.15, (stock_price - strike) * 0.5 + 2.1),
                delta=call_delta,
                volume=100,
                open_interest=500,
                implied_volatility=0.30
            ))

        return OptionChain(
            underlying_symbol=symbol,
            underlying_price=stock_price,
            timestamp=datetime.now(),
            calls=calls,
            puts=puts
        )

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Return mock account info."""
        return AccountInfo(
            account_id="MOCK123",
            account_value=100000.0,
            cash_balance=50000.0,
            buying_power=100000.0,
            day_trading_buying_power=250000.0,
            positions_value=50000.0
        )

    async def get_positions(self) -> List[Position]:
        """Return empty positions list."""
        return []

    def is_connected(self) -> bool:
        """Always returns True for mock provider."""
        return self._connected

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Mock Provider"
