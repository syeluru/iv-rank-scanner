"""
Schwab implementation of MarketDataProvider.

Wraps the Schwab API client to conform to the broker-agnostic interface.
"""

from typing import List, Dict, Any, Optional
from datetime import date, datetime
from loguru import logger

from data.providers.base_provider import (
    MarketDataProvider,
    Quote,
    OptionContract,
    OptionChain,
    AccountInfo,
    Position
)
from execution.broker_api.schwab_client import schwab_client


class SchwabMarketDataProvider(MarketDataProvider):
    """
    Schwab implementation of market data provider.

    Adapts Schwab API responses to standardized data structures.
    """

    def __init__(self):
        """Initialize Schwab provider."""
        self.client = schwab_client
        logger.info("SchwabMarketDataProvider initialized")

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get quote from Schwab API.

        Args:
            symbol: Stock symbol

        Returns:
            Quote object or None
        """
        try:
            quotes = await self.client.get_quotes([symbol])

            if not quotes or symbol not in quotes:
                return None

            schwab_quote = quotes[symbol]
            quote_data = schwab_quote.get('quote', {})

            return Quote(
                symbol=symbol,
                bid=quote_data.get('bidPrice', 0.0),
                ask=quote_data.get('askPrice', 0.0),
                last=quote_data.get('lastPrice', 0.0),
                volume=int(quote_data.get('totalVolume', 0)),
                timestamp=datetime.now(),
                open=quote_data.get('openPrice'),
                high=quote_data.get('highPrice'),
                low=quote_data.get('lowPrice'),
                close=quote_data.get('closePrice'),
                bid_size=quote_data.get('bidSize'),
                ask_size=quote_data.get('askSize')
            )

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols from Schwab.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to Quote
        """
        try:
            schwab_quotes = await self.client.get_quotes(symbols)

            quotes = {}
            for symbol, schwab_data in schwab_quotes.items():
                quote_data = schwab_data.get('quote', {})

                quotes[symbol] = Quote(
                    symbol=symbol,
                    bid=quote_data.get('bidPrice', 0.0),
                    ask=quote_data.get('askPrice', 0.0),
                    last=quote_data.get('lastPrice', 0.0),
                    volume=int(quote_data.get('totalVolume', 0)),
                    timestamp=datetime.now(),
                    open=quote_data.get('openPrice'),
                    high=quote_data.get('highPrice'),
                    low=quote_data.get('lowPrice'),
                    close=quote_data.get('closePrice'),
                    bid_size=quote_data.get('bidSize'),
                    ask_size=quote_data.get('askSize')
                )

            return quotes

        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}

    async def get_option_chain(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        contract_type: Optional[str] = None,
        strike_count: Optional[int] = None
    ) -> Optional[OptionChain]:
        """
        Get option chain from Schwab API.

        Args:
            symbol: Underlying symbol
            from_date: Start date for expirations
            to_date: End date for expirations
            contract_type: 'CALL', 'PUT', or None for both
            strike_count: Number of strikes around ATM

        Returns:
            OptionChain object or None
        """
        try:
            chain_data = await self.client.get_option_chain(
                symbol=symbol,
                contract_type=contract_type,
                strike_count=strike_count,
                from_date=from_date,
                to_date=to_date
            )

            if not chain_data:
                return None

            # Get underlying price
            underlying_price = chain_data.get('underlyingPrice', 0.0)

            # Parse puts
            puts = []
            put_map = chain_data.get('putExpDateMap', {})
            for exp_date_str, strikes in put_map.items():
                expiration = datetime.strptime(
                    exp_date_str.split(':')[0],
                    '%Y-%m-%d'
                ).date()

                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)

                    for contract in contracts:
                        puts.append(OptionContract(
                            symbol=contract.get('symbol', ''),
                            underlying_symbol=symbol,
                            strike=strike,
                            expiration=expiration,
                            option_type='PUT',
                            bid=contract.get('bid', 0.0),
                            ask=contract.get('ask', 0.0),
                            last=contract.get('last', 0.0),
                            delta=contract.get('delta'),
                            gamma=contract.get('gamma'),
                            theta=contract.get('theta'),
                            vega=contract.get('vega'),
                            rho=contract.get('rho'),
                            volume=contract.get('totalVolume'),
                            open_interest=contract.get('openInterest'),
                            implied_volatility=contract.get('volatility')
                        ))

            # Parse calls
            calls = []
            call_map = chain_data.get('callExpDateMap', {})
            for exp_date_str, strikes in call_map.items():
                expiration = datetime.strptime(
                    exp_date_str.split(':')[0],
                    '%Y-%m-%d'
                ).date()

                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)

                    for contract in contracts:
                        calls.append(OptionContract(
                            symbol=contract.get('symbol', ''),
                            underlying_symbol=symbol,
                            strike=strike,
                            expiration=expiration,
                            option_type='CALL',
                            bid=contract.get('bid', 0.0),
                            ask=contract.get('ask', 0.0),
                            last=contract.get('last', 0.0),
                            delta=contract.get('delta'),
                            gamma=contract.get('gamma'),
                            theta=contract.get('theta'),
                            vega=contract.get('vega'),
                            rho=contract.get('rho'),
                            volume=contract.get('totalVolume'),
                            open_interest=contract.get('openInterest'),
                            implied_volatility=contract.get('volatility')
                        ))

            return OptionChain(
                underlying_symbol=symbol,
                underlying_price=underlying_price,
                timestamp=datetime.now(),
                calls=calls,
                puts=puts
            )

        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}: {e}")
            return None

    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information from Schwab.

        Returns:
            AccountInfo object or None
        """
        try:
            account_data = await self.client.get_account()

            if not account_data:
                return None

            # Extract account info
            account_id = account_data.get('securitiesAccount', {}).get('accountNumber', '')
            initial_balances = account_data.get('securitiesAccount', {}).get('initialBalances', {})
            current_balances = account_data.get('securitiesAccount', {}).get('currentBalances', {})

            return AccountInfo(
                account_id=account_id,
                account_value=current_balances.get('liquidationValue', 0.0),
                cash_balance=current_balances.get('cashBalance', 0.0),
                buying_power=current_balances.get('buyingPower', 0.0),
                day_trading_buying_power=current_balances.get('dayTradingBuyingPower'),
                maintenance_requirement=current_balances.get('maintenanceRequirement'),
                positions_value=current_balances.get('longMarketValue', 0.0)
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    async def get_positions(self) -> List[Position]:
        """
        Get positions from Schwab account.

        Returns:
            List of Position objects
        """
        try:
            positions_data = await self.client.get_account_positions()

            positions = []
            for pos_data in positions_data:
                instrument = pos_data.get('instrument', {})
                symbol = instrument.get('symbol', '')

                if not symbol:
                    continue

                positions.append(Position(
                    symbol=symbol,
                    quantity=int(pos_data.get('longQuantity', 0)),
                    average_price=float(pos_data.get('averagePrice', 0.0)),
                    current_price=float(pos_data.get('marketValue', 0.0)) / max(1, int(pos_data.get('longQuantity', 1))),
                    market_value=float(pos_data.get('marketValue', 0.0)),
                    unrealized_pnl=float(pos_data.get('currentDayProfitLoss', 0.0)),
                    unrealized_pnl_pct=float(pos_data.get('currentDayProfitLossPercentage', 0.0)),
                    instrument_type=instrument.get('assetType'),
                    asset_type=pos_data.get('assetType')
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def is_connected(self) -> bool:
        """
        Check if Schwab client is authenticated.

        Returns:
            True if authenticated
        """
        try:
            return self.client.is_authenticated()
        except Exception:
            return False

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Schwab"


# Global Schwab provider instance
schwab_provider = SchwabMarketDataProvider()
