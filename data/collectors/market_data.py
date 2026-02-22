"""
Market data collector for quotes and option chains.

This module handles async collection of real-time market data from Schwab API
and stores it in DuckDB.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, List
import pandas as pd
from loguru import logger

from execution.broker_api.schwab_client import schwab_client
from data.storage.database import market_data_db
from config.settings import settings


class MarketDataCollector:
    """
    Async market data collector for quotes and option chains.

    Collects real-time quotes and option chain data for multiple symbols
    concurrently and stores them in DuckDB.
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize market data collector.

        Args:
            symbols: List of symbols to collect (defaults to settings)
        """
        self.symbols = symbols or settings.DEFAULT_SYMBOLS
        logger.info(f"MarketDataCollector initialized with {len(self.symbols)} symbols")

    async def collect_quote(self, symbol: str) -> Optional[dict]:
        """
        Collect real-time quote for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data or None if failed
        """
        try:
            quote_data = await schwab_client.get_quote(symbol)

            # Extract relevant fields from Schwab response
            quote = quote_data.get(symbol, {}).get('quote', {})

            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'open': quote.get('openPrice'),
                'high': quote.get('highPrice'),
                'low': quote.get('lowPrice'),
                'close': quote.get('lastPrice'),
                'volume': quote.get('totalVolume')
            }

        except Exception as e:
            logger.error(f"Failed to collect quote for {symbol}: {e}")
            return None

    async def collect_quotes(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect real-time quotes for multiple symbols concurrently.

        Args:
            symbols: List of symbols (defaults to instance symbols)

        Returns:
            DataFrame with quote data
        """
        symbols = symbols or self.symbols

        logger.info(f"Collecting quotes for {len(symbols)} symbols...")

        # Collect all quotes concurrently
        tasks = [self.collect_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures and convert to DataFrame
        quotes = [r for r in results if r is not None and not isinstance(r, Exception)]

        if not quotes:
            logger.error("No quotes collected successfully")
            return pd.DataFrame()

        df = pd.DataFrame(quotes)
        logger.info(f"Collected {len(df)} quotes successfully")

        return df

    async def collect_option_chain(
        self,
        symbol: str,
        dte_min: int = 20,
        dte_max: int = 60,
        strike_count: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        Collect option chain for a symbol.

        Args:
            symbol: Underlying symbol
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            strike_count: Number of strikes around ATM

        Returns:
            DataFrame with option chain data or None if failed
        """
        try:
            # Calculate date range
            today = date.today()
            from_date = today + timedelta(days=dte_min)
            to_date = today + timedelta(days=dte_max)

            # Fetch option chain
            chain_data = await schwab_client.get_option_chain(
                symbol=symbol,
                strike_count=strike_count,
                from_date=from_date,
                to_date=to_date
            )

            # Parse option chain data
            options = []
            timestamp = datetime.now()

            # Process call options
            for exp_date, strikes in chain_data.get('callExpDateMap', {}).items():
                for strike_price, contracts in strikes.items():
                    for contract in contracts:
                        options.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'expiration': datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d').date(),
                            'strike': float(strike_price),
                            'option_type': 'CALL',
                            'bid': contract.get('bid'),
                            'ask': contract.get('ask'),
                            'last': contract.get('last'),
                            'volume': contract.get('totalVolume'),
                            'open_interest': contract.get('openInterest'),
                            'implied_volatility': contract.get('volatility'),
                            'delta': contract.get('delta'),
                            'gamma': contract.get('gamma'),
                            'theta': contract.get('theta'),
                            'vega': contract.get('vega'),
                            'rho': contract.get('rho')
                        })

            # Process put options
            for exp_date, strikes in chain_data.get('putExpDateMap', {}).items():
                for strike_price, contracts in strikes.items():
                    for contract in contracts:
                        options.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'expiration': datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d').date(),
                            'strike': float(strike_price),
                            'option_type': 'PUT',
                            'bid': contract.get('bid'),
                            'ask': contract.get('ask'),
                            'last': contract.get('last'),
                            'volume': contract.get('totalVolume'),
                            'open_interest': contract.get('openInterest'),
                            'implied_volatility': contract.get('volatility'),
                            'delta': contract.get('delta'),
                            'gamma': contract.get('gamma'),
                            'theta': contract.get('theta'),
                            'vega': contract.get('vega'),
                            'rho': contract.get('rho')
                        })

            if not options:
                logger.warning(f"No options found for {symbol}")
                return None

            df = pd.DataFrame(options)
            logger.debug(f"Collected {len(df)} options for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Failed to collect option chain for {symbol}: {e}")
            return None

    async def collect_option_chains(
        self,
        symbols: Optional[List[str]] = None,
        dte_min: int = 20,
        dte_max: int = 60
    ) -> pd.DataFrame:
        """
        Collect option chains for multiple symbols concurrently.

        Args:
            symbols: List of symbols (defaults to instance symbols)
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration

        Returns:
            DataFrame with all option chain data
        """
        symbols = symbols or self.symbols

        logger.info(f"Collecting option chains for {len(symbols)} symbols...")

        # Collect all option chains concurrently
        tasks = [
            self.collect_option_chain(symbol, dte_min, dte_max)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures and concatenate
        chains = [r for r in results if r is not None and not isinstance(r, Exception)]

        if not chains:
            logger.error("No option chains collected successfully")
            return pd.DataFrame()

        df = pd.concat(chains, ignore_index=True)
        logger.info(f"Collected {len(df)} total options across {len(chains)} symbols")

        return df

    def store_quotes(self, quotes_df: pd.DataFrame) -> None:
        """
        Store quotes in database.

        Args:
            quotes_df: DataFrame with quote data
        """
        if quotes_df.empty:
            logger.warning("No quotes to store")
            return

        try:
            market_data_db.insert_df(quotes_df, 'quotes', if_exists='append')
            logger.info(f"Stored {len(quotes_df)} quotes in database")
        except Exception as e:
            logger.error(f"Failed to store quotes: {e}")
            raise

    def store_option_chains(self, chains_df: pd.DataFrame) -> None:
        """
        Store option chains in database.

        Args:
            chains_df: DataFrame with option chain data
        """
        if chains_df.empty:
            logger.warning("No option chains to store")
            return

        try:
            market_data_db.insert_df(chains_df, 'option_chains', if_exists='append')
            logger.info(f"Stored {len(chains_df)} options in database")
        except Exception as e:
            logger.error(f"Failed to store option chains: {e}")
            raise

    async def collect_and_store_all(self) -> dict:
        """
        Collect all market data and store it.

        Returns:
            Dictionary with collection statistics
        """
        logger.info("Starting full market data collection...")

        stats = {
            'quotes_collected': 0,
            'options_collected': 0,
            'errors': []
        }

        try:
            # Collect quotes
            quotes_df = await self.collect_quotes()
            if not quotes_df.empty:
                self.store_quotes(quotes_df)
                stats['quotes_collected'] = len(quotes_df)

            # Collect option chains
            chains_df = await self.collect_option_chains()
            if not chains_df.empty:
                self.store_option_chains(chains_df)
                stats['options_collected'] = len(chains_df)

            logger.info(f"Collection complete: {stats}")

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            stats['errors'].append(str(e))

        return stats


# Convenience function for running standalone
async def main():
    """Run market data collection standalone."""
    collector = MarketDataCollector()
    stats = await collector.collect_and_store_all()

    print("\n" + "=" * 60)
    print("Market Data Collection Results")
    print("=" * 60)
    print(f"Quotes collected: {stats['quotes_collected']}")
    print(f"Options collected: {stats['options_collected']}")
    print(f"Errors: {len(stats['errors'])}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
