"""
Historical IV data collector and IV Rank calculator.

This module handles collection of historical implied volatility data
and calculates true IV Rank and IV Percentile (fixing the placeholder logic).
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger

from execution.broker_api.schwab_client import schwab_client
from data.storage.database import market_data_db
from config.settings import settings


class HistoricalIVCollector:
    """
    Collects historical IV data and calculates IV metrics.

    This fixes the placeholder logic from the original scanner by
    storing actual historical IV values and calculating true IV Rank.
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize historical IV collector.

        Args:
            lookback_days: Number of days to look back for IV Rank calculation (default: 252 = 1 year)
        """
        self.lookback_days = lookback_days
        logger.info(f"HistoricalIVCollector initialized with {lookback_days} day lookback")

    async def get_atm_iv_from_chain(
        self,
        symbol: str,
        dte_target: int = 30
    ) -> Optional[float]:
        """
        Get at-the-money implied volatility from option chain.

        Args:
            symbol: Stock symbol
            dte_target: Target days to expiration (default: 30)

        Returns:
            ATM IV or None if not available
        """
        try:
            # Get option chain for target expiration
            today = date.today()
            from_date = today + timedelta(days=dte_target - 5)
            to_date = today + timedelta(days=dte_target + 5)

            chain_data = await schwab_client.get_option_chain(
                symbol=symbol,
                strike_count=3,  # Just need ATM strikes
                from_date=from_date,
                to_date=to_date,
                include_underlying_quote=True
            )

            # Get underlying price
            underlying_price = chain_data.get('underlyingPrice', 0)
            if not underlying_price:
                logger.warning(f"No underlying price for {symbol}")
                return None

            # Find ATM call and put IVs
            call_ivs = []
            put_ivs = []

            # Process calls
            for exp_date, strikes in chain_data.get('callExpDateMap', {}).items():
                for strike_price, contracts in strikes.items():
                    strike = float(strike_price)
                    # Find ATM strikes (within 2% of underlying)
                    if abs(strike - underlying_price) / underlying_price < 0.02:
                        for contract in contracts:
                            iv = contract.get('volatility')
                            if iv and iv > 0:
                                call_ivs.append(iv)

            # Process puts
            for exp_date, strikes in chain_data.get('putExpDateMap', {}).items():
                for strike_price, contracts in strikes.items():
                    strike = float(strike_price)
                    # Find ATM strikes (within 2% of underlying)
                    if abs(strike - underlying_price) / underlying_price < 0.02:
                        for contract in contracts:
                            iv = contract.get('volatility')
                            if iv and iv > 0:
                                put_ivs.append(iv)

            # Average ATM IVs
            all_ivs = call_ivs + put_ivs
            if not all_ivs:
                logger.warning(f"No ATM IV found for {symbol}")
                return None

            atm_iv = np.mean(all_ivs) / 100.0  # Convert from percentage to decimal
            logger.debug(f"{symbol} ATM IV: {atm_iv:.4f}")

            return atm_iv

        except Exception as e:
            logger.error(f"Failed to get ATM IV for {symbol}: {e}")
            return None

    def calculate_historical_volatility(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> float:
        """
        Calculate historical volatility from price data.

        Args:
            prices: Series of closing prices
            window: Number of days for calculation

        Returns:
            Annualized historical volatility
        """
        if len(prices) < window + 1:
            return np.nan

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate standard deviation and annualize
        hv = log_returns.rolling(window=window).std() * np.sqrt(252)

        return hv.iloc[-1]

    async def collect_iv_data_point(
        self,
        symbol: str,
        target_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Collect IV data for a specific symbol and date.

        Args:
            symbol: Stock symbol
            target_date: Date for data point (defaults to today)

        Returns:
            Dictionary with IV data or None if failed
        """
        target_date = target_date or date.today()

        try:
            # Get current ATM IV
            atm_iv = await self.get_atm_iv_from_chain(symbol)
            if atm_iv is None:
                return None

            # Get historical prices for HV calculation
            price_history = await schwab_client.get_price_history(
                symbol=symbol,
                period_type='month',
                period=3,  # 3 months of data
                frequency_type='daily',
                frequency=1
            )

            # Convert to DataFrame
            candles = price_history.get('candles', [])
            if not candles:
                logger.warning(f"No price history for {symbol}")
                return None

            df = pd.DataFrame(candles)
            df['date'] = pd.to_datetime(df['datetime'], unit='ms')
            df = df.set_index('date')

            # Calculate historical volatility
            hv_20 = self.calculate_historical_volatility(df['close'], window=20)
            hv_30 = self.calculate_historical_volatility(df['close'], window=30)

            current_price = df['close'].iloc[-1]

            return {
                'symbol': symbol,
                'date': target_date,
                'iv_30': atm_iv,
                'hv_20': hv_20,
                'hv_30': hv_30,
                'stock_price': current_price,
                'iv_rank': None,  # Will be calculated after we have history
                'iv_percentile': None
            }

        except Exception as e:
            logger.error(f"Failed to collect IV data for {symbol}: {e}")
            return None

    def calculate_iv_rank(
        self,
        current_iv: float,
        iv_history: pd.Series
    ) -> tuple[float, float]:
        """
        Calculate IV Rank and IV Percentile from historical data.

        This replaces the placeholder logic (lines 258-261 of original scanner).

        Args:
            current_iv: Current implied volatility
            iv_history: Series of historical IV values

        Returns:
            Tuple of (iv_rank, iv_percentile)
        """
        if len(iv_history) < 10:
            logger.warning("Not enough IV history for rank calculation")
            return (None, None)

        # IV Rank: (current - min) / (max - min) * 100
        iv_min = iv_history.min()
        iv_max = iv_history.max()

        if iv_max == iv_min:
            iv_rank = 50.0  # If no range, assume middle
        else:
            iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100.0

        # IV Percentile: percentage of days current IV is above
        iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100.0

        return (iv_rank, iv_percentile)

    def update_iv_metrics(self, symbol: str) -> bool:
        """
        Update IV Rank and Percentile for a symbol using stored history.

        Args:
            symbol: Stock symbol

        Returns:
            True if updated successfully
        """
        try:
            # Get historical IV data from database
            query = f"""
            SELECT date, iv_30
            FROM iv_history
            WHERE symbol = '{symbol}'
            AND date >= CURRENT_DATE - INTERVAL '{self.lookback_days} days'
            ORDER BY date
            """

            iv_history_df = market_data_db.fetch_df(query)

            if len(iv_history_df) < 10:
                logger.warning(f"Not enough IV history for {symbol}")
                return False

            # Get current IV (most recent date)
            current_iv = iv_history_df['iv_30'].iloc[-1]
            current_date = iv_history_df['date'].iloc[-1]

            # Calculate metrics
            iv_rank, iv_percentile = self.calculate_iv_rank(
                current_iv,
                iv_history_df['iv_30']
            )

            # Update database
            update_query = f"""
            UPDATE iv_history
            SET iv_rank = {iv_rank}, iv_percentile = {iv_percentile}
            WHERE symbol = '{symbol}' AND date = '{current_date}'
            """

            market_data_db.execute(update_query)

            logger.debug(
                f"{symbol}: IV Rank = {iv_rank:.1f}, "
                f"IV Percentile = {iv_percentile:.1f}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update IV metrics for {symbol}: {e}")
            return False

    async def collect_and_store(
        self,
        symbols: List[str],
        target_date: Optional[date] = None
    ) -> dict:
        """
        Collect IV data for multiple symbols and store in database.

        Args:
            symbols: List of stock symbols
            target_date: Date for data collection (defaults to today)

        Returns:
            Dictionary with collection statistics
        """
        target_date = target_date or date.today()

        logger.info(f"Collecting IV data for {len(symbols)} symbols...")

        # Collect IV data for all symbols concurrently
        tasks = [
            self.collect_iv_data_point(symbol, target_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        iv_data = [r for r in results if r is not None and not isinstance(r, Exception)]

        if not iv_data:
            logger.error("No IV data collected")
            return {'collected': 0, 'stored': 0, 'updated': 0}

        # Convert to DataFrame and store
        df = pd.DataFrame(iv_data)

        try:
            market_data_db.insert_df(df, 'iv_history', if_exists='append')
            logger.info(f"Stored {len(df)} IV data points")
        except Exception as e:
            logger.error(f"Failed to store IV data: {e}")
            return {'collected': len(df), 'stored': 0, 'updated': 0}

        # Update IV Rank and Percentile for each symbol
        updated_count = 0
        for symbol in df['symbol'].unique():
            if self.update_iv_metrics(symbol):
                updated_count += 1

        stats = {
            'collected': len(df),
            'stored': len(df),
            'updated': updated_count
        }

        logger.info(f"IV collection complete: {stats}")
        return stats


# Convenience function for running standalone
async def main():
    """Run historical IV collection standalone."""
    collector = HistoricalIVCollector()

    # Use default symbols from settings
    symbols = settings.DEFAULT_SYMBOLS[:10]  # Test with first 10

    print(f"\nCollecting IV data for {len(symbols)} symbols...")
    stats = await collector.collect_and_store(symbols)

    print("\n" + "=" * 60)
    print("Historical IV Collection Results")
    print("=" * 60)
    print(f"IV data points collected: {stats['collected']}")
    print(f"Data points stored: {stats['stored']}")
    print(f"IV metrics updated: {stats['updated']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
