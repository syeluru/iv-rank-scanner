"""
IV Data Provider - Fetches option data and calculates live IV.

This module bridges the gap between Schwab's option chain API
and the IV calculator to provide real-time IV data.
"""

import asyncio
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
from loguru import logger

from strategies.indicators.iv_calculator import iv_calculator, IVCalculator
from execution.broker_api.schwab_client import schwab_client


class IVDataProvider:
    """
    Provides real-time IV data by fetching option chains and calculating IV.

    Uses Schwab API for option data and mibian for IV calculations.
    """

    # Index symbols that need $ prefix for Schwab API
    INDEX_SYMBOLS = {'SPX', 'NDX', 'RUT', 'VIX', 'DJX'}

    def __init__(self, iv_calc: IVCalculator = None):
        """
        Initialize IV Data Provider.

        Args:
            iv_calc: IVCalculator instance (uses global if not provided)
        """
        self.iv_calc = iv_calc or iv_calculator
        self._last_update: Dict[str, datetime] = {}
        self._update_interval_seconds = 60  # Minimum seconds between updates per symbol

        logger.info("IVDataProvider initialized")

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for Schwab API.

        Index symbols require $ prefix.
        """
        clean_symbol = symbol.upper().replace('$', '')
        if clean_symbol in self.INDEX_SYMBOLS:
            return f"${clean_symbol}"
        return clean_symbol

    async def get_live_iv(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get live IV for a symbol.

        Fetches ATM option prices and calculates IV.

        Args:
            symbol: Stock/ETF symbol
            force_refresh: Force refresh even if cached

        Returns:
            Current IV as percentage, or None if unavailable
        """
        # Check cache first
        if not force_refresh:
            cached_iv = self.iv_calc.get_cached_iv(symbol)
            if cached_iv is not None:
                return cached_iv

        # Check rate limiting
        now = datetime.now()
        last_update = self._last_update.get(symbol)
        if last_update and (now - last_update).total_seconds() < self._update_interval_seconds:
            # Return cached or None if we're rate limited
            return self.iv_calc.get_cached_iv(symbol)

        try:
            # Fetch ATM options and calculate IV
            iv = await self._fetch_and_calculate_iv(symbol)

            if iv is not None:
                # Cache the result
                self.iv_calc.set_cached_iv(symbol, iv)

                # Update history for percentile calculations
                self.iv_calc.update_iv_history(symbol, iv)

                self._last_update[symbol] = now
                logger.debug(f"Live IV for {symbol}: {iv:.2f}%")

            return iv

        except Exception as e:
            logger.error(f"Failed to get live IV for {symbol}: {e}")
            return None

    async def _fetch_and_calculate_iv(self, symbol: str) -> Optional[float]:
        """
        Fetch option chain and calculate ATM IV.

        Args:
            symbol: Stock symbol

        Returns:
            Calculated IV or None
        """
        try:
            # Normalize symbol for API
            api_symbol = self._normalize_symbol(symbol)

            # Get quote for current price
            quote_data = await schwab_client.get_quote(api_symbol)

            if not quote_data or api_symbol not in quote_data:
                logger.warning(f"No quote data for {symbol}")
                return None

            quote = quote_data[api_symbol].get('quote', {})
            current_price = quote.get('lastPrice') or quote.get('mark') or quote.get('closePrice')

            if not current_price:
                logger.warning(f"No price data for {symbol}")
                return None

            # Get option chain - focus on near-term ATM options
            # Use 30-45 DTE for stable IV reading
            from_date = date.today() + timedelta(days=25)
            to_date = date.today() + timedelta(days=50)

            chain_data = await schwab_client.get_option_chain(
                symbol=api_symbol,
                contract_type='ALL',
                strike_count=5,  # Get 5 strikes around ATM
                from_date=from_date,
                to_date=to_date
            )

            if not chain_data:
                logger.warning(f"No option chain data for {symbol}")
                return None

            # Find ATM options
            atm_iv = self._calculate_atm_iv_from_chain(chain_data, current_price)

            return atm_iv

        except Exception as e:
            logger.error(f"Error fetching option data for {symbol}: {e}")
            return None

    def _calculate_atm_iv_from_chain(
        self,
        chain_data: Dict,
        current_price: float
    ) -> Optional[float]:
        """
        Calculate ATM IV from option chain data.

        Args:
            chain_data: Option chain response from Schwab
            current_price: Current underlying price

        Returns:
            ATM IV or None
        """
        try:
            call_map = chain_data.get('callExpDateMap', {})
            put_map = chain_data.get('putExpDateMap', {})

            if not call_map and not put_map:
                return None

            best_iv = None
            best_distance = float('inf')

            # Iterate through expirations
            for exp_date, strikes in call_map.items():
                # Parse expiration date
                exp_parts = exp_date.split(':')
                exp_date_str = exp_parts[0] if exp_parts else exp_date
                try:
                    expiration = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    dte = (expiration - date.today()).days
                except:
                    continue

                if dte <= 0:
                    continue

                # Find ATM strike
                for strike_str, options in strikes.items():
                    try:
                        strike = float(strike_str)
                    except:
                        continue

                    distance = abs(strike - current_price)

                    # Only consider strikes within 2% of current price
                    if distance / current_price > 0.02:
                        continue

                    if distance < best_distance:
                        # Get call option data
                        if options and len(options) > 0:
                            call_opt = options[0]
                            call_bid = call_opt.get('bid', 0)
                            call_ask = call_opt.get('ask', 0)
                            call_mid = (call_bid + call_ask) / 2 if call_bid and call_ask else 0

                            # Get corresponding put
                            put_mid = 0
                            if exp_date in put_map and strike_str in put_map[exp_date]:
                                put_opts = put_map[exp_date][strike_str]
                                if put_opts and len(put_opts) > 0:
                                    put_opt = put_opts[0]
                                    put_bid = put_opt.get('bid', 0)
                                    put_ask = put_opt.get('ask', 0)
                                    put_mid = (put_bid + put_ask) / 2 if put_bid and put_ask else 0

                            # Calculate IV from straddle if we have both
                            if call_mid > 0 and put_mid > 0:
                                iv = self.iv_calc.calculate_iv_from_straddle(
                                    current_price, strike, dte, call_mid, put_mid
                                )
                            elif call_mid > 0:
                                iv = self.iv_calc.calculate_iv_from_option(
                                    current_price, strike, dte, call_mid, 'CALL'
                                )
                            elif put_mid > 0:
                                iv = self.iv_calc.calculate_iv_from_option(
                                    current_price, strike, dte, put_mid, 'PUT'
                                )
                            else:
                                continue

                            if iv is not None:
                                best_iv = iv
                                best_distance = distance

            return best_iv

        except Exception as e:
            logger.error(f"Error calculating ATM IV from chain: {e}")
            return None

    async def get_iv_percentile(self, symbol: str) -> Optional[float]:
        """
        Get IV percentile for a symbol.

        First ensures we have current IV, then calculates percentile.

        Args:
            symbol: Stock symbol

        Returns:
            IV percentile (0-100) or None
        """
        # Ensure we have current IV
        current_iv = await self.get_live_iv(symbol)

        if current_iv is None:
            return None

        # Calculate percentile from historical data
        percentile = self.iv_calc.get_iv_percentile(symbol, current_iv)

        return percentile

    async def get_iv_rank(self, symbol: str) -> Optional[float]:
        """
        Get IV rank for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            IV rank (0-100) or None
        """
        current_iv = await self.get_live_iv(symbol)

        if current_iv is None:
            return None

        return self.iv_calc.get_iv_rank(symbol, current_iv)

    async def get_iv_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive IV statistics.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with IV stats or None
        """
        # Ensure current IV is fetched
        await self.get_live_iv(symbol)

        return self.iv_calc.get_iv_stats(symbol)

    async def update_iv_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Update IV for multiple symbols.

        Args:
            symbols: List of symbols to update

        Returns:
            Dict mapping symbol to IV
        """
        results = {}

        for symbol in symbols:
            try:
                iv = await self.get_live_iv(symbol)
                if iv is not None:
                    results[symbol] = iv
            except Exception as e:
                logger.error(f"Failed to update IV for {symbol}: {e}")

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return results

    async def get_vix_level(self) -> Optional[float]:
        """
        Get current VIX level.

        The VIX is itself an IV measure, so we just need the price.

        Returns:
            Current VIX level
        """
        try:
            quote_data = await schwab_client.get_quote('$VIX')

            if not quote_data or '$VIX' not in quote_data:
                return None

            quote = quote_data['$VIX'].get('quote', {})
            vix = quote.get('lastPrice') or quote.get('mark') or quote.get('closePrice')

            return float(vix) if vix else None

        except Exception as e:
            logger.error(f"Failed to get VIX: {e}")
            return None


# Global IV data provider instance
iv_data_provider = IVDataProvider()
