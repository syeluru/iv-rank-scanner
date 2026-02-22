"""
Option Contract Selector for Algo Trading Strategies.

Automatically selects appropriate option contracts based on directional signals
from algo trading strategies (ORB, VWAP, RSI, etc.).

Selection Criteria:
- Days to Expiration (DTE): 30-45 days (configurable)
- Target Delta: ~0.30 for entries (configurable)
- Liquidity: Minimum open interest and volume
- Strike Selection: Based on delta and current price
"""

from typing import Optional, Dict, Any, List
from datetime import date, timedelta
from loguru import logger

from strategies.base_strategy import SignalDirection
from data.providers.base_provider import MarketDataProvider
from execution.broker_api.schwab_client import schwab_client


class OptionSelector:
    """
    Selects optimal option contracts based on trading signals.

    This class bridges the gap between stock-based algo signals and
    options execution by automatically selecting appropriate strikes
    and expirations.
    """

    def __init__(
        self,
        min_dte: int = 30,
        max_dte: int = 45,
        target_delta: float = 0.30,
        min_open_interest: int = 100,
        min_volume: int = 10
    ):
        """
        Initialize option selector.

        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            target_delta: Target delta for option selection (0.30 = 30 delta)
            min_open_interest: Minimum open interest for liquidity
            min_volume: Minimum daily volume for liquidity
        """
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.target_delta = target_delta
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume

        logger.info(
            f"OptionSelector initialized: DTE={min_dte}-{max_dte}, "
            f"target_delta={target_delta}, min_OI={min_open_interest}"
        )

    async def select_option(
        self,
        provider: MarketDataProvider,
        symbol: str,
        direction: SignalDirection,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Select best option contract for the signal using Schwab API.

        Args:
            provider: Market data provider (unused, kept for compatibility)
            symbol: Underlying symbol
            direction: LONG (calls) or SHORT (puts)
            current_price: Current stock price

        Returns:
            Dictionary with option details or None if no suitable option found
            {
                'option_symbol': str,
                'option_type': str,  # 'CALL' or 'PUT'
                'strike': float,
                'expiration': date,
                'delta': float,
                'bid': float,
                'ask': float,
                'mid': float,
                'open_interest': int,
                'volume': int
            }
        """
        try:
            # Determine option type based on direction
            option_type = 'CALL' if direction == SignalDirection.LONG else 'PUT'

            # Calculate date range for option chain
            today = date.today()
            from_date = today + timedelta(days=self.min_dte)
            to_date = today + timedelta(days=self.max_dte)

            logger.debug(
                f"Fetching {option_type} chain for {symbol} from Schwab: "
                f"DTE {self.min_dte}-{self.max_dte}"
            )

            # Get option chain from Schwab
            chain_data = await schwab_client.get_option_chain(
                symbol=symbol,
                contract_type=option_type,
                strike_count=20,  # Get strikes around current price
                include_underlying_quote=True,
                from_date=from_date,
                to_date=to_date
            )

            if not chain_data:
                logger.warning(f"No option chain data for {symbol}")
                return None

            # Parse Schwab response
            underlying_price = chain_data.get('underlyingPrice', current_price)

            # Get the appropriate expiration map
            exp_map = chain_data.get('callExpDateMap' if option_type == 'CALL' else 'putExpDateMap', {})

            if not exp_map:
                logger.warning(f"No {option_type} expirations found for {symbol}")
                return None

            # Find best option across all valid expirations
            best_option = None
            best_delta_diff = float('inf')

            for exp_key, strikes in exp_map.items():
                # Parse expiration: format is "YYYY-MM-DD:DTE"
                parts = exp_key.split(':')
                if len(parts) != 2:
                    continue

                exp_date_str, dte_str = parts
                dte = int(dte_str)

                # Skip if outside DTE range
                if not (self.min_dte <= dte <= self.max_dte):
                    continue

                # Parse expiration date
                from datetime import datetime
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()

                # Check each strike
                for strike_str, contracts in strikes.items():
                    strike = float(strike_str)

                    # Get first contract (there's usually only one per strike)
                    contract = contracts[0] if isinstance(contracts, list) else list(contracts.values())[0]

                    # Check liquidity
                    open_interest = contract.get('openInterest', 0)
                    volume = contract.get('totalVolume', 0)

                    if open_interest < self.min_open_interest or volume < self.min_volume:
                        continue

                    # Check for valid quotes
                    bid = contract.get('bid', 0)
                    ask = contract.get('ask', 0)

                    if not bid or not ask:
                        continue

                    # Get delta
                    delta = contract.get('delta', 0)
                    if not delta:
                        continue

                    # Find closest to target delta
                    delta_diff = abs(abs(delta) - self.target_delta)

                    if delta_diff < best_delta_diff:
                        best_delta_diff = delta_diff
                        best_option = {
                            'option_symbol': contract.get('symbol', ''),
                            'option_type': option_type,
                            'strike': strike,
                            'expiration': exp_date,
                            'delta': delta,
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2.0,
                            'open_interest': open_interest,
                            'volume': volume,
                            'underlying_price': underlying_price
                        }

            if not best_option:
                logger.warning(
                    f"No suitable {option_type} contract found for {symbol} "
                    f"with target delta {self.target_delta}"
                )
                return None

            logger.info(
                f"Selected {option_type} from Schwab: {symbol} ${best_option['strike']} "
                f"exp {best_option['expiration']} delta {best_option['delta']:.3f} "
                f"@ ${best_option['mid']:.2f} "
                f"(OI: {best_option['open_interest']}, Vol: {best_option['volume']})"
            )

            return best_option

        except Exception as e:
            logger.error(f"Error selecting option for {symbol} from Schwab: {e}")
            logger.exception(e)
            return None

    def _find_best_by_delta(
        self,
        contracts: List,
        target_delta: float,
        option_type: str
    ) -> Optional[Any]:
        """
        Find contract with delta closest to target.

        Args:
            contracts: List of option contracts
            target_delta: Target delta value
            option_type: 'CALL' or 'PUT'

        Returns:
            Best matching contract or None
        """
        # For puts, delta is negative, so we use absolute value
        target = abs(target_delta)

        best_contract = None
        best_delta_diff = float('inf')

        for contract in contracts:
            if not contract.delta:
                continue

            delta = abs(contract.delta)
            delta_diff = abs(delta - target)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_contract = contract

        return best_contract


# Global singleton instance with default parameters
option_selector = OptionSelector(
    min_dte=30,
    max_dte=45,
    target_delta=0.30,
    min_open_interest=50,  # Lowered for testing
    min_volume=5  # Lowered for testing
)
