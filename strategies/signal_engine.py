"""
Broker-agnostic signal generation engine.

This engine identifies trading signals without requiring broker connection.
It uses an abstract MarketDataProvider interface, allowing it to work with:
- Real broker data (Schwab, IBKR, etc.)
- Mock data for testing
- Historical data for backtesting
- Any custom data source

The signal engine is completely separated from order execution,
making it easy to:
1. Test strategies without broker connection
2. Switch between brokers
3. Run signals in read-only mode
4. Backtest with historical data
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, PositionState
from data.providers.base_provider import MarketDataProvider, MockMarketDataProvider


class SignalEngine:
    """
    Broker-agnostic signal generation engine.

    Runs strategies to identify trading opportunities without
    requiring broker connection or order execution capability.
    """

    def __init__(self, provider: Optional[MarketDataProvider] = None):
        """
        Initialize signal engine.

        Args:
            provider: Market data provider (defaults to mock provider)
        """
        self.provider = provider or MockMarketDataProvider()
        self.strategies: Dict[str, BaseStrategy] = {}

        logger.info(
            f"SignalEngine initialized with provider: "
            f"{self.provider.get_provider_name()}"
        )

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance to add
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Remove a strategy from the engine.

        Args:
            strategy_name: Name of strategy to remove

        Returns:
            Removed strategy or None
        """
        strategy = self.strategies.pop(strategy_name, None)
        if strategy:
            logger.info(f"Removed strategy: {strategy_name}")
        return strategy

    async def generate_signals(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, List[Signal]]:
        """
        Generate signals from all strategies.

        This method:
        1. Gathers market data from provider
        2. Runs each strategy's signal generation logic
        3. Returns signals WITHOUT executing them

        Args:
            symbols: Symbols to analyze (None = use strategy defaults)

        Returns:
            Dictionary mapping strategy name to list of signals
        """
        logger.info("=" * 60)
        logger.info("Generating signals...")
        logger.info(f"Provider: {self.provider.get_provider_name()}")
        logger.info(f"Strategies: {len(self.strategies)}")
        logger.info("=" * 60)

        all_signals = {}
        stats = {
            'start_time': datetime.now(),
            'strategies_run': 0,
            'total_signals': 0,
            'errors': []
        }

        try:
            # Check provider connection
            if not self.provider.is_connected():
                logger.warning(
                    f"Provider {self.provider.get_provider_name()} "
                    f"not connected - signals may be limited"
                )

            # Gather market data
            market_data = await self._gather_market_data(symbols)

            # Get current positions (empty for signal-only mode)
            positions = await self._get_positions()

            # Run each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    logger.info(f"\nRunning strategy: {strategy_name}")

                    # Generate signals
                    signals = await strategy.generate_signals(market_data, positions)

                    logger.info(f"  Generated {len(signals)} signals")

                    # Log signal details
                    for signal in signals:
                        logger.info(
                            f"    • {signal.symbol}: {signal.signal_type.value} "
                            f"{signal.direction.value} @ ${signal.entry_price:.2f} "
                            f"(confidence: {signal.confidence:.0%})"
                        )

                    all_signals[strategy_name] = signals
                    stats['total_signals'] += len(signals)
                    stats['strategies_run'] += 1

                except Exception as e:
                    logger.error(f"Error running strategy {strategy_name}: {e}")
                    stats['errors'].append(f"{strategy_name}: {str(e)}")
                    all_signals[strategy_name] = []

            stats['end_time'] = datetime.now()
            stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

            # Log summary
            logger.info("\n" + "=" * 60)
            logger.info("Signal Generation Complete")
            logger.info("=" * 60)
            logger.info(f"Strategies run: {stats['strategies_run']}")
            logger.info(f"Total signals: {stats['total_signals']}")
            logger.info(f"Duration: {stats['duration']:.1f}s")

            if stats['errors']:
                logger.warning(f"Errors: {len(stats['errors'])}")

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            stats['errors'].append(str(e))

        return all_signals

    async def generate_signals_for_strategy(
        self,
        strategy_name: str,
        symbols: Optional[List[str]] = None
    ) -> List[Signal]:
        """
        Generate signals for a specific strategy.

        Args:
            strategy_name: Name of strategy to run
            symbols: Symbols to analyze

        Returns:
            List of signals
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy not found: {strategy_name}")
            return []

        strategy = self.strategies[strategy_name]

        try:
            # Gather market data
            market_data = await self._gather_market_data(symbols)

            # Get positions
            positions = await self._get_positions()

            # Generate signals
            signals = await strategy.generate_signals(market_data, positions)

            logger.info(
                f"Strategy '{strategy_name}' generated {len(signals)} signals"
            )

            return signals

        except Exception as e:
            logger.error(f"Failed to generate signals for {strategy_name}: {e}")
            return []

    async def _gather_market_data(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Gather market data from provider.

        Args:
            symbols: Symbols to get data for

        Returns:
            Dictionary with market data
        """
        if symbols is None:
            # Use default symbols from config if available
            from config.settings import settings
            symbols = settings.DEFAULT_SYMBOLS[:20]  # Limit for performance

        try:
            logger.info(f"Gathering market data for {len(symbols)} symbols...")

            # Get quotes
            quotes = await self.provider.get_quotes(symbols)

            logger.info(f"  Retrieved {len(quotes)} quotes")

            # Get account info (if available)
            account_info = await self.provider.get_account_info()

            return {
                'symbols': symbols,
                'quotes': quotes,
                'account_info': account_info,
                'timestamp': datetime.now(),
                'provider': self.provider.get_provider_name()
            }

        except Exception as e:
            logger.error(f"Failed to gather market data: {e}")
            return {
                'symbols': symbols or [],
                'quotes': {},
                'account_info': None,
                'timestamp': datetime.now(),
                'provider': self.provider.get_provider_name()
            }

    async def _get_positions(self) -> List[PositionState]:
        """
        Get current positions from provider.

        Returns:
            List of position states (empty if provider doesn't support it)
        """
        try:
            positions_data = await self.provider.get_positions()

            # Convert to PositionState objects
            positions = []
            for pos in positions_data:
                # Create basic PositionState
                # Note: This is simplified - real implementation would
                # need to map provider positions to strategy positions
                positions.append(PositionState(
                    position_id=f"{pos.symbol}_{datetime.now().strftime('%Y%m%d')}",
                    strategy_name='unknown',
                    symbol=pos.symbol,
                    direction=None,  # Would determine from position
                    quantity=pos.quantity,
                    entry_price=pos.average_price,
                    entry_date=datetime.now(),
                    current_price=pos.current_price,
                    current_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct
                ))

            return positions

        except Exception as e:
            logger.debug(f"Could not get positions (OK for signal-only mode): {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all strategies.

        Returns:
            Dictionary with strategy statistics
        """
        stats = {
            'provider': self.provider.get_provider_name(),
            'provider_connected': self.provider.is_connected(),
            'total_strategies': len(self.strategies),
            'strategies': {}
        }

        for name, strategy in self.strategies.items():
            stats['strategies'][name] = strategy.get_statistics()

        return stats

    async def validate_provider(self) -> Dict[str, Any]:
        """
        Validate that provider is working correctly.

        Returns:
            Dictionary with validation results
        """
        results = {
            'provider': self.provider.get_provider_name(),
            'connected': False,
            'can_get_quotes': False,
            'can_get_option_chain': False,
            'can_get_account': False,
            'errors': []
        }

        try:
            # Check connection
            results['connected'] = self.provider.is_connected()

            # Test quote retrieval
            try:
                quote = await self.provider.get_quote('AAPL')
                results['can_get_quotes'] = quote is not None
            except Exception as e:
                results['errors'].append(f"Quote error: {e}")

            # Test option chain retrieval
            try:
                from datetime import date, timedelta
                from_date = date.today() + timedelta(days=30)
                to_date = date.today() + timedelta(days=45)

                chain = await self.provider.get_option_chain(
                    'AAPL',
                    from_date=from_date,
                    to_date=to_date
                )
                results['can_get_option_chain'] = chain is not None
            except Exception as e:
                results['errors'].append(f"Option chain error: {e}")

            # Test account info retrieval
            try:
                account = await self.provider.get_account_info()
                results['can_get_account'] = account is not None
            except Exception as e:
                results['errors'].append(f"Account error: {e}")

        except Exception as e:
            results['errors'].append(f"Validation error: {e}")

        return results


# Convenience function for quick signal generation
async def generate_signals_quick(
    strategy: BaseStrategy,
    symbols: Optional[List[str]] = None,
    provider: Optional[MarketDataProvider] = None
) -> List[Signal]:
    """
    Quick signal generation for a single strategy.

    Args:
        strategy: Strategy to run
        symbols: Symbols to analyze
        provider: Market data provider (defaults to mock)

    Returns:
        List of signals
    """
    engine = SignalEngine(provider)
    engine.add_strategy(strategy)

    all_signals = await engine.generate_signals(symbols)

    return all_signals.get(strategy.name, [])
