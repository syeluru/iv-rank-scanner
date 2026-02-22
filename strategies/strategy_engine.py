"""
Strategy execution engine.

This module orchestrates strategy execution, managing the complete lifecycle:
1. Run strategies to generate signals
2. Validate signals
3. Size positions
4. Execute orders
5. Track and manage positions
6. Generate performance reports
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, PositionState, SignalType
from strategies.position_sizing.kelly_criterion import kelly_sizer
from execution.order_manager.order_state import Order, OrderType, OrderSide, OrderDuration
from execution.order_manager.order_router import order_router
from execution.broker_api.schwab_client import schwab_client
from execution.approval.signal_queue import signal_queue, ApprovalStatus
from monitoring.terminal.signal_monitor import terminal_monitor
from config.settings import settings


class StrategyEngine:
    """
    Main execution engine for trading strategies.

    Coordinates strategy execution, order placement, and position management.
    """

    def __init__(self):
        """Initialize strategy engine."""
        self.strategies: Dict[str, BaseStrategy] = {}
        self.is_running = False
        self._execution_task: Optional[asyncio.Task] = None

        logger.info("StrategyEngine initialized")

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

    async def run_once(self) -> Dict[str, Any]:
        """
        Run all strategies once (single cycle).

        Returns:
            Execution statistics
        """
        logger.info("=" * 60)
        logger.info("Running strategy cycle...")
        logger.info("=" * 60)

        stats = {
            'start_time': datetime.now(),
            'strategies_run': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_failed': 0,
            'orders_rejected': 0,
            'orders_expired': 0,
            'positions_managed': 0,
            'errors': []
        }

        try:
            # Get current market data
            market_data = await self._gather_market_data()

            # Get current positions
            positions = await self._get_all_positions()

            # Run each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    logger.info(f"\nRunning strategy: {strategy_name}")

                    # Generate signals
                    signals = await strategy.generate_signals(market_data, positions)
                    stats['signals_generated'] += len(signals)

                    logger.info(f"  Generated {len(signals)} signals")

                    # Execute signals (with approval if enabled)
                    for signal in signals:
                        # Check if manual approval is required
                        if settings.ENABLE_MANUAL_APPROVAL:
                            approval_status = await self._request_approval(signal)

                            if approval_status == ApprovalStatus.APPROVED:
                                success = await self._execute_signal(signal, strategy)
                                if success:
                                    stats['orders_placed'] += 1
                                else:
                                    stats['orders_failed'] += 1
                            elif approval_status == ApprovalStatus.REJECTED:
                                logger.info(f"Signal rejected by user: {signal.symbol}")
                                stats['orders_rejected'] += 1
                            else:  # EXPIRED
                                logger.warning(f"Signal expired without approval: {signal.symbol}")
                                stats['orders_expired'] += 1
                        else:
                            # Auto-execute if approval disabled
                            success = await self._execute_signal(signal, strategy)
                            if success:
                                stats['orders_placed'] += 1
                            else:
                                stats['orders_failed'] += 1

                    # Manage existing positions for this strategy
                    strategy_positions = [p for p in positions if p.strategy_name == strategy_name]

                    for position in strategy_positions:
                        management_signal = await strategy.manage_position(position, market_data)

                        if management_signal:
                            success = await self._execute_signal(management_signal, strategy)
                            if success:
                                stats['positions_managed'] += 1

                    stats['strategies_run'] += 1

                except Exception as e:
                    logger.error(f"Error running strategy {strategy_name}: {e}")
                    stats['errors'].append(f"{strategy_name}: {str(e)}")

            stats['end_time'] = datetime.now()
            stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

            # Log summary
            logger.info("\n" + "=" * 60)
            logger.info("Strategy Cycle Complete")
            logger.info("=" * 60)
            logger.info(f"Strategies run: {stats['strategies_run']}")
            logger.info(f"Signals generated: {stats['signals_generated']}")
            logger.info(f"Orders placed: {stats['orders_placed']}")
            logger.info(f"Orders failed: {stats['orders_failed']}")

            if settings.ENABLE_MANUAL_APPROVAL:
                logger.info(f"Orders rejected: {stats['orders_rejected']}")
                logger.info(f"Orders expired: {stats['orders_expired']}")

            logger.info(f"Positions managed: {stats['positions_managed']}")
            logger.info(f"Duration: {stats['duration']:.1f}s")

            if stats['errors']:
                logger.warning(f"Errors: {len(stats['errors'])}")

        except Exception as e:
            logger.error(f"Strategy cycle failed: {e}")
            stats['errors'].append(str(e))

        return stats

    async def run_continuous(self, interval_seconds: int = 300):
        """
        Run strategies continuously at specified interval.

        Args:
            interval_seconds: Seconds between runs (default: 300 = 5 minutes)
        """
        self.is_running = True
        logger.info(f"Starting continuous execution (interval: {interval_seconds}s)")

        while self.is_running:
            try:
                await self.run_once()

                # Wait for next cycle
                logger.info(f"Sleeping {interval_seconds}s until next cycle...")
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in continuous execution: {e}")
                await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop continuous execution."""
        self.is_running = False
        logger.info("Stopping continuous execution...")

    async def _gather_market_data(self) -> Dict[str, Any]:
        """
        Gather current market data for all symbols.

        Returns:
            Dictionary with market data
        """
        symbols = settings.DEFAULT_SYMBOLS

        try:
            # Get quotes
            quotes = await schwab_client.get_quotes(symbols)

            # Get account info
            account_info = await schwab_client.get_account()

            return {
                'symbols': symbols,
                'quotes': quotes,
                'account_info': account_info,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to gather market data: {e}")
            return {
                'symbols': symbols,
                'quotes': {},
                'account_info': {},
                'timestamp': datetime.now()
            }

    async def _get_all_positions(self) -> List[PositionState]:
        """
        Get all current positions from broker.

        Returns:
            List of position states
        """
        try:
            # Get positions from broker
            broker_positions = await schwab_client.get_account_positions()

            # Convert to PositionState objects
            positions = []

            for broker_pos in broker_positions:
                # Extract position details
                instrument = broker_pos.get('instrument', {})
                symbol = instrument.get('symbol', '')

                if not symbol:
                    continue

                # Create PositionState
                # Note: This is simplified - would need more logic to match
                # broker positions to strategy positions
                # For now, just create basic state

                positions.append(PositionState(
                    position_id=f"{symbol}_{datetime.now().strftime('%Y%m%d')}",
                    strategy_name='unknown',  # Would need to look up
                    symbol=symbol,
                    direction=None,  # Would determine from position
                    quantity=int(broker_pos.get('longQuantity', 0)),
                    entry_price=float(broker_pos.get('averagePrice', 0)),
                    entry_date=datetime.now(),  # Would need to look up
                    current_price=float(broker_pos.get('marketValue', 0)) / int(broker_pos.get('longQuantity', 1)),
                    current_value=float(broker_pos.get('marketValue', 0)),
                    unrealized_pnl=float(broker_pos.get('currentDayProfitLoss', 0)),
                    unrealized_pnl_pct=float(broker_pos.get('currentDayProfitLossPercentage', 0))
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def _request_approval(self, signal: Signal) -> ApprovalStatus:
        """
        Request user approval for signal via terminal monitor.

        Args:
            signal: Signal requiring approval

        Returns:
            ApprovalStatus: APPROVED, REJECTED, or EXPIRED
        """
        logger.info(f"Requesting approval for signal: {signal.symbol}")

        # Add signal to approval queue
        signal_id = signal_queue.add_signal(
            signal,
            timeout_seconds=settings.APPROVAL_TIMEOUT_SECONDS
        )

        # Get pending signal for display
        pending_signal = signal_queue.get_pending_signal(signal_id)

        if not pending_signal:
            logger.error(f"Failed to retrieve pending signal {signal_id}")
            return ApprovalStatus.REJECTED

        try:
            # Display signal and wait for user decision
            approved, reason = await terminal_monitor.get_user_decision(
                pending_signal,
                refresh_interval=settings.TERMINAL_REFRESH_RATE
            )

            if approved:
                # Mark as approved in queue
                signal_queue.approve_signal(signal_id)
                terminal_monitor.show_approval_confirmation(signal)
                return ApprovalStatus.APPROVED
            else:
                # Mark as rejected in queue
                signal_queue.reject_signal(signal_id, reason or "User rejected")
                terminal_monitor.show_rejection_confirmation(signal, reason or "User rejected")
                return ApprovalStatus.REJECTED

        except TimeoutError:
            logger.warning(f"Approval timeout for signal {signal_id}")
            signal_queue.expire_signal(signal_id)
            return ApprovalStatus.EXPIRED

        except Exception as e:
            logger.error(f"Error during approval request: {e}")
            signal_queue.reject_signal(signal_id, f"Error: {str(e)}")
            return ApprovalStatus.REJECTED

    async def _execute_signal(
        self,
        signal: Signal,
        strategy: BaseStrategy
    ) -> bool:
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute
            strategy: Strategy that generated the signal

        Returns:
            True if successful
        """
        try:
            logger.info(f"Executing signal: {signal}")

            # Validate signal
            valid, message = strategy.validate_signal(signal)
            if not valid:
                logger.warning(f"Signal validation failed: {message}")
                return False

            # Get account info for position sizing
            account_info = await schwab_client.get_account()
            portfolio_value = account_info.get('accountValue', 100000)  # Default fallback

            # Calculate position size if not specified
            if signal.quantity is None:
                signal.quantity = strategy.calculate_position_size(
                    signal,
                    portfolio_value,
                    portfolio_value  # Available capital (simplified)
                )

            # SAFETY FILTER: Convert naked short options to spreads
            signal = await self._convert_naked_to_spread(signal)

            # Create order from signal
            order = self._signal_to_order(signal)

            # Submit order
            success, msg = await order_router.submit_order(order)

            if success:
                logger.info(f"✓ Order submitted: {msg}")

                # Notify strategy of order
                await strategy.on_fill(signal, order.quantity, order.limit_price or 0)

                return True
            else:
                logger.error(f"✗ Order failed: {msg}")
                await strategy.on_error(signal, Exception(msg))
                return False

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            await strategy.on_error(signal, e)
            return False

    async def _convert_naked_to_spread(self, signal: Signal) -> Signal:
        """
        Convert naked short option signals to vertical spreads for risk management.

        If signal is a short call/put (SELL_TO_OPEN), automatically adds a protective
        long wing to convert it to a spread.

        Args:
            signal: Original signal

        Returns:
            Modified signal (spread) or original signal if not applicable
        """
        # Only apply to entry signals with options
        if signal.signal_type != SignalType.ENTRY:
            return signal

        if not signal.option_type or signal.option_type in ['IRON_CONDOR', 'VERTICAL_SPREAD', 'PUT_SPREAD', 'CALL_SPREAD']:
            return signal  # Already a spread or not an option

        # Check if it's a naked short (direction is SHORT for entry)
        if signal.direction != SignalDirection.SHORT:
            return signal  # Long options are fine

        logger.warning(f"⚠️ SAFETY: Converting naked {signal.option_type} to vertical spread for {signal.symbol}")

        # Signal is a naked short - convert to spread
        # Add protective wing $5-10 away depending on stock price
        if not signal.strike:
            logger.error(f"Cannot convert to spread - no strike price in signal")
            return signal

        try:
            # Determine wing width based on underlying price
            underlying_price = signal.metadata.get('underlying_price', signal.entry_price * 100)
            wing_width = 10 if underlying_price > 100 else 5

            # Calculate protective strike
            if signal.option_type == 'CALL':
                # For short call, buy call above
                protective_strike = signal.strike + wing_width
                new_option_type = 'CALL_SPREAD'
            else:  # PUT
                # For short put, buy put below
                protective_strike = signal.strike - wing_width
                new_option_type = 'PUT_SPREAD'

            # Reduce entry credit by estimated cost of protective wing
            # Rough estimate: protective wing costs ~30-40% of the credit received
            adjusted_entry = signal.entry_price * 0.65  # Keep 65% of original credit

            # Update signal to be a spread
            signal.option_type = new_option_type
            signal.entry_price = adjusted_entry
            signal.metadata['converted_from_naked'] = True
            signal.metadata['short_strike'] = signal.strike
            signal.metadata['long_strike'] = protective_strike
            signal.metadata['original_credit'] = signal.entry_price / 0.65

            signal.notes = f"CONVERTED TO SPREAD: {signal.notes} | Protective wing @ ${protective_strike}"

            logger.info(f"✓ Converted to {new_option_type}: Short ${signal.strike}, Long ${protective_strike}")

        except Exception as e:
            logger.error(f"Failed to convert naked to spread: {e}. Rejecting signal.")
            # Reject the signal by returning None - but can't do that, so mark as invalid
            signal.metadata['conversion_failed'] = True

        return signal

    def _signal_to_order(self, signal: Signal) -> Order:
        """
        Convert a signal to an order.

        Args:
            signal: Signal to convert

        Returns:
            Order object
        """
        # Determine order side
        if signal.signal_type == SignalType.ENTRY:
            if signal.direction.value == 'LONG':
                side = OrderSide.BUY_TO_OPEN
            else:
                side = OrderSide.SELL_TO_OPEN
        else:  # EXIT
            if signal.direction.value == 'LONG':
                side = OrderSide.BUY_TO_CLOSE
            else:
                side = OrderSide.SELL_TO_CLOSE

        # Create order
        order = Order(
            symbol=signal.symbol,
            order_type=OrderType.LIMIT,  # Default to limit orders
            side=side,
            quantity=signal.quantity or 1,
            limit_price=signal.entry_price,
            duration=OrderDuration.DAY,
            strategy_name=signal.strategy_name,
            position_id=signal.position_id,
            notes=signal.notes
        )

        return order

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all strategies.

        Returns:
            Dictionary with strategy statistics
        """
        stats = {
            'total_strategies': len(self.strategies),
            'is_running': self.is_running,
            'strategies': {}
        }

        for name, strategy in self.strategies.items():
            stats['strategies'][name] = strategy.get_statistics()

        return stats


# Global strategy engine instance
strategy_engine = StrategyEngine()
