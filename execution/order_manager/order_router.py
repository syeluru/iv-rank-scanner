"""
Order routing and execution management.

This module manages the complete order lifecycle from validation through
submission, tracking, and completion.
"""

import asyncio
from typing import Optional
from datetime import datetime
import pandas as pd
from loguru import logger

from execution.order_manager.order_state import Order, OrderStatus, OrderType, OrderSide
from execution.broker_api.schwab_client import schwab_client
from execution.risk_manager.pre_trade_checks import pre_trade_checker
from execution.risk_manager.position_limits import position_limit_manager
from data.storage.database import trades_db
from strategies.base_strategy import Signal, SignalDirection
from config.settings import settings


class OrderRouter:
    """
    Routes orders through validation, submission, and tracking.

    This is the main interface for order execution, handling:
    1. Pre-trade validation
    2. Order submission to broker
    3. Order status polling
    4. Fill tracking
    5. Database persistence
    """

    def __init__(self):
        """Initialize order router."""
        self.active_orders: dict[str, Order] = {}
        self._polling_tasks: dict[str, asyncio.Task] = {}
        logger.info("OrderRouter initialized")

    async def submit_signal(self, signal: Signal) -> dict:
        """
        Submit a signal from algo trading strategy.

        Converts the Signal to an Order and submits it.

        Args:
            signal: Trading signal from strategy

        Returns:
            Dict with 'success' and 'order_id' or 'error'
        """
        try:
            logger.info(f"Converting signal to order: {signal.symbol} {signal.direction.value}")

            # Convert signal to order
            order = self._signal_to_order(signal)

            # Submit order
            success, message = await self.submit_order(order, skip_validation=False)

            if success:
                return {
                    'success': True,
                    'order_id': order.order_id,
                    'message': message
                }
            else:
                return {
                    'success': False,
                    'error': message
                }

        except Exception as e:
            logger.error(f"Error submitting signal: {e}")
            logger.exception(e)
            return {
                'success': False,
                'error': str(e)
            }

    def _signal_to_order(self, signal: Signal) -> Order:
        """
        Convert a Signal to an Order.

        Args:
            signal: Trading signal

        Returns:
            Order object
        """
        # Determine order side based on signal type and direction
        # For options:
        #   - ENTRY + LONG = BUY_TO_OPEN (buying options)
        #   - ENTRY + SHORT = SELL_TO_OPEN (selling/writing options)
        #   - EXIT + LONG = SELL_TO_CLOSE (closing long options)
        #   - EXIT + SHORT = BUY_TO_CLOSE (closing short options)
        from strategies.base_strategy import SignalType

        if signal.signal_type == SignalType.ENTRY:
            if signal.direction == SignalDirection.LONG:
                side = OrderSide.BUY_TO_OPEN
            else:  # SHORT - selling/writing options
                side = OrderSide.SELL_TO_OPEN
        else:  # EXIT or ADJUST
            if signal.direction == SignalDirection.LONG:
                side = OrderSide.SELL_TO_CLOSE
            else:  # SHORT - buying back short options
                side = OrderSide.BUY_TO_CLOSE

        # Determine order type - for options we'll use LIMIT
        order_type = OrderType.LIMIT

        # Build notes with signal info
        notes = f"Signal: {signal.signal_id[:8]} | {signal.signal_type.value} | Confidence: {signal.confidence:.2f}"

        # Add option info to notes if present
        if signal.option_type:
            notes += f" | {signal.option_type} ${signal.strike} exp {signal.expiration}"

        # Create order with option details
        order = Order(
            symbol=signal.symbol,  # Will be overridden with OCC symbol for options
            underlying_symbol=signal.symbol,  # Keep track of underlying
            side=side,
            order_type=order_type,
            quantity=signal.quantity or 1,
            limit_price=signal.entry_price,
            stop_price=signal.stop_loss if signal.stop_loss else None,
            option_type=signal.option_type,
            strike=signal.strike,
            expiration=signal.expiration,
            strategy_name=signal.strategy_name,
            notes=notes
        )

        # For options, set the symbol to the OCC symbol
        if order.is_option:
            order.symbol = order.occ_symbol
            logger.info(f"Built option order with OCC symbol: {order.symbol}")

        logger.debug(f"Converted signal to order: {order.order_id}")
        return order

    async def submit_order(
        self,
        order: Order,
        skip_validation: bool = False
    ) -> tuple[bool, str]:
        """
        Submit an order with full validation and tracking.

        Args:
            order: Order to submit
            skip_validation: Skip pre-trade checks (not recommended)

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Submitting order: {order}")

        # Validate order is in correct state
        if order.status != OrderStatus.PENDING:
            msg = f"Order must be in PENDING state, currently {order.status}"
            logger.error(msg)
            return False, msg

        # Run pre-trade checks
        if not skip_validation:
            order.transition_to(OrderStatus.VALIDATING)

            # Get account info for checks
            try:
                account_info = await schwab_client.get_account()
                positions = await schwab_client.get_account_positions()
            except Exception as e:
                order.transition_to(OrderStatus.FAILED)
                order.record_error(f"Failed to get account info: {e}")
                return False, str(e)

            # Run validation checks
            check_results = await pre_trade_checker.validate_order(
                order,
                account_info,
                positions
            )

            # Check for errors
            if pre_trade_checker.has_errors(check_results):
                error_msg = pre_trade_checker.get_error_message(check_results)
                order.transition_to(OrderStatus.REJECTED, error_msg)
                order.record_error(error_msg)
                logger.error(f"Order {order.order_id} rejected: {error_msg}")
                return False, error_msg

            order.transition_to(OrderStatus.VALIDATED)

        # Submit to broker
        order.transition_to(OrderStatus.SUBMITTING)

        try:
            # Build order spec for Schwab API
            order_spec = self._build_schwab_order_spec(order)

            # Submit order
            response = await schwab_client.place_order(order_spec)

            # Check if paper trading
            if settings.PAPER_TRADING:
                logger.info(f"Paper trading mode - order simulated: {order.order_id}")
                order.broker_order_id = f"PAPER_{order.order_id[:8]}"
                order.transition_to(OrderStatus.SUBMITTED)

                # Store in database
                self._store_order(order)

                # Add to active orders
                self.active_orders[order.order_id] = order

                # Start polling (will simulate fills in paper trading)
                self._start_polling(order)

                return True, "Order submitted (paper trading)"

            # Real trading - extract broker order ID
            broker_order_id = response.get('orderId')
            if not broker_order_id:
                # WARNING: Order may have been placed but we didn't get the ID back!
                # This is a critical issue - the order might be open at the broker
                logger.error(
                    f"CRITICAL: Order submitted but no broker order ID received! "
                    f"Response: {response}. CHECK BROKER ACCOUNT FOR ORPHAN ORDERS!"
                )
                order.transition_to(OrderStatus.FAILED)
                order.record_error("No broker order ID received - check broker for orphan orders")
                return False, "No broker order ID received - check broker account manually"

            order.broker_order_id = broker_order_id
            order.transition_to(OrderStatus.SUBMITTED)

            # Store in database
            self._store_order(order)

            # Add to active orders and start polling
            self.active_orders[order.order_id] = order
            self._start_polling(order)

            logger.info(f"Order submitted successfully: {order.order_id}")
            return True, "Order submitted successfully"

        except Exception as e:
            order.transition_to(OrderStatus.FAILED)
            order.record_error(str(e))
            logger.error(f"Failed to submit order: {e}")
            return False, str(e)

    async def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Tuple of (success, message)
        """
        order = self.active_orders.get(order_id)
        if not order:
            return False, f"Order {order_id} not found"

        if not order.is_active:
            return False, f"Order {order_id} is not active (status: {order.status})"

        logger.info(f"Cancelling order: {order_id}")

        order.transition_to(OrderStatus.CANCELLING)

        try:
            # Cancel with broker
            if not settings.PAPER_TRADING and order.broker_order_id:
                success = await schwab_client.cancel_order(order.broker_order_id)
                if not success:
                    order.transition_to(OrderStatus.FAILED)
                    return False, "Failed to cancel with broker"

            order.transition_to(OrderStatus.CANCELLED)

            # Update in database
            self._store_order(order)

            # Stop polling
            self._stop_polling(order_id)

            logger.info(f"Order cancelled: {order_id}")
            return True, "Order cancelled successfully"

        except Exception as e:
            order.record_error(str(e))
            logger.error(f"Failed to cancel order: {e}")
            return False, str(e)

    def _start_polling(self, order: Order) -> None:
        """
        Start polling for order status updates.

        Args:
            order: Order to poll
        """
        if order.order_id in self._polling_tasks:
            logger.warning(f"Already polling order {order.order_id}")
            return

        task = asyncio.create_task(self._poll_order_status(order))
        self._polling_tasks[order.order_id] = task

        logger.debug(f"Started polling for order {order.order_id}")

    def _stop_polling(self, order_id: str) -> None:
        """
        Stop polling for an order.

        Args:
            order_id: Order ID
        """
        task = self._polling_tasks.pop(order_id, None)
        if task and not task.done():
            task.cancel()
            logger.debug(f"Stopped polling for order {order_id}")

    async def _poll_order_status(self, order: Order) -> None:
        """
        Poll broker for order status updates.

        Args:
            order: Order to poll
        """
        poll_interval = 2  # seconds
        max_polls = 300  # 10 minutes max
        poll_count = 0

        while poll_count < max_polls and not order.is_terminal_state:
            await asyncio.sleep(poll_interval)
            poll_count += 1

            try:
                # Paper trading simulation
                if settings.PAPER_TRADING:
                    # Simulate order fill after a few polls
                    if poll_count >= 3 and order.status != OrderStatus.FILLED:
                        # Simulate full fill
                        fill_price = order.limit_price if order.limit_price else 100.0
                        order.add_fill(
                            quantity=order.quantity,
                            price=fill_price,
                            commission=0.0,
                            fees=0.0
                        )
                        self._store_order(order)
                        logger.info(f"Order {order.order_id} filled (simulated)")
                        break
                    continue

                # Real trading - get order status from broker
                if not order.broker_order_id:
                    logger.error(f"No broker order ID for {order.order_id}")
                    break

                order_data = await schwab_client.get_order(order.broker_order_id)

                # Parse order status
                broker_status = order_data.get('status', '').upper()

                # Update order based on broker status
                if broker_status == 'FILLED':
                    # Extract fill details
                    filled_qty = order_data.get('filledQuantity', order.quantity)
                    avg_price = order_data.get('averagePrice', order.limit_price or 0)

                    if filled_qty > order.filled_quantity:
                        order.add_fill(
                            quantity=filled_qty - order.filled_quantity,
                            price=avg_price,
                            commission=0.0,  # Schwab commission structure
                            fees=0.0
                        )

                    self._store_order(order)
                    break

                elif broker_status == 'CANCELED':
                    order.transition_to(OrderStatus.CANCELLED)
                    self._store_order(order)
                    break

                elif broker_status == 'REJECTED':
                    # Extract rejection reason from broker response
                    rejection_reason = order_data.get('statusDescription', 'Unknown reason')
                    close_time = order_data.get('closeTime', '')
                    logger.error(
                        f"Order {order.order_id} REJECTED by broker: {rejection_reason} "
                        f"(closeTime: {close_time})"
                    )
                    logger.error(f"Full rejection response: {order_data}")
                    order.transition_to(OrderStatus.REJECTED, rejection_reason)
                    order.record_error(f"Broker rejected: {rejection_reason}")
                    self._store_order(order)
                    break

                elif broker_status in ['WORKING', 'QUEUED', 'ACCEPTED']:
                    if order.status != OrderStatus.WORKING:
                        order.transition_to(OrderStatus.WORKING)

            except Exception as e:
                logger.error(f"Error polling order {order.order_id}: {e}")
                order.record_error(str(e))

        # Cleanup
        self._stop_polling(order.order_id)

        if poll_count >= max_polls:
            logger.warning(f"Polling timed out for order {order.order_id}")

    def _build_schwab_order_spec(self, order: Order) -> dict:
        """
        Build Schwab API order specification for options orders.

        Args:
            order: Order object (must be an options order)

        Returns:
            Schwab order specification dict

        Raises:
            ValueError: If order is not an options order
        """
        if not order.is_option:
            raise ValueError(
                f"Only options orders are supported. "
                f"Order {order.order_id} is missing option details "
                f"(option_type={order.option_type}, strike={order.strike}, expiration={order.expiration})"
            )

        # Map OrderSide to Schwab instruction
        # Schwab uses: BUY_TO_OPEN, SELL_TO_OPEN, BUY_TO_CLOSE, SELL_TO_CLOSE
        instruction_map = {
            OrderSide.BUY_TO_OPEN: 'BUY_TO_OPEN',
            OrderSide.SELL_TO_OPEN: 'SELL_TO_OPEN',
            OrderSide.BUY_TO_CLOSE: 'BUY_TO_CLOSE',
            OrderSide.SELL_TO_CLOSE: 'SELL_TO_CLOSE',
            # Fallback mappings for generic BUY/SELL (treat as opening positions)
            OrderSide.BUY: 'BUY_TO_OPEN',
            OrderSide.SELL: 'SELL_TO_OPEN',
        }

        instruction = instruction_map.get(order.side)
        if not instruction:
            raise ValueError(f"Unsupported order side for options: {order.side}")

        # Build the order specification
        order_spec = {
            'orderType': order.order_type.value,
            'session': 'NORMAL',
            'duration': order.duration.value,
            'orderStrategyType': 'SINGLE',
            'orderLegCollection': [
                {
                    'instruction': instruction,
                    'quantity': order.quantity,
                    'instrument': {
                        'symbol': order.occ_symbol,
                        'assetType': 'OPTION'
                    }
                }
            ]
        }

        # Add price if limit order
        if order.order_type == OrderType.LIMIT and order.limit_price:
            order_spec['price'] = str(round(order.limit_price, 2))

        # Add stop price if stop order
        if order.order_type == OrderType.STOP and order.stop_price:
            order_spec['stopPrice'] = str(round(order.stop_price, 2))

        logger.info(
            f"Built Schwab option order: {instruction} {order.quantity}x {order.occ_symbol} "
            f"@ ${order.limit_price or 'MKT'}"
        )

        return order_spec

    def _store_order(self, order: Order) -> None:
        """
        Store order in database.

        Args:
            order: Order to store
        """
        try:
            df = pd.DataFrame([order.to_dict()])
            trades_db.insert_df(df, 'orders', if_exists='append')
            logger.debug(f"Stored order {order.order_id} in database")
        except Exception as e:
            logger.error(f"Failed to store order {order.order_id}: {e}")

    def get_active_orders(self) -> list[Order]:
        """
        Get list of active orders.

        Returns:
            List of orders that are not in terminal state
        """
        return [order for order in self.active_orders.values() if order.is_active]

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        return self.active_orders.get(order_id)

    async def close_all_positions(self) -> list[tuple[bool, str]]:
        """
        Emergency function to close all open positions.

        Returns:
            List of (success, message) tuples for each close attempt
        """
        logger.warning("Closing all positions (emergency)")

        try:
            positions = await schwab_client.get_account_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return [(False, str(e))]

        results = []

        for position in positions:
            try:
                symbol = position.get('instrument', {}).get('symbol')
                quantity = position.get('longQuantity', 0) - position.get('shortQuantity', 0)

                if quantity == 0:
                    continue

                # Create close order
                close_order = Order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if quantity > 0 else OrderSide.BUY,
                    quantity=abs(int(quantity)),
                    strategy_name="emergency_close"
                )

                # Submit order
                success, message = await self.submit_order(close_order, skip_validation=True)
                results.append((success, f"{symbol}: {message}"))

            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {e}")
                results.append((False, f"{symbol}: {str(e)}"))

        return results


# Global order router instance
order_router = OrderRouter()
