"""
Bracket Order Manager - Automatically places stop loss and take profit orders.

When a main order fills, this module creates protective orders:
1. Stop Loss - Limits downside risk
2. Take Profit - Locks in gains at target

Schwab supports OCO (One-Cancels-Other) orders which we use for bracket orders.
"""

import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

from execution.order_manager.order_state import Order, OrderStatus, OrderType, OrderSide
from execution.broker_api.schwab_client import schwab_client
from strategies.base_strategy import Signal, SignalDirection
from config.settings import settings


class BracketOrderManager:
    """
    Manages bracket orders (stop loss + take profit) for filled positions.

    Creates OCO (One-Cancels-Other) orders after main entry order fills.
    """

    def __init__(self):
        """Initialize bracket order manager."""
        self.active_brackets: Dict[str, Dict[str, Any]] = {}
        logger.info("BracketOrderManager initialized")

    def validate_signal_has_exits(self, signal: Signal) -> Tuple[bool, str]:
        """
        Validate that a signal has proper exit levels defined.

        Args:
            signal: Signal to validate

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        # Check stop loss
        if signal.stop_loss is None:
            issues.append("No stop_loss defined")

        # Check take profit
        if signal.take_profit is None:
            issues.append("No take_profit defined")

        # For options, we may use alternative exit strategies
        if signal.option_type:
            # Options often use delta/premium-based exits stored in metadata
            has_alternative = (
                signal.metadata.get('delta_stop') or
                signal.metadata.get('profit_target_pct') or
                signal.metadata.get('premium_target')
            )
            if has_alternative:
                return True, "Using options-specific exit strategy"

        if issues:
            return False, f"Missing exit levels: {', '.join(issues)}"

        return True, "Exit levels defined"

    async def place_bracket_orders(
        self,
        signal: Signal,
        filled_order_id: str,
        filled_price: float,
        filled_quantity: int
    ) -> Dict[str, Any]:
        """
        Place bracket orders (stop loss and take profit) after main order fills.

        Args:
            signal: Original signal with stop/profit levels
            filled_order_id: ID of the filled entry order
            filled_price: Actual fill price
            filled_quantity: Quantity filled

        Returns:
            Dict with bracket order details
        """
        result = {
            'stop_loss_order_id': None,
            'take_profit_order_id': None,
            'errors': []
        }

        # Skip for paper trading simulation
        if settings.PAPER_TRADING:
            logger.info(f"Paper trading - simulating bracket orders for {signal.symbol}")
            result['stop_loss_order_id'] = f"PAPER_SL_{filled_order_id[:8]}"
            result['take_profit_order_id'] = f"PAPER_TP_{filled_order_id[:8]}"
            return result

        # Determine exit side (opposite of entry)
        if signal.direction == SignalDirection.LONG:
            exit_side = OrderSide.SELL
        else:
            exit_side = OrderSide.BUY

        # Adjust stop/profit based on actual fill price if needed
        stop_loss = self._adjust_stop_for_fill(signal, filled_price)
        take_profit = self._adjust_profit_for_fill(signal, filled_price)

        # Try to place OCO bracket order (preferred)
        if stop_loss and take_profit:
            try:
                oco_result = await self._place_oco_order(
                    symbol=signal.symbol,
                    quantity=filled_quantity,
                    exit_side=exit_side,
                    stop_price=stop_loss,
                    limit_price=take_profit
                )

                if oco_result.get('success'):
                    result['stop_loss_order_id'] = oco_result.get('stop_order_id')
                    result['take_profit_order_id'] = oco_result.get('limit_order_id')
                    result['oco_order_id'] = oco_result.get('oco_order_id')

                    logger.info(
                        f"Bracket OCO placed for {signal.symbol}: "
                        f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}"
                    )
                    return result
                else:
                    logger.warning(f"OCO order failed, falling back to separate orders")

            except Exception as e:
                logger.warning(f"OCO order failed: {e}, placing separate orders")

        # Fallback: Place separate stop loss and take profit orders
        if stop_loss:
            try:
                sl_result = await self._place_stop_loss_order(
                    symbol=signal.symbol,
                    quantity=filled_quantity,
                    exit_side=exit_side,
                    stop_price=stop_loss
                )
                result['stop_loss_order_id'] = sl_result.get('orderId')
                logger.info(f"Stop loss placed for {signal.symbol} @ ${stop_loss:.2f}")
            except Exception as e:
                error_msg = f"Failed to place stop loss: {e}"
                result['errors'].append(error_msg)
                logger.error(error_msg)

        if take_profit:
            try:
                tp_result = await self._place_take_profit_order(
                    symbol=signal.symbol,
                    quantity=filled_quantity,
                    exit_side=exit_side,
                    limit_price=take_profit
                )
                result['take_profit_order_id'] = tp_result.get('orderId')
                logger.info(f"Take profit placed for {signal.symbol} @ ${take_profit:.2f}")
            except Exception as e:
                error_msg = f"Failed to place take profit: {e}"
                result['errors'].append(error_msg)
                logger.error(error_msg)

        # Track active brackets
        if result['stop_loss_order_id'] or result['take_profit_order_id']:
            self.active_brackets[filled_order_id] = {
                'symbol': signal.symbol,
                'entry_order_id': filled_order_id,
                'stop_loss_order_id': result['stop_loss_order_id'],
                'take_profit_order_id': result['take_profit_order_id'],
                'stop_price': stop_loss,
                'take_profit_price': take_profit,
                'quantity': filled_quantity,
                'created_at': datetime.now()
            }

        return result

    def _adjust_stop_for_fill(self, signal: Signal, filled_price: float) -> Optional[float]:
        """
        Adjust stop loss based on actual fill price.

        If the fill was at a different price than expected, maintain
        the same risk distance.
        """
        if signal.stop_loss is None:
            return None

        if signal.entry_price is None:
            return signal.stop_loss

        # Calculate original risk distance
        original_risk = abs(signal.entry_price - signal.stop_loss)

        # Apply same risk distance to actual fill
        if signal.direction == SignalDirection.LONG:
            adjusted_stop = filled_price - original_risk
        else:
            adjusted_stop = filled_price + original_risk

        # Round to 2 decimal places
        return round(adjusted_stop, 2)

    def _adjust_profit_for_fill(self, signal: Signal, filled_price: float) -> Optional[float]:
        """
        Adjust take profit based on actual fill price.

        If the fill was at a different price than expected, maintain
        the same reward distance.
        """
        if signal.take_profit is None:
            return None

        if signal.entry_price is None:
            return signal.take_profit

        # Calculate original reward distance
        original_reward = abs(signal.take_profit - signal.entry_price)

        # Apply same reward distance to actual fill
        if signal.direction == SignalDirection.LONG:
            adjusted_profit = filled_price + original_reward
        else:
            adjusted_profit = filled_price - original_reward

        # Round to 2 decimal places
        return round(adjusted_profit, 2)

    async def _place_oco_order(
        self,
        symbol: str,
        quantity: int,
        exit_side: OrderSide,
        stop_price: float,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Place OCO (One-Cancels-Other) bracket order.

        This is the preferred method as it ensures when one leg fills,
        the other is automatically cancelled.
        """
        # Schwab OCO order specification
        oco_spec = {
            'orderStrategyType': 'OCO',
            'childOrderStrategies': [
                # Stop loss leg
                {
                    'orderType': 'STOP',
                    'session': 'NORMAL',
                    'duration': 'GOOD_TILL_CANCEL',
                    'orderStrategyType': 'SINGLE',
                    'stopPrice': stop_price,
                    'orderLegCollection': [
                        {
                            'instruction': exit_side.value,
                            'quantity': quantity,
                            'instrument': {
                                'symbol': symbol,
                                'assetType': 'EQUITY'
                            }
                        }
                    ]
                },
                # Take profit leg
                {
                    'orderType': 'LIMIT',
                    'session': 'NORMAL',
                    'duration': 'GOOD_TILL_CANCEL',
                    'orderStrategyType': 'SINGLE',
                    'price': limit_price,
                    'orderLegCollection': [
                        {
                            'instruction': exit_side.value,
                            'quantity': quantity,
                            'instrument': {
                                'symbol': symbol,
                                'assetType': 'EQUITY'
                            }
                        }
                    ]
                }
            ]
        }

        response = await schwab_client.place_order(oco_spec)

        return {
            'success': 'orderId' in response or response.get('status') == 'success',
            'oco_order_id': response.get('orderId'),
            'response': response
        }

    async def _place_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        exit_side: OrderSide,
        stop_price: float
    ) -> Dict[str, Any]:
        """Place a standalone stop loss order."""
        order_spec = {
            'orderType': 'STOP',
            'session': 'NORMAL',
            'duration': 'GOOD_TILL_CANCEL',
            'orderStrategyType': 'SINGLE',
            'stopPrice': stop_price,
            'orderLegCollection': [
                {
                    'instruction': exit_side.value,
                    'quantity': quantity,
                    'instrument': {
                        'symbol': symbol,
                        'assetType': 'EQUITY'
                    }
                }
            ]
        }

        return await schwab_client.place_order(order_spec)

    async def _place_take_profit_order(
        self,
        symbol: str,
        quantity: int,
        exit_side: OrderSide,
        limit_price: float
    ) -> Dict[str, Any]:
        """Place a standalone take profit (limit) order."""
        order_spec = {
            'orderType': 'LIMIT',
            'session': 'NORMAL',
            'duration': 'GOOD_TILL_CANCEL',
            'orderStrategyType': 'SINGLE',
            'price': limit_price,
            'orderLegCollection': [
                {
                    'instruction': exit_side.value,
                    'quantity': quantity,
                    'instrument': {
                        'symbol': symbol,
                        'assetType': 'EQUITY'
                    }
                }
            ]
        }

        return await schwab_client.place_order(order_spec)

    async def cancel_bracket(self, entry_order_id: str) -> bool:
        """
        Cancel all bracket orders for a position.

        Use this when manually closing a position or on error.
        """
        bracket = self.active_brackets.get(entry_order_id)
        if not bracket:
            logger.warning(f"No bracket found for order {entry_order_id}")
            return False

        success = True

        # Cancel stop loss
        if bracket.get('stop_loss_order_id'):
            try:
                await schwab_client.cancel_order(bracket['stop_loss_order_id'])
                logger.info(f"Cancelled stop loss order {bracket['stop_loss_order_id']}")
            except Exception as e:
                logger.error(f"Failed to cancel stop loss: {e}")
                success = False

        # Cancel take profit
        if bracket.get('take_profit_order_id'):
            try:
                await schwab_client.cancel_order(bracket['take_profit_order_id'])
                logger.info(f"Cancelled take profit order {bracket['take_profit_order_id']}")
            except Exception as e:
                logger.error(f"Failed to cancel take profit: {e}")
                success = False

        # Remove from tracking
        if success:
            del self.active_brackets[entry_order_id]

        return success

    def get_bracket_info(self, entry_order_id: str) -> Optional[Dict[str, Any]]:
        """Get bracket order information for a position."""
        return self.active_brackets.get(entry_order_id)

    def get_all_brackets(self) -> Dict[str, Dict[str, Any]]:
        """Get all active bracket orders."""
        return self.active_brackets.copy()


# Global bracket order manager instance
bracket_manager = BracketOrderManager()
