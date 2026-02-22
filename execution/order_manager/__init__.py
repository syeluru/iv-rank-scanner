"""
Order Manager module.

Provides order routing, state management, and bracket order functionality.
"""

from execution.order_manager.order_router import order_router, OrderRouter
from execution.order_manager.order_state import Order, OrderStatus, OrderType, OrderSide
from execution.order_manager.bracket_orders import bracket_manager, BracketOrderManager

__all__ = [
    'order_router',
    'OrderRouter',
    'Order',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'bracket_manager',
    'BracketOrderManager',
]
