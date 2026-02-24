"""
Order Manager module.

Provides iron condor order building for 0DTE SPX strategies.
"""

from execution.order_manager.ic_order_builder import IronCondorOrderBuilder

__all__ = [
    'IronCondorOrderBuilder',
]
