"""
Broker factory — returns the configured broker client.

Usage:
    from execution.broker_api.broker_factory import broker_client
    # broker_client has the same interface regardless of which broker is active
"""

from loguru import logger
from config.settings import settings


def get_broker_client():
    """
    Get the broker client based on BROKER setting.

    Returns:
        SchwabClient or TradierClient instance (same async interface)
    """
    broker = settings.BROKER.lower()

    if broker == "tradier":
        from execution.broker_api.tradier_client import get_tradier_client
        logger.info("Using Tradier broker")
        return get_tradier_client()
    elif broker == "schwab":
        from execution.broker_api.schwab_client import schwab_client
        logger.info("Using Schwab broker")
        return schwab_client
    else:
        raise ValueError(f"Unknown broker: {broker}. Use 'schwab' or 'tradier'.")


# Global broker client instance — lazy loaded
broker_client = get_broker_client()
