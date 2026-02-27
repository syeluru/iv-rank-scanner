"""
Trade Match Persistence - Stub for dashboard compatibility.

Wraps the trades database for trade matching data storage/retrieval.
"""

from loguru import logger
from data.storage.database import DatabaseManager
from config.settings import settings


class TradeMatchPersistence:
    """Persistence layer for matched trades."""

    def __init__(self):
        self.db = DatabaseManager(settings.trades_db_path)

    def save_matches(self, matched_trades):
        """Save matched trades to database."""
        logger.debug(f"save_matches called with {len(matched_trades) if matched_trades else 0} trades")
