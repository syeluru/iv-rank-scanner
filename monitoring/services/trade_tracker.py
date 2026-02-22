"""
Trade ID tracking service for managing persistent trade identifiers across roll chains.

This module provides a TradeTracker service that assigns unique IDs to trades and maintains
the mapping between orders and trades. This eliminates ambiguity when multiple positions
have identical strikes/quantities.
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from filelock import FileLock

logger = logging.getLogger(__name__)


@dataclass
class LegInfo:
    """Information about a single option leg."""
    symbol: str
    quantity: int
    side: str  # "LONG" or "SHORT"
    strike: float
    option_type: str  # "PUT" or "CALL"
    added_by_order: str


@dataclass
class TradeRecord:
    """Complete record of a trade and its history."""
    trade_id: str
    order_ids: List[str]  # Chronological
    underlying: str
    strategy: str  # "IRON_CONDOR", "VERTICAL_SPREAD", etc.
    quantity: int
    opened_at: str
    opened_order_id: str
    current_legs: Dict[str, Dict]  # symbol -> LegInfo as dict
    closed_legs: Dict[str, Dict]  # symbol -> {closed_at, closed_by_order, quantity}
    status: str  # "OPEN" or "CLOSED"


class TradeTracker:
    """Persistent trade ID tracking service."""

    def __init__(self, storage_path: str = "data/storage/trade_tracking.json"):
        self.storage_path = Path(storage_path)
        self.trades: Dict[str, TradeRecord] = {}
        self.order_to_trade_map: Dict[str, str] = {}
        self.load()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """
        Normalize option symbol for matching purposes.

        SPXW and SPX should match, NDXP and NDX should match, etc.
        Also normalize spacing.
        """
        if not symbol:
            return symbol

        # Replace weekly/mini variants with standard symbol
        normalized = symbol.replace('SPXW', 'SPX').replace('NDXP', 'NDX').replace('RUTW', 'RUT')

        # Normalize spacing (multiple spaces to single space)
        normalized = ' '.join(normalized.split())

        return normalized

    def assign_trade_id(self, order: Dict, underlying: str, strategy: str,
                       quantity: int, current_legs: Dict[str, LegInfo]) -> str:
        """Assign new trade ID to an opening order."""
        order_id = order.get('orderId')

        # SAFETY CHECK: Prevent duplicate trade creation
        # This should never happen if _assign_or_lookup_trade_id is working correctly,
        # but adding as defensive programming
        existing_trade_id = self.order_to_trade_map.get(order_id)
        if existing_trade_id:
            logger.warning(f"⚠️  Order {order_id} already has trade ID {existing_trade_id}! Returning existing ID instead of creating duplicate.")
            return existing_trade_id

        order_time = order.get('closeTime') or order.get('enteredTime') or datetime.now().isoformat()

        # Parse timestamp for ID generation
        try:
            dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
            timestamp = dt.strftime('%Y%m%d_%H%M%S')
        except Exception:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate ID: trade_{timestamp}_{uuid[:8]}
        trade_id = f"trade_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Convert LegInfo objects to dicts
        current_legs_dict = {k: asdict(v) for k, v in current_legs.items()}

        # Create TradeRecord
        trade_record = TradeRecord(
            trade_id=trade_id,
            order_ids=[order_id],
            underlying=underlying,
            strategy=strategy,
            quantity=quantity,
            opened_at=order_time,
            opened_order_id=order_id,
            current_legs=current_legs_dict,
            closed_legs={},
            status="OPEN"
        )

        # Store in memory
        self.trades[trade_id] = trade_record
        self.order_to_trade_map[order_id] = trade_id

        # Persist to disk
        self.save()

        logger.info(f"Assigned new trade ID {trade_id} to order {order_id} ({underlying} {strategy} x{quantity})")
        return trade_id

    def find_trade_for_closing_legs(self, closed_symbols: List[str],
                                    order_time: str) -> Optional[str]:
        """
        Find which trade owns these closing symbols.

        Match criteria:
        1. All closed symbols exist in trade's current_legs
        2. Trade status is OPEN
        3. Trade opened BEFORE this order
        4. Score by: symbol matches + quantity match bonus
        5. Return highest scoring trade (most recent if tied)
        """
        if not closed_symbols:
            return None

        logger.debug(f"🔍 TRADE MATCHING: Looking for trade with closing symbols: {closed_symbols}")
        logger.debug(f"   Order time: {order_time}")
        logger.debug(f"   Total trades in tracker: {len(self.trades)}")

        candidates = []

        for trade_id, trade in self.trades.items():
            logger.debug(f"   Checking trade {trade_id}:")
            logger.debug(f"     - Status: {trade.status}")
            logger.debug(f"     - Opened at: {trade.opened_at}")
            logger.debug(f"     - Current legs: {list(trade.current_legs.keys())}")

            # Must be OPEN
            if trade.status != "OPEN":
                logger.debug(f"     ❌ Skipped: Not OPEN")
                continue

            # Must have opened before this order
            if order_time and trade.opened_at > order_time:
                logger.debug(f"     ❌ Skipped: Opened after this order ({trade.opened_at} > {order_time})")
                continue

            # Check how many symbols match (using normalized symbols)
            # Map normalized symbols back to original symbols
            trade_symbol_map = {self._normalize_symbol(s): s for s in trade.current_legs.keys()}
            current_normalized = set(trade_symbol_map.keys())

            closed_normalized = {self._normalize_symbol(s): s for s in closed_symbols}
            closed_normalized_set = set(closed_normalized.keys())

            logger.debug(f"     - Normalized trade symbols: {current_normalized}")
            logger.debug(f"     - Normalized closing symbols: {closed_normalized_set}")

            matches = current_normalized.intersection(closed_normalized_set)
            logger.debug(f"     - Matches: {matches}")

            if not matches:
                logger.debug(f"     ❌ Skipped: No symbol matches")
                continue

            # Score: number of matching symbols
            score = len(matches)

            # Bonus: all closed symbols found in this trade
            if closed_normalized_set.issubset(current_normalized):
                score += 100  # Strong bonus for complete match
                logger.debug(f"     ✅ Complete match! Score: {score}")
            else:
                logger.debug(f"     ⚠️  Partial match. Score: {score}")

            candidates.append((trade_id, score, trade.opened_at))

        if not candidates:
            logger.warning(f"❌ NO MATCHING TRADE FOUND for symbols: {closed_symbols}")
            logger.warning(f"   This will create a NEW trade instead of linking to existing!")
            return None

        # Sort by score (desc), then by opened_at (desc for most recent)
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        best_trade_id = candidates[0][0]
        best_score = candidates[0][1]

        logger.info(f"✅ MATCHED trade {best_trade_id} for closing symbols {closed_symbols} (score: {best_score})")
        return best_trade_id

    def link_order_to_trade(self, trade_id: str, order_id: str,
                           new_opening_legs: Dict[str, LegInfo],
                           closed_symbols: List[str] = None):
        """Link a roll/close order to existing trade."""
        if trade_id not in self.trades:
            logger.warning(f"Trade {trade_id} not found for linking order {order_id}")
            return

        trade = self.trades[trade_id]

        # Add order_id to trade's order_ids list
        if order_id not in trade.order_ids:
            trade.order_ids.append(order_id)

        # Move closed legs to closed_legs (use normalized symbol matching)
        if closed_symbols:
            # Build mapping of normalized symbols to actual symbols in trade
            trade_symbol_map = {self._normalize_symbol(s): s for s in trade.current_legs.keys()}

            for closing_symbol in closed_symbols:
                normalized = self._normalize_symbol(closing_symbol)

                # Find the matching leg in trade using normalized symbol
                if normalized in trade_symbol_map:
                    actual_symbol = trade_symbol_map[normalized]
                    leg_info = trade.current_legs[actual_symbol]

                    # Store under the actual symbol from the trade
                    trade.closed_legs[actual_symbol] = {
                        'closed_at': datetime.now().isoformat(),
                        'closed_by_order': order_id,
                        'quantity': leg_info['quantity']
                    }
                    del trade.current_legs[actual_symbol]
                    logger.debug(f"  Closed leg {actual_symbol} (matched from {closing_symbol})")
                else:
                    logger.warning(f"  Could not find leg matching {closing_symbol} (normalized: {normalized}) in trade {trade_id}")

        # Add new opening legs to current_legs
        for symbol, leg_info in new_opening_legs.items():
            trade.current_legs[symbol] = asdict(leg_info)

        # Check if all legs are closed
        if not trade.current_legs:
            trade.status = "CLOSED"
            logger.info(f"Trade {trade_id} marked as CLOSED (all legs closed)")

        # Update order_to_trade_map
        self.order_to_trade_map[order_id] = trade_id

        # Persist to disk
        self.save()

        logger.info(f"Linked order {order_id} to trade {trade_id}")

    def get_trade_id_for_order(self, order_id: str) -> Optional[str]:
        """Lookup which trade an order belongs to."""
        return self.order_to_trade_map.get(order_id)

    def get_trade_orders(self, trade_id: str) -> List[str]:
        """Get all order IDs for a trade (chronological)."""
        if trade_id in self.trades:
            return self.trades[trade_id].order_ids.copy()
        return []

    def mark_trade_closed(self, trade_id: str):
        """Mark a trade as fully closed."""
        if trade_id in self.trades:
            self.trades[trade_id].status = "CLOSED"
            self.save()
            logger.info(f"Trade {trade_id} marked as CLOSED")

    def load(self):
        """Load trade tracking data from JSON."""
        if not self.storage_path.exists():
            logger.info(f"Trade tracking file not found at {self.storage_path}, starting fresh")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Deserialize trades
            trades_data = data.get('trades', {})
            for trade_id, trade_dict in trades_data.items():
                self.trades[trade_id] = TradeRecord(**trade_dict)

            # Deserialize order mapping
            self.order_to_trade_map = data.get('order_to_trade_map', {})

            logger.info(f"Loaded {len(self.trades)} trades from {self.storage_path}")

        except Exception as e:
            logger.error(f"Error loading trade tracking data: {e}", exc_info=True)
            logger.info("Starting with fresh trade tracking state")

    def save(self):
        """Save trade tracking data to JSON with atomic write and file locking."""
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize data
        data = {
            'trades': {k: asdict(v) for k, v in self.trades.items()},
            'order_to_trade_map': self.order_to_trade_map
        }

        # Use file locking to prevent corruption
        lock_path = str(self.storage_path) + '.lock'
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                # Atomic write: write to temp file, then rename
                temp_path = str(self.storage_path) + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)

                # Atomic rename
                Path(temp_path).replace(self.storage_path)

            logger.debug(f"Saved {len(self.trades)} trades to {self.storage_path}")

        except Exception as e:
            logger.error(f"Error saving trade tracking data: {e}", exc_info=True)


# Global singleton instance
trade_tracker = TradeTracker()
