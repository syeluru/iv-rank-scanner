"""
Trade Grouper Service

Groups individual order executions into cohesive multi-leg trade structures.
Identifies strategy types (Iron Condor, Vertical Spread, Straddle, etc.)
and calculates max profit/loss for each grouped trade.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from loguru import logger

from data.storage.database import trades_db, positions_db


class StrategyType(Enum):
    """Recognized multi-leg strategy types."""
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    VERTICAL_SPREAD = "VERTICAL_SPREAD"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    BUTTERFLY = "BUTTERFLY"
    CONDOR = "CONDOR"
    COVERED_CALL = "COVERED_CALL"
    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    SINGLE_LEG = "SINGLE_LEG"
    UNKNOWN = "UNKNOWN"


@dataclass
class TradeLeg:
    """Individual leg of a trade."""
    symbol: str
    underlying: str
    option_type: Optional[str]  # 'CALL', 'PUT', or None for stock
    strike: Optional[float]
    expiration: Optional[datetime]
    quantity: int  # Positive for long, negative for short
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_timestamp: datetime
    order_id: Optional[str] = None


@dataclass
class TradeGroup:
    """Grouped multi-leg trade."""
    trade_group_id: str
    underlying: str
    strategy_type: StrategyType
    legs: List[TradeLeg] = field(default_factory=list)
    opened_at: datetime = field(default_factory=datetime.now)
    expiration: Optional[datetime] = None

    # Computed metrics
    entry_credit: float = 0.0
    entry_debit: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0

    # Metadata
    strategy_name: Optional[str] = None
    notes: Optional[str] = None


class TradeGrouper:
    """
    Groups individual trades into cohesive multi-leg structures.

    Grouping Logic:
    - Orders within 2-second window on same underlying are grouped
    - Rolling exception: BTC + STO within 5 seconds = new trade object
    - Identifies strategy type based on leg configuration
    """

    # Grouping time windows
    DEFAULT_GROUPING_WINDOW_SECONDS = 2
    ROLL_GROUPING_WINDOW_SECONDS = 5

    def __init__(self):
        """Initialize trade grouper."""
        self._pending_legs: Dict[str, List[TradeLeg]] = {}  # underlying -> legs
        self._pending_timestamps: Dict[str, datetime] = {}  # underlying -> first leg timestamp
        logger.info("TradeGrouper initialized")

    def process_trade(
        self,
        symbol: str,
        underlying: str,
        option_type: Optional[str],
        strike: Optional[float],
        expiration: Optional[datetime],
        quantity: int,
        price: float,
        action: str,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> Optional[TradeGroup]:
        """
        Process a single trade execution and attempt to group it.

        Args:
            symbol: Full option symbol or underlying for stock
            underlying: Underlying ticker (e.g., 'SPX', 'AAPL')
            option_type: 'CALL', 'PUT', or None for stock
            strike: Option strike price
            expiration: Option expiration date
            quantity: Number of contracts (always positive)
            price: Execution price per contract
            action: Trade action ('BTO', 'STO', 'BTC', 'STC')
            timestamp: Execution timestamp (defaults to now)
            order_id: Optional order reference
            strategy_name: Optional strategy attribution

        Returns:
            TradeGroup if grouping window closed, None if still collecting
        """
        timestamp = timestamp or datetime.now()

        # Determine side from action
        side = 'SHORT' if action in ['STO', 'STC'] else 'LONG'
        signed_quantity = -quantity if action in ['STO', 'BTC'] else quantity

        leg = TradeLeg(
            symbol=symbol,
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            quantity=signed_quantity,
            side=side,
            entry_price=price,
            entry_timestamp=timestamp,
            order_id=order_id
        )

        # Check for roll detection (BTC followed by STO)
        is_roll = self._detect_roll(underlying, action, timestamp)

        if is_roll:
            # Finalize existing group first, then start new one
            existing_group = self._finalize_group(underlying, strategy_name)
            self._pending_legs[underlying] = [leg]
            self._pending_timestamps[underlying] = timestamp
            return existing_group

        # Check if we should start a new group or add to existing
        if underlying not in self._pending_legs:
            # Start new pending group
            self._pending_legs[underlying] = [leg]
            self._pending_timestamps[underlying] = timestamp
            return None

        # Check if within grouping window
        first_timestamp = self._pending_timestamps[underlying]
        window = timedelta(seconds=self.DEFAULT_GROUPING_WINDOW_SECONDS)

        if timestamp - first_timestamp <= window:
            # Add to existing pending group
            self._pending_legs[underlying].append(leg)
            return None
        else:
            # Window closed - finalize existing group and start new
            completed_group = self._finalize_group(underlying, strategy_name)
            self._pending_legs[underlying] = [leg]
            self._pending_timestamps[underlying] = timestamp
            return completed_group

    def flush_pending(self, underlying: Optional[str] = None, strategy_name: Optional[str] = None) -> List[TradeGroup]:
        """
        Force finalization of pending groups.

        Args:
            underlying: Specific underlying to flush, or None for all
            strategy_name: Strategy attribution for groups

        Returns:
            List of finalized TradeGroups
        """
        groups = []

        if underlying:
            if underlying in self._pending_legs:
                group = self._finalize_group(underlying, strategy_name)
                if group:
                    groups.append(group)
        else:
            # Flush all
            underlyings = list(self._pending_legs.keys())
            for u in underlyings:
                group = self._finalize_group(u, strategy_name)
                if group:
                    groups.append(group)

        return groups

    def _detect_roll(self, underlying: str, action: str, timestamp: datetime) -> bool:
        """
        Detect if this trade is part of a roll (close + open).

        A roll is detected when:
        - BTC (buy to close) followed by STO (sell to open) within 5 seconds
        - On the same underlying

        Args:
            underlying: Underlying symbol
            action: Current trade action
            timestamp: Current trade timestamp

        Returns:
            True if this appears to be a new position after a roll
        """
        if action != 'STO' or underlying not in self._pending_legs:
            return False

        # Check if previous leg was a BTC
        pending = self._pending_legs[underlying]
        if not pending:
            return False

        # Look for BTC in recent pending legs
        for leg in reversed(pending):
            if leg.side == 'LONG' and leg.quantity > 0:  # BTC results in positive quantity (buying back short)
                time_diff = timestamp - leg.entry_timestamp
                if time_diff <= timedelta(seconds=self.ROLL_GROUPING_WINDOW_SECONDS):
                    return True

        return False

    def _finalize_group(self, underlying: str, strategy_name: Optional[str] = None) -> Optional[TradeGroup]:
        """
        Finalize a pending group into a TradeGroup.

        Args:
            underlying: Underlying to finalize
            strategy_name: Strategy attribution

        Returns:
            Finalized TradeGroup or None if no pending legs
        """
        if underlying not in self._pending_legs:
            return None

        legs = self._pending_legs.pop(underlying)
        self._pending_timestamps.pop(underlying, None)

        if not legs:
            return None

        # Create trade group
        trade_group_id = f"tg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Identify strategy type
        strategy_type = self._identify_strategy(legs)

        # Find expiration (use earliest if multiple)
        expirations = [leg.expiration for leg in legs if leg.expiration]
        expiration = min(expirations) if expirations else None

        group = TradeGroup(
            trade_group_id=trade_group_id,
            underlying=underlying,
            strategy_type=strategy_type,
            legs=legs,
            opened_at=min(leg.entry_timestamp for leg in legs),
            expiration=expiration,
            strategy_name=strategy_name
        )

        # Calculate entry credit/debit and max profit/loss
        self._calculate_metrics(group)

        # Save to database
        self._save_to_database(group)

        logger.info(
            f"Created trade group: {trade_group_id} ({strategy_type.value}) "
            f"with {len(legs)} legs on {underlying}"
        )

        return group

    def _identify_strategy(self, legs: List[TradeLeg]) -> StrategyType:
        """
        Identify the strategy type based on leg configuration.

        Args:
            legs: List of trade legs

        Returns:
            Identified StrategyType
        """
        option_legs = [leg for leg in legs if leg.option_type]
        stock_legs = [leg for leg in legs if not leg.option_type]

        # Single leg
        if len(legs) == 1:
            if stock_legs:
                return StrategyType.SINGLE_LEG
            elif option_legs[0].option_type == 'PUT' and option_legs[0].side == 'SHORT':
                return StrategyType.CASH_SECURED_PUT
            return StrategyType.SINGLE_LEG

        # Covered call: long stock + short call
        if len(stock_legs) == 1 and len(option_legs) == 1:
            if stock_legs[0].quantity > 0 and option_legs[0].option_type == 'CALL' and option_legs[0].side == 'SHORT':
                return StrategyType.COVERED_CALL

        # All option strategies from here
        if not option_legs:
            return StrategyType.UNKNOWN

        calls = [leg for leg in option_legs if leg.option_type == 'CALL']
        puts = [leg for leg in option_legs if leg.option_type == 'PUT']

        # Check expirations
        expirations = set(leg.expiration for leg in option_legs if leg.expiration)
        same_expiration = len(expirations) <= 1

        # Straddle: 1 call + 1 put, same strike, same expiration
        if len(calls) == 1 and len(puts) == 1 and same_expiration:
            if calls[0].strike == puts[0].strike:
                return StrategyType.STRADDLE

        # Strangle: 1 call + 1 put, different strikes, same expiration
        if len(calls) == 1 and len(puts) == 1 and same_expiration:
            if calls[0].strike != puts[0].strike:
                return StrategyType.STRANGLE

        # Vertical spread: 2 options same type, different strikes, same exp
        if len(option_legs) == 2 and same_expiration:
            if len(calls) == 2 or len(puts) == 2:
                return StrategyType.VERTICAL_SPREAD

        # Iron Condor: 4 legs - put spread + call spread
        if len(option_legs) == 4 and same_expiration:
            if len(calls) == 2 and len(puts) == 2:
                call_strikes = sorted([c.strike for c in calls])
                put_strikes = sorted([p.strike for p in puts])
                # Verify it's a proper iron condor structure
                if put_strikes[1] < call_strikes[0]:
                    return StrategyType.IRON_CONDOR

        # Iron Butterfly: 4 legs with shared middle strike
        if len(option_legs) == 4 and same_expiration:
            if len(calls) == 2 and len(puts) == 2:
                call_strikes = set(c.strike for c in calls)
                put_strikes = set(p.strike for p in puts)
                if len(call_strikes & put_strikes) == 1:  # Shared middle strike
                    return StrategyType.IRON_BUTTERFLY

        # Butterfly: 3 strikes, 4 legs same type
        if len(option_legs) == 4 and same_expiration:
            if len(calls) == 4 or len(puts) == 4:
                return StrategyType.BUTTERFLY

        # Calendar spread: same strike, different expirations
        if len(option_legs) == 2 and not same_expiration:
            if option_legs[0].strike == option_legs[1].strike:
                return StrategyType.CALENDAR_SPREAD

        # Diagonal spread: different strikes, different expirations
        if len(option_legs) == 2 and not same_expiration:
            if option_legs[0].strike != option_legs[1].strike:
                return StrategyType.DIAGONAL_SPREAD

        return StrategyType.UNKNOWN

    def _calculate_metrics(self, group: TradeGroup) -> None:
        """
        Calculate entry credit/debit and max profit/loss.

        Args:
            group: TradeGroup to calculate metrics for
        """
        # Calculate net premium
        net_premium = 0.0
        for leg in group.legs:
            # For options: quantity * price * 100 (multiplier)
            # Positive quantity (long) = debit, Negative quantity (short) = credit
            if leg.option_type:
                leg_value = leg.quantity * leg.entry_price * 100
            else:
                # Stock: quantity * price
                leg_value = leg.quantity * leg.entry_price
            net_premium += leg_value

        # Credit if net_premium < 0 (we received money), debit if > 0 (we paid)
        if net_premium < 0:
            group.entry_credit = abs(net_premium)
            group.entry_debit = 0.0
        else:
            group.entry_credit = 0.0
            group.entry_debit = net_premium

        # Calculate max profit/loss based on strategy
        self._calculate_max_profit_loss(group)

    def _calculate_max_profit_loss(self, group: TradeGroup) -> None:
        """
        Calculate maximum profit and loss based on strategy type.

        Args:
            group: TradeGroup to calculate for
        """
        strategy = group.strategy_type
        legs = group.legs
        option_legs = [leg for leg in legs if leg.option_type]

        if strategy == StrategyType.CASH_SECURED_PUT:
            # Max profit = premium received
            # Max loss = strike * 100 - premium
            leg = option_legs[0]
            group.max_profit = group.entry_credit
            group.max_loss = (leg.strike * 100) - group.entry_credit

        elif strategy == StrategyType.COVERED_CALL:
            # Max profit = premium + (call_strike - stock_entry) * 100
            # Max loss = stock_entry * 100 - premium
            stock_leg = [leg for leg in legs if not leg.option_type][0]
            call_leg = option_legs[0]
            stock_cost = stock_leg.entry_price * abs(stock_leg.quantity)
            premium = group.entry_credit
            group.max_profit = premium + (call_leg.strike - stock_leg.entry_price) * 100
            group.max_loss = stock_cost - premium

        elif strategy == StrategyType.VERTICAL_SPREAD:
            # For credit spreads: max profit = credit, max loss = width - credit
            # For debit spreads: max profit = width - debit, max loss = debit
            strikes = sorted([leg.strike for leg in option_legs])
            width = (strikes[1] - strikes[0]) * 100

            if group.entry_credit > 0:  # Credit spread
                group.max_profit = group.entry_credit
                group.max_loss = width - group.entry_credit
            else:  # Debit spread
                group.max_profit = width - group.entry_debit
                group.max_loss = group.entry_debit

        elif strategy == StrategyType.IRON_CONDOR:
            # Max profit = net credit
            # Max loss = width of wider spread - credit
            calls = [leg for leg in option_legs if leg.option_type == 'CALL']
            puts = [leg for leg in option_legs if leg.option_type == 'PUT']

            call_strikes = sorted([c.strike for c in calls])
            put_strikes = sorted([p.strike for p in puts])

            call_width = (call_strikes[1] - call_strikes[0]) * 100
            put_width = (put_strikes[1] - put_strikes[0]) * 100
            max_width = max(call_width, put_width)

            group.max_profit = group.entry_credit
            group.max_loss = max_width - group.entry_credit

        elif strategy == StrategyType.STRADDLE:
            short_legs = [leg for leg in option_legs if leg.quantity < 0]
            if short_legs:  # Short straddle
                group.max_profit = group.entry_credit
                group.max_loss = float('inf')  # Unlimited on call side
            else:  # Long straddle
                group.max_profit = float('inf')
                group.max_loss = group.entry_debit

        elif strategy == StrategyType.STRANGLE:
            short_legs = [leg for leg in option_legs if leg.quantity < 0]
            if short_legs:  # Short strangle
                group.max_profit = group.entry_credit
                group.max_loss = float('inf')
            else:  # Long strangle
                group.max_profit = float('inf')
                group.max_loss = group.entry_debit

        elif strategy == StrategyType.SINGLE_LEG:
            if option_legs:
                leg = option_legs[0]
                if leg.quantity < 0:  # Short option
                    group.max_profit = group.entry_credit
                    if leg.option_type == 'PUT':
                        group.max_loss = (leg.strike * 100) - group.entry_credit
                    else:  # Call
                        group.max_loss = float('inf')
                else:  # Long option
                    group.max_profit = float('inf') if leg.option_type == 'CALL' else (leg.strike * 100) - group.entry_debit
                    group.max_loss = group.entry_debit
            else:
                # Stock position
                stock_leg = legs[0]
                if stock_leg.quantity > 0:  # Long stock
                    group.max_profit = float('inf')
                    group.max_loss = stock_leg.entry_price * stock_leg.quantity
                else:  # Short stock
                    group.max_profit = stock_leg.entry_price * abs(stock_leg.quantity)
                    group.max_loss = float('inf')

        else:
            # For unknown/complex strategies, estimate based on premium
            group.max_profit = group.entry_credit if group.entry_credit > 0 else 0
            group.max_loss = group.entry_debit if group.entry_debit > 0 else 0

    def _save_to_database(self, group: TradeGroup) -> None:
        """
        Save trade group and legs to database.

        Args:
            group: TradeGroup to save
        """
        # Insert trade group
        group_query = """
            INSERT INTO trade_groups (
                trade_group_id, underlying, strategy_type, status, opened_at,
                expiration, entry_credit, entry_debit, max_profit, max_loss,
                strategy_name, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        group_values = (
            group.trade_group_id,
            group.underlying,
            group.strategy_type.value,
            'OPEN',
            group.opened_at,
            group.expiration,
            group.entry_credit,
            group.entry_debit,
            group.max_profit if group.max_profit != float('inf') else None,
            group.max_loss if group.max_loss != float('inf') else None,
            group.strategy_name,
            group.notes
        )

        try:
            trades_db.execute(group_query, group_values)
        except Exception as e:
            logger.error(f"Error saving trade group: {e}")
            raise

        # Insert legs
        leg_query = """
            INSERT INTO trade_group_legs (
                leg_id, trade_group_id, symbol, underlying, option_type,
                strike, expiration, quantity, side, entry_price, entry_timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for i, leg in enumerate(group.legs):
            leg_id = f"{group.trade_group_id}_leg{i}"
            leg_values = (
                leg_id,
                group.trade_group_id,
                leg.symbol,
                leg.underlying,
                leg.option_type,
                leg.strike,
                leg.expiration,
                leg.quantity,
                leg.side,
                leg.entry_price,
                leg.entry_timestamp,
                'OPEN'
            )

            try:
                trades_db.execute(leg_query, leg_values)
            except Exception as e:
                logger.error(f"Error saving trade leg: {e}")
                raise

    def get_open_trade_groups(self, underlying: Optional[str] = None) -> List[Dict]:
        """
        Get all open trade groups from database.

        Args:
            underlying: Optional filter by underlying

        Returns:
            List of trade group dictionaries
        """
        query = """
            SELECT * FROM trade_groups
            WHERE status = 'OPEN'
        """
        params = []

        if underlying:
            query += " AND underlying = ?"
            params.append(underlying)

        query += " ORDER BY opened_at DESC"

        try:
            result = trades_db.execute(query, tuple(params) if params else None)
            df = result.fetchdf()
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            logger.error(f"Error fetching trade groups: {e}")
            return []

    def get_trade_group_legs(self, trade_group_id: str) -> List[Dict]:
        """
        Get all legs for a trade group.

        Args:
            trade_group_id: Trade group ID

        Returns:
            List of leg dictionaries
        """
        query = """
            SELECT * FROM trade_group_legs
            WHERE trade_group_id = ?
            ORDER BY strike ASC
        """

        try:
            result = trades_db.execute(query, (trade_group_id,))
            df = result.fetchdf()
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            logger.error(f"Error fetching trade group legs: {e}")
            return []


# Global instance
trade_grouper = TradeGrouper()
