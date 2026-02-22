"""
Trade Aggregator Service

Aggregates live metrics for trade groups:
- Fetches current quotes for all open positions
- Calculates P&L, profit capture %, day P&L
- Determines alert colors based on risk thresholds
- Computes theta-per-hour metrics
"""

from datetime import datetime, date, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import pytz
from loguru import logger

from data.storage.database import trades_db, market_data_db
from monitoring.services.trade_grouper import trade_grouper


class AlertColor(Enum):
    """Alert color codes for trade cards."""
    GREEN = "GREEN"      # 50%+ profit capture
    AMBER = "AMBER"      # Price within 25% of short strike
    ORANGE = "ORANGE"    # Any leg delta > 0.25
    RED = "RED"          # Short strike breached
    NONE = "NONE"        # No alerts


@dataclass
class AggregatedTradeGroup:
    """Aggregated metrics for a trade group."""
    trade_group_id: str
    underlying: str
    strategy_type: str
    expiration: Optional[date]
    days_to_expiration: int

    # Entry metrics
    entry_credit: float
    entry_debit: float
    max_profit: float
    max_loss: float

    # Current metrics
    current_mark: float
    unrealized_pnl: float
    day_pnl: float
    profit_capture_pct: float

    # Greeks
    total_delta: float
    total_theta: float
    total_gamma: float
    total_vega: float
    theta_per_hour: float

    # Alert status
    alert_color: AlertColor
    short_strike_breached: bool

    # Display helpers (fields with defaults must come after non-default fields)
    pnl_history: List[float] = field(default_factory=list)
    legs: List[Dict] = field(default_factory=list)

    # Underlying price for payoff diagrams
    underlying_price: Optional[float] = None

    # P&L from closed legs
    realized_pnl: float = 0

    # Per-contract entry/mark values (for display)
    entry_per_contract: float = 0  # Adjusted entry (open legs only for partial closes)
    original_entry_per_contract: float = 0  # Original entry at trade open
    mark_per_contract: float = 0
    num_contracts: int = 1

    # Commission and fees
    commission: float = 0  # Opening commission
    fees: float = 0  # Opening regulatory fees
    closing_commission: float = 0  # Closing commission (if closed)
    closing_fees: float = 0  # Closing regulatory fees (if closed)
    total_costs: float = 0  # Total commission + fees (opening + closing)

    # Timestamps
    opened_at: Optional[datetime] = None
    status: str = "OPEN"
    strategy_name: Optional[str] = None


class TradeAggregator:
    """
    Aggregates live metrics for all open trade groups.

    Features:
    - Calculates current mark prices from live quotes
    - Computes P&L metrics (total, day, profit capture %)
    - Determines alert colors based on risk thresholds
    - Tracks theta-per-hour (theta / 6.5 market hours)
    - Resets day P&L at 9:30 AM ET
    """

    # Market hours for theta calculation
    MARKET_HOURS_PER_DAY = 6.5  # 9:30 AM - 4:00 PM ET

    # Alert thresholds
    PROFIT_CAPTURE_GREEN_THRESHOLD = 50.0  # 50%+
    DELTA_ORANGE_THRESHOLD = 0.25
    STRIKE_PROXIMITY_AMBER_THRESHOLD = 0.25  # 25% of distance to strike

    # Timezone
    ET_TIMEZONE = pytz.timezone('US/Eastern')

    def __init__(self):
        """Initialize trade aggregator."""
        self._day_pnl_baseline: Dict[str, float] = {}  # trade_group_id -> baseline P&L
        self._last_reset_date: Optional[date] = None
        logger.info("TradeAggregator initialized")

    def get_aggregated_trades(self) -> List[AggregatedTradeGroup]:
        """
        Get all open trade groups with aggregated metrics.

        Returns:
            List of AggregatedTradeGroup with current metrics
        """
        # Check for day P&L reset
        self._check_day_pnl_reset()

        # Get all open trade groups
        groups = trade_grouper.get_open_trade_groups()

        aggregated = []
        for group in groups:
            try:
                agg = self._aggregate_trade_group(group)
                if agg:
                    aggregated.append(agg)
            except Exception as e:
                logger.error(f"Error aggregating trade group {group.get('trade_group_id')}: {e}")

        return aggregated

    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get global KPI metrics for the dashboard header.

        Returns:
            Dictionary with:
            - daily_realized: Total realized P&L today
            - open_unrealized: Total unrealized P&L
            - theta_pulse: Total theta per hour
            - market_time: Current market time (ET)
        """
        aggregated = self.get_aggregated_trades()

        # Calculate totals
        total_unrealized = sum(t.unrealized_pnl for t in aggregated)
        total_day_pnl = sum(t.day_pnl for t in aggregated)
        total_theta_per_hour = sum(t.theta_per_hour for t in aggregated)

        # Get realized P&L from closed trades today
        daily_realized = self._get_daily_realized_pnl()

        # Get market time
        now_et = datetime.now(self.ET_TIMEZONE)
        market_time = now_et.strftime('%H:%M:%S ET')

        return {
            'daily_realized': daily_realized,
            'open_unrealized': total_unrealized,
            'theta_pulse': total_theta_per_hour,
            'market_time': market_time,
            'day_pnl': total_day_pnl,
            'num_open_trades': len(aggregated)
        }

    def _aggregate_trade_group(self, group: Dict) -> Optional[AggregatedTradeGroup]:
        """
        Aggregate metrics for a single trade group.

        Args:
            group: Trade group dictionary from database

        Returns:
            AggregatedTradeGroup with current metrics
        """
        trade_group_id = group['trade_group_id']

        # Get legs
        legs = trade_grouper.get_trade_group_legs(trade_group_id)
        if not legs:
            return None

        # Get current quotes for all legs
        legs_with_quotes = self._get_leg_quotes(legs)

        # Calculate current mark
        current_mark = self._calculate_current_mark(legs_with_quotes)

        # Calculate P&L
        entry_value = group.get('entry_credit', 0) - group.get('entry_debit', 0)
        exit_value = current_mark  # What we'd receive/pay to close

        # For credit trades: profit = entry_credit - exit_cost
        # For debit trades: profit = exit_value - entry_debit
        if group.get('entry_credit', 0) > 0:
            unrealized_pnl = group['entry_credit'] - abs(current_mark)
        else:
            unrealized_pnl = abs(current_mark) - group.get('entry_debit', 0)

        # Calculate day P&L
        day_pnl = self._calculate_day_pnl(trade_group_id, unrealized_pnl)

        # Calculate profit capture percentage
        max_profit = group.get('max_profit') or 0
        if max_profit and max_profit > 0:
            profit_capture_pct = (unrealized_pnl / max_profit) * 100
        else:
            profit_capture_pct = 0.0

        # Aggregate Greeks
        total_delta = sum(leg.get('delta', 0) or 0 for leg in legs_with_quotes)
        total_theta = sum(leg.get('theta', 0) or 0 for leg in legs_with_quotes)
        total_gamma = sum(leg.get('gamma', 0) or 0 for leg in legs_with_quotes)
        total_vega = sum(leg.get('vega', 0) or 0 for leg in legs_with_quotes)

        # Theta per hour
        theta_per_hour = abs(total_theta) / self.MARKET_HOURS_PER_DAY if total_theta else 0

        # Calculate DTE
        expiration = group.get('expiration')
        if expiration:
            if isinstance(expiration, str):
                expiration = datetime.strptime(expiration, '%Y-%m-%d').date()
            elif isinstance(expiration, datetime):
                expiration = expiration.date()
            days_to_expiration = (expiration - date.today()).days
        else:
            days_to_expiration = 0

        # Determine alert color
        alert_color, strike_breached = self._determine_alert_color(
            legs_with_quotes, profit_capture_pct, group['underlying']
        )

        # Get P&L history
        pnl_history = self._get_pnl_history(trade_group_id)

        # Update database with current metrics
        self._update_database_metrics(
            trade_group_id, current_mark, unrealized_pnl, day_pnl,
            profit_capture_pct, days_to_expiration,
            total_delta, total_theta, total_gamma, total_vega,
            alert_color.value, strike_breached, pnl_history
        )

        return AggregatedTradeGroup(
            trade_group_id=trade_group_id,
            underlying=group['underlying'],
            strategy_type=group['strategy_type'],
            expiration=expiration,
            days_to_expiration=days_to_expiration,
            entry_credit=group.get('entry_credit', 0),
            entry_debit=group.get('entry_debit', 0),
            max_profit=max_profit,
            max_loss=group.get('max_loss') or 0,
            current_mark=current_mark,
            unrealized_pnl=unrealized_pnl,
            day_pnl=day_pnl,
            profit_capture_pct=profit_capture_pct,
            total_delta=total_delta,
            total_theta=total_theta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            theta_per_hour=theta_per_hour,
            alert_color=alert_color,
            short_strike_breached=strike_breached,
            pnl_history=pnl_history,
            legs=legs_with_quotes,
            opened_at=group.get('opened_at'),
            status=group.get('status', 'OPEN'),
            strategy_name=group.get('strategy_name')
        )

    def _get_leg_quotes(self, legs: List[Dict]) -> List[Dict]:
        """
        Get current quotes for all legs.

        Args:
            legs: List of leg dictionaries

        Returns:
            Legs with current quote data added
        """
        legs_with_quotes = []

        for leg in legs:
            leg_copy = dict(leg)

            # Try to get current quote from database
            symbol = leg['symbol']
            quote = self._get_latest_quote(symbol)

            if quote:
                leg_copy['current_bid'] = quote.get('bid')
                leg_copy['current_ask'] = quote.get('ask')
                leg_copy['current_mark'] = (quote.get('bid', 0) + quote.get('ask', 0)) / 2
                leg_copy['current_last'] = quote.get('last')
                leg_copy['delta'] = quote.get('delta')
                leg_copy['gamma'] = quote.get('gamma')
                leg_copy['theta'] = quote.get('theta')
                leg_copy['vega'] = quote.get('vega')
                leg_copy['implied_volatility'] = quote.get('implied_volatility')
            else:
                # Use entry price as fallback
                leg_copy['current_mark'] = leg.get('entry_price', 0)
                leg_copy['delta'] = leg.get('delta', 0)
                leg_copy['theta'] = leg.get('theta', 0)
                leg_copy['gamma'] = leg.get('gamma', 0)
                leg_copy['vega'] = leg.get('vega', 0)

            legs_with_quotes.append(leg_copy)

        return legs_with_quotes

    def _get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Option or stock symbol

        Returns:
            Quote dictionary or None
        """
        # For options, check option_chains table
        # For stocks, check quotes table

        try:
            # Try option chains first
            query = """
                SELECT bid, ask, last, delta, gamma, theta, vega, implied_volatility
                FROM option_chains
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = market_data_db.execute(query, (symbol,))
            df = result.fetchdf()

            if not df.empty:
                return df.iloc[0].to_dict()

            # Try quotes table for stocks
            query = """
                SELECT close as last, close as bid, close as ask
                FROM quotes
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = market_data_db.execute(query, (symbol,))
            df = result.fetchdf()

            if not df.empty:
                return df.iloc[0].to_dict()

        except Exception as e:
            logger.debug(f"Could not get quote for {symbol}: {e}")

        return None

    def _calculate_current_mark(self, legs: List[Dict]) -> float:
        """
        Calculate the current mark value for a trade group.

        For closing the position:
        - Short legs: we'd pay the ask to close
        - Long legs: we'd receive the bid to close

        Args:
            legs: List of legs with current quotes

        Returns:
            Net mark value (negative = we'd pay to close, positive = we'd receive)
        """
        total = 0.0

        for leg in legs:
            quantity = leg.get('quantity', 0)
            mark = leg.get('current_mark', leg.get('entry_price', 0))

            if leg.get('option_type'):
                # Options: multiply by 100
                leg_value = quantity * mark * 100
            else:
                # Stock
                leg_value = quantity * mark

            # Flip sign because we're calculating closing value
            # Short position (neg quantity): closing = buying = pay ask (positive cost)
            # Long position (pos quantity): closing = selling = receive bid (negative cost)
            total -= leg_value

        return total

    def _calculate_day_pnl(self, trade_group_id: str, current_pnl: float) -> float:
        """
        Calculate P&L since market open (9:30 AM ET).

        Args:
            trade_group_id: Trade group ID
            current_pnl: Current unrealized P&L

        Returns:
            Day P&L
        """
        baseline = self._day_pnl_baseline.get(trade_group_id)

        if baseline is None:
            # First time seeing this trade today, set baseline
            self._day_pnl_baseline[trade_group_id] = current_pnl
            return 0.0

        return current_pnl - baseline

    def _check_day_pnl_reset(self) -> None:
        """Check if we need to reset day P&L at 9:30 AM ET."""
        now_et = datetime.now(self.ET_TIMEZONE)
        today = now_et.date()
        market_open = time(9, 30)

        # Reset if:
        # 1. New day
        # 2. Or we're past 9:30 AM and haven't reset today
        if self._last_reset_date != today and now_et.time() >= market_open:
            logger.info("Resetting day P&L baselines at market open")
            self._day_pnl_baseline.clear()
            self._last_reset_date = today

    def _determine_alert_color(
        self,
        legs: List[Dict],
        profit_capture_pct: float,
        underlying: str
    ) -> Tuple[AlertColor, bool]:
        """
        Determine alert color based on risk thresholds.

        Priority (highest to lowest):
        1. RED: Short strike breached
        2. GREEN: 50%+ profit capture
        3. ORANGE: Any leg delta > 0.25
        4. AMBER: Price within 25% of short strike
        5. NONE: No alerts

        Args:
            legs: List of legs with current quotes
            profit_capture_pct: Current profit capture percentage
            underlying: Underlying symbol

        Returns:
            Tuple of (AlertColor, strike_breached)
        """
        # Get underlying price
        underlying_price = self._get_underlying_price(underlying)

        # Check for breached short strikes
        short_legs = [leg for leg in legs if leg.get('quantity', 0) < 0 and leg.get('option_type')]
        strike_breached = False

        for leg in short_legs:
            strike = leg.get('strike')
            option_type = leg.get('option_type')

            if strike and underlying_price:
                if option_type == 'CALL' and underlying_price > strike:
                    strike_breached = True
                    break
                elif option_type == 'PUT' and underlying_price < strike:
                    strike_breached = True
                    break

        if strike_breached:
            return AlertColor.RED, True

        # Check for 50%+ profit
        if profit_capture_pct >= self.PROFIT_CAPTURE_GREEN_THRESHOLD:
            return AlertColor.GREEN, False

        # Check for high delta
        for leg in legs:
            delta = abs(leg.get('delta', 0) or 0)
            if delta > self.DELTA_ORANGE_THRESHOLD:
                return AlertColor.ORANGE, False

        # Check for price near short strike
        if underlying_price:
            for leg in short_legs:
                strike = leg.get('strike')
                if strike:
                    distance = abs(underlying_price - strike)
                    threshold_distance = strike * self.STRIKE_PROXIMITY_AMBER_THRESHOLD
                    if distance < threshold_distance:
                        return AlertColor.AMBER, False

        return AlertColor.NONE, False

    def _get_underlying_price(self, underlying: str) -> Optional[float]:
        """
        Get current price of underlying.

        Args:
            underlying: Underlying symbol

        Returns:
            Current price or None
        """
        try:
            query = """
                SELECT close
                FROM quotes
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = market_data_db.execute(query, (underlying,))
            df = result.fetchdf()

            if not df.empty:
                return df.iloc[0]['close']

        except Exception as e:
            logger.debug(f"Could not get underlying price for {underlying}: {e}")

        return None

    def _get_daily_realized_pnl(self) -> float:
        """
        Get total realized P&L from trades closed today.

        Returns:
            Total realized P&L today
        """
        try:
            today = date.today()
            query = """
                SELECT COALESCE(SUM(realized_pnl), 0) as total
                FROM trade_groups
                WHERE DATE(closed_at) = ?
            """
            result = trades_db.execute(query, (today,))
            df = result.fetchdf()

            if not df.empty:
                return df.iloc[0]['total'] or 0.0

        except Exception as e:
            logger.debug(f"Could not get daily realized P&L: {e}")

        return 0.0

    def _get_pnl_history(self, trade_group_id: str) -> List[float]:
        """
        Get P&L history for sparkline.

        Args:
            trade_group_id: Trade group ID

        Returns:
            List of P&L values for sparkline
        """
        try:
            query = """
                SELECT pnl_history
                FROM trade_groups
                WHERE trade_group_id = ?
            """
            result = trades_db.execute(query, (trade_group_id,))
            df = result.fetchdf()

            if not df.empty and df.iloc[0]['pnl_history']:
                history_json = df.iloc[0]['pnl_history']
                if isinstance(history_json, str):
                    return json.loads(history_json)
                return history_json or []

        except Exception as e:
            logger.debug(f"Could not get P&L history for {trade_group_id}: {e}")

        return []

    def _update_database_metrics(
        self,
        trade_group_id: str,
        current_mark: float,
        unrealized_pnl: float,
        day_pnl: float,
        profit_capture_pct: float,
        days_to_expiration: int,
        total_delta: float,
        total_theta: float,
        total_gamma: float,
        total_vega: float,
        alert_color: str,
        strike_breached: bool,
        pnl_history: List[float]
    ) -> None:
        """
        Update trade group metrics in database.

        Args:
            trade_group_id: Trade group ID
            ... other metrics
        """
        # Append current P&L to history (keep last 50 points)
        pnl_history = pnl_history[-49:] + [unrealized_pnl]

        query = """
            UPDATE trade_groups SET
                current_mark = ?,
                unrealized_pnl = ?,
                day_pnl = ?,
                profit_capture_pct = ?,
                days_to_expiration = ?,
                total_delta = ?,
                total_theta = ?,
                total_gamma = ?,
                total_vega = ?,
                alert_color = ?,
                short_strike_breached = ?,
                pnl_history = ?
            WHERE trade_group_id = ?
        """

        values = (
            current_mark,
            unrealized_pnl,
            day_pnl,
            profit_capture_pct,
            days_to_expiration,
            total_delta,
            total_theta,
            total_gamma,
            total_vega,
            alert_color,
            strike_breached,
            json.dumps(pnl_history),
            trade_group_id
        )

        try:
            trades_db.execute(query, values)
        except Exception as e:
            logger.debug(f"Could not update metrics for {trade_group_id}: {e}")


# Global instance
trade_aggregator = TradeAggregator()
