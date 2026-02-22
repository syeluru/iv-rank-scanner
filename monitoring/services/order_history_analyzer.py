"""
Order History Analyzer Service

Fetches orders from Schwab API, calculates P&L, and caches results in DuckDB.
Provides daily P&L summaries for the History tab dashboard.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import pandas as pd

from execution.broker_api.schwab_client import SchwabClient
from monitoring.services.pnl_calculator import (
    extract_fill_prices,
    calculate_order_pnl,
    classify_order_type,
    get_order_underlying,
    get_order_strategy_type,
    OrderType
)
from data.storage.database import DatabaseManager
from config.settings import settings

logger = logging.getLogger(__name__)


class OrderHistoryAnalyzer:
    """
    Analyzes order history and calculates realized P&L.

    Fetches orders from Schwab API, processes them to calculate P&L,
    and caches results in DuckDB for fast dashboard queries.
    """

    def __init__(self, lookback_days: int = 14):
        """
        Initialize the analyzer.

        Args:
            lookback_days: Number of days of history to analyze (default: 14)
        """
        self.lookback_days = lookback_days
        self.schwab_client = SchwabClient()
        self.db = DatabaseManager(settings.trades_db_path)

        # Cache of entry prices for matching closing orders
        self._entry_price_cache: Dict[str, float] = {}

    async def fetch_orders(self) -> List[dict]:
        """
        Fetch orders from Schwab API for the lookback period.

        Returns:
            List of order dictionaries
        """
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self.lookback_days)

        logger.info(f"Fetching orders from {from_date.date()} to {to_date.date()}")

        try:
            orders = await self.schwab_client.get_orders_for_account(
                status='FILLED',
                from_entered_datetime=from_date,
                to_entered_datetime=to_date
            )

            logger.info(f"Fetched {len(orders)} filled orders")
            return orders

        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    def _build_entry_price_cache(self, orders: List[dict]) -> None:
        """
        Build cache of entry prices from opening orders.

        Args:
            orders: List of all orders (sorted by time)
        """
        self._entry_price_cache.clear()

        for order in orders:
            order_type = classify_order_type(order)

            if order_type == OrderType.OPENING:
                # Extract fill prices and store
                fill_prices = extract_fill_prices(order)
                self._entry_price_cache.update(fill_prices)

                logger.debug(
                    f"Cached entry prices from order {order.get('orderId')}: "
                    f"{len(fill_prices)} legs"
                )

    def _get_order_date(self, order: dict) -> date:
        """
        Extract order date, using closeTime for consistency.

        Args:
            order: Schwab order dictionary

        Returns:
            Order date
        """
        # Use closeTime (when order was filled) not enteredTime
        close_time_str = order.get('closeTime')

        if close_time_str:
            # Parse ISO format: "2024-03-15T14:30:45+0000"
            dt = datetime.fromisoformat(close_time_str.replace('+0000', ''))
            return dt.date()
        else:
            # Fall back to entered time or today
            entered_time_str = order.get('enteredTime')
            if entered_time_str:
                dt = datetime.fromisoformat(entered_time_str.replace('+0000', ''))
                return dt.date()
            else:
                return datetime.now().date()

    def _get_filled_timestamp(self, order: dict) -> datetime:
        """
        Extract filled timestamp from order.

        Args:
            order: Schwab order dictionary

        Returns:
            Filled timestamp
        """
        close_time_str = order.get('closeTime')

        if close_time_str:
            return datetime.fromisoformat(close_time_str.replace('+0000', ''))
        else:
            return datetime.now()

    def _process_orders(self, orders: List[dict]) -> pd.DataFrame:
        """
        Process orders and calculate P&L for each.

        Args:
            orders: List of order dictionaries

        Returns:
            DataFrame with columns: order_id, order_date, underlying, strategy_type,
                                   realized_pnl, commission, fees, order_type,
                                   leg_details, filled_at, has_entry_prices
        """
        # Sort by close time to process in chronological order
        sorted_orders = sorted(
            orders,
            key=lambda o: o.get('closeTime', o.get('enteredTime', ''))
        )

        # Build entry price cache first
        self._build_entry_price_cache(sorted_orders)

        records = []

        for order in sorted_orders:
            order_id = order.get('orderId', '')
            order_type = classify_order_type(order)

            # Calculate P&L for closing orders
            if order_type == OrderType.CLOSING:
                realized_pnl, details = calculate_order_pnl(
                    order,
                    entry_prices=self._entry_price_cache
                )
            else:
                # Opening orders have no realized P&L yet
                realized_pnl = 0.0
                details = {
                    'order_type': order_type,
                    'realized_pnl': 0.0,
                    'commission': 0.0,
                    'fees': 0.0,
                    'legs': [],
                    'has_entry_prices': False
                }

            record = {
                'order_id': order_id,
                'order_date': self._get_order_date(order),
                'underlying': get_order_underlying(order),
                'strategy_type': get_order_strategy_type(order),
                'realized_pnl': realized_pnl,
                'commission': details.get('commission', 0.0),
                'fees': details.get('fees', 0.0),
                'order_type': order_type,
                'leg_details': json.dumps(details.get('legs', [])),
                'filled_at': self._get_filled_timestamp(order),
                'has_entry_prices': details.get('has_entry_prices', False)
            }

            records.append(record)

        df = pd.DataFrame(records)

        logger.info(f"Processed {len(df)} orders")
        logger.info(f"Closing orders: {len(df[df['order_type'] == OrderType.CLOSING])}")
        logger.info(f"Total realized P&L: ${df['realized_pnl'].sum():.2f}")

        return df

    def _aggregate_daily_pnl(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate orders by day to create daily summary.

        Args:
            order_df: DataFrame from _process_orders()

        Returns:
            DataFrame with daily summaries
        """
        # Only include closing orders for P&L calculation
        closing_df = order_df[order_df['order_type'] == OrderType.CLOSING].copy()

        if closing_df.empty:
            # No closing orders - return empty summary
            return pd.DataFrame(columns=[
                'date', 'total_pnl', 'num_trades', 'num_winning_trades',
                'num_losing_trades', 'commission_total', 'fees_total',
                'largest_win', 'largest_loss', 'order_ids'
            ])

        # Group by date
        daily = closing_df.groupby('order_date').agg({
            'realized_pnl': ['sum', 'max', 'min'],
            'order_id': lambda x: json.dumps(list(x)),
            'commission': 'sum',
            'fees': 'sum'
        }).reset_index()

        # Flatten multi-level columns
        daily.columns = ['date', 'total_pnl', 'largest_win', 'largest_loss',
                        'order_ids', 'commission_total', 'fees_total']

        # Calculate win/loss counts
        def count_winners(date_val):
            day_orders = closing_df[closing_df['order_date'] == date_val]
            return len(day_orders[day_orders['realized_pnl'] > 0])

        def count_losers(date_val):
            day_orders = closing_df[closing_df['order_date'] == date_val]
            return len(day_orders[day_orders['realized_pnl'] < 0])

        def count_total(date_val):
            return len(closing_df[closing_df['order_date'] == date_val])

        daily['num_winning_trades'] = daily['date'].apply(count_winners)
        daily['num_losing_trades'] = daily['date'].apply(count_losers)
        daily['num_trades'] = daily['date'].apply(count_total)

        logger.info(f"Aggregated {len(daily)} days of P&L")

        return daily

    async def refresh_cache(self) -> None:
        """
        Fetch orders from API and refresh database cache.

        This is the main entry point for updating the history data.
        """
        logger.info(f"Refreshing order history cache (last {self.lookback_days} days)")

        # Fetch orders
        orders = await self.fetch_orders()

        if not orders:
            logger.warning("No orders fetched - cache not updated")
            return

        # Process orders
        order_df = self._process_orders(orders)

        # Aggregate daily P&L
        daily_df = self._aggregate_daily_pnl(order_df)

        # Update database
        with self.db.get_connection() as conn:
            # Clear existing data for this period
            from_date = datetime.now().date() - timedelta(days=self.lookback_days)

            conn.execute(
                "DELETE FROM order_pnl_cache WHERE order_date >= ?",
                (from_date,)
            )

            conn.execute(
                "DELETE FROM daily_pnl_summary WHERE date >= ?",
                (from_date,)
            )

            # Insert new data
            if not order_df.empty:
                conn.execute("""
                    INSERT INTO order_pnl_cache (
                        order_id, order_date, underlying, strategy_type,
                        realized_pnl, commission, fees, order_type,
                        leg_details, filled_at, has_entry_prices
                    )
                    SELECT
                        order_id, order_date, underlying, strategy_type,
                        realized_pnl, commission, fees, order_type,
                        leg_details, filled_at, has_entry_prices
                    FROM order_df
                """)
                logger.info(f"Inserted {len(order_df)} orders into cache")

            if not daily_df.empty:
                conn.execute("""
                    INSERT INTO daily_pnl_summary (
                        date, total_pnl, num_trades, num_winning_trades,
                        num_losing_trades, commission_total, fees_total,
                        largest_win, largest_loss, order_ids
                    )
                    SELECT
                        date, total_pnl, num_trades, num_winning_trades,
                        num_losing_trades, commission_total, fees_total,
                        largest_win, largest_loss, order_ids
                    FROM daily_df
                """)
                logger.info(f"Inserted {len(daily_df)} daily summaries")

        logger.info("Cache refresh complete")

    def get_daily_summary(self) -> pd.DataFrame:
        """
        Get daily P&L summary from cache.

        Returns:
            DataFrame with daily aggregates, sorted by date descending
        """
        with self.db.get_connection() as conn:
            query = """
                SELECT
                    date,
                    total_pnl,
                    num_trades,
                    num_winning_trades,
                    num_losing_trades,
                    commission_total,
                    fees_total,
                    largest_win,
                    largest_loss,
                    order_ids
                FROM daily_pnl_summary
                ORDER BY date DESC
            """

            df = conn.execute(query).fetchdf()

        return df

    def get_orders_for_date(self, target_date: date) -> pd.DataFrame:
        """
        Get all orders for a specific date.

        Args:
            target_date: Date to query

        Returns:
            DataFrame with order details
        """
        with self.db.get_connection() as conn:
            query = """
                SELECT
                    order_id,
                    underlying,
                    strategy_type,
                    realized_pnl,
                    commission,
                    fees,
                    order_type,
                    leg_details,
                    filled_at
                FROM order_pnl_cache
                WHERE order_date = ?
                ORDER BY filled_at
            """

            df = conn.execute(query, (target_date,)).fetchdf()

        return df

    def get_summary_stats(self) -> Dict[str, any]:
        """
        Get overall summary statistics.

        Returns:
            Dictionary with total P&L, win rate, best/worst days, etc.
        """
        daily_df = self.get_daily_summary()

        if daily_df.empty:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'best_day': 0.0,
                'worst_day': 0.0,
                'avg_daily_pnl': 0.0,
                'num_days': 0
            }

        total_pnl = daily_df['total_pnl'].sum()
        total_trades = daily_df['num_trades'].sum()
        total_wins = daily_df['num_winning_trades'].sum()

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        return {
            'total_pnl': total_pnl,
            'total_trades': int(total_trades),
            'win_rate': win_rate,
            'best_day': daily_df['total_pnl'].max(),
            'worst_day': daily_df['total_pnl'].min(),
            'avg_daily_pnl': daily_df['total_pnl'].mean(),
            'num_days': len(daily_df)
        }
