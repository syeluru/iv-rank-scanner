"""
Strategy Performance Tracker

Tracks and aggregates performance metrics by strategy name.
Links trades → positions → P&L → strategy cards in dashboard.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from loguru import logger

from data.storage.database import trades_db, positions_db


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""
    strategy_name: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_pnl_per_trade: float
    best_trade: float
    worst_trade: float
    current_positions: int
    unrealized_pnl: float
    realized_pnl: float
    total_return_pct: float


class StrategyTracker:
    """
    Track and aggregate performance by strategy.

    This is the core service that connects:
    - Trades (closed positions)
    - Positions (open positions)
    - Strategy metrics (aggregated stats)
    - Dashboard cards (visual display)
    """

    def __init__(self):
        """Initialize strategy tracker."""
        # Don't store connections - get fresh ones each time
        logger.info("StrategyTracker initialized")

    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """
        Get performance metrics for all strategies.

        Returns:
            Dictionary mapping strategy_name to StrategyMetrics
        """
        # Get all closed trades
        trades_df = self._get_closed_trades()

        # Get all open positions
        positions_df = self._get_open_positions()

        # Get unique strategy names from both
        strategies = set()
        if not trades_df.empty:
            strategies.update(trades_df['strategy_name'].unique())
        if not positions_df.empty:
            strategies.update(positions_df['strategy_name'].unique())

        # Calculate metrics for each strategy
        metrics = {}
        for strategy_name in strategies:
            metrics[strategy_name] = self._calculate_strategy_metrics(
                strategy_name, trades_df, positions_df
            )

        return metrics

    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """
        Get metrics for a specific strategy.

        Args:
            strategy_name: Name of strategy to get metrics for

        Returns:
            StrategyMetrics or None if strategy not found
        """
        all_metrics = self.get_all_strategy_metrics()
        return all_metrics.get(strategy_name)

    def _get_closed_trades(self) -> pd.DataFrame:
        """
        Get all closed trades from database.

        Returns:
            DataFrame with columns: symbol, strategy_name, pnl, timestamp, etc.
        """
        # Note: Current trades table schema is a simple execution ledger
        # P&L is calculated from positions when they close
        # For now, return empty DataFrame since we don't have closed trades with P&L yet

        try:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'symbol', 'strategy_name', 'pnl', 'quantity',
                'entry_price', 'exit_price', 'entry_timestamp',
                'exit_timestamp', 'duration_days', 'return_pct'
            ])
        except Exception as e:
            logger.error(f"Error creating closed trades DataFrame: {e}")
            return pd.DataFrame()

    def _get_open_positions(self) -> pd.DataFrame:
        """
        Get all open positions from database.

        Returns:
            DataFrame with columns: symbol, strategy_name, unrealized_pnl, etc.
        """
        query = """
            SELECT
                symbol,
                strategy_name,
                quantity,
                avg_cost as entry_price,
                current_price,
                unrealized_pnl,
                opened_at as entry_timestamp
            FROM positions
            WHERE status = 'OPEN'
        """

        try:
            from data.storage.database import positions_db
            result = positions_db.execute(query)
            df = result.fetchdf()
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return pd.DataFrame()

    def _calculate_strategy_metrics(
        self,
        strategy_name: str,
        trades_df: pd.DataFrame,
        positions_df: pd.DataFrame
    ) -> StrategyMetrics:
        """
        Calculate performance metrics for a strategy.

        Args:
            strategy_name: Name of strategy
            trades_df: DataFrame of closed trades
            positions_df: DataFrame of open positions

        Returns:
            StrategyMetrics with all calculated values
        """
        # Filter to this strategy
        strategy_trades = trades_df[trades_df['strategy_name'] == strategy_name] if not trades_df.empty else pd.DataFrame()
        strategy_positions = positions_df[positions_df['strategy_name'] == strategy_name] if not positions_df.empty else pd.DataFrame()

        # Realized P&L from closed trades
        realized_pnl = strategy_trades['pnl'].sum() if not strategy_trades.empty else 0.0

        # Unrealized P&L from open positions
        unrealized_pnl = strategy_positions['unrealized_pnl'].sum() if not strategy_positions.empty else 0.0

        # Total P&L
        total_pnl = realized_pnl + unrealized_pnl

        # Trade statistics
        total_trades = len(strategy_trades)

        if total_trades > 0:
            winning_trades = len(strategy_trades[strategy_trades['pnl'] > 0])
            losing_trades = len(strategy_trades[strategy_trades['pnl'] < 0])
            win_rate = winning_trades / total_trades

            # Profit factor = gross profits / gross losses
            gross_profits = strategy_trades[strategy_trades['pnl'] > 0]['pnl'].sum()
            gross_losses = abs(strategy_trades[strategy_trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

            avg_pnl = strategy_trades['pnl'].mean()
            best_trade = strategy_trades['pnl'].max()
            worst_trade = strategy_trades['pnl'].min()
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_pnl = 0.0
            best_trade = 0.0
            worst_trade = 0.0

        # Position statistics
        current_positions = len(strategy_positions)

        # Total return percentage (simplistic - based on total P&L vs capital deployed)
        # For now, just use P&L as percentage (can be improved later)
        total_return_pct = (total_pnl / 10000.0) * 100 if total_pnl != 0 else 0.0  # Assume $10k starting capital

        return StrategyMetrics(
            strategy_name=strategy_name,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_pnl_per_trade=avg_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            current_positions=current_positions,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_return_pct=total_return_pct
        )

    def record_trade(
        self,
        symbol: str,
        strategy_name: str,
        action: str,
        quantity: int,
        price: float,
        order_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Record a trade in the database.

        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy that generated this trade
            action: Trade action (BUY/SELL/STO/BTC/etc)
            quantity: Number of contracts/shares
            price: Execution price
            order_id: Optional order ID reference
            notes: Optional notes about the trade

        Returns:
            Trade ID
        """
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
        position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"

        # Insert into trades table (matching actual schema)
        query = """
            INSERT INTO trades (
                trade_id, position_id, order_id, symbol, side,
                quantity, price, executed_at, strategy_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            trade_id,
            position_id,
            order_id or trade_id,  # Use trade_id if no order_id provided
            symbol,
            action,  # side column
            quantity,
            price,
            datetime.now(),
            strategy_name
        )

        try:
            from data.storage.database import trades_db
            trades_db.execute(query, values)
            logger.info(f"Recorded trade: {trade_id} ({strategy_name})")
            return trade_id
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            raise

    def update_position(
        self,
        symbol: str,
        strategy_name: str,
        quantity: int,
        entry_price: float,
        current_price: float
    ) -> None:
        """
        Update or create a position.

        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy that owns this position
            quantity: Position size (negative for short)
            entry_price: Average entry price
            current_price: Current market price
        """
        # Calculate unrealized P&L
        if quantity < 0:  # Short position
            unrealized_pnl = (entry_price - current_price) * abs(quantity) * 100  # Options multiplier
        else:  # Long position
            unrealized_pnl = (current_price - entry_price) * quantity * 100

        # Check if position exists
        from data.storage.database import positions_db
        check_query = "SELECT position_id FROM positions WHERE symbol = ? AND strategy_name = ? AND status = 'OPEN'"
        result = positions_db.execute(check_query, (symbol, strategy_name))
        existing = result.fetchdf()

        if existing is not None and not existing.empty:
            # Update existing position
            update_query = """
                UPDATE positions
                SET quantity = ?, current_price = ?, unrealized_pnl = ?
                WHERE symbol = ? AND strategy_name = ? AND status = 'OPEN'
            """
            values = (quantity, current_price, unrealized_pnl, symbol, strategy_name)
        else:
            # Create new position
            position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            update_query = """
                INSERT INTO positions (
                    position_id, symbol, position_type, strategy_name, quantity,
                    avg_cost, current_price, unrealized_pnl,
                    opened_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = (
                position_id, symbol, 'OPTION', strategy_name, quantity,
                entry_price, current_price, unrealized_pnl,
                datetime.now(), 'OPEN'
            )

        try:
            positions_db.execute(update_query, values)
            logger.debug(f"Updated position: {symbol} ({strategy_name})")
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            raise


# Global instance
strategy_tracker = StrategyTracker()
