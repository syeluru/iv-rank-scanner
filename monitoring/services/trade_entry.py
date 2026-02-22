"""
Manual Trade Entry Service

Handles manual trade entry from dashboard with full strategy attribution
and database logging.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from data.storage.database import trades_db, positions_db
from monitoring.services.strategy_tracker import strategy_tracker
from config.settings import settings


@dataclass
class TradeEntry:
    """Manual trade entry data."""
    symbol: str
    action: str  # BUY, SELL, STO, BTC, etc.
    quantity: int
    price: float
    strategy_name: str
    option_type: Optional[str] = None  # CALL or PUT
    strike: Optional[float] = None
    expiration: Optional[str] = None
    notes: Optional[str] = None


class TradeEntryService:
    """
    Service for entering trades manually from dashboard.

    Workflow:
    1. User fills out trade form in dashboard
    2. Trade is validated
    3. Order is created (with strategy attribution)
    4. In paper trading mode: Simulate execution immediately
    5. In live mode: Submit to broker API
    6. Record trade in database
    7. Update positions
    8. Dashboard refreshes with new data
    """

    def __init__(self):
        """Initialize trade entry service."""
        self.paper_trading = settings.PAPER_TRADING
        logger.info(f"TradeEntryService initialized (paper_trading={self.paper_trading})")

    def enter_trade(self, trade: TradeEntry) -> Dict[str, Any]:
        """
        Enter a manual trade.

        Args:
            trade: TradeEntry with all trade details

        Returns:
            Dictionary with result status and details
        """
        try:
            # Validate trade
            validation_result = self._validate_trade(trade)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'trade_id': None
                }

            # Format symbol for options
            formatted_symbol = self._format_symbol(trade)

            # In paper trading mode, simulate immediate execution
            if self.paper_trading:
                result = self._simulate_execution(trade, formatted_symbol)
            else:
                # TODO: Submit to broker API when ready
                result = self._submit_to_broker(trade, formatted_symbol)

            if result['success']:
                # Record trade in database
                trade_id = strategy_tracker.record_trade(
                    symbol=formatted_symbol,
                    strategy_name=trade.strategy_name,
                    action=trade.action,
                    quantity=trade.quantity,
                    price=result['fill_price'],
                    notes=trade.notes
                )

                # Update position
                self._update_position_from_trade(trade, formatted_symbol, result['fill_price'])

                logger.info(
                    f"Trade entered: {formatted_symbol} {trade.action} {trade.quantity} @ ${result['fill_price']:.2f} "
                    f"({trade.strategy_name})"
                )

                return {
                    'success': True,
                    'trade_id': trade_id,
                    'fill_price': result['fill_price'],
                    'message': f"Trade executed: {trade.action} {trade.quantity} {formatted_symbol} @ ${result['fill_price']:.2f}"
                }
            else:
                return {
                    'success': False,
                    'error': result['error'],
                    'trade_id': None
                }

        except Exception as e:
            logger.error(f"Error entering trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'trade_id': None
            }

    def _validate_trade(self, trade: TradeEntry) -> Dict[str, Any]:
        """
        Validate trade parameters.

        Args:
            trade: TradeEntry to validate

        Returns:
            Dictionary with 'valid' (bool) and 'error' (str) if invalid
        """
        # Required fields
        if not trade.symbol:
            return {'valid': False, 'error': 'Symbol is required'}

        if not trade.action:
            return {'valid': False, 'error': 'Action is required'}

        if trade.quantity <= 0:
            return {'valid': False, 'error': 'Quantity must be positive'}

        if trade.price <= 0:
            return {'valid': False, 'error': 'Price must be positive'}

        if not trade.strategy_name:
            return {'valid': False, 'error': 'Strategy name is required'}

        # Validate action
        valid_actions = ['BUY', 'SELL', 'STO', 'BTC', 'STC', 'BTO']
        if trade.action.upper() not in valid_actions:
            return {'valid': False, 'error': f'Action must be one of: {", ".join(valid_actions)}'}

        # For options, validate additional fields
        if trade.option_type:
            if trade.option_type.upper() not in ['CALL', 'PUT']:
                return {'valid': False, 'error': 'Option type must be CALL or PUT'}

            if not trade.strike:
                return {'valid': False, 'error': 'Strike price is required for options'}

            if not trade.expiration:
                return {'valid': False, 'error': 'Expiration date is required for options'}

        return {'valid': True, 'error': None}

    def _format_symbol(self, trade: TradeEntry) -> str:
        """
        Format symbol with option details if applicable.

        Args:
            trade: TradeEntry

        Returns:
            Formatted symbol (e.g., "AAPL" or "AAPL 150P 02/28/26")
        """
        if trade.option_type and trade.strike and trade.expiration:
            # Format: AAPL 150P 02/28/26
            option_code = 'P' if trade.option_type.upper() == 'PUT' else 'C'
            return f"{trade.symbol} {trade.strike}{option_code} {trade.expiration}"
        else:
            return trade.symbol

    def _simulate_execution(self, trade: TradeEntry, formatted_symbol: str) -> Dict[str, Any]:
        """
        Simulate trade execution for paper trading.

        In paper trading mode, we immediately "fill" the trade at the specified price.

        Args:
            trade: TradeEntry
            formatted_symbol: Formatted symbol string

        Returns:
            Dictionary with 'success', 'fill_price', 'error'
        """
        # In paper trading, assume immediate fill at specified price
        # In real world, might add some slippage simulation
        fill_price = trade.price

        logger.debug(f"[PAPER TRADING] Simulated fill: {formatted_symbol} @ ${fill_price:.2f}")

        return {
            'success': True,
            'fill_price': fill_price,
            'error': None
        }

    def _submit_to_broker(self, trade: TradeEntry, formatted_symbol: str) -> Dict[str, Any]:
        """
        Submit trade to broker API (for live trading).

        Args:
            trade: TradeEntry
            formatted_symbol: Formatted symbol string

        Returns:
            Dictionary with 'success', 'fill_price', 'error'
        """
        # TODO: Implement broker API submission when API access is ready
        logger.warning("Live trading not yet implemented - use paper trading mode")
        return {
            'success': False,
            'fill_price': None,
            'error': 'Live trading not yet implemented'
        }

    def _update_position_from_trade(self, trade: TradeEntry, symbol: str, fill_price: float) -> None:
        """
        Update position after trade execution.

        Args:
            trade: TradeEntry
            symbol: Formatted symbol
            fill_price: Actual fill price
        """
        # Determine position change based on action
        action = trade.action.upper()

        if action in ['SELL', 'STO']:  # Opening short position
            position_delta = -trade.quantity
        elif action in ['BUY', 'BTO']:  # Opening long position
            position_delta = trade.quantity
        elif action in ['BTC', 'STC']:  # Closing position
            # For closing trades, we need to look up existing position
            # For now, assume it's closing a short if BTC, long if STC
            position_delta = trade.quantity if action == 'BTC' else -trade.quantity
        else:
            position_delta = 0

        # Update position in database
        # For simplicity, using fill_price as both entry and current price
        # In production, would fetch current market price
        strategy_tracker.update_position(
            symbol=symbol,
            strategy_name=trade.strategy_name,
            quantity=position_delta,
            entry_price=fill_price,
            current_price=fill_price
        )

    def get_available_strategies(self) -> list[str]:
        """
        Get list of available strategy names for dropdown.

        Returns:
            List of strategy names
        """
        # These should match the strategies implemented in the system
        return [
            "wheel_strategy",
            "iron_condor",
            "credit_spreads",
            "orb_0dte",
            "straddle",
            "calendar_spread",
            "manual"  # For ad-hoc trades not tied to a strategy
        ]


# Global instance
trade_entry_service = TradeEntryService()
