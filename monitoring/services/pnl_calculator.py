"""
P&L Calculator Service

Extracts fill prices from Schwab order data and calculates realized P&L for closing orders.
Handles the critical 1-indexed legId pattern from orderActivityCollection.executionLegs.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderType:
    """Order type classifications"""
    OPENING = "OPENING"
    CLOSING = "CLOSING"
    ROLL = "ROLL"
    UNKNOWN = "UNKNOWN"


def extract_fill_prices(order: dict) -> Dict[str, float]:
    """
    Extract actual fill prices from Schwab order executionLegs.

    CRITICAL: legId in executionLegs is 1-indexed, not 0-indexed!
    Must subtract 1 to get the correct orderLegCollection index.

    Args:
        order: Schwab order dictionary containing orderActivityCollection

    Returns:
        Dictionary mapping symbol -> fill_price

    Example order structure:
        {
            'orderLegCollection': [
                {'instrument': {'symbol': 'SPX_240315C05000'}, ...},  # index 0
                {'instrument': {'symbol': 'SPX_240315P04900'}, ...}   # index 1
            ],
            'orderActivityCollection': [
                {
                    'executionLegs': [
                        {'legId': 1, 'price': 12.50},  # References index 0!
                        {'legId': 2, 'price': 15.30}   # References index 1!
                    ]
                }
            ]
        }
    """
    fill_prices = {}

    order_legs = order.get('orderLegCollection', [])
    activities = order.get('orderActivityCollection', [])

    if not activities:
        logger.debug(f"Order {order.get('orderId')} has no activities")
        return fill_prices

    # Use first activity for consistency (in case of multiple fills)
    activity = activities[0]
    execution_legs = activity.get('executionLegs', [])

    for exec_leg in execution_legs:
        leg_id = exec_leg.get('legId', 0)  # 1-indexed!
        price = exec_leg.get('price', 0)

        if leg_id > 0 and leg_id <= len(order_legs):
            # Convert 1-indexed legId to 0-indexed array position
            leg_index = leg_id - 1
            leg = order_legs[leg_index]

            symbol = leg.get('instrument', {}).get('symbol', '')
            if symbol:
                fill_prices[symbol] = price
                logger.debug(f"Extracted fill price for {symbol}: ${price:.4f} (legId={leg_id}, index={leg_index})")
        else:
            logger.warning(f"Invalid legId {leg_id} (order has {len(order_legs)} legs)")

    return fill_prices


def classify_order_type(order: dict) -> str:
    """
    Classify order as OPENING, CLOSING, or ROLL based on leg instructions.

    Args:
        order: Schwab order dictionary

    Returns:
        OrderType constant (OPENING, CLOSING, ROLL, UNKNOWN)
    """
    legs = order.get('orderLegCollection', [])

    if not legs:
        return OrderType.UNKNOWN

    instructions = [leg.get('instruction', '') for leg in legs]

    # Count instruction types
    has_buy_to_open = any('BUY_TO_OPEN' in inst for inst in instructions)
    has_sell_to_open = any('SELL_TO_OPEN' in inst for inst in instructions)
    has_buy_to_close = any('BUY_TO_CLOSE' in inst for inst in instructions)
    has_sell_to_close = any('SELL_TO_CLOSE' in inst for inst in instructions)

    has_opening = has_buy_to_open or has_sell_to_open
    has_closing = has_buy_to_close or has_sell_to_close

    if has_opening and has_closing:
        return OrderType.ROLL
    elif has_closing:
        return OrderType.CLOSING
    elif has_opening:
        return OrderType.OPENING
    else:
        return OrderType.UNKNOWN


def extract_commission(order: dict) -> float:
    """
    Extract total commission from order.

    Args:
        order: Schwab order dictionary

    Returns:
        Total commission amount (positive number)
    """
    activities = order.get('orderActivityCollection', [])

    total_commission = 0.0
    for activity in activities:
        execution_legs = activity.get('executionLegs', [])
        for leg in execution_legs:
            # Commission is typically reported per leg
            commission = leg.get('commission', 0) or 0
            total_commission += abs(commission)

    return total_commission


def extract_fees(order: dict) -> float:
    """
    Extract regulatory fees from order.

    Args:
        order: Schwab order dictionary

    Returns:
        Total fees amount (positive number)
    """
    activities = order.get('orderActivityCollection', [])

    total_fees = 0.0
    for activity in activities:
        # Fees are often at the activity level
        fees = activity.get('fees', 0) or 0
        total_fees += abs(fees)

        # Also check execution legs for per-leg fees
        execution_legs = activity.get('executionLegs', [])
        for leg in execution_legs:
            leg_fees = leg.get('regulatoryFees', 0) or 0
            total_fees += abs(leg_fees)

    return total_fees


def calculate_order_pnl(
    order: dict,
    entry_prices: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, any]]:
    """
    Calculate realized P&L for a closing order.

    For credit spreads: profit = credit_received - cost_to_close
    For debit spreads: profit = value_at_close - debit_paid

    Args:
        order: Schwab order dictionary (must be a CLOSING order)
        entry_prices: Dictionary mapping symbol -> entry_price (from opening order)
                     If None, will return incomplete P&L

    Returns:
        Tuple of (realized_pnl, details_dict)
        - realized_pnl: Net profit/loss including commissions and fees
        - details_dict: Breakdown with per-leg P&L and costs
    """
    order_type = classify_order_type(order)

    if order_type != OrderType.CLOSING:
        # Not a closing order - no realized P&L yet
        return 0.0, {
            'order_type': order_type,
            'realized_pnl': 0.0,
            'reason': 'Not a closing order'
        }

    # Extract fill prices from this closing order
    closing_prices = extract_fill_prices(order)

    if not closing_prices:
        logger.warning(f"Order {order.get('orderId')} has no fill prices")
        return 0.0, {
            'order_type': order_type,
            'realized_pnl': 0.0,
            'reason': 'No fill prices available'
        }

    # Calculate P&L for each leg
    total_pnl = 0.0
    leg_details = []

    legs = order.get('orderLegCollection', [])

    for leg in legs:
        symbol = leg.get('instrument', {}).get('symbol', '')
        quantity = leg.get('quantity', 0)
        instruction = leg.get('instruction', '')

        closing_price = closing_prices.get(symbol, 0)

        # Get entry price (if available)
        entry_price = 0.0
        if entry_prices and symbol in entry_prices:
            entry_price = entry_prices[symbol]
        else:
            logger.warning(f"No entry price available for {symbol}")

        # Calculate P&L for this leg
        # P&L = (close - open) × quantity × multiplier
        # For shorts (negative qty via SELL_TO_OPEN): profit when close < open
        # For longs (positive qty via BUY_TO_OPEN): profit when close > open

        # Determine sign based on closing instruction
        if 'BUY_TO_CLOSE' in instruction:
            # Closing a short position - we're buying back
            # Original was SELL_TO_OPEN (received credit)
            # P&L = entry_credit - closing_cost = entry - close (per share)
            leg_pnl = (entry_price - closing_price) * quantity * 100
        elif 'SELL_TO_CLOSE' in instruction:
            # Closing a long position - we're selling
            # Original was BUY_TO_OPEN (paid debit)
            # P&L = closing_value - entry_cost = close - entry (per share)
            leg_pnl = (closing_price - entry_price) * quantity * 100
        else:
            leg_pnl = 0.0

        total_pnl += leg_pnl

        leg_details.append({
            'symbol': symbol,
            'quantity': quantity,
            'instruction': instruction,
            'entry_price': entry_price,
            'closing_price': closing_price,
            'leg_pnl': leg_pnl
        })

        logger.debug(
            f"Leg P&L: {symbol} {instruction} "
            f"entry=${entry_price:.4f} close=${closing_price:.4f} "
            f"qty={quantity} P&L=${leg_pnl:.2f}"
        )

    # Subtract costs
    commission = extract_commission(order)
    fees = extract_fees(order)

    net_pnl = total_pnl - commission - fees

    details = {
        'order_type': order_type,
        'gross_pnl': total_pnl,
        'commission': commission,
        'fees': fees,
        'realized_pnl': net_pnl,
        'legs': leg_details,
        'has_entry_prices': bool(entry_prices)
    }

    logger.info(
        f"Order P&L: gross=${total_pnl:.2f} "
        f"commission=${commission:.2f} fees=${fees:.2f} "
        f"net=${net_pnl:.2f}"
    )

    return net_pnl, details


def normalize_symbol(symbol: str) -> str:
    """
    Normalize option symbols for matching.

    Handles variants like SPXW→SPX, NDXP→NDX

    Args:
        symbol: Option symbol (possibly OCC format)

    Returns:
        Normalized symbol
    """
    # Extract underlying from OCC symbol if needed
    # Schwab OCC format: "SPXW  260212C06870000" (6-char ticker + date + C/P + strike)
    # Legacy format: "SPXW_260212C06870000" (ticker_date...)

    if '_' in symbol:
        # Legacy format with underscore
        underlying = symbol.split('_')[0]
    elif len(symbol) > 6 and symbol[6:12].isdigit():
        # Schwab format: first 6 chars are underlying (may have trailing spaces)
        underlying = symbol[:6].rstrip()
    else:
        # Plain symbol
        underlying = symbol

    # Normalize common variants
    normalizations = {
        'SPXW': 'SPX',
        'NDXP': 'NDX',
        'MRUT': 'RUT',
        'RUTW': 'RUT'
    }

    return normalizations.get(underlying, underlying)


def get_order_underlying(order: dict) -> str:
    """
    Extract underlying symbol from order.

    Args:
        order: Schwab order dictionary

    Returns:
        Underlying symbol (e.g., 'SPX', 'SPY')
    """
    legs = order.get('orderLegCollection', [])

    if not legs:
        return ''

    # Get first leg's symbol
    first_leg = legs[0]
    symbol = first_leg.get('instrument', {}).get('symbol', '')

    return normalize_symbol(symbol)


def get_order_strategy_type(order: dict) -> str:
    """
    Classify order strategy type based on leg structure.

    Args:
        order: Schwab order dictionary

    Returns:
        Strategy type (e.g., 'IRON_CONDOR', 'VERTICAL_SPREAD', 'SINGLE_LEG')
    """
    legs = order.get('orderLegCollection', [])
    num_legs = len(legs)

    if num_legs == 1:
        return 'SINGLE_LEG'
    elif num_legs == 2:
        return 'VERTICAL_SPREAD'
    elif num_legs == 3:
        return 'BUTTERFLY'
    elif num_legs == 4:
        # Could be iron condor or iron butterfly
        # Check if we have 2 calls and 2 puts
        call_count = sum(1 for leg in legs
                        if leg.get('instrument', {}).get('putCall') == 'CALL')
        put_count = sum(1 for leg in legs
                       if leg.get('instrument', {}).get('putCall') == 'PUT')

        if call_count == 2 and put_count == 2:
            return 'IRON_CONDOR'
        else:
            return 'FOUR_LEG'
    else:
        return f'{num_legs}_LEG'
