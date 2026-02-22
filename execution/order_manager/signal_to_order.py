"""
Signal to Order Converter

Converts algo trading signals into orders that can be placed via
the place_schwab_order.py script.
"""

import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from strategies.base_strategy import Signal, SignalDirection


def signal_to_order_dict(signal: Signal) -> Dict[str, Any]:
    """
    Convert a Signal object to an order dictionary.

    Args:
        signal: Signal from algo strategy

    Returns:
        Dictionary that can be passed to place_schwab_order.py
    """
    order_dict = {
        'symbol': signal.symbol,
        'action': 'BUY' if signal.direction == SignalDirection.LONG else 'SELL',
        'quantity': signal.quantity or 1,
        'entry_price': signal.entry_price,
        'strategy_name': signal.strategy_name,
        'signal_id': signal.signal_id,
    }

    # Add option-specific fields if present
    if signal.option_type:
        order_dict.update({
            'type': signal.option_type,
            'strike': signal.strike,
            'expiration': signal.expiration.isoformat() if signal.expiration else None,
            'delta': signal.delta,
            'option_symbol': signal.metadata.get('option_symbol', ''),
            'option_bid': signal.metadata.get('option_bid'),
            'option_ask': signal.metadata.get('option_ask'),
        })

    return order_dict


async def place_order_via_script(
    signal: Signal,
    auto_approve: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Place order by calling the place_schwab_order.py script.

    Args:
        signal: Signal to convert to order
        auto_approve: Skip approval prompt
        dry_run: Preview only, don't place real order

    Returns:
        True if order placed successfully
    """
    try:
        # Convert signal to order dict
        order_dict = signal_to_order_dict(signal)

        # Build command
        script_path = Path(__file__).parent.parent.parent / "scripts" / "place_schwab_order.py"
        cmd = [
            "python",
            str(script_path),
            "--order-json", json.dumps(order_dict)
        ]

        if auto_approve:
            cmd.append("--auto-approve")
        if dry_run:
            cmd.append("--dry-run")

        logger.info(f"Calling order placement script for signal {signal.signal_id}")

        # Call the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"Order placed successfully for signal {signal.signal_id}")
            return True
        else:
            logger.error(f"Order placement failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Order placement timed out")
        return False
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return False


def format_signal_for_display(signal: Signal) -> str:
    """
    Format signal details for display.

    Args:
        signal: Signal to format

    Returns:
        Formatted string
    """
    if signal.option_type:
        return (
            f"{signal.symbol} ${signal.strike} {signal.option_type} "
            f"exp {signal.expiration} @ ${signal.entry_price:.2f}"
        )
    else:
        return (
            f"{signal.symbol} @ ${signal.entry_price:.2f}"
        )
