"""
Schwab Positions Service

Fetches real positions from Schwab and converts them to trade groups
for display in the Trade-Nexus dashboard.

Groups positions by strategy based on filled orders, and separates
options trades from stock positions.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from loguru import logger

from execution.broker_api.schwab_client import schwab_client
from monitoring.services.trade_aggregator import AggregatedTradeGroup, AlertColor
from monitoring.services.trade_tracker import trade_tracker, LegInfo


@dataclass
class PositionGroup:
    """A group of positions that belong to the same strategy/order."""
    order_id: str
    underlying: str
    strategy_type: str
    positions: List[Dict]
    order_data: Optional[Dict] = None
    is_stock: bool = False
    order_fill_price: float = 0.0  # Net credit/debit for the order
    commission: float = 0.0  # Opening commission
    fees: float = 0.0  # Opening regulatory fees
    closing_commission: float = 0.0  # Closing commission (if closed)
    closing_fees: float = 0.0  # Closing regulatory fees (if closed)
    trade_id: Optional[str] = None  # Unique trade ID for tracking across rolls


class SchwabPositionsService:
    """
    Fetches and processes positions from Schwab brokerage account.

    Groups positions by strategy based on filled orders and separates
    options trades from stock positions.
    """

    def __init__(self):
        """Initialize the positions service."""
        self._positions_cache: List[Dict] = []
        self._orders_cache: List[Dict] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl_seconds = 5  # Refresh every 5 seconds
        logger.info("SchwabPositionsService initialized")

    async def get_positions(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get current positions from Schwab.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            List of position dictionaries
        """
        now = datetime.now()

        # Check cache
        if not force_refresh and self._last_fetch:
            age = (now - self._last_fetch).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._positions_cache

        try:
            positions = await schwab_client.get_account_positions()
            self._positions_cache = positions
            self._last_fetch = now
            logger.debug(f"Fetched {len(positions)} positions from Schwab")
            return positions
        except Exception as e:
            logger.error(f"Error fetching Schwab positions: {e}")
            return self._positions_cache  # Return cached data on error

    async def get_open_orders(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get open (working) orders from Schwab.

        Returns:
            List of open order dictionaries
        """
        try:
            # Fetch open orders (WORKING, QUEUED, PENDING_ACTIVATION, etc.)
            orders = await schwab_client.get_orders_for_account(
                status='WORKING'
            )
            logger.info(f"Fetched {len(orders)} open orders from Schwab")
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    async def get_filled_orders(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get filled orders from Schwab (last 60 days).

        Returns:
            List of filled order dictionaries
        """
        now = datetime.now()

        # Check cache
        if not force_refresh and self._orders_cache and self._last_fetch:
            age = (now - self._last_fetch).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._orders_cache

        try:
            # Fetch filled orders from last 60 days
            # Use UTC timezone to ensure we get all recent orders
            from_date = datetime.now() - timedelta(days=60)
            to_date = datetime.now() + timedelta(hours=24)  # Extended to ensure we get today's orders

            logger.info(f"Fetching orders from {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}")

            orders = await schwab_client.get_orders_for_account(
                status='FILLED',
                from_entered_datetime=from_date,
                to_entered_datetime=to_date
            )
            self._orders_cache = orders
            logger.info(f"Fetched {len(orders)} filled orders from Schwab")

            # Log recent order dates for debugging
            if orders:
                recent_orders = sorted(orders, key=lambda o: o.get('closeTime', ''), reverse=True)[:5]
                for order in recent_orders:
                    close_time = order.get('closeTime', 'N/A')[:19]
                    order_id = order.get('orderId', 'N/A')
                    logger.debug(f"  Recent order: {order_id} closed at {close_time}")

            return orders
        except Exception as e:
            logger.error(f"Error fetching Schwab orders: {e}")
            return self._orders_cache

    def _get_position_key(self, pos: Dict) -> str:
        """Get a unique key for a position based on its symbol."""
        instrument = pos.get('instrument', {})
        return instrument.get('symbol', '')

    def _convert_orders_to_trade_groups(self, orders: List[Dict], position_map: Dict[str, Dict], used_positions: set) -> List[PositionGroup]:
        """
        Convert filled orders to trade groups (pure order-based view).

        Each order becomes a separate trade card. For each leg:
        - If it's a closing leg (BTC/STC), mark as closed
        - If it's an opening leg (BTO/STO) with current position, mark as open
        - If it's an opening leg without current position, mark as closed

        IMPORTANT: Multiple orders can reference the same position symbol (overlapping strikes).
        Each order independently checks if its legs have current positions.
        No exclusive allocation - this allows multiple ICs with same strikes to display correctly.

        Args:
            orders: List of filled orders
            position_map: Map of symbol -> position data
            used_positions: Set of symbols already allocated (only used for stocks to avoid duplicates)

        Returns:
            List of PositionGroup objects (one per order)
        """
        logger.info("="*100)
        logger.info(f"🔧 _convert_orders_to_trade_groups CALLED with {len(orders)} orders")
        logger.info("="*100)

        # Deduplicate and filter orders
        seen_order_ids = set()
        unique_orders = []
        for order in orders:
            order_id = order.get('orderId')
            if order_id and order_id not in seen_order_ids:
                seen_order_ids.add(order_id)
                unique_orders.append(order)

        # Filter out child orders
        parent_order_ids = {o.get('orderId') for o in unique_orders if o.get('orderId')}
        filtered_orders = []
        for order in unique_orders:
            parent_id = order.get('parentOrderId')
            if parent_id and parent_id in parent_order_ids:
                logger.debug(f"Skipping child order {order.get('orderId')} (parent: {parent_id})")
                continue
            filtered_orders.append(order)

        # Sort chronologically (oldest first) for trade ID tracking
        # Original trades must be processed before rolls to enable proper linking
        sorted_orders = sorted(
            filtered_orders,
            key=lambda o: (o.get('closeTime', ''), o.get('orderId', '')),
            reverse=False  # Oldest first for proper roll linking
        )

        logger.info(f"🔄 Converting {len(sorted_orders)} orders to trade groups (ORDER-BASED VIEW)")
        logger.info(f"📦 Available positions in position_map: {len(position_map)} ({list(position_map.keys())[:5]}...)")

        # First pass: Track which symbols were closed, when, and at what price
        # This helps us determine if an opening order's legs have been closed
        closing_orders = {}  # symbol -> list of (close_time, order_id, fill_price)
        closing_commissions = {}  # order_id -> {'commission': float, 'fees': float}

        for order in sorted_orders:
            order_legs = order.get('orderLegCollection', [])
            close_time = order.get('closeTime', '')
            order_id = order.get('orderId')

            # Extract fill prices from this closing order
            closing_fill_prices = self._extract_fill_prices_from_order(order)

            # Check if this order has any closing legs
            has_closing_leg = any('CLOSE' in leg.get('instruction', '') for leg in order_legs)
            if has_closing_leg:
                # Extract commission for closing orders
                commission_data = self._extract_commission_from_order(order)
                closing_commissions[order_id] = commission_data

            for leg in order_legs:
                instruction = leg.get('instruction', '')
                if 'CLOSE' in instruction:
                    symbol = leg.get('instrument', {}).get('symbol', '')
                    if symbol:
                        # Get the fill price for this specific leg
                        fill_price = closing_fill_prices.get(symbol, 0)
                        if symbol not in closing_orders:
                            closing_orders[symbol] = []
                        closing_orders[symbol].append((close_time, order_id, fill_price))
                        logger.info(f"  📍 Closing leg tracked: {symbol[:15]} @ ${fill_price:.2f} (from order {order_id})")

        logger.info(f"📋 Found {len(closing_orders)} symbols with closing orders, {len(closing_commissions)} orders with closing commissions")

        position_groups = []

        # Pre-fetch all DB data once to avoid creating connections per-order
        db_roll_order_ids = set()
        db_matches_by_opening_order = {}  # opening_order_id -> match_id
        db_legs_by_match = {}  # match_id -> list of leg dicts
        db_rolls_by_match = {}  # match_id -> list of roll dicts
        db_open_leg_counts = {}  # opening_order_id -> count of OPEN legs
        try:
            from monitoring.services.trade_match_persistence import TradeMatchPersistence
            _persistence = TradeMatchPersistence()
            with _persistence.db.get_connection() as _conn:
                # All roll order IDs
                for row in _conn.execute("SELECT roll_order_id FROM trade_rolls").fetchall():
                    db_roll_order_ids.add(row[0])

                # All matches keyed by opening_order_id
                for row in _conn.execute("SELECT match_id, opening_order_id FROM trade_matches").fetchall():
                    db_matches_by_opening_order[row[1]] = row[0]

                # All legs keyed by match_id
                all_legs_rows = _conn.execute("SELECT * FROM trade_legs ORDER BY entry_time").fetchdf().to_dict('records')
                for leg in all_legs_rows:
                    mid = leg['match_id']
                    db_legs_by_match.setdefault(mid, []).append(leg)

                # All rolls keyed by match_id
                all_rolls_rows = _conn.execute("SELECT * FROM trade_rolls ORDER BY roll_time").fetchdf().to_dict('records')
                for roll in all_rolls_rows:
                    mid = roll['match_id']
                    db_rolls_by_match.setdefault(mid, []).append(roll)

                # Open leg counts per opening_order_id
                counts = _conn.execute("""
                    SELECT tm.opening_order_id, COUNT(*)
                    FROM trade_legs tl
                    JOIN trade_matches tm ON tl.match_id = tm.match_id
                    WHERE tl.status = 'OPEN'
                    GROUP BY tm.opening_order_id
                """).fetchall()
                for row in counts:
                    db_open_leg_counts[row[0]] = row[1]

            logger.info(f"📦 Pre-fetched DB data: {len(db_roll_order_ids)} roll orders, {len(db_matches_by_opening_order)} matches, {len(db_legs_by_match)} match leg groups")
        except Exception as e:
            logger.warning(f"Could not pre-fetch DB data: {e}")

        for order in sorted_orders:
            order_id = order.get('orderId')
            order_legs = order.get('orderLegCollection', [])

            # Assign or lookup trade ID for this order
            trade_id = self._assign_or_lookup_trade_id(order, closing_orders)

            if not order_legs:
                continue

            # Skip pure closing orders (all legs are closes)
            # For options: instruction contains 'TO_CLOSE'
            # For stocks: plain 'SELL' (without '_TO_OPEN') is also a close
            def _is_opening_leg(leg):
                instruction = leg.get('instruction', '')
                if 'CLOSE' in instruction:
                    return False
                asset_type = leg.get('instrument', {}).get('assetType', '')
                if asset_type in ('EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND'):
                    # For stocks, SELL without _TO_OPEN is a closing order
                    return 'BUY' in instruction
                return True

            has_opening_leg = any(_is_opening_leg(leg) for leg in order_legs)
            if not has_opening_leg:
                logger.debug(f"Order {order_id}: All legs are closing - skipping")
                continue

            # Skip roll orders (mix of opening and closing legs)
            # Rolls are already incorporated into the original trade via the database
            has_closing_leg = any('CLOSE' in leg.get('instruction', '') for leg in order_legs)
            if has_opening_leg and has_closing_leg:
                logger.info(f"Order {order_id}: ROLL order (mix of opening/closing) - skipping to avoid duplicate display")
                continue

            # ALSO skip if this order is a roll order in the database
            if order_id in db_roll_order_ids:
                logger.info(f"Order {order_id}: Found in trade_rolls table - skipping to avoid duplicate display")
                continue

            # Create positions for all legs
            split_positions = []
            has_open_position = False  # Track if this order has at least one open position

            # Extract fill prices
            leg_fill_prices = self._extract_fill_prices_from_order(order)
            order_fill_price = order.get('price', 0)

            # Log order leg details
            logger.debug(f"📦 Processing Order {order_id}: {len(order_legs)} legs")
            for i, leg in enumerate(order_legs):
                symbol = leg.get('instrument', {}).get('symbol', '')
                qty = leg.get('quantity', 0)
                instruction = leg.get('instruction', '')
                logger.debug(f"  Leg {i+1}: {symbol[:20]} qty={qty} {instruction}")

            # Extract commission and fees
            commission_data = self._extract_commission_from_order(order)
            order_commission = commission_data['commission']
            order_fees = commission_data['fees']

            # Get order close time for comparison
            order_close_time = order.get('closeTime', '')

            # Track closing order ID and commission if this order was closed
            closing_order_id_used = None
            order_closing_commission = 0.0
            order_closing_fees = 0.0

            for leg in order_legs:
                instrument = leg.get('instrument', {})
                symbol = instrument.get('symbol', '')
                leg_qty = abs(leg.get('quantity', 0))
                instruction = leg.get('instruction', '')

                if not symbol or leg_qty == 0:
                    logger.warning(f"Order {order_id}: Skipping leg with symbol={symbol}, qty={leg_qty}")
                    continue

                # Check if this is a closing leg
                is_closing_leg = 'CLOSE' in instruction

                # Check if this leg was closed AFTER this order was placed
                # If there's a closing order for this symbol with time > this order's time, the leg is closed
                was_closed_later = False
                closing_fill_price = None
                if not is_closing_leg and symbol in closing_orders and order_close_time:
                    for close_time, close_order_id, fill_price in closing_orders[symbol]:
                        if close_time > order_close_time:
                            was_closed_later = True
                            closing_fill_price = fill_price
                            closing_order_id_used = close_order_id
                            logger.debug(f"  ✓ Order {order_id}: Leg {symbol[:15]} closed later by order {close_order_id} at FILL PRICE ${fill_price:.2f}")
                            break

                # For opening legs, check if position currently exists
                # Note: Multiple orders can reference the same position (no exclusive allocation)
                current_position = position_map.get(symbol)

                # Debug logging for position matching
                if not is_closing_leg:
                    logger.debug(f"  Opening leg {symbol[:20]}: position_found={current_position is not None}, was_closed_later={was_closed_later}")

                if not is_closing_leg and current_position and not was_closed_later:
                    # Opening leg with current position - check if position quantity covers this leg
                    current_qty = current_position.get('longQuantity', 0) + current_position.get('shortQuantity', 0)

                    # Position exists and has quantity - this leg is open
                    # (Even if multiple orders reference the same position)
                    has_open_position = True

                    split_pos = current_position.copy()
                    split_pos['instrument'] = current_position.get('instrument', {}).copy()

                    # Use order-specific leg fill price if available
                    if symbol in leg_fill_prices:
                        split_pos['averagePrice'] = leg_fill_prices[symbol]

                    # For stocks, use FULL position quantity
                    # For options, use THIS ORDER's quantity (for display purposes)
                    asset_type = instrument.get('assetType', '')
                    if asset_type in ['EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND']:
                        # Stock: use full position quantity
                        split_pos['longQuantity'] = current_position.get('longQuantity', 0)
                        split_pos['shortQuantity'] = current_position.get('shortQuantity', 0)
                        split_pos['marketValue'] = current_position.get('marketValue', 0)
                        # Mark as used only for stocks (to avoid duplicates)
                        used_positions.add(symbol)
                    else:
                        # Option: use THIS ORDER's quantity to show the leg properly
                        if 'BUY' in instruction:
                            split_pos['longQuantity'] = leg_qty
                            split_pos['shortQuantity'] = 0
                        else:
                            split_pos['longQuantity'] = 0
                            split_pos['shortQuantity'] = leg_qty

                        # Calculate market value proportionally
                        if current_qty > 0:
                            proportion = min(1.0, leg_qty / current_qty)
                            split_pos['marketValue'] = current_position.get('marketValue', 0) * proportion

                        # Mark position as used to prevent it showing up as unmatched
                        # NOTE: Multiple orders can still reference this before it's marked used
                        used_positions.add(symbol)
                        logger.debug(f"  ✓ Marked option position {symbol[:20]} as USED (order {order_id})")

                    split_pos['_is_closed'] = False
                    logger.debug(f"Order {order_id}: Leg {symbol[:30]} is OPEN (current position: {current_qty}, order qty: {leg_qty})")

                else:
                    # Closing leg OR opening leg with no current position OR leg was closed later - mark as closed
                    split_pos = {
                        'instrument': instrument.copy(),
                        'averagePrice': leg_fill_prices.get(symbol, 0),  # Entry price
                        'longQuantity': leg_qty if 'BUY' in instruction else 0,
                        'shortQuantity': leg_qty if 'SELL' in instruction else 0,
                        'marketValue': 0,
                        '_is_closed': True,
                        '_closing_price': closing_fill_price if was_closed_later else None  # Exit price
                    }
                    reason = 'closing leg' if is_closing_leg else ('closed by later order' if was_closed_later else 'no current position')
                    logger.debug(f"Order {order_id}: Leg {symbol[:30]} marked as CLOSED ({reason})")
                    if was_closed_later and closing_fill_price:
                        logger.debug(f"  💰 Leg {symbol[:30]}: Entry=${leg_fill_prices.get(symbol, 0):.2f}, Exit=${closing_fill_price:.2f}")

                split_positions.append(split_pos)

            # Determine underlying and strategy first (needed for skip logic)
            underlying = self._extract_underlying_from_order(order)
            strategy_type = self._extract_strategy_from_order(order)

            # Skip this order if it has no open positions (fully closed)
            # BUT: if the DB has a match with open legs (e.g., all original legs were rolled),
            # don't skip — the roll lookup below will add the actual live legs.
            if not has_open_position:
                db_has_open_legs = db_open_leg_counts.get(order_id, 0) > 0

                if not db_has_open_legs:
                    logger.debug(f"Order {order_id}: No open positions and no DB open legs - skipping (fully closed trade)")
                    continue
                else:
                    logger.info(f"Order {order_id}: No original positions but DB has open legs (all legs rolled) - continuing for roll lookup")

            # Check if this is a stock order
            is_stock = all(
                pos.get('instrument', {}).get('assetType') in ['EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND']
                for pos in split_positions
            )

            # Look up closing commission if this order was closed
            if closing_order_id_used and closing_order_id_used in closing_commissions:
                closing_comm_data = closing_commissions[closing_order_id_used]
                order_closing_commission = closing_comm_data['commission']
                order_closing_fees = closing_comm_data['fees']

            logger.debug(f"✓ Order {order_id}: {underlying} {strategy_type} with {len(split_positions)} legs @ ${order_fill_price:.2f} (commission: ${order_commission:.2f}, fees: ${order_fees:.2f})")
            logger.info(f"  📋 Original order had {len(order_legs)} legs, processed {len(split_positions)} legs for display")

            # Check pre-fetched DB data for rolled legs that belong to this trade
            match_id = db_matches_by_opening_order.get(order_id)
            if match_id:
                logger.info(f"  🔍 Found match {match_id[:8]} for order {order_id}")
                all_legs = db_legs_by_match.get(match_id, [])
                logger.info(f"  🔍 DB has {len(all_legs)} total legs for match {match_id[:8]}")

                existing_symbols = {p.get('instrument', {}).get('symbol') for p in split_positions}
                db_all_symbols = {leg['symbol'] for leg in all_legs}
                missing_from_display = db_all_symbols - existing_symbols

                if missing_from_display:
                    logger.info(f"  🔄 Found {len(missing_from_display)} extra legs in database for order {order_id}")
                    rolls = db_rolls_by_match.get(match_id, [])

                    import json
                    for db_leg in all_legs:
                        if db_leg['symbol'] not in existing_symbols:
                            roll_number = 0
                            roll_order_id = None
                            roll_timestamp = None

                            for idx, roll in enumerate(rolls, 1):
                                opened_symbols = json.loads(roll.get('opened_leg_symbols', '[]') or '[]')
                                if db_leg['symbol'] in opened_symbols:
                                    roll_number = idx
                                    roll_order_id = roll['roll_order_id']
                                    roll_timestamp = roll['roll_time']
                                    break

                            is_closed_leg = db_leg['status'] == 'CLOSED'

                            current_position = position_map.get(db_leg['symbol'])
                            if current_position:
                                rolled_pos = current_position.copy()
                                rolled_pos['instrument'] = current_position.get('instrument', {}).copy()
                                rolled_pos['averagePrice'] = db_leg['entry_price']
                            else:
                                qty = db_leg['current_quantity']
                                long_qty = qty if db_leg['side'] == 'LONG' else 0
                                short_qty = qty if db_leg['side'] == 'SHORT' else 0
                                rolled_pos = {
                                    'instrument': {
                                        'symbol': db_leg['symbol'],
                                        'assetType': 'OPTION',
                                        'putCall': db_leg['option_type'],
                                        'strikePrice': db_leg['strike'],
                                    },
                                    'longQuantity': long_qty,
                                    'shortQuantity': short_qty,
                                    'averagePrice': db_leg['entry_price'],
                                    'marketValue': 0,
                                }

                            rolled_pos['_is_closed'] = is_closed_leg
                            if is_closed_leg:
                                rolled_pos['_closing_price'] = db_leg.get('exit_price', 0) or 0
                            rolled_pos['_is_rolled_leg'] = True
                            rolled_pos['_roll_number'] = roll_number
                            rolled_pos['_roll_order_id'] = roll_order_id
                            rolled_pos['_roll_timestamp'] = str(roll_timestamp) if roll_timestamp else ''

                            if db_leg['side'] == 'LONG':
                                rolled_pos['longQuantity'] = db_leg['current_quantity']
                                rolled_pos['shortQuantity'] = 0
                            else:
                                rolled_pos['longQuantity'] = 0
                                rolled_pos['shortQuantity'] = db_leg['current_quantity']

                            split_positions.append(rolled_pos)
                            used_positions.add(db_leg['symbol'])
                            status_label = "CLOSED" if is_closed_leg else "OPEN"
                            logger.info(f"    ✅ Added {'closed' if is_closed_leg else 'rolled'} leg: {db_leg['symbol'][:25]} {db_leg['side']} ({status_label})")
            else:
                logger.debug(f"  🔍 No match found in DB for opening_order_id={order_id}")

            # Mark strategy as rolled if we have rolled legs
            has_rolled_legs = any(p.get('_is_rolled_leg', False) for p in split_positions)
            display_strategy_type = strategy_type
            if has_rolled_legs and "(Rolled)" not in strategy_type:
                display_strategy_type = f"{strategy_type} (Rolled)"
                logger.info(f"  ✅ Marked strategy as rolled: {display_strategy_type}")

            position_groups.append(PositionGroup(
                order_id=str(order_id),
                underlying=underlying,
                strategy_type=display_strategy_type,
                positions=split_positions,
                order_data=order,
                is_stock=is_stock,
                order_fill_price=order_fill_price,
                commission=order_commission,
                fees=order_fees,
                closing_commission=order_closing_commission,
                closing_fees=order_closing_fees,
                trade_id=trade_id
            ))

        logger.info(f"✓ Converted {len(position_groups)} orders to position groups")
        logger.info(f"📍 Positions marked as used: {len(used_positions)} ({list(used_positions)[:10]}...)")

        # Group position groups by trade ID
        position_groups = self._group_by_trade_id(position_groups)

        return position_groups

    def _merge_roll_trades(self, position_groups: List[PositionGroup]) -> List[PositionGroup]:
        """
        Merge roll trades into their original position groups.

        When a position is rolled, we want to show all legs (original + rolled) in the same trade card.

        Enhanced logic: When multiple ICs have the same strikes (e.g., two ICs both using 6965C/6990C),
        we match by BOTH symbols AND quantities to avoid merging rolls into the wrong IC.
        """
        rolls_to_merge = []

        # Find all roll trades
        for i, group in enumerate(position_groups):
            if group.strategy_type.startswith('ROLL'):
                rolls_to_merge.append((i, group))

        if not rolls_to_merge:
            return position_groups

        logger.info(f"🔄 Found {len(rolls_to_merge)} roll trades to merge")

        merged_groups = position_groups.copy()
        groups_to_remove = []

        for roll_idx, roll_group in rolls_to_merge:
            # Get the closed symbols AND quantities from this roll
            closed_legs_info = []  # List of (symbol, quantity) tuples
            for pos in roll_group.positions:
                if pos.get('_is_closed'):
                    symbol = pos.get('instrument', {}).get('symbol', '')
                    quantity = abs(pos.get('longQuantity', 0) + pos.get('shortQuantity', 0))
                    if symbol:
                        closed_legs_info.append((symbol, quantity))

            closed_symbols = set(sym for sym, _ in closed_legs_info)
            roll_quantity = max((qty for _, qty in closed_legs_info), default=0)

            logger.info(f"  Roll {roll_group.order_id} ({roll_group.underlying}): Looking for original with symbols {closed_symbols}, quantity {roll_quantity}")

            # Find the original position that contains these symbols with matching quantity
            candidates = []
            for i, group in enumerate(merged_groups):
                if i == roll_idx or group.strategy_type.startswith('ROLL'):
                    continue  # Skip self and other rolls

                # Must match same underlying
                if group.underlying != roll_group.underlying:
                    continue

                # Check if this group has any of the closed symbols
                group_symbols = set()
                group_quantities = set()
                for pos in group.positions:
                    symbol = pos.get('instrument', {}).get('symbol', '')
                    quantity = abs(pos.get('longQuantity', 0) + pos.get('shortQuantity', 0))
                    if symbol:
                        group_symbols.add(symbol)
                        group_quantities.add(quantity)

                # If symbols match, this is a candidate
                matching_symbols = closed_symbols & group_symbols
                if matching_symbols:
                    # Calculate match score: higher if quantities match exactly
                    quantity_match = roll_quantity in group_quantities
                    match_score = len(matching_symbols)
                    if quantity_match:
                        match_score += 100  # Heavily favor quantity matches

                    candidates.append((i, group, match_score, matching_symbols, quantity_match))
                    logger.debug(f"    Candidate {group.order_id}: symbols {matching_symbols}, qty_match={quantity_match}, score={match_score}")

            # Pick the best candidate
            # Sort by: 1) highest score, 2) already-rolled status, 3) earliest timestamp
            if candidates:
                # Enhanced sort with proper tiebreakers
                def sort_key(candidate):
                    idx, group, score, symbols, qty_match = candidate
                    is_already_rolled = '(Rolled)' in group.strategy_type
                    # Use enteredTime (when order was placed) instead of closeTime
                    # closeTime may be empty for orders that aren't fully closed yet
                    order_timestamp = group.order_data.get('enteredTime', 'z') if group.order_data else 'z'  # Default to 'z' for sorting

                    # Return tuple for sorting:
                    # - score: higher is better (sort descending)
                    # - is_already_rolled: True is better (sort descending)
                    # - order_timestamp: EARLIER is better (sort ASCENDING), so negate the reverse
                    return (
                        -score,  # Negate for descending (higher scores first)
                        not is_already_rolled,  # Negate for descending (True first)
                        order_timestamp  # Ascending (earlier timestamps first)
                    )

                candidates.sort(key=sort_key)  # Sort ascending (because we negated the values we want descending)
                original_idx, original_group, best_score, matching_symbols, qty_match = candidates[0]

                logger.info(f"    ✓ Found original position: {original_group.order_id} ({original_group.strategy_type})")
                logger.info(f"      Matched: symbols={matching_symbols}, quantity_match={qty_match}, score={best_score}, already_rolled={'(Rolled)' in original_group.strategy_type}")
                if original_group.order_data:
                    logger.info(f"      Order entered time: {original_group.order_data.get('enteredTime', 'N/A')}")

                # Merge only the NEW opening legs from the roll (not the closing legs which are duplicates)
                new_opening_legs = [pos for pos in roll_group.positions if not pos.get('_is_closed')]
                logger.info(f"    Merging {len(new_opening_legs)} new opening legs into original position (skipping {len(roll_group.positions) - len(new_opening_legs)} duplicate closing legs)")

                # Determine roll number (how many rolls have been merged so far into this position)
                existing_roll_numbers = [leg.get('_roll_number', 0) for leg in merged_groups[original_idx].positions]
                next_roll_number = max(existing_roll_numbers, default=0) + 1

                # Mark the new legs as rolled with roll sequence metadata
                for leg in new_opening_legs:
                    leg['_is_rolled_leg'] = True
                    leg['_roll_number'] = next_roll_number  # 1 = first roll, 2 = second roll, etc.
                    leg['_roll_order_id'] = roll_group.order_id
                    leg['_roll_timestamp'] = roll_group.order_data.get('closeTime', '') if roll_group.order_data else ''

                logger.info(f"    Marked new legs as roll #{next_roll_number}")

                # Add only the new opening legs to original
                merged_groups[original_idx].positions.extend(new_opening_legs)

                # Update strategy to show it's been rolled
                original_strategy = original_group.strategy_type
                if not original_strategy.endswith('(Rolled)'):
                    merged_groups[original_idx].strategy_type = f"{original_strategy} (Rolled)"

                # Mark roll group for removal
                groups_to_remove.append(roll_idx)
            else:
                logger.warning(f"    ⚠️ Could not find original position for roll {roll_group.order_id}")

        # Remove merged roll groups (in reverse order to maintain indices)
        for idx in sorted(groups_to_remove, reverse=True):
            logger.info(f"  Removing standalone roll trade card: {merged_groups[idx].order_id}")
            merged_groups.pop(idx)

        logger.info(f"✓ Merged {len(groups_to_remove)} roll trades into original positions")

        # Final cleanup: Remove any fully-closed ICs that had no rolls merged
        # (These were kept as potential roll targets but weren't actually used)
        final_groups = []
        for group in merged_groups:
            has_open_leg = any(not pos.get('_is_closed', False) for pos in group.positions)
            if has_open_leg or '(Rolled)' in group.strategy_type:
                final_groups.append(group)
            else:
                logger.info(f"  Removing fully-closed IC with no rolls: {group.order_id}")

        return final_groups

    def _assign_or_lookup_trade_id(self, order: Dict, closing_orders: Dict) -> str:
        """Assign or lookup trade ID for an order using TradeTracker."""
        order_id = order.get('orderId')

        logger.debug(f"")
        logger.debug(f"{'='*80}")
        logger.debug(f"🔑 TRADE ID ASSIGNMENT: Order {order_id}")
        logger.debug(f"{'='*80}")

        # Check if already mapped
        existing_id = trade_tracker.get_trade_id_for_order(order_id)
        if existing_id:
            logger.debug(f"✅ Order {order_id}: Already mapped to trade ID {existing_id}")
            return existing_id

        # Analyze order legs
        order_legs = order.get('orderLegCollection', [])
        closing_symbols = []
        opening_legs_info = {}

        logger.debug(f"📋 Order has {len(order_legs)} legs:")

        # Extract fill prices for this order
        leg_fill_prices = self._extract_fill_prices_from_order(order)

        for i, leg in enumerate(order_legs):
            symbol = leg.get('instrument', {}).get('symbol', '')
            instruction = leg.get('instruction', '')
            quantity = abs(leg.get('quantity', 0))

            logger.debug(f"   Leg {i+1}: {symbol[:30]} | {instruction} | qty={quantity}")

            if 'CLOSE' in instruction:
                closing_symbols.append(symbol)
                logger.debug(f"      ↳ 🔴 CLOSING leg")
            else:
                # Opening leg
                logger.debug(f"      ↳ 🟢 OPENING leg")
                opening_legs_info[symbol] = LegInfo(
                    symbol=symbol,
                    quantity=quantity,
                    side='SHORT' if 'SELL' in instruction else 'LONG',
                    strike=leg.get('instrument', {}).get('strikePrice', 0),
                    option_type=leg.get('instrument', {}).get('putCall', ''),
                    added_by_order=order_id
                )

        logger.debug(f"")
        logger.debug(f"📊 Summary:")
        logger.debug(f"   - Closing symbols: {closing_symbols}")
        logger.debug(f"   - Opening symbols: {list(opening_legs_info.keys())}")

        # If has closing legs, find the trade being closed/rolled
        if closing_symbols:
            logger.debug(f"")
            logger.debug(f"🔍 This is a ROLL/CLOSE order - searching for existing trade...")
            order_time = order.get('closeTime') or order.get('enteredTime')
            trade_id = trade_tracker.find_trade_for_closing_legs(closing_symbols, order_time)

            if trade_id:
                logger.info(f"✅ Order {order_id}: Linked to trade {trade_id}")
                trade_tracker.link_order_to_trade(trade_id, order_id, opening_legs_info, closing_symbols)
                return trade_id
            else:
                logger.warning(f"⚠️  Order {order_id}: Could not find matching trade for closing legs {closing_symbols} - creating new trade")

        # No match found or pure opening order - create new trade
        logger.debug(f"🆕 Creating NEW trade for order {order_id}")
        underlying = self._extract_underlying_from_order(order)
        strategy = self._extract_strategy_from_order(order)
        quantity = max((abs(leg.get('quantity', 0)) for leg in order_legs), default=0)

        trade_id = trade_tracker.assign_trade_id(
            order=order,
            underlying=underlying,
            strategy=strategy,
            quantity=quantity,
            current_legs=opening_legs_info
        )

        logger.info(f"✅ Order {order_id}: Created new trade {trade_id}")
        return trade_id

    def _group_by_trade_id(self, position_groups: List[PositionGroup]) -> List[PositionGroup]:
        """Group position groups by trade ID."""
        from collections import defaultdict

        trade_groups = defaultdict(list)

        # Group by trade_id
        for group in position_groups:
            if group.trade_id:
                trade_groups[group.trade_id].append(group)
            else:
                # Shouldn't happen, but handle gracefully
                logger.warning(f"Order {group.order_id} has no trade_id!")
                trade_groups[group.order_id].append(group)

        # Merge groups with same trade_id
        merged_groups = []
        for trade_id, groups in trade_groups.items():
            if len(groups) == 1:
                merged_groups.append(groups[0])
            else:
                # Multiple orders in same trade - merge them
                logger.info(f"🔄 Trade {trade_id}: Merging {len(groups)} orders into one position group")
                merged = self._merge_position_groups(groups, trade_id)
                merged_groups.append(merged)

        logger.info(f"✓ Grouped into {len(merged_groups)} trade groups")

        # Filter out fully closed position groups (only show active positions)
        active_groups = []
        for group in merged_groups:
            has_open_leg = any(not pos.get('_is_closed', False) for pos in group.positions)
            if has_open_leg:
                active_groups.append(group)
            else:
                logger.debug(f"  Filtering out fully closed trade: {group.order_id} ({group.underlying} {group.strategy_type})")

        logger.info(f"✓ Active position groups: {len(active_groups)} (filtered out {len(merged_groups) - len(active_groups)} closed)")
        return active_groups

    def _merge_position_groups(self, groups: List[PositionGroup], trade_id: str) -> PositionGroup:
        """Merge multiple PositionGroups from same trade into one."""
        # Use first group as base
        base = groups[0]

        # Collect all positions from all groups
        all_positions = []
        total_commission = 0
        total_fees = 0
        total_closing_commission = 0
        total_closing_fees = 0

        # Track roll numbers for proper sequencing
        roll_number = 0

        # Track which symbols we've already seen to avoid duplicates
        seen_positions = {}  # symbol -> position

        for i, group in enumerate(groups):
            for pos in group.positions:
                symbol = pos.get('instrument', {}).get('symbol', '')
                is_closed = pos.get('_is_closed', False)

                # For subsequent orders (rolls), mark new opening legs as rolled
                if i > 0 and not is_closed:
                    roll_number = i  # Roll number based on order sequence
                    pos['_is_rolled_leg'] = True
                    pos['_roll_number'] = roll_number
                    pos['_roll_order_id'] = group.order_id
                    pos['_roll_timestamp'] = group.order_data.get('closeTime', '') if group.order_data else ''

                # Deduplicate: If we already have this symbol, decide which to keep
                if symbol in seen_positions:
                    existing = seen_positions[symbol]
                    # Keep the original position, but update to closed if needed
                    if is_closed and not existing.get('_is_closed'):
                        existing['_is_closed'] = True
                        # Keep the original entry data but mark as closed
                    # Skip adding this duplicate
                    continue

                # Add this position
                seen_positions[symbol] = pos
                all_positions.append(pos)

            total_commission += group.commission
            total_fees += group.fees
            total_closing_commission += group.closing_commission
            total_closing_fees += group.closing_fees

        # Update strategy label to show it's rolled
        strategy_type = base.strategy_type
        if len(groups) > 1 and not strategy_type.endswith('(Rolled)'):
            strategy_type = f"{strategy_type} (Rolled)"

        logger.info(f"  ✓ Merged trade {trade_id}: {len(groups)} orders → {len(all_positions)} total positions")

        return PositionGroup(
            order_id=base.order_id,  # Use first order as primary
            underlying=base.underlying,
            strategy_type=strategy_type,
            positions=all_positions,
            order_data=base.order_data,
            is_stock=base.is_stock,
            order_fill_price=base.order_fill_price,
            commission=total_commission,
            fees=total_fees,
            closing_commission=total_closing_commission,
            closing_fees=total_closing_fees,
            trade_id=trade_id
        )

    def _get_order_leg_symbols(self, order: Dict) -> List[str]:
        """Extract all leg symbols from an order."""
        symbols = []
        for leg in order.get('orderLegCollection', []):
            instrument = leg.get('instrument', {})
            symbol = instrument.get('symbol', '')
            if symbol:
                symbols.append(symbol)
        return symbols

    def _match_positions_to_orders(
        self,
        positions: List[Dict],
        orders: List[Dict]
    ) -> Tuple[List[PositionGroup], List[Dict]]:
        """
        Match positions to filled orders to determine strategy grouping.

        Handles quantity-based matching since Schwab aggregates positions.
        If you place two separate orders for the same strikes, this will
        create two separate trade groups.

        Returns:
            Tuple of (matched position groups, unmatched positions)
        """
        # Create a map of position symbols to positions with remaining quantities
        position_map = {}
        remaining_quantities = {}  # Track remaining qty for each symbol

        for pos in positions:
            key = self._get_position_key(pos)
            if key:
                position_map[key] = pos
                # Get total quantity (long or short)
                long_qty = pos.get('longQuantity', 0)
                short_qty = pos.get('shortQuantity', 0)
                total_qty = long_qty + short_qty
                remaining_quantities[key] = total_qty

        position_groups = []

        # Deduplicate orders by order ID (in case API returns duplicates)
        seen_order_ids = set()
        unique_orders = []
        for order in orders:
            order_id = order.get('orderId')
            if order_id and order_id not in seen_order_ids:
                seen_order_ids.add(order_id)
                unique_orders.append(order)
            elif order_id:
                logger.debug(f"Skipping duplicate order {order_id}")

        # Filter out child orders - only process parent orders
        # Child orders are individual legs that are already included in the parent's orderLegCollection
        parent_order_ids = {o.get('orderId') for o in unique_orders if o.get('orderId')}
        filtered_orders = []

        for order in unique_orders:
            parent_id = order.get('parentOrderId')
            order_id = order.get('orderId')
            legs = order.get('orderLegCollection', [])
            # Skip if this is a child order (has a parent that's in our list)
            if parent_id and parent_id in parent_order_ids:
                if len(legs) >= 4:
                    logger.warning(f"⚠️  Skipping 4-LEG child order {order_id} (parent: {parent_id})")
                else:
                    logger.debug(f"Skipping child order {order_id} (parent: {parent_id})")
                continue
            filtered_orders.append(order)

        if len(filtered_orders) < len(orders):
            logger.info(f"Filtered out {len(orders) - len(filtered_orders)} duplicate/child orders")

        # Sort orders by close time (oldest first, so we match in chronological order)
        # This ensures trades are separated in the order they were placed
        sorted_orders = sorted(
            filtered_orders,
            key=lambda o: (o.get('closeTime', ''), o.get('orderId', '')),
            reverse=False  # False = oldest first (chronological)
        )

        logger.info(f"Processing {len(sorted_orders)} orders for position matching (filtered from {len(orders)} total)")

        # Debug: Log order details and check for duplicates
        order_ids_seen = set()
        for i, order in enumerate(sorted_orders[:15]):
            order_id = order.get('orderId')
            parent_id = order.get('parentOrderId')
            strategy = order.get('orderStrategyType')
            complex_strategy = order.get('complexOrderStrategyType')
            close_time = order.get('closeTime', '')[:19] if order.get('closeTime') else 'N/A'
            legs = order.get('orderLegCollection', [])
            child_orders = order.get('childOrderStrategies', [])

            # Check for duplicates
            if order_id in order_ids_seen:
                logger.warning(f"DUPLICATE Order {i+1}: ID={order_id}")
            else:
                order_ids_seen.add(order_id)

            # Get leg symbols
            leg_symbols = [leg.get('instrument', {}).get('symbol', '')[:20] for leg in legs[:4]]
            logger.info(f"Order {i+1}: ID={order_id}, Parent={parent_id}, Strategy={strategy}/{complex_strategy}, Legs={len(legs)} {leg_symbols}, Children={len(child_orders)}, Time={close_time}")

        # Match orders to positions
        for order in sorted_orders:
            order_id = order.get('orderId')
            order_legs = order.get('orderLegCollection', [])
            close_time = order.get('closeTime', '')[:19]

            if not order_legs:
                continue

            # ENHANCED DEBUG: Log every order being processed
            if len(order_legs) >= 4:
                logger.info(f"🔍 PROCESSING 4-LEG ORDER {order_id} (closed {close_time}):")
                for i, leg in enumerate(order_legs):
                    symbol = leg.get('instrument', {}).get('symbol', '')[:30]
                    qty = leg.get('quantity', 0)
                    instruction = leg.get('instruction', '')
                    available = remaining_quantities.get(leg.get('instrument', {}).get('symbol', ''), 0)
                    logger.info(f"  Leg {i+1}: {instruction} {qty} {symbol} (available: {available})")

            # Separate opening vs closing legs for tracking
            opening_legs = []
            closing_legs = []
            for leg in order_legs:
                instruction = leg.get('instruction', '')
                if 'CLOSE' in instruction:
                    closing_legs.append(leg)
                else:
                    opening_legs.append(leg)

            # If this was purely a closing order (no opening legs), skip it
            if not opening_legs:
                logger.debug(f"Order {order_id}: All legs are closing - skipping")
                continue

            # Log if we have closing legs (rolling order)
            if closing_legs:
                logger.info(f"Order {order_id}: Rolling order detected - {len(opening_legs)} opening legs + {len(closing_legs)} closing legs")

            # Check if all OPENING legs have matching positions with sufficient quantity
            # (We only match opening legs to positions, but we'll include closing legs in the display)
            matching_data = []
            all_matched = True
            failed_reason = None

            for leg in opening_legs:
                instrument = leg.get('instrument', {})
                symbol = instrument.get('symbol', '')
                leg_qty = abs(leg.get('quantity', 0))

                if not symbol or leg_qty == 0:
                    all_matched = False
                    failed_reason = f"leg missing symbol or quantity"
                    logger.debug(f"Order {order_id}: {failed_reason}")
                    break

                # Check if we have remaining quantity for this symbol
                available_qty = remaining_quantities.get(symbol, 0)
                if symbol in position_map and available_qty >= leg_qty:
                    matching_data.append({
                        'symbol': symbol,
                        'position': position_map[symbol],
                        'quantity': leg_qty,
                        'instruction': leg.get('instruction', '')
                    })
                else:
                    all_matched = False
                    if symbol not in position_map:
                        failed_reason = f"leg {symbol[:30]} not found in positions"
                        logger.warning(f"❌ Order {order_id}: {failed_reason}")
                    else:
                        failed_reason = f"leg {symbol[:30]} insufficient qty (need {leg_qty}, have {available_qty})"
                        logger.warning(f"❌ Order {order_id}: {failed_reason}")
                    break

            # Log if 4-leg order failed to match
            if len(order_legs) >= 4 and not all_matched:
                logger.error(f"🚨 4-LEG ORDER {order_id} FAILED TO MATCH: {failed_reason}")

            if all_matched and matching_data:
                # Create split positions for this order's quantities
                split_positions = []

                # Get order-level fill price (net credit/debit for spreads)
                order_fill_price = order.get('price', 0)
                logger.info(f"Order {order.get('orderId')} fill price: ${order_fill_price:.2f}")

                # Extract individual leg fill prices
                leg_fill_prices = self._extract_fill_prices_from_order(order)

                for idx, match in enumerate(matching_data):
                    symbol = match['symbol']
                    original_pos = match['position']
                    order_qty = match['quantity']

                    # Create a copy of the position with the order's quantity
                    split_pos = original_pos.copy()
                    split_pos['instrument'] = original_pos.get('instrument', {}).copy()

                    # Use order-specific leg fill price if available
                    if symbol in leg_fill_prices:
                        split_pos['averagePrice'] = leg_fill_prices[symbol]
                        logger.debug(f"  Using leg fill price for {symbol[:30]}: ${leg_fill_prices[symbol]:.2f}")
                    else:
                        logger.debug(f"  Using position average for {symbol[:30]}: ${split_pos.get('averagePrice', 0):.2f}")

                    # Determine if this is long or short
                    if 'BUY' in match['instruction']:
                        split_pos['longQuantity'] = order_qty
                        split_pos['shortQuantity'] = 0
                    else:
                        split_pos['longQuantity'] = 0
                        split_pos['shortQuantity'] = order_qty

                    # Adjust market value proportionally
                    original_qty = original_pos.get('longQuantity', 0) + original_pos.get('shortQuantity', 0)
                    if original_qty > 0:
                        proportion = order_qty / original_qty
                        original_market_value = original_pos.get('marketValue', 0)
                        split_pos['marketValue'] = original_market_value * proportion

                    split_positions.append(split_pos)

                    # Deduct from remaining quantity
                    remaining_quantities[symbol] -= order_qty
                    logger.debug(f"Matched order {order.get('orderId')} leg {symbol[:30]}: {order_qty} contracts (remaining: {remaining_quantities[symbol]})")

                # Add closing legs as synthetic positions (they're already closed, so no current position)
                for closing_leg in closing_legs:
                    instrument = closing_leg.get('instrument', {})
                    symbol = instrument.get('symbol', '')
                    leg_qty = abs(closing_leg.get('quantity', 0))
                    instruction = closing_leg.get('instruction', '')

                    # Create a synthetic position for display purposes
                    # This position was closed, so market value is 0
                    synthetic_pos = {
                        'instrument': instrument.copy(),
                        'averagePrice': leg_fill_prices.get(symbol, 0),  # Exit price
                        'longQuantity': leg_qty if 'BUY' in instruction else 0,
                        'shortQuantity': leg_qty if 'SELL' in instruction else 0,
                        'marketValue': 0,  # Position is closed
                        '_is_closed': True  # Flag to mark this as a closed leg
                    }
                    split_positions.append(synthetic_pos)
                    logger.debug(f"Added closing leg {symbol[:30]}: {instruction} {leg_qty} contracts @ ${leg_fill_prices.get(symbol, 0):.2f}")

                # Determine underlying and strategy type from order
                underlying = self._extract_underlying_from_order(order)
                strategy_type = self._extract_strategy_from_order(order)
                # Check if all positions are equity-like (stocks, ETFs, mutual funds)
                is_stock = all(
                    pos.get('instrument', {}).get('assetType') in ['EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND']
                    for pos in split_positions
                )

                opening_leg_count = len(opening_legs) if 'opening_legs' in locals() else len(matching_data)
                total_leg_count = len(order.get('orderLegCollection', []))
                if opening_leg_count < total_leg_count:
                    logger.info(f"✓ Matched order {order.get('orderId')}: {underlying} {strategy_type} with {len(split_positions)} opening legs (+ {total_leg_count - opening_leg_count} closing legs) @ ${order_fill_price:.2f}")
                else:
                    logger.info(f"✓ Matched order {order.get('orderId')}: {underlying} {strategy_type} with {len(split_positions)} legs, {len(matching_data)} total contracts @ ${order_fill_price:.2f}")

                position_groups.append(PositionGroup(
                    order_id=str(order.get('orderId', '')),
                    underlying=underlying,
                    strategy_type=strategy_type,
                    positions=split_positions,
                    order_data=order,
                    is_stock=is_stock,
                    order_fill_price=order_fill_price
                ))

        # Collect unmatched positions (positions with remaining quantity)
        unmatched = []
        for symbol, pos in position_map.items():
            remaining_qty = remaining_quantities.get(symbol, 0)
            if remaining_qty > 0:
                # Create a position for the remaining quantity
                remaining_pos = pos.copy()
                remaining_pos['instrument'] = pos.get('instrument', {}).copy()

                original_qty = pos.get('longQuantity', 0) + pos.get('shortQuantity', 0)
                if original_qty > 0:
                    proportion = remaining_qty / original_qty
                    original_market_value = pos.get('marketValue', 0)
                    remaining_pos['marketValue'] = original_market_value * proportion

                    # Adjust quantities
                    if pos.get('longQuantity', 0) > 0:
                        remaining_pos['longQuantity'] = remaining_qty
                        remaining_pos['shortQuantity'] = 0
                    else:
                        remaining_pos['longQuantity'] = 0
                        remaining_pos['shortQuantity'] = remaining_qty

                unmatched.append(remaining_pos)
                instrument = pos.get('instrument', {})
                asset_type = instrument.get('assetType', '')
                put_call = instrument.get('putCall', '')
                logger.info(f"⚠ Unmatched: {symbol[:30]} - {remaining_qty} contracts ({asset_type} {put_call})")

        logger.info(f"✓ Summary: {len(position_groups)} matched groups, {len(unmatched)} unmatched positions")

        # Log what strategy types were identified
        strategy_counts = {}
        for group in position_groups:
            strategy_counts[group.strategy_type] = strategy_counts.get(group.strategy_type, 0) + 1
        logger.info(f"  Matched strategies: {strategy_counts}")

        return position_groups, unmatched

    def _extract_fill_prices_from_order(self, order: Dict) -> Dict[str, float]:
        """
        Extract actual fill prices for each leg from order execution data.

        Args:
            order: Order dictionary from Schwab

        Returns:
            Dictionary mapping symbol to fill price
        """
        fill_prices = {}
        order_id = order.get('orderId', 'N/A')

        try:
            # Check orderActivityCollection for execution details
            activities = order.get('orderActivityCollection', [])
            if not activities:
                return fill_prices

            order_legs = order.get('orderLegCollection', [])
            if not order_legs:
                return fill_prices

            # Process execution legs from first activity
            # (For multi-fill orders, we'll use the first fill's prices)
            activity = activities[0]
            execution_legs = activity.get('executionLegs', [])

            for exec_leg in execution_legs:
                # IMPORTANT: legId is 1-indexed, not 0-indexed!
                leg_id = exec_leg.get('legId', 0)
                price = exec_leg.get('price', 0)

                if leg_id > 0 and price > 0:
                    # Map to order leg (legId is 1-indexed)
                    leg_index = leg_id - 1
                    if leg_index < len(order_legs):
                        leg = order_legs[leg_index]
                        symbol = leg.get('instrument', {}).get('symbol', '')
                        if symbol:
                            fill_prices[symbol] = price
                            logger.debug(f"Order {order_id}: {symbol[:30]} filled @ ${price:.2f}")

            if fill_prices:
                logger.debug(f"✓ Extracted {len(fill_prices)} leg prices for order {order_id}")
            else:
                logger.debug(f"No leg prices extracted from order {order_id}")

        except Exception as e:
            logger.error(f"Error extracting fill prices from order {order_id}: {e}")
            import traceback
            traceback.print_exc()

        return fill_prices

    def _extract_commission_from_order(self, order: Dict) -> Dict[str, float]:
        """
        Extract commission and fees from order execution data.

        Returns:
            Dict with 'commission' and 'fees' keys
        """
        order_id = order.get('orderId', 'UNKNOWN')
        commission = 0.0
        fees = 0.0

        try:
            # Check orderActivityCollection for commission details
            activities = order.get('orderActivityCollection', [])
            if not activities:
                logger.debug(f"Order {order_id}: No orderActivityCollection found")
                return {'commission': 0.0, 'fees': 0.0}

            # Log the first activity structure to see what fields are available
            if activities:
                sample_activity = activities[0]
                logger.debug(f"📋 Order {order_id} activity keys: {list(sample_activity.keys())}")
                # Log full activity structure for debugging (only for first few orders)
                import json
                logger.debug(f"📋 Full activity structure: {json.dumps(sample_activity, indent=2, default=str)}")

            # Sum commissions/fees across all activities (for multi-fill orders)
            for activity in activities:
                # Extract commission (Schwab's trading commission)
                activity_commission = activity.get('commissionAmount', 0.0)
                if activity_commission:
                    commission += float(activity_commission)

                # Extract regulatory fees (SEC, FINRA, etc.)
                activity_fees = activity.get('chargeAmount', 0.0)
                if activity_fees:
                    fees += float(activity_fees)

                # Try alternative field names
                if not activity_commission:
                    activity_commission = activity.get('commission', 0.0)
                    if activity_commission:
                        commission += float(activity_commission)
                        logger.debug(f"Found commission in 'commission' field: ${activity_commission:.2f}")

                if not activity_fees:
                    activity_fees = activity.get('fees', 0.0) or activity.get('regulatoryFees', 0.0)
                    if activity_fees:
                        fees += float(activity_fees)
                        logger.debug(f"Found fees in alternative field: ${activity_fees:.2f}")

            if commission > 0 or fees > 0:
                logger.info(f"💰 Order {order_id}: commission=${commission:.2f}, fees=${fees:.2f}")
            else:
                logger.debug(f"⚠️ Order {order_id}: No commission/fees found in API response")

        except Exception as e:
            logger.error(f"Error extracting commission from order {order_id}: {e}")
            import traceback
            traceback.print_exc()

        return {'commission': commission, 'fees': fees}

    def _extract_underlying_from_order(self, order: Dict) -> str:
        """Extract underlying symbol from order."""
        for leg in order.get('orderLegCollection', []):
            instrument = leg.get('instrument', {})
            # For options, get underlying
            underlying = instrument.get('underlyingSymbol', '')
            if underlying:
                return underlying.replace('$', '')
            # For stocks, use the symbol
            symbol = instrument.get('symbol', '')
            if symbol:
                return symbol.replace('$', '')
        return "UNKNOWN"

    def _extract_strategy_from_order(self, order: Dict) -> str:
        """Extract strategy type from order."""
        # Check order strategy type field
        order_strategy = order.get('orderStrategyType', '')
        complex_strategy = order.get('complexOrderStrategyType', '')

        # Analyze legs to determine strategy - consider ALL legs to get accurate count
        legs = order.get('orderLegCollection', [])

        # Detect ROLL trades (mixed CLOSE and OPEN instructions in same order)
        closing_legs = [leg for leg in legs if 'CLOSE' in leg.get('instruction', '')]
        opening_legs = [leg for leg in legs if 'CLOSE' not in leg.get('instruction', '')]

        if closing_legs and opening_legs:
            # This is a ROLL trade - determine what's being rolled
            num_closing = len(closing_legs)
            num_opening = len(opening_legs)

            # Check if it's puts or calls
            closing_types = set(leg.get('instrument', {}).get('putCall', '') for leg in closing_legs)
            opening_types = set(leg.get('instrument', {}).get('putCall', '') for leg in opening_legs)

            if num_closing == 2 and num_opening == 2:
                # Rolling a spread (2 legs)
                if 'CALL' in closing_types and 'CALL' in opening_types:
                    return 'ROLL - Call Spread'
                elif 'PUT' in closing_types and 'PUT' in opening_types:
                    return 'ROLL - Put Spread'
                else:
                    return 'ROLL - Mixed Spread'
            elif num_closing == 4 and num_opening == 4:
                # Rolling entire iron condor
                return 'ROLL - Iron Condor'
            else:
                return f'ROLL - {num_closing} to {num_opening} legs'

        if complex_strategy:
            # Map Schwab complex strategy types
            strategy_map = {
                'IRON_CONDOR': 'IRON_CONDOR',
                'VERTICAL': 'VERTICAL_SPREAD',
                'STRADDLE': 'STRADDLE',
                'STRANGLE': 'STRANGLE',
                'BUTTERFLY': 'BUTTERFLY',
                'CONDOR': 'CONDOR',
                'COVERED': 'COVERED_CALL',
                'COLLAR': 'COLLAR',
            }
            for key, value in strategy_map.items():
                if key in complex_strategy.upper():
                    return value
        if len(legs) == 1:
            instrument = legs[0].get('instrument', {})
            asset_type = instrument.get('assetType', '')
            put_call = instrument.get('putCall', '')

            # Check if it's a stock/ETF (equity-like instruments)
            if asset_type in ['EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND']:
                return 'STOCK'

            # If it's an option, classify by instruction
            if put_call in ['PUT', 'CALL'] or asset_type == 'OPTION':
                instruction = legs[0].get('instruction', '')
                if 'SELL' in instruction:
                    if put_call == 'PUT':
                        return 'CASH_SECURED_PUT'
                    elif put_call == 'CALL':
                        return 'NAKED_CALL'
                return 'SINGLE_LEG'

            # Default to STOCK for unknown single-leg instruments (better than SINGLE_LEG)
            return 'STOCK'

        if len(legs) == 4:
            # Likely iron condor
            return 'IRON_CONDOR'

        if len(legs) == 2:
            # Could be vertical spread or strangle
            instruments = [leg.get('instrument', {}) for leg in legs]
            put_calls = [i.get('putCall', '') for i in instruments]
            if put_calls[0] == put_calls[1]:
                return 'VERTICAL_SPREAD'
            else:
                return 'STRANGLE'

        return 'MULTI_LEG'

    def _create_group_for_unmatched(self, positions: List[Dict]) -> List[PositionGroup]:
        """Create position groups for unmatched positions.

        Groups positions by underlying AND expiration to keep separate
        trades separate (e.g., 3 iron condors on same underlying but
        different expirations become 3 groups).
        """
        groups = []

        # Group by underlying first
        by_underlying = {}
        for pos in positions:
            instrument = pos.get('instrument', {})
            asset_type = instrument.get('assetType', '')

            if asset_type == 'OPTION':
                underlying = instrument.get('underlyingSymbol', '').replace('$', '')
                # Normalize underlying for display (SPXW -> SPX)
                underlying = self._normalize_underlying(underlying)
            else:
                underlying = instrument.get('symbol', '').replace('$', '')

            if not underlying:
                continue

            if underlying not in by_underlying:
                by_underlying[underlying] = []
            by_underlying[underlying].append(pos)

        # For each underlying, further group by expiration
        for underlying, pos_list in by_underlying.items():
            # Separate stocks/ETFs from options
            stock_positions = [p for p in pos_list if p.get('instrument', {}).get('assetType') in ['EQUITY', 'COLLECTIVE_INVESTMENT', 'ETF', 'MUTUAL_FUND']]
            option_positions = [p for p in pos_list if p.get('instrument', {}).get('assetType') == 'OPTION']

            # Debug logging for SPY
            if underlying == 'SPY':
                logger.debug(f"SPY positions: {len(pos_list)} total")
                for p in pos_list:
                    asset_type = p.get('instrument', {}).get('assetType')
                    logger.debug(f"  - assetType={asset_type}, symbol={p.get('instrument', {}).get('symbol')}")
                logger.debug(f"  Stock positions: {len(stock_positions)}, Option positions: {len(option_positions)}")

            # Create group for stocks
            if stock_positions:
                groups.append(PositionGroup(
                    order_id=f"unmatched_{underlying}_stock",
                    underlying=underlying,
                    strategy_type='STOCK',
                    positions=stock_positions,
                    order_data=None,
                    is_stock=True
                ))

            # Group options by expiration
            by_expiration = {}
            for pos in option_positions:
                exp_date = self._parse_expiration(pos.get('instrument', {}))
                exp_key = exp_date.isoformat() if exp_date else 'unknown'
                if exp_key not in by_expiration:
                    by_expiration[exp_key] = []
                by_expiration[exp_key].append(pos)

            # Now try to identify strategies within each expiration group
            for exp_key, exp_positions in by_expiration.items():
                # Try to identify iron condors (4 legs: 2 puts + 2 calls)
                strategy_groups = self._identify_strategy_groups(exp_positions)

                for idx, strategy_pos in enumerate(strategy_groups):
                    strategy_type = self._identify_strategy_from_positions(strategy_pos)
                    groups.append(PositionGroup(
                        order_id=f"unmatched_{underlying}_{exp_key}_{idx}",
                        underlying=underlying,
                        strategy_type=strategy_type,
                        positions=strategy_pos,
                        order_data=None,
                        is_stock=False
                    ))

        return groups

    def _identify_strategy_groups(self, positions: List[Dict]) -> List[List[Dict]]:
        """
        Try to identify and group positions into strategies.

        For iron condors: Look for groups of 4 legs (2 puts + 2 calls)
        with matching quantities.

        Important: This should only be called for unmatched positions.
        Matched positions are already grouped by order.
        """
        if len(positions) == 0:
            return []

        if len(positions) <= 4:
            # Single strategy or less, return as one group
            return [positions]

        # Separate by put/call
        puts = []
        calls = []
        for pos in positions:
            instrument = pos.get('instrument', {})
            put_call = instrument.get('putCall', '')
            if put_call == 'PUT':
                puts.append(pos)
            elif put_call == 'CALL':
                calls.append(pos)

        # Try to match iron condors by quantity
        # An iron condor has: long put, short put, short call, long call
        # All with same absolute quantity

        # Group by quantity
        puts_by_qty = {}
        for p in puts:
            qty = p.get('longQuantity', 0) + p.get('shortQuantity', 0)
            if qty > 0:
                if qty not in puts_by_qty:
                    puts_by_qty[qty] = []
                puts_by_qty[qty].append(p)

        calls_by_qty = {}
        for c in calls:
            qty = c.get('longQuantity', 0) + c.get('shortQuantity', 0)
            if qty > 0:
                if qty not in calls_by_qty:
                    calls_by_qty[qty] = []
                calls_by_qty[qty].append(c)

        # Find matching iron condors
        strategy_groups = []
        used_positions = set()

        for qty in sorted(puts_by_qty.keys(), reverse=True):  # Process larger quantities first
            if qty in calls_by_qty:
                qty_puts = [p for p in puts_by_qty[qty] if id(p) not in used_positions]
                qty_calls = [c for c in calls_by_qty[qty] if id(c) not in used_positions]

                # Try to form iron condors (need 2 puts and 2 calls of same qty)
                while len(qty_puts) >= 2 and len(qty_calls) >= 2:
                    # Take 2 puts and 2 calls
                    ic_legs = qty_puts[:2] + qty_calls[:2]
                    qty_puts = qty_puts[2:]
                    qty_calls = qty_calls[2:]

                    # Mark as used
                    for leg in ic_legs:
                        used_positions.add(id(leg))

                    strategy_groups.append(ic_legs)

                # If odd number of puts/calls remain, group them as vertical spreads
                if len(qty_puts) >= 2:
                    for i in range(0, len(qty_puts) - 1, 2):
                        spread_legs = qty_puts[i:i+2]
                        for leg in spread_legs:
                            used_positions.add(id(leg))
                        strategy_groups.append(spread_legs)

                if len(qty_calls) >= 2:
                    for i in range(0, len(qty_calls) - 1, 2):
                        spread_legs = qty_calls[i:i+2]
                        for leg in spread_legs:
                            used_positions.add(id(leg))
                        strategy_groups.append(spread_legs)

        # Add any remaining positions individually
        remaining = [p for p in positions if id(p) not in used_positions]
        for p in remaining:
            strategy_groups.append([p])

        return strategy_groups if strategy_groups else [positions]

    def _identify_strategy_from_positions(self, positions: List[Dict]) -> str:
        """Identify strategy type from position data."""
        option_positions = []
        stock_positions = []

        for pos in positions:
            instrument = pos.get('instrument', {})
            asset_type = instrument.get('assetType', '')
            if asset_type == 'OPTION':
                option_positions.append(pos)
            else:
                stock_positions.append(pos)

        if not option_positions and stock_positions:
            return 'STOCK'

        if len(option_positions) == 1 and not stock_positions:
            instrument = option_positions[0].get('instrument', {})
            put_call = instrument.get('putCall', '')
            quantity = option_positions[0].get('shortQuantity', 0)
            if quantity > 0:
                if put_call == 'PUT':
                    return 'CASH_SECURED_PUT'
                return 'NAKED_CALL'
            return 'SINGLE_LEG'

        if len(option_positions) == 4:
            return 'IRON_CONDOR'

        if len(option_positions) == 2:
            instruments = [pos.get('instrument', {}) for pos in option_positions]
            put_calls = [i.get('putCall', '') for i in instruments]
            if put_calls[0] == put_calls[1]:
                return 'VERTICAL_SPREAD'
            return 'STRANGLE'

        if stock_positions and option_positions:
            return 'COVERED_CALL'

        return 'MULTI_LEG'

    async def get_trades_closed_on_date(self, target_date: date) -> Tuple[List[AggregatedTradeGroup], List[AggregatedTradeGroup]]:
        """
        Get trades that were closed on a specific date.

        Args:
            target_date: The date to filter by

        Returns:
            Tuple of (options_trades, stock_positions) closed on that date
        """
        positions = await self.get_positions()

        # Get filled orders closed on this date
        all_orders = await self.get_filled_orders()

        # Filter orders closed on target date
        filtered_orders = []
        for order in all_orders:
            close_time_str = order.get('closeTime', '')
            if close_time_str:
                try:
                    # Parse ISO format: 2026-02-09T14:30:00+0000
                    close_dt = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                    if close_dt.date() == target_date:
                        filtered_orders.append(order)
                except Exception as e:
                    logger.debug(f"Error parsing close time {close_time_str}: {e}")

        logger.info(f"📅 Found {len(filtered_orders)} orders closed on {target_date}")

        if not filtered_orders:
            return [], []

        # Create position lookup map
        position_map = {}
        for pos in positions:
            key = self._get_position_key(pos)
            if key:
                position_map[key] = pos

        # Track which positions were matched to orders
        used_positions = set()

        # Convert orders to trade groups
        order_groups = self._convert_orders_to_trade_groups(filtered_orders, position_map, used_positions)

        # Fetch underlying prices
        unique_underlyings = set(g.underlying for g in order_groups)
        underlying_prices = await self._fetch_underlying_prices(list(unique_underlyings))

        # Convert to AggregatedTradeGroup and separate options from stocks
        options_trades = []
        stock_positions = []

        for group in order_groups:
            try:
                underlying_price = underlying_prices.get(group.underlying)
                trade_group = self._create_trade_group(group, underlying_price)
                if trade_group:
                    if group.is_stock:
                        stock_positions.append(trade_group)
                    else:
                        options_trades.append(trade_group)
            except Exception as e:
                logger.error(f"Error creating trade group for {group.underlying}: {e}")

        return options_trades, stock_positions

    async def get_aggregated_trades(self) -> Tuple[List[AggregatedTradeGroup], List[AggregatedTradeGroup]]:
        """
        Get positions as aggregated trade groups for the dashboard.

        Auto-syncs trade matches database to properly handle rolls.

        Returns:
            Tuple of (options_trades, stock_positions)
        """
        positions = await self.get_positions()
        orders = await self.get_filled_orders()

        if not orders:
            return [], []

        # Auto-sync trade matches database (runs trade matcher on recent orders)
        # This ensures rolls are properly tracked and won't show as duplicate cards
        try:
            from monitoring.services.trade_matcher import TradeMatchingEngine
            from monitoring.services.trade_match_persistence import TradeMatchPersistence

            logger.info("🔄 Auto-syncing trade matches database...")
            matcher = TradeMatchingEngine()
            matched_trades = matcher.match_orders(orders)

            persistence = TradeMatchPersistence()
            persistence.save_matches(matched_trades)

            logger.info(f"✅ Synced {len(matched_trades)} trade matches to database")
        except Exception as e:
            logger.error(f"Failed to sync trade matches: {e}")
            # Continue anyway - dashboard will work without database sync

        # Create position lookup map
        position_map = {}
        for pos in positions:
            key = self._get_position_key(pos)
            if key:
                position_map[key] = pos

        # Track which positions were matched to orders
        used_positions = set()

        # Convert orders to trade groups (order-based view)
        order_groups = self._convert_orders_to_trade_groups(orders, position_map, used_positions)

        # Create groups for unmatched positions (positions not tied to any displayed order)
        unmatched_positions = [pos for key, pos in position_map.items() if key not in used_positions]
        logger.info(f"🔍 Unmatched positions: {len(unmatched_positions)} (total positions: {len(position_map)}, used: {len(used_positions)})")

        # Filter out positions that are rolled legs in the database
        # These belong to existing trades and shouldn't be shown as unmatched
        if unmatched_positions:
            unmatched_symbols = [p.get('instrument', {}).get('symbol', '') for p in unmatched_positions]
            logger.info(f"   Unmatched symbols: {unmatched_symbols}")
            logger.info(f"   Used symbols: {sorted(list(used_positions))[:20]}...")

            # Check database for rolled legs
            try:
                from monitoring.services.trade_match_persistence import TradeMatchPersistence
                persistence = TradeMatchPersistence()
                with persistence.db.get_connection() as conn:
                    # Get all open legs from database (includes rolled legs)
                    db_symbols = conn.execute("""
                        SELECT DISTINCT symbol FROM trade_legs
                        WHERE status = 'OPEN'
                    """).fetchall()
                    db_symbols_set = {row[0] for row in db_symbols}

                    # Filter out positions that are in the database
                    filtered_unmatched = []
                    for pos in unmatched_positions:
                        symbol = pos.get('instrument', {}).get('symbol', '')
                        if symbol not in db_symbols_set:
                            filtered_unmatched.append(pos)
                        else:
                            logger.info(f"   ✓ Skipping {symbol[:30]} - found in database as rolled leg")

                    unmatched_positions = filtered_unmatched
                    logger.info(f"   → {len(unmatched_positions)} truly unmatched (after filtering rolled legs)")
            except Exception as e:
                logger.debug(f"Could not filter rolled legs from unmatched: {e}")

        unmatched_groups = self._create_group_for_unmatched(unmatched_positions)

        # Combine order-based groups and unmatched position groups
        all_groups = order_groups + unmatched_groups

        logger.info(f"✓ Trade groups: {len(order_groups)} from orders + {len(unmatched_groups)} from unmatched positions = {len(all_groups)} total")

        # Fetch underlying prices for all unique underlyings
        unique_underlyings = set(g.underlying for g in all_groups)
        underlying_prices = await self._fetch_underlying_prices(list(unique_underlyings))

        # Collect all open option symbols across all groups for Greeks fetch
        all_option_symbols = []
        for group in all_groups:
            if not group.is_stock:
                for pos in group.positions:
                    instrument = pos.get('instrument', {})
                    if instrument.get('assetType') == 'OPTION' and not pos.get('_is_closed', False):
                        sym = instrument.get('symbol', '')
                        if sym:
                            all_option_symbols.append(sym)

        # Fetch live Greeks for all open option symbols
        option_greeks = await self._fetch_option_greeks(list(set(all_option_symbols)))

        # Fetch persisted entry deltas from DB and persist new ones
        entry_deltas = {}
        try:
            from monitoring.services.trade_match_persistence import TradeMatchPersistence
            persistence = TradeMatchPersistence()
            with persistence.db.get_connection() as conn:
                # Add entry_delta column if it doesn't exist yet
                try:
                    conn.execute("ALTER TABLE trade_legs ADD COLUMN IF NOT EXISTS entry_delta DOUBLE")
                except Exception:
                    pass  # Column already exists

                # Fetch all persisted entry deltas
                rows = conn.execute("""
                    SELECT symbol, entry_delta FROM trade_legs
                    WHERE status = 'OPEN' AND entry_delta IS NOT NULL
                """).fetchall()
                entry_deltas = {row[0]: row[1] for row in rows}
                logger.info(f"📊 Loaded {len(entry_deltas)} persisted entry deltas from DB")

                # Persist entry delta for any open legs that don't have one yet
                if option_greeks:
                    null_delta_rows = conn.execute("""
                        SELECT leg_id, symbol FROM trade_legs
                        WHERE status = 'OPEN' AND entry_delta IS NULL
                    """).fetchall()
                    updated_count = 0
                    for leg_id, symbol in null_delta_rows:
                        if symbol in option_greeks:
                            delta_val = option_greeks[symbol].get('delta', 0)
                            if delta_val:
                                conn.execute(
                                    "UPDATE trade_legs SET entry_delta = ? WHERE leg_id = ?",
                                    (delta_val, leg_id)
                                )
                                entry_deltas[symbol] = delta_val
                                updated_count += 1
                    if updated_count:
                        logger.info(f"📊 Persisted entry delta for {updated_count} legs (first sync)")
        except Exception as e:
            logger.warning(f"Could not fetch/persist entry deltas: {e}")

        # Convert to AggregatedTradeGroup and separate options from stocks
        options_trades = []
        stock_positions = []

        for group in all_groups:
            try:
                underlying_price = underlying_prices.get(group.underlying)
                trade_group = self._create_trade_group(group, underlying_price, option_greeks, entry_deltas)
                if trade_group:
                    if group.is_stock:
                        stock_positions.append(trade_group)
                    else:
                        options_trades.append(trade_group)
            except Exception as e:
                logger.error(f"Error creating trade group for {group.underlying}: {e}")
                import traceback
                traceback.print_exc()

        return options_trades, stock_positions

    def _normalize_underlying(self, symbol: str) -> str:
        """Normalize underlying symbol for price lookups.

        SPXW options trade on SPX, so map SPXW -> SPX for quotes.
        """
        # Map weekly/mini symbols to their main underlying
        symbol_map = {
            'SPXW': 'SPX',
            'NDXP': 'NDX',
            'RUTW': 'RUT',
        }
        return symbol_map.get(symbol, symbol)

    async def _fetch_underlying_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for underlying symbols."""
        prices = {}
        if not symbols:
            return prices

        try:
            # Normalize symbols and convert to Schwab format
            schwab_symbols = []
            symbol_mapping = {}  # Maps schwab symbol back to original

            for s in symbols:
                normalized = self._normalize_underlying(s)
                if normalized in ('SPX', 'NDX', 'RUT', 'VIX'):
                    schwab_sym = f'${normalized}'
                else:
                    schwab_sym = normalized
                schwab_symbols.append(schwab_sym)
                symbol_mapping[schwab_sym] = s  # Map back to original
                # Also map the normalized version
                if s != normalized:
                    symbol_mapping[schwab_sym] = s

            # Deduplicate
            schwab_symbols = list(set(schwab_symbols))

            logger.info(f"📊 Fetching quotes for {len(schwab_symbols)} symbols: {schwab_symbols}")

            quotes = await schwab_client.get_quotes(schwab_symbols)

            logger.info(f"📊 Received {len(quotes)} quotes from Schwab")

            for schwab_sym, data in quotes.items():
                quote_data = data.get('quote', {})
                price = quote_data.get('lastPrice') or quote_data.get('mark') or quote_data.get('closePrice')
                if price:
                    # Store under the normalized symbol (without $)
                    clean = schwab_sym.replace('$', '')
                    prices[clean] = price
                    logger.info(f"  ✓ {schwab_sym} → ${price:.2f}")
                    # Also store under original symbols that map to this
                    for orig_sym in symbols:
                        if self._normalize_underlying(orig_sym) == clean:
                            prices[orig_sym] = price
                else:
                    logger.warning(f"  ✗ {schwab_sym}: No price found in quote data")
                    logger.debug(f"    Quote data: {quote_data}")

            if not prices:
                logger.error(f"🚨 Failed to fetch any prices! Requested: {symbols}")
                logger.error(f"    Quote response: {quotes}")

        except Exception as e:
            logger.error(f"Error fetching underlying prices: {e}")
            import traceback
            traceback.print_exc()

        return prices

    async def _fetch_option_greeks(self, option_symbols: List[str]) -> Dict[str, Dict]:
        """Batch-fetch Greeks (delta, gamma, theta, vega) for option symbols via Schwab quotes API.

        Args:
            option_symbols: List of full OCC option symbols

        Returns:
            Dict mapping symbol -> {'delta': float, 'gamma': float, 'theta': float, 'vega': float}
        """
        greeks = {}
        if not option_symbols:
            return greeks

        try:
            # Schwab quotes API accepts option symbols directly
            logger.info(f"📊 Fetching Greeks for {len(option_symbols)} option symbols")
            quotes = await schwab_client.get_quotes(option_symbols)

            for symbol, data in quotes.items():
                quote_data = data.get('quote', {})
                greeks[symbol] = {
                    'delta': quote_data.get('delta', 0) or 0,
                    'gamma': quote_data.get('gamma', 0) or 0,
                    'theta': quote_data.get('theta', 0) or 0,
                    'vega': quote_data.get('vega', 0) or 0,
                }
                logger.debug(f"  Greeks for {symbol[:30]}: delta={greeks[symbol]['delta']:.4f}")

            logger.info(f"✓ Fetched Greeks for {len(greeks)}/{len(option_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error fetching option Greeks: {e}")
            import traceback
            traceback.print_exc()

        return greeks

    def _create_trade_group(self, group: PositionGroup, underlying_price: Optional[float] = None, option_greeks: Optional[Dict] = None, entry_deltas: Optional[Dict] = None) -> Optional[AggregatedTradeGroup]:
        """
        Create an AggregatedTradeGroup from a PositionGroup.

        Args:
            group: PositionGroup containing matched positions
            underlying_price: Current price of the underlying
            option_greeks: Dict of symbol -> greeks from Schwab quotes API
            entry_deltas: Dict of symbol -> entry delta from database

        Returns:
            AggregatedTradeGroup or None
        """
        legs = []
        total_market_value = 0
        total_cost = 0
        total_delta = 0
        total_theta = 0
        total_gamma = 0
        total_vega = 0
        expiration = None

        # Track per-contract values (sum of entry prices per contract)
        total_entry_per_contract = 0
        total_mark_per_contract = 0
        num_contracts = 0

        # Track realized P&L from closed legs
        realized_pnl = 0

        # Track entry/cost for OPEN legs only (for adjusted entry display)
        open_legs_cost = 0
        open_legs_entry_per_contract = 0
        open_legs_count = 0

        # Use order fill price if available (for matched orders)
        # This is the actual filled price for this specific order
        order_fill_price = group.order_fill_price if group.order_fill_price > 0 else None
        if order_fill_price:
            logger.info(f"Using order fill price for {group.underlying} {group.strategy_type}: ${order_fill_price:.2f}")

        for pos in group.positions:
            instrument = pos.get('instrument', {})
            asset_type = instrument.get('assetType', '')

            # Extract leg info
            quantity = pos.get('longQuantity', 0) - pos.get('shortQuantity', 0)
            avg_price = pos.get('averagePrice', 0)
            market_value = pos.get('marketValue', 0)

            # Calculate per-contract mark
            if asset_type == 'OPTION':
                # Options: market_value / quantity / 100 (per contract)
                mark_per_contract = abs(market_value / quantity / 100) if quantity != 0 else 0
            else:
                # Stocks: market_value / quantity (current price per share)
                mark_per_contract = abs(market_value / quantity) if quantity != 0 else 0

            leg_data = {
                'symbol': instrument.get('symbol', ''),
                'underlying': group.underlying,
                'quantity': int(quantity),
                'entry_price': avg_price,
                'current_mark': mark_per_contract,
                '_is_closed': pos.get('_is_closed', False),  # Flag for closed legs
                '_is_rolled_leg': pos.get('_is_rolled_leg', False),  # Flag for rolled legs
                '_closing_price': pos.get('_closing_price', 0),  # Closing fill price
                '_roll_number': pos.get('_roll_number', 0),  # Which roll (1, 2, 3, etc.)
                '_roll_order_id': pos.get('_roll_order_id', ''),  # Roll order ID
                '_roll_timestamp': pos.get('_roll_timestamp', ''),  # When the roll happened
            }

            if asset_type == 'OPTION':
                leg_data['option_type'] = instrument.get('putCall', '')

                # Get strike - try strikePrice field first, then parse from symbol
                strike = instrument.get('strikePrice', 0)
                if not strike or strike == 0:
                    strike = self._parse_strike_from_symbol(instrument.get('symbol', ''))
                leg_data['strike'] = strike

                # Parse expiration
                exp_date = self._parse_expiration(instrument)
                if exp_date:
                    leg_data['expiration'] = exp_date
                    if expiration is None or exp_date < expiration:
                        expiration = exp_date

                # Estimate Greeks (Schwab doesn't provide live Greeks for positions)
                # Theta estimation: credit spreads lose ~1/DTE of their value per day
                estimated_theta = 0
                if exp_date and quantity != 0:
                    leg_dte = (exp_date - date.today()).days
                    if leg_dte > 0:
                        # For short options: positive theta (earn money from decay)
                        # For long options: negative theta (lose money from decay)
                        # Rough estimate: theta ≈ option_value / days_to_expiration
                        option_value = mark_per_contract * abs(quantity)
                        if quantity < 0:  # Short position - positive theta
                            estimated_theta = option_value / leg_dte
                        else:  # Long position - negative theta
                            estimated_theta = -option_value / leg_dte

                        # Accumulate total theta
                        total_theta += estimated_theta

                # Get live Greeks from Schwab quotes API
                sym = leg_data['symbol']
                if option_greeks and sym in option_greeks:
                    sym_greeks = option_greeks[sym]
                    leg_data['delta'] = sym_greeks.get('delta', 0)
                    leg_data['gamma'] = sym_greeks.get('gamma', 0)
                    leg_data['vega'] = sym_greeks.get('vega', 0)
                    # Use live theta if available, fall back to estimated
                    live_theta = sym_greeks.get('theta', 0)
                    if live_theta:
                        leg_data['theta'] = live_theta
                        # Override estimated with live theta for totals
                        total_theta = total_theta - estimated_theta + live_theta
                    else:
                        leg_data['theta'] = estimated_theta
                    total_delta += leg_data['delta'] * quantity
                else:
                    leg_data['delta'] = 0
                    leg_data['theta'] = estimated_theta
                    leg_data['gamma'] = 0
                    leg_data['vega'] = 0

                # Entry delta from database (persisted on first sync)
                if entry_deltas and sym in entry_deltas:
                    leg_data['entry_delta'] = entry_deltas[sym]
                else:
                    leg_data['entry_delta'] = None

                # For options, entry price is per contract (premium)
                # Short positions have negative entry (credit), long have positive (debit)
                if quantity < 0:
                    total_entry_per_contract += avg_price  # Credit (we received this)
                    total_mark_per_contract += mark_per_contract
                else:
                    total_entry_per_contract -= avg_price  # Debit (we paid this)
                    total_mark_per_contract -= mark_per_contract

                # Track number of contracts (use max quantity seen)
                num_contracts = max(num_contracts, abs(quantity))
            else:
                # Stock positions
                leg_data['option_type'] = None
                leg_data['strike'] = None

                # For stocks, entry is purchase price per share, mark is current price
                # Add to totals for display (stocks use simple addition, not credit/debit logic)
                total_entry_per_contract += avg_price
                total_mark_per_contract += mark_per_contract

                # Track number of shares (for display purposes, treat as "contracts")
                num_contracts = max(num_contracts, abs(quantity))

            legs.append(leg_data)

            # Calculate cost basis properly accounting for direction
            # Short positions: we RECEIVED premium (negative cost = credit)
            # Long positions: we PAID premium (positive cost = debit)
            multiplier = 100 if asset_type == 'OPTION' else 1
            leg_cost = 0
            if quantity < 0:  # Short - received premium
                leg_cost = -avg_price * abs(quantity) * multiplier
            else:  # Long - paid premium
                leg_cost = avg_price * abs(quantity) * multiplier

            # Accumulate totals
            total_cost += leg_cost

            # Check if this leg is closed
            is_closed = pos.get('_is_closed', False)

            if is_closed:
                # For closed legs, calculate realized P&L properly
                # Need both entry and exit prices
                closing_price = pos.get('_closing_price', 0)

                if closing_price and closing_price > 0:
                    # We have the actual closing price - calculate P&L correctly
                    # For shorts (negative qty): profit when close < open
                    # For longs (positive qty): profit when close > open
                    # P&L = (close - open) * qty * multiplier
                    leg_realized_pnl = (closing_price - avg_price) * quantity * 100
                    realized_pnl += leg_realized_pnl
                    logger.info(f"  💰 REALIZED P&L: {instrument.get('symbol', '')[:30]}")
                    logger.info(f"      Entry: ${avg_price:.4f}, Exit: ${closing_price:.4f}")
                    logger.info(f"      Quantity: {quantity:.0f}, Multiplier: 100")
                    logger.info(f"      Calculation: ({closing_price:.4f} - {avg_price:.4f}) × {quantity:.0f} × 100 = ${leg_realized_pnl:.2f}")
                    logger.info(f"      Running total realized: ${realized_pnl:.2f}")
                else:
                    # No closing price available - use old method (entry only)
                    # This will be inaccurate but better than nothing
                    realized_pnl += -leg_cost
                    logger.debug(f"  Closed leg {instrument.get('symbol', '')[:20]}: realized P&L = ${-leg_cost:.2f} (estimated, no exit price)")
            else:
                # Accumulate for open legs only
                total_market_value += market_value
                open_legs_cost += leg_cost
                open_legs_count += 1

                if asset_type == 'OPTION':
                    if quantity < 0:
                        open_legs_entry_per_contract += avg_price  # Credit
                    else:
                        open_legs_entry_per_contract -= avg_price  # Debit

        if not legs:
            return None

        # Sort legs by strike price (puts first, then calls, both sorted by strike ascending)
        # This makes iron condors display as: long put, short put, short call, long call
        def sort_key(leg):
            strike = leg.get('strike', 0) or 0
            option_type = leg.get('option_type', '')
            # Puts come before calls, both sorted by strike
            # Use 0 for puts, 1 for calls to sort puts first
            type_order = 0 if option_type == 'PUT' else 1
            return (type_order, strike)

        legs = sorted(legs, key=sort_key)

        # Calculate P&L
        # For credit positions: total_cost is negative (credit received)
        # market_value for shorts is negative (liability)
        # P&L = what we received + current market value (which is negative for liability)
        # e.g., received $200 credit, now worth -$150 liability = $200 + (-$150) = $50 profit
        # But actually: P&L = credit_received - amount_to_close = -total_cost - abs(market_value)
        # Let's use: P&L = -total_cost + total_market_value for credit strategies
        # Or simpler: P&L = -(current_value_to_close - credit_received) = credit - cost_to_close

        # Using Schwab's standard: unrealized P&L = market_value - cost_basis
        # But now only for OPEN legs
        unrealized_pnl = total_market_value - open_legs_cost

        # Calculate total commission costs (will be displayed separately)
        total_opening_costs = group.commission + group.fees
        total_closing_costs = group.closing_commission + group.closing_fees

        # Total P&L = realized (from closed legs) + unrealized (from open legs)
        # NOTE: This is GROSS P&L (before commissions) - commissions will be displayed separately
        total_pnl = realized_pnl + unrealized_pnl

        if total_opening_costs > 0 or total_closing_costs > 0:
            logger.info(f"  💸 Commissions tracked: opening=${total_opening_costs:.2f}, closing=${total_closing_costs:.2f}")
            logger.info(f"  📊 Gross P&L: ${total_pnl:.2f}, Net P&L (after costs): ${total_pnl - total_opening_costs - total_closing_costs:.2f}")

        # Calculate days to expiration
        dte = (expiration - date.today()).days if expiration else 0

        # Determine if this is a credit or debit strategy
        # If we have order_fill_price, use that to determine
        # Otherwise fall back to total_cost calculation
        if order_fill_price:
            # For spreads: positive price = net credit, negative = net debit
            # Actually, Schwab returns credit as positive for credit spreads
            # So we need to check the order type
            is_credit_strategy = total_cost < 0  # Our calculated cost is negative for credits
            if is_credit_strategy:
                entry_credit = order_fill_price * num_contracts * 100
                entry_debit = 0
            else:
                entry_credit = 0
                entry_debit = order_fill_price * num_contracts * 100
        else:
            # Entry credit/debit (total position value)
            # total_cost < 0 means net credit, total_cost > 0 means net debit
            entry_credit = abs(total_cost) if total_cost < 0 else 0
            entry_debit = total_cost if total_cost > 0 else 0

        # Profit capture percentage (how much of max profit captured)
        # Use TOTAL P&L (realized + unrealized) for profit capture
        # For credit strategies, max_profit = credit received
        max_profit = entry_credit if entry_credit > 0 else entry_debit
        if max_profit > 0 and entry_credit > 0:
            # For credit strategies: profit capture = total_profit / max_possible_profit
            profit_capture_pct = (total_pnl / max_profit) * 100
        else:
            profit_capture_pct = 0

        # Alert color
        alert_color = self._determine_alert_color(profit_capture_pct, legs)

        # Per-contract values for display
        # IMPORTANT: For partial closes, track both original and adjusted entry
        original_entry_per_contract = order_fill_price if order_fill_price else abs(total_entry_per_contract)

        if order_fill_price and not group.is_stock:
            # If we have closed legs, calculate adjusted entry for open legs only
            if open_legs_count > 0 and open_legs_count < len(legs):
                # Partial close: use adjusted entry for remaining open legs
                # Calculate entry per contract for just the open legs
                entry_per_contract = abs(open_legs_entry_per_contract)
                logger.info(f"  Original entry: ${original_entry_per_contract:.2f}/contract, Adjusted entry (open legs): ${entry_per_contract:.2f}/contract")
            else:
                # All legs open or all legs closed: use order fill price
                entry_per_contract = order_fill_price

            # Calculate current mark per contract from market value (open legs only)
            mark_per_contract = abs(total_market_value) / (num_contracts * 100) if num_contracts > 0 else 0
        else:
            # Fallback to calculated values (or for stocks, use direct values)
            if open_legs_count > 0 and open_legs_count < len(legs):
                # Partial close: use adjusted entry
                entry_per_contract = abs(open_legs_entry_per_contract)
            else:
                entry_per_contract = abs(total_entry_per_contract) if total_entry_per_contract > 0 else abs(total_entry_per_contract)
            mark_per_contract = abs(total_mark_per_contract)

            # For stocks, adjust mark_per_contract calculation if using order_fill_price path
            if group.is_stock and order_fill_price:
                entry_per_contract = order_fill_price
                # For stocks: market value / num shares (not * 100)
                mark_per_contract = abs(total_market_value) / num_contracts if num_contracts > 0 else 0

        # Debug logging
        logger.info(f"📊 Trade group {group.underlying} {group.strategy_type} (Order: {group.order_id}):")
        logger.info(f"  ✓ Total legs: {len(legs)} (open: {open_legs_count}, closed: {len(legs) - open_legs_count})")
        logger.info(f"  ✓ Realized P&L: ${realized_pnl:.2f}")
        logger.info(f"  ✓ Unrealized P&L: ${unrealized_pnl:.2f}")
        logger.info(f"  ✓ Total P&L: ${total_pnl:.2f}")
        logger.debug(f"  order_fill_price={order_fill_price if order_fill_price else 'N/A'}")
        logger.debug(f"  total_cost={total_cost:.2f}, open_legs_cost={open_legs_cost:.2f}")
        logger.debug(f"  total_market_value={total_market_value:.2f}")
        logger.debug(f"  entry_credit={entry_credit:.2f}, open_legs={open_legs_count}/{len(legs)}")
        logger.debug(f"  entry_per_contract={entry_per_contract:.2f}, mark_per_contract={mark_per_contract:.2f}")
        logger.debug(f"  num_contracts={num_contracts}")

        # Extract order time if available
        order_time = None
        if group.order_data:
            close_time = group.order_data.get('closeTime', '')
            if close_time:
                try:
                    order_time = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                except:
                    order_time = datetime.now()

        return AggregatedTradeGroup(
            trade_group_id=f"schwab_{group.order_id}",
            underlying=group.underlying,
            strategy_type=group.strategy_type,
            expiration=expiration,
            days_to_expiration=dte,
            entry_credit=entry_credit,
            entry_debit=entry_debit,
            max_profit=max_profit,
            max_loss=0,
            current_mark=abs(total_market_value),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            day_pnl=0,
            profit_capture_pct=profit_capture_pct,
            total_delta=total_delta,
            total_theta=total_theta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            theta_per_hour=abs(total_theta) / 6.5 if total_theta else 0,
            alert_color=alert_color,
            short_strike_breached=False,
            pnl_history=[total_pnl],  # Show total P&L (realized + unrealized) in sparkline
            legs=legs,
            underlying_price=underlying_price,
            entry_per_contract=entry_per_contract,
            original_entry_per_contract=original_entry_per_contract,
            mark_per_contract=mark_per_contract,
            num_contracts=num_contracts if num_contracts > 0 else 1,
            commission=group.commission,
            fees=group.fees,
            closing_commission=group.closing_commission,
            closing_fees=group.closing_fees,
            total_costs=group.commission + group.fees + group.closing_commission + group.closing_fees,
            opened_at=order_time if order_time else datetime.now(),
            status="OPEN",
            strategy_name=group.strategy_type
        )

    def _parse_expiration(self, instrument: Dict) -> Optional[date]:
        """Parse expiration date from Schwab instrument data."""
        exp_str = instrument.get('expirationDate')
        if exp_str:
            try:
                return datetime.strptime(exp_str[:10], '%Y-%m-%d').date()
            except:
                pass

        # Parse from OCC symbol
        symbol = instrument.get('symbol', '')
        match = re.search(r'(\d{6})[PC]', symbol)
        if match:
            date_str = match.group(1)
            try:
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                return date(year, month, day)
            except:
                pass

        return None

    def _parse_strike_from_symbol(self, symbol: str) -> float:
        """
        Parse strike price from OCC option symbol.

        OCC format: SPXW  260204P06805000
        The last 8 digits after P/C are the strike:
        - First 5 digits: integer part (06805 = 6805)
        - Last 3 digits: decimal part (000 = .000)
        So 06805000 = 6805.000

        Args:
            symbol: OCC option symbol

        Returns:
            Strike price as float, or 0 if parsing fails
        """
        if not symbol:
            return 0

        # Match: 6 digits (date) + P or C + 8 digits (strike)
        match = re.search(r'\d{6}[PC](\d{8})', symbol)
        if match:
            strike_str = match.group(1)
            try:
                # First 5 digits integer, last 3 decimal
                integer_part = int(strike_str[:5])
                decimal_part = int(strike_str[5:]) / 1000
                return integer_part + decimal_part
            except:
                pass

        return 0

    def _determine_alert_color(self, profit_capture_pct: float, legs: List[Dict]) -> AlertColor:
        """Determine alert color based on position status."""
        if profit_capture_pct >= 50:
            return AlertColor.GREEN

        for leg in legs:
            delta = abs(leg.get('delta', 0))
            if delta > 0.25:
                return AlertColor.ORANGE

        if profit_capture_pct < 0:
            return AlertColor.AMBER

        return AlertColor.NONE

    async def get_global_metrics(self) -> Dict:
        """Get global portfolio metrics."""
        options_trades, stock_positions = await self.get_aggregated_trades()
        all_trades = options_trades + stock_positions

        # Calculate separate unrealized for options and stocks
        options_unrealized = sum(t.unrealized_pnl for t in options_trades)
        stocks_unrealized = sum(t.unrealized_pnl for t in stock_positions)
        total_unrealized = options_unrealized + stocks_unrealized

        # Calculate realized P&L from all trades
        total_realized = sum(t.realized_pnl for t in all_trades)

        total_theta_per_hour = sum(t.theta_per_hour for t in all_trades)

        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)

        return {
            'daily_realized': total_realized,  # Now includes realized from partial closes
            'options_unrealized': options_unrealized,
            'stocks_unrealized': stocks_unrealized,
            'open_unrealized': total_unrealized,
            'theta_pulse': total_theta_per_hour,
            'market_time': now_et.strftime('%H:%M:%S ET'),
            'day_pnl': 0,
            'num_open_trades': len(all_trades),
            'num_options_trades': len(options_trades),
            'num_stock_positions': len(stock_positions)
        }


# Global instance
schwab_positions = SchwabPositionsService()
