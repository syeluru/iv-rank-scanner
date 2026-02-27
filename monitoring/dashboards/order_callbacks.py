"""Callbacks for order placement functionality."""
from dash import callback, Input, Output, State, ALL, ctx, no_update, html
import asyncio
from loguru import logger

from execution.broker_api.schwab_client import schwab_client


def register_order_callbacks(app):
    """Register all order placement callbacks."""

    # Enable/disable close button based on checkbox selections
    @app.callback(
        Output({'type': 'close-legs-btn', 'index': ALL}, 'disabled'),
        Input({'type': 'leg-checkbox', 'trade': ALL, 'leg': ALL}, 'value'),
        prevent_initial_call=True
    )
    def toggle_close_button(checkbox_values):
        """Enable close button if any legs are selected."""
        logger.info(f"🔘 toggle_close_button called with {len(checkbox_values)} checkbox values")

        # Group checkboxes by trade_id
        from collections import defaultdict
        trades = defaultdict(list)

        for prop_id in ctx.inputs_list[0]:
            trade_id = prop_id['id']['trade']
            idx = ctx.inputs_list[0].index(prop_id)
            value = checkbox_values[idx]
            trades[trade_id].append(len(value) > 0)  # True if checked

        # Return disabled=False if any checkbox is selected for each trade
        result = [not any(trades[trade_id]) for trade_id in trades]
        logger.info(f"Button disabled states: {result}")
        return result

    # Open order modal when close button is clicked
    @app.callback(
        [Output('order-modal', 'is_open'),
         Output('order-legs-summary', 'children'),
         Output('suggested-price', 'children'),
         Output('selected-legs-store', 'data')],
        [Input({'type': 'close-legs-btn', 'index': ALL}, 'n_clicks')],
        [State({'type': 'leg-checkbox', 'trade': ALL, 'leg': ALL}, 'value'),
         State('live-positions-store', 'data'),
         State('open-orders-store', 'data')],
        prevent_initial_call=True
    )
    def open_order_modal(n_clicks, checkbox_values, live_positions_data, open_orders_data):
        """Open order modal and populate with selected legs."""
        logger.info(f"📋 open_order_modal called: n_clicks={n_clicks}, checkbox_values={len(checkbox_values) if checkbox_values else 0}")

        if not n_clicks or not any(n_clicks):
            logger.warning("No button clicks detected")
            return no_update, no_update, no_update, no_update

        # Find which button was clicked
        logger.info(f"Button clicked! Finding which one...")
        button_idx = next((i for i, n in enumerate(n_clicks) if n), None)
        if button_idx is None:
            logger.error("Could not find clicked button")
            return no_update, no_update, no_update, no_update

        logger.info(f"Button {button_idx} was clicked")

        # Get all trades (live + open orders)
        all_trades = []
        if live_positions_data:
            all_trades.extend(live_positions_data)
        if open_orders_data:
            all_trades.extend(open_orders_data)

        # Get selected legs with full data
        selected_legs = []
        suggested_price = 0.0

        for prop_id in ctx.states_list[0]:
            trade_id = prop_id['id']['trade']
            leg_idx = prop_id['id']['leg']
            idx = ctx.states_list[0].index(prop_id)
            value = checkbox_values[idx]

            if len(value) > 0:  # Checkbox is checked
                # Find the trade and leg data
                trade = next((t for t in all_trades if t.get('trade_group_id') == trade_id), None)
                if trade and 'legs' in trade:
                    legs = trade['legs']
                    if leg_idx < len(legs):
                        leg_data = legs[leg_idx]

                        # Store full leg data for order creation
                        selected_legs.append({
                            'trade_id': trade_id,
                            'leg_idx': leg_idx,
                            'symbol': leg_data.get('symbol'),
                            'quantity': leg_data.get('quantity', 0),
                            'current_mark': leg_data.get('current_mark', 0),
                            'option_type': leg_data.get('option_type'),
                            'strike': leg_data.get('strike')
                        })

                        # Calculate suggested price
                        # For closing: if short (negative qty), we pay to close; if long (positive), we receive
                        qty = leg_data.get('quantity', 0)
                        mark = leg_data.get('current_mark', 0)
                        if qty < 0:  # Short position - we pay to close
                            suggested_price += abs(mark)
                        else:  # Long position - we receive to close
                            suggested_price -= mark

        if not selected_legs:
            logger.warning("No legs were selected")
            return no_update, no_update, no_update, no_update

        logger.info(f"✓ Found {len(selected_legs)} selected legs")

        # Create summary of selected legs
        summary_items = []
        for leg in selected_legs:
            qty = leg['quantity']
            strike = leg.get('strike', 0)
            opt_type = leg.get('option_type', '')
            position_type = "Short" if qty < 0 else "Long"

            if strike and opt_type:
                leg_desc = f"{position_type} {abs(qty):.0f} x ${strike:.0f} {opt_type} ({leg['symbol']})"
            else:
                leg_desc = f"{position_type} {abs(qty)} x {leg['symbol']}"

            summary_items.append(html.Li(leg_desc, style={'marginBottom': '4px'}))

        summary = html.Div([
            html.H6(f"{len(selected_legs)} leg(s) selected for closing:", style={'marginBottom': '12px'}),
            html.Ul(summary_items, style={'listStyleType': 'none', 'paddingLeft': '0'})
        ])

        # Format suggested price
        # Net debit (we pay) is positive, net credit (we receive) is negative
        if suggested_price > 0:
            suggested_price_text = f"${abs(suggested_price):.2f} debit"
        else:
            suggested_price_text = f"${abs(suggested_price):.2f} credit"

        logger.info(f"✅ Opening modal with {len(selected_legs)} legs, price: {suggested_price_text}")
        return True, summary, suggested_price_text, selected_legs

    # Pre-populate limit price with suggested price
    @app.callback(
        Output('limit-price-input', 'value'),
        Input('selected-legs-store', 'data'),
        prevent_initial_call=True
    )
    def populate_limit_price(selected_legs):
        """Pre-populate limit price based on current marks."""
        if not selected_legs:
            return None

        # Calculate suggested price per contract
        net_price = 0.0
        for leg in selected_legs:
            qty = leg.get('quantity', 0)
            mark = leg.get('current_mark', 0)

            # Short positions: we pay to close
            # Long positions: we receive to close
            if qty < 0:
                net_price += abs(mark)
            else:
                net_price -= mark

        # Return absolute value as the limit price
        # (the instruction determines if it's debit or credit)
        return round(abs(net_price), 2)

    # Show/hide limit price based on order type
    @app.callback(
        Output('limit-price-section', 'style'),
        Input('order-type-input', 'value')
    )
    def toggle_limit_price(order_type):
        """Show limit price input only for LIMIT orders."""
        if order_type == 'LIMIT':
            return {'marginBottom': '16px'}
        return {'display': 'none'}

    # Update order summary
    @app.callback(
        Output('order-summary-text', 'children'),
        [Input('order-type-input', 'value'),
         Input('limit-price-input', 'value'),
         Input('order-duration-input', 'value')],
        State('selected-legs-store', 'data')
    )
    def update_order_summary(order_type, limit_price, duration, selected_legs):
        """Update the order summary text."""
        if not selected_legs:
            return "No legs selected"

        # Calculate total quantity and estimated value
        total_contracts = 0
        estimated_value = 0.0

        for leg in selected_legs:
            qty = abs(leg.get('quantity', 0))
            mark = leg.get('current_mark', 0)
            original_qty = leg.get('quantity', 0)

            total_contracts += qty

            # Calculate value: short positions = pay to close, long = receive
            if original_qty < 0:
                estimated_value += mark * qty * 100
            else:
                estimated_value -= mark * qty * 100

        summary_lines = [
            f"Order Type: {order_type}",
            f"Duration: {duration}",
            f"Legs to close: {len(selected_legs)}",
            f"Total contracts: {int(total_contracts)}",
        ]

        if order_type == 'LIMIT' and limit_price:
            try:
                price_val = float(limit_price)
                total_value = price_val * total_contracts * 100
                summary_lines.append(f"Limit Price: ${price_val:.2f} per contract")
                if estimated_value >= 0:
                    summary_lines.append(f"Total Debit: ${total_value:.2f}")
                else:
                    summary_lines.append(f"Total Credit: ${total_value:.2f}")
            except (ValueError, TypeError):
                summary_lines.append(f"Limit Price: (enter price)")
        elif order_type == 'MARKET':
            if estimated_value >= 0:
                summary_lines.append(f"Est. Total Debit: ~${abs(estimated_value):.2f}")
            else:
                summary_lines.append(f"Est. Total Credit: ~${abs(estimated_value):.2f}")

        return html.Div([html.P(line, style={'margin': '4px 0'}) for line in summary_lines])

    # Place order when submit is clicked
    @app.callback(
        [Output('confirmation-modal', 'is_open'),
         Output('confirmation-title', 'children'),
         Output('confirmation-message', 'children')],
        Input('order-submit-btn', 'n_clicks'),
        [State('order-type-input', 'value'),
         State('limit-price-input', 'value'),
         State('order-duration-input', 'value'),
         State('selected-legs-store', 'data')],
        prevent_initial_call=True
    )
    def place_order(n_clicks, order_type, limit_price, duration, selected_legs):
        """Place the order via Schwab API."""
        if not n_clicks or not selected_legs:
            return False, "", ""

        try:
            # Validate limit price for LIMIT orders
            if order_type == 'LIMIT':
                if not limit_price or float(limit_price) <= 0:
                    return True, "Order Failed", html.Div([
                        html.P("❌ Invalid limit price", style={'color': '#ff4444', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                        html.P("Please enter a valid limit price for LIMIT orders.", style={'marginTop': '12px', 'color': '#ff4444'}),
                    ])

            # Build order legs
            order_legs = []
            for leg_data in selected_legs:
                symbol = leg_data.get('symbol')
                quantity = abs(leg_data.get('quantity', 0))  # Always positive for quantity
                original_qty = leg_data.get('quantity', 0)

                if not symbol or quantity == 0:
                    continue

                # Determine instruction based on position direction
                # Short positions (negative qty) -> BUY_TO_CLOSE
                # Long positions (positive qty) -> SELL_TO_CLOSE
                instruction = "BUY_TO_CLOSE" if original_qty < 0 else "SELL_TO_CLOSE"

                order_legs.append({
                    "instruction": instruction,
                    "quantity": int(quantity),
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "OPTION"
                    }
                })

            if not order_legs:
                return True, "Order Failed", html.Div([
                    html.P("❌ No valid legs to close", style={'color': '#ff4444', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                    html.P("Please select at least one leg to close.", style={'marginTop': '12px', 'color': '#ff4444'}),
                ])

            # Determine if this is a multi-leg order (spread)
            is_multi_leg = len(order_legs) > 1

            # Build Schwab order
            schwab_order = {
                "session": "NORMAL",
                "duration": duration,
                "orderStrategyType": "SINGLE",
                "orderLegCollection": order_legs
            }

            # For multi-leg orders, Schwab requires NET_DEBIT or NET_CREDIT
            # Single leg orders use LIMIT or MARKET
            if order_type == 'LIMIT':
                if is_multi_leg:
                    # For spreads: the price is the NET price for the spread
                    # Determine if we're paying (NET_DEBIT) or receiving (NET_CREDIT)
                    # Count BUY vs SELL instructions
                    buy_count = sum(1 for leg in order_legs if 'BUY' in leg['instruction'])
                    sell_count = sum(1 for leg in order_legs if 'SELL' in leg['instruction'])

                    # For closing spreads:
                    # - If closing a credit spread (short + long), we typically pay to close -> NET_DEBIT
                    # - Use the price as entered (should be net price for the spread)
                    if buy_count >= sell_count:
                        schwab_order["orderType"] = "NET_DEBIT"
                    else:
                        schwab_order["orderType"] = "NET_CREDIT"

                    logger.info(f"Multi-leg order: {buy_count} BUY legs, {sell_count} SELL legs -> {schwab_order['orderType']}")
                else:
                    schwab_order["orderType"] = "LIMIT"

                schwab_order["price"] = f"{float(limit_price):.2f}"
            else:
                # Market order
                schwab_order["orderType"] = "MARKET"

            logger.info(f"Placing order: {schwab_order}")

            # Submit order via Schwab client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(schwab_client.place_order(order=schwab_order))
            finally:
                loop.close()

            # Extract order ID from response
            order_id = response.get('orderId', 'Unknown')

            # Check for paper trading mode
            if response.get('status') == 'PAPER_TRADING':
                return True, "Order Simulated (Paper Trading)", html.Div([
                    html.P("⚠️ Paper trading mode - order simulated", style={'color': '#ffaa00', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                    html.P("Order was validated but not submitted to live market.", style={'marginTop': '12px'}),
                    html.P(f"Type: {order_type}", style={'margin': '4px 0'}),
                    html.P(f"Legs: {len(order_legs)}", style={'margin': '4px 0'}),
                    html.P(f"Duration: {duration}", style={'margin': '4px 0'}),
                ])

            return True, "Order Placed Successfully", html.Div([
                html.P("✅ Your order has been placed!", style={'color': '#00ff88', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                html.P(f"Order ID: {order_id}", style={'marginTop': '12px'}),
                html.P(f"Type: {order_type}", style={'margin': '4px 0'}),
                html.P(f"Legs: {len(order_legs)}", style={'margin': '4px 0'}),
                html.P(f"Duration: {duration}", style={'margin': '4px 0'}),
                html.Hr(style={'margin': '12px 0'}),
                html.P("The dashboard will update on next refresh.", style={'fontSize': '0.9em', 'color': '#888'}),
            ])

        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            logger.error(f"Order that failed: {schwab_order if 'schwab_order' in locals() else 'Order not built yet'}")

            # Try to extract more details from the error
            error_detail = str(e)
            if hasattr(e, '__dict__'):
                logger.error(f"Error attributes: {e.__dict__}")

            return True, "Order Failed", html.Div([
                html.P("❌ Failed to place order", style={'color': '#ff4444', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                html.P(f"Error: {error_detail}", style={'marginTop': '12px', 'color': '#ff4444', 'fontSize': '0.9em'}),
                html.P("Check the logs for more details.", style={'marginTop': '8px', 'fontSize': '0.85em', 'color': '#888'}),
            ])

    # Close modals
    @app.callback(
        Output('order-modal', 'is_open', allow_duplicate=True),
        Input('order-cancel-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_order_modal(n_clicks):
        if n_clicks:
            return False
        return no_update

    @app.callback(
        Output('confirmation-modal', 'is_open', allow_duplicate=True),
        Input('confirmation-close-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_confirmation_modal(n_clicks):
        if n_clicks:
            return False
        return no_update
