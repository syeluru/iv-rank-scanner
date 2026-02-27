"""Helper to create spread breakdown display for iron condors and multi-leg strategies."""
from dash import html
from typing import List, Dict, Tuple


def calculate_spread_metrics(legs: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Calculate entry and mark for put spread and call spread separately.

    Returns:
        Tuple of (put_spread_metrics, call_spread_metrics)
        Each dict contains: entry, mark, is_open, quantity
    """
    put_legs = [leg for leg in legs if leg.get('option_type') == 'PUT']
    call_legs = [leg for leg in legs if leg.get('option_type') == 'CALL']

    def calc_spread(legs_subset):
        if not legs_subset:
            return None

        entry = 0
        mark = 0
        quantity = 0
        all_closed = all(leg.get('_is_closed', False) for leg in legs_subset)

        for leg in legs_subset:
            qty = abs(leg.get('quantity', 0))
            entry_price = leg.get('entry_price') or 0
            is_closed = leg.get('_is_closed', False)
            is_short = leg.get('quantity', 0) < 0
            symbol = leg.get('symbol', '')

            # Use closing price if leg is closed, otherwise use current mark
            if is_closed:
                closing_price = leg.get('_closing_price') or 0
                leg_mark = closing_price or 0
                from loguru import logger
                logger.debug(f"  Closed leg {symbol[:10]}: entry=${entry_price:.2f}, closing_price=${closing_price:.2f}, using mark=${leg_mark:.2f}")
            else:
                leg_mark = leg.get('current_mark') or 0

            # For credit spreads: short - long = net credit
            if is_short:
                entry += entry_price
                mark += leg_mark
            else:
                entry -= entry_price
                mark -= leg_mark

            quantity = max(quantity, qty)

        return {
            'entry': abs(entry),
            'mark': abs(mark),
            'is_open': not all_closed,
            'quantity': quantity
        }

    put_spread = calc_spread(put_legs)
    call_spread = calc_spread(call_legs)

    return put_spread, call_spread


def create_spread_breakdown(legs: List[Dict], realized_pnl: float = 0, unrealized_pnl: float = 0, num_contracts: int = 1,
                           commission: float = 0, fees: float = 0, closing_commission: float = 0, closing_fees: float = 0,
                           show_spreads: bool = True, show_totals: bool = True) -> html.Div:
    """
    Create a two-column spread breakdown showing put and call spreads.

    Args:
        legs: List of leg dictionaries
        realized_pnl: Realized P&L
        unrealized_pnl: Unrealized P&L
        num_contracts: Number of contracts
        commission: Opening commission
        fees: Opening regulatory fees
        closing_commission: Closing commission (if closed)
        closing_fees: Closing regulatory fees (if closed)

    Returns:
        Dash HTML component
    """
    put_spread, call_spread = calculate_spread_metrics(legs)

    def create_spread_column(spread_data, spread_type: str):
        if not spread_data:
            return html.Div()

        is_open = spread_data['is_open']
        entry = spread_data['entry']
        mark = spread_data['mark']
        quantity = spread_data['quantity']

        # Colors based on open/closed status
        label_color = '#00bfff' if is_open else '#666'
        entry_color = '#00bfff' if is_open else '#888'
        mark_color = '#00ff88' if is_open else '#666'

        return html.Div([
            html.Div(
                spread_type,
                style={
                    'fontSize': '0.9em',
                    'fontWeight': 'bold',
                    'color': label_color,
                    'marginBottom': '6px',
                    'textTransform': 'uppercase'
                }
            ),
            html.Div([
                html.Span("Entry: ", style={'fontSize': '0.85em', 'color': '#888'}),
                html.Span(
                    f"${entry:.2f}",
                    style={'fontSize': '0.9em', 'color': entry_color, 'fontWeight': 'bold'}
                ),
            ], style={'marginBottom': '4px'}),
            html.Div([
                html.Span("Mark: ", style={'fontSize': '0.85em', 'color': '#888'}),
                html.Span(
                    f"${mark:.2f}",
                    style={
                        'fontSize': '0.9em',
                        'color': mark_color,
                        'fontWeight': 'bold'
                    }
                ),
                html.Span(
                    " (closed)" if not is_open else "",
                    style={'fontSize': '0.75em', 'color': '#666', 'marginLeft': '5px', 'fontStyle': 'italic'}
                ),
            ], style={'marginBottom': '4px'}),
            html.Div([
                html.Span(
                    "OPEN" if is_open else "CLOSED",
                    style={
                        'fontSize': '0.75em',
                        'color': '#00ff88' if is_open else '#666',
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgba(0, 255, 136, 0.1)' if is_open else 'rgba(102, 102, 102, 0.1)',
                        'padding': '2px 6px',
                        'borderRadius': '3px'
                    }
                ),
            ]),
        ], style={
            'padding': '10px',
            'backgroundColor': 'rgba(0, 191, 255, 0.05)' if is_open else 'rgba(102, 102, 102, 0.05)',
            'borderRadius': '4px',
            'border': f'1px solid {"rgba(0, 191, 255, 0.3)" if is_open else "rgba(102, 102, 102, 0.2)"}'
        })

    # Calculate totals - use passed-in values for accuracy
    gross_pnl = realized_pnl + unrealized_pnl
    total_costs = commission + fees + closing_commission + closing_fees
    net_pnl = gross_pnl - total_costs

    # Calculate entry/mark per contract
    total_entry = 0
    total_mark = 0

    if put_spread:
        total_entry += put_spread['entry']
        if put_spread['is_open']:
            total_mark += put_spread['mark']

    if call_spread:
        total_entry += call_spread['entry']
        if call_spread['is_open']:
            total_mark += call_spread['mark']

    return html.Div([
        # Spread columns (conditionally shown)
        html.Div([
            html.Div([
                create_spread_column(put_spread, "Put Spread")
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                create_spread_column(call_spread, "Call Spread")
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
        ]) if show_spreads else None,

        # Totals section (conditionally shown)
        html.Div([
            html.Div("TOTALS", style={
                'fontSize': '0.85em',
                'fontWeight': 'bold',
                'color': '#888',
                'marginTop': '12px',
                'marginBottom': '6px',
                'textTransform': 'uppercase'
            }),
            html.Div([
                html.Div([
                    html.Span("Entry: ", style={'fontSize': '0.85em', 'color': '#888'}),
                    html.Span(f"${total_entry:.2f}", style={'fontSize': '0.9em', 'color': '#00bfff', 'fontWeight': 'bold'}),
                ], style={'marginBottom': '4px'}),
                html.Div([
                    html.Span("Mark: ", style={'fontSize': '0.85em', 'color': '#888'}),
                    html.Span(f"${total_mark:.2f}", style={'fontSize': '0.9em', 'color': '#00ff88', 'fontWeight': 'bold'}),
                ], style={'marginBottom': '4px'}),
                # Realized P&L (if any)
                html.Div([
                    html.Span("Realized: ", style={'fontSize': '0.85em', 'color': '#888'}),
                    html.Span(
                        f"+${realized_pnl:,.0f}" if realized_pnl >= 0 else f"-${abs(realized_pnl):,.0f}",
                        style={
                            'fontSize': '0.9em',
                            'color': '#00ff88' if realized_pnl >= 0 else '#ff4444',
                            'fontWeight': 'normal'
                        }
                    ),
                ], style={'marginBottom': '4px'}) if realized_pnl != 0 else None,

                # Unrealized P&L
                html.Div([
                    html.Span("Unrealized: ", style={'fontSize': '0.85em', 'color': '#888'}),
                    html.Span(
                        f"+${unrealized_pnl:,.0f}" if unrealized_pnl >= 0 else f"-${abs(unrealized_pnl):,.0f}",
                        style={
                            'fontSize': '0.9em',
                            'color': '#00ff88' if unrealized_pnl >= 0 else '#ff4444',
                            'fontWeight': 'normal'
                        }
                    ),
                ], style={'marginBottom': '4px'}),

                # Gross P&L (trading profit before costs)
                html.Div([
                    html.Span("Gross P&L: ", style={'fontSize': '0.85em', 'color': '#888'}),
                    html.Span(
                        f"+${gross_pnl:,.0f}" if gross_pnl >= 0 else f"-${abs(gross_pnl):,.0f}",
                        style={
                            'fontSize': '0.95em',
                            'color': '#00ff88' if gross_pnl >= 0 else '#ff4444',
                            'fontWeight': 'bold'
                        }
                    ),
                ], style={'marginBottom': '8px', 'paddingBottom': '8px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)'}),

                # Commission costs breakdown (displayed separately)
                html.Div([
                    html.Div([
                        html.Span("Commission Costs", style={'fontSize': '0.85em', 'color': '#888', 'fontWeight': 'bold', 'textDecoration': 'underline'}),
                    ], style={'marginBottom': '4px'}),
                    html.Div([
                        html.Span("  Opening: ", style={'fontSize': '0.8em', 'color': '#888'}),
                        html.Span(
                            f"-${commission + fees:.2f}",
                            style={'fontSize': '0.8em', 'color': '#ff8844'}
                        ),
                        html.Span(
                            f" (comm: ${commission:.2f}, fees: ${fees:.2f})",
                            style={'fontSize': '0.7em', 'color': '#666', 'marginLeft': '5px'}
                        ),
                    ], style={'marginBottom': '2px'}),
                    html.Div([
                        html.Span("  Closing: ", style={'fontSize': '0.8em', 'color': '#888'}),
                        html.Span(
                            f"-${closing_commission + closing_fees:.2f}",
                            style={'fontSize': '0.8em', 'color': '#ff8844'}
                        ),
                        html.Span(
                            f" (comm: ${closing_commission:.2f}, fees: ${closing_fees:.2f})",
                            style={'fontSize': '0.7em', 'color': '#666', 'marginLeft': '5px'}
                        ),
                    ], style={'marginBottom': '2px'}) if (closing_commission + closing_fees) > 0 else None,
                    html.Div([
                        html.Span("  Total Costs: ", style={'fontSize': '0.85em', 'color': '#888', 'fontWeight': 'bold'}),
                        html.Span(
                            f"-${total_costs:.2f}",
                            style={'fontSize': '0.85em', 'color': '#ff6644', 'fontWeight': 'bold'}
                        ),
                    ], style={'marginTop': '4px'}),
                ], style={'marginBottom': '8px', 'paddingBottom': '8px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)'}) if total_costs > 0 else None,

                # Net P&L (after commissions) - most prominent
                html.Div([
                    html.Span("Net P&L: ", style={'fontSize': '0.9em', 'color': '#aaa', 'fontWeight': 'bold'}),
                    html.Span(
                        f"+${net_pnl:,.0f}" if net_pnl >= 0 else f"-${abs(net_pnl):,.0f}",
                        style={
                            'fontSize': '1.1em',
                            'color': '#00ff88' if net_pnl >= 0 else '#ff4444',
                            'fontWeight': 'bold'
                        }
                    ),
                    html.Span(
                        " (after costs)",
                        style={'fontSize': '0.75em', 'color': '#666', 'marginLeft': '5px', 'fontStyle': 'italic'}
                    ),
                ], style={'marginTop': '8px'}),
            ]),
        ], style={
            'padding': '10px',
            'backgroundColor': 'rgba(255, 255, 255, 0.02)',
            'borderRadius': '4px',
            'marginTop': '8px'
        }) if show_totals else None,
    ])
