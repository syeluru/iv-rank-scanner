"""
Trade-Nexus Dashboard - Portfolio View with Grouped Trade Cards

A trade-centric dashboard that groups multi-leg options strategies into
cohesive trade cards with:
- Hierarchical accordion view (Trade → P&L Sparkline → Legs)
- Entry vs. Mark visualizer with profit capture %
- Global KPI bar (Daily Realized, Open Unrealized, Theta-Pulse)
- Color-coded alerts (Green 50%+, Amber near strike, Red breached)
- Payoff diagrams per trade card
- Market time P&L reset at 9:30 AM ET

Run with: python monitoring/dashboards/command_center.py
Access at: http://127.0.0.1:8051
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime, date
from typing import Dict, Optional
import pandas as pd
import numpy as np
import re
from loguru import logger
import sys
from pathlib import Path
import json

# Configure logger to show only INFO and above (suppress DEBUG spam)
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from monitoring.services.trade_aggregator import trade_aggregator, AggregatedTradeGroup, AlertColor
from monitoring.services.trade_grouper import trade_grouper, StrategyType
from monitoring.services.payoff_diagram import payoff_generator
from monitoring.services.schwab_positions import schwab_positions
from monitoring.dashboards.spread_breakdown import create_spread_breakdown
from monitoring.dashboards.order_placement import create_order_modal, create_confirmation_modal, create_close_button
from monitoring.dashboards.order_callbacks import register_order_callbacks
from monitoring.dashboards.history_tab import create_history_tab_layout
from monitoring.dashboards import history_callbacks  # Import to register callbacks
import asyncio


# Initialize Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

app.title = "Trade-Nexus - Portfolio Dashboard"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value: float, show_sign: bool = True) -> str:
    """Format currency value with commas and optional sign."""
    if value is None:
        return "$0"
    # Format with commas, no decimals for cleaner display
    formatted = f"{abs(value):,.0f}"
    if show_sign:
        return f"+${formatted}" if value >= 0 else f"-${formatted}"
    return f"${formatted}"


def format_percentage(value: float) -> str:
    """Format percentage value."""
    if value is None:
        return "0%"
    return f"{value:.1f}%"


def get_alert_class(alert_color: AlertColor) -> str:
    """Get CSS class for alert color."""
    mapping = {
        AlertColor.GREEN: "trade-card-green",
        AlertColor.AMBER: "trade-card-amber",
        AlertColor.ORANGE: "trade-card-orange",
        AlertColor.RED: "trade-card-red",
        AlertColor.NONE: "trade-card-neutral"
    }
    return mapping.get(alert_color, "trade-card-neutral")


def get_strategy_display_name(strategy_type: str) -> str:
    """Get human-readable strategy name."""
    mapping = {
        "IRON_CONDOR": "Iron Condor",
        "IRON_BUTTERFLY": "Iron Butterfly",
        "VERTICAL_SPREAD": "Vertical Spread",
        "CALENDAR_SPREAD": "Calendar Spread",
        "DIAGONAL_SPREAD": "Diagonal Spread",
        "STRADDLE": "Straddle",
        "STRANGLE": "Strangle",
        "BUTTERFLY": "Butterfly",
        "CONDOR": "Condor",
        "COVERED_CALL": "Covered Call",
        "CASH_SECURED_PUT": "Cash Secured Put",
        "SINGLE_LEG": "Single Leg",
        "ROLL - Call Spread": "🔄 Roll - Call Spread",
        "ROLL - Put Spread": "🔄 Roll - Put Spread",
        "ROLL - Mixed Spread": "🔄 Roll - Mixed Spread",
        "ROLL - Iron Condor": "🔄 Roll - Iron Condor",
        "UNKNOWN": "Unknown"
    }
    # Handle dynamic roll labels (e.g., "ROLL - 2 to 2 legs")
    if strategy_type.startswith("ROLL"):
        return f"🔄 {strategy_type}"
    return mapping.get(strategy_type, strategy_type)


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_kpi_bar():
    """Create global KPI metrics bar."""
    return html.Div([
        dbc.Row([
            # Daily Realized
            dbc.Col([
                html.Div([
                    html.Div("DAILY REALIZED", className="kpi-label"),
                    html.Div(id="kpi-daily-realized", className="kpi-value"),
                ], className="kpi-item")
            ], width=2),

            # Options Unrealized
            dbc.Col([
                html.Div([
                    html.Div("OPTIONS UNREALIZED", className="kpi-label"),
                    html.Div(id="kpi-options-unrealized", className="kpi-value"),
                ], className="kpi-item")
            ], width=2),

            # Stocks Unrealized
            dbc.Col([
                html.Div([
                    html.Div("STOCKS UNREALIZED", className="kpi-label"),
                    html.Div(id="kpi-stocks-unrealized", className="kpi-value"),
                ], className="kpi-item")
            ], width=2),

            # Total Unrealized
            dbc.Col([
                html.Div([
                    html.Div("TOTAL UNREALIZED", className="kpi-label"),
                    html.Div(id="kpi-total-unrealized", className="kpi-value"),
                ], className="kpi-item")
            ], width=2),

            # Theta-Pulse
            dbc.Col([
                html.Div([
                    html.Div("THETA-PULSE", className="kpi-label"),
                    html.Div(id="kpi-theta-pulse", className="kpi-value kpi-theta"),
                ], className="kpi-item")
            ], width=2),

            # Market Time
            dbc.Col([
                html.Div([
                    html.Div("MARKET TIME", className="kpi-label"),
                    html.Div(id="kpi-market-time", className="kpi-value kpi-time"),
                ], className="kpi-item")
            ], width=2),
        ], className="kpi-bar")
    ], className="kpi-container")


def create_sparkline(pnl_history: list, pnl: float) -> go.Figure:
    """Create P&L sparkline mini chart."""
    fig = go.Figure()

    if not pnl_history:
        pnl_history = [0]

    color = '#00ff88' if pnl >= 0 else '#ff4444'

    fig.add_trace(go.Scatter(
        y=pnl_history,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(0, 255, 136, 0.2)' if pnl >= 0 else 'rgba(255, 68, 68, 0.2)',
        hoverinfo='skip'
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=40
    )

    return fig


def create_dual_bar_gauge(
    entry_per_contract: float,
    mark_per_contract: float,
    total_profit: float,
    profit_capture_pct: float,
    num_contracts: int
) -> html.Div:
    """Create Entry vs Mark dual-bar gauge showing per-contract values."""
    # Calculate bar widths based on entry (entry is 100%, mark shows relative)
    if entry_per_contract and entry_per_contract > 0:
        entry_pct = 100
        mark_pct = min(100, (mark_per_contract / entry_per_contract) * 100) if mark_per_contract else 0
    else:
        entry_pct = 50
        mark_pct = 50

    return html.Div([
        # Entry bar (per contract)
        html.Div([
            html.Span("Entry", className="gauge-label"),
            html.Div([
                html.Div(
                    className="gauge-bar gauge-bar-entry",
                    style={'width': f'{min(100, abs(entry_pct))}%'}
                )
            ], className="gauge-track"),
            html.Span(f"${entry_per_contract:.2f}", className="gauge-value")
        ], className="gauge-row"),

        # Mark bar (per contract)
        html.Div([
            html.Span("Mark", className="gauge-label"),
            html.Div([
                html.Div(
                    className="gauge-bar gauge-bar-mark",
                    style={'width': f'{min(100, abs(mark_pct))}%'}
                )
            ], className="gauge-track"),
            html.Span(f"${mark_per_contract:.2f}", className="gauge-value")
        ], className="gauge-row"),

        # Per contract label
        html.Div([
            html.Span(f"× {num_contracts} contracts", className="gauge-contracts")
        ], className="gauge-contracts-row"),

        # Total profit captured
        html.Div([
            html.Span(
                f"Total Profit: {format_currency(total_profit)} ({profit_capture_pct:.0f}%)",
                className="gauge-capture",
                style={'color': '#00ff88' if profit_capture_pct >= 50 else '#ffcc00' if profit_capture_pct > 0 else '#ff4444'}
            )
        ], className="gauge-capture-row")
    ], className="dual-bar-gauge")


def create_legs_table(legs: list, trade_id: str = None) -> html.Div:
    """Create expandable legs table with checkboxes for closing positions."""
    if not legs:
        return html.Div("No legs", className="text-muted")

    rows = []
    for idx, leg in enumerate(legs):
        option_type = leg.get('option_type', '')
        strike = leg.get('strike', 0)
        quantity = leg.get('quantity', 0)
        entry = leg.get('entry_price', 0)
        mark = leg.get('current_mark', entry)
        delta = leg.get('delta', 0)
        entry_delta = leg.get('entry_delta')
        symbol = leg.get('symbol', '')

        # Parse strike from OCC symbol if not available
        # OCC format: SPXW  260204P06805000 -> strike is 6805.000
        if (not strike or strike == 0) and symbol and option_type:
            # Match pattern: 6 digits for date, P or C, then 8 digits for strike (5 integer + 3 decimal)
            match = re.search(r'\d{6}[PC](\d{8})', symbol)
            if match:
                strike_str = match.group(1)
                # First 5 digits are integer part, last 3 are decimal
                strike = int(strike_str[:5]) + int(strike_str[5:]) / 1000

        # Format strike display
        if option_type:
            strike_display = f"{strike:.0f}{option_type[0]}" if strike else f"?{option_type[0]}"
        else:
            strike_display = "Stock"

        # Check if this leg is closed or pending
        is_closed = leg.get('_is_closed', False) or (mark == 0 and entry > 0)
        is_pending = leg.get('_is_pending', False)

        # Determine side
        side = "Short" if quantity < 0 else "Long"
        side_class = "leg-short" if quantity < 0 else "leg-long"

        if is_closed:
            side_class += " leg-closed"
            side += " (CLOSED)"
        elif is_pending:
            side_class += " leg-pending"
            side += " (PENDING)"

        # Calculate leg P&L
        # For closed legs, P&L is realized (entry is exit price, we don't show P&L here)
        # For pending legs, no P&L yet
        # For open legs:
        #   SHORT = sell at entry, buy back at mark → P&L = (entry - mark) * contracts * 100
        #   LONG = buy at entry, sell at mark → P&L = (mark - entry) * contracts * 100
        if is_closed:
            leg_pnl = 0  # Don't show P&L for closed legs (realized elsewhere)
            pnl_class = "text-muted"
            mark_display = "CLOSED"
            pnl_display = "-"
        elif is_pending:
            leg_pnl = 0
            pnl_class = "text-muted"
            mark_display = "PENDING"
            pnl_display = "-"
        else:
            if option_type:
                if quantity < 0:  # Short
                    leg_pnl = (entry - mark) * abs(quantity) * 100
                else:  # Long
                    leg_pnl = (mark - entry) * quantity * 100
            else:
                leg_pnl = (mark - entry) * quantity

            pnl_class = "pnl-positive" if leg_pnl >= 0 else "pnl-negative"
            mark_display = f"${mark:.2f}"
            pnl_display = format_currency(leg_pnl)

        # Add checkbox for open legs (not closed or pending)
        checkbox_cell = html.Td()
        if not is_closed and not is_pending and trade_id:
            checkbox_cell = html.Td(
                dcc.Checklist(
                    id={'type': 'leg-checkbox', 'trade': trade_id, 'leg': idx},
                    options=[{'label': '', 'value': 'selected'}],
                    value=[],
                    style={'margin': '0'}
                ),
                style={'textAlign': 'center'}
            )

        rows.append(html.Tr([
            checkbox_cell,
            html.Td(strike_display, className="leg-symbol"),
            html.Td(side, className=side_class),
            html.Td(str(abs(quantity)), className="leg-qty"),
            html.Td(f"${entry:.2f}", className="leg-entry"),
            html.Td(mark_display, className="leg-mark"),
            html.Td(f"{delta:.2f}" if delta and not is_closed and not is_pending else "-", className="leg-delta"),
            html.Td(f"{entry_delta:.2f}" if entry_delta is not None and not is_closed and not is_pending else "-", className="leg-delta"),
            html.Td(pnl_display, className=pnl_class),
        ]))

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("", style={'width': '40px'}),  # Checkbox column
                html.Th("Strike"),
                html.Th("Side"),
                html.Th("Qty"),
                html.Th("Entry"),
                html.Th("Mark"),
                html.Th("Curr \u0394"),
                html.Th("Entry \u0394"),
                html.Th("P&L"),
            ])
        ]),
        html.Tbody(rows)
    ], className="legs-table")


def create_trade_card(trade: AggregatedTradeGroup, is_expanded: bool = False) -> html.Div:
    """Create a single trade card component.

    Args:
        trade: The aggregated trade group data
        is_expanded: Whether the legs accordion should be expanded
    """
    # Format expiration
    if trade.expiration:
        exp_str = trade.expiration.strftime('%m/%d') if isinstance(trade.expiration, date) else str(trade.expiration)[:5]
        dte_str = f"{trade.days_to_expiration} DTE"
    else:
        exp_str = "N/A"
        dte_str = ""

    # Get alert class
    alert_class = get_alert_class(trade.alert_color)
    if trade.short_strike_breached:
        alert_class += " trade-card-pulsing"

    # Strategy name
    strategy_name = get_strategy_display_name(trade.strategy_type)

    # Order timestamp (convert to ET)
    import pytz
    if trade.opened_at:
        et = pytz.timezone('US/Eastern')
        # If timestamp is naive, assume it's UTC
        if trade.opened_at.tzinfo is None:
            utc = pytz.UTC
            opened_at_utc = utc.localize(trade.opened_at)
        else:
            opened_at_utc = trade.opened_at
        opened_at_et = opened_at_utc.astimezone(et)
        order_time_str = opened_at_et.strftime('%I:%M %p ET')
    else:
        order_time_str = ""

    # P&L formatting - use total P&L (realized + unrealized)
    total_pnl = trade.realized_pnl + trade.unrealized_pnl
    pnl_class = "pnl-positive" if total_pnl >= 0 else "pnl-negative"

    # Create sparkline with total P&L
    sparkline = create_sparkline(trade.pnl_history, total_pnl)

    # Check if this is a rolled position
    is_rolled = "(Rolled)" in trade.strategy_type

    # Generate payoff diagram(s)
    if is_rolled:
        # For rolled positions, create diagrams for EACH stage of the roll progression
        # Stage 0: Original position (all original legs)
        # Stage 1: After first roll (original legs + roll #1 legs)
        # Stage 2: After second roll (original legs + roll #1 legs + roll #2 legs)
        # etc.

        # Separate legs by type
        original_legs_all = [leg for leg in trade.legs if not leg.get('_is_rolled_leg', False)]
        rolled_legs = [leg for leg in trade.legs if leg.get('_is_rolled_leg', False)]

        # DEBUG: Check if roll metadata exists
        logger.info(f"📊 DEBUG: Total legs in trade: {len(trade.legs)}")
        logger.info(f"📊 DEBUG: Original legs: {len(original_legs_all)}")
        logger.info(f"📊 DEBUG: Rolled legs: {len(rolled_legs)}")
        for i, leg in enumerate(trade.legs):
            logger.info(f"  Leg {i}: symbol={leg.get('symbol', '')[:15]}, "
                       f"_is_rolled_leg={leg.get('_is_rolled_leg', False)}, "
                       f"_roll_number={leg.get('_roll_number', 'NOT SET')}")

        # Find how many rolls occurred
        max_roll_number = max([leg.get('_roll_number', 0) for leg in rolled_legs], default=0)

        logger.info(f"📊 Rolled position {trade.underlying}: {max_roll_number} roll(s) detected")

        # Build position at each stage
        roll_stages = []

        # Stage 0: Original position
        roll_stages.append({
            'label': 'Original Position',
            'legs': original_legs_all,
            'roll_number': 0
        })

        # Stage 1, 2, 3, ... : After each roll
        for roll_num in range(1, max_roll_number + 1):
            # Get legs up to this roll
            legs_at_stage = original_legs_all.copy()
            for rolled_leg in rolled_legs:
                if rolled_leg.get('_roll_number', 0) <= roll_num:
                    legs_at_stage.append(rolled_leg)

            roll_stages.append({
                'label': f'After Roll #{roll_num}',
                'legs': legs_at_stage,
                'roll_number': roll_num
            })

        # Calculate combined axis ranges for all stages
        all_strikes = []
        for stage in roll_stages:
            for leg in stage['legs']:
                if leg.get('strike'):
                    all_strikes.append(leg.get('strike'))

        if all_strikes:
            min_strike = min(all_strikes)
            max_strike = max(all_strikes)
            price_range = max_strike - min_strike
            if price_range == 0:
                price_range = min_strike * 0.2

            # Calculate shared x-axis range
            x_range = (
                min_strike - price_range * 0.3,
                max_strike + price_range * 0.3
            )

            # Calculate shared y-axis range by getting min/max payoff from all stages
            import numpy as np
            prices = np.linspace(x_range[0], x_range[1], 100)

            all_payoffs = []
            for stage in roll_stages:
                stage_payoffs = [payoff_generator._calculate_payoff_at_price(p, stage['legs']) for p in prices]
                all_payoffs.extend(stage_payoffs)

            # Combined y-axis range
            y_range = (
                min(all_payoffs),
                max(all_payoffs)
            )
        else:
            x_range = None
            y_range = None

        # Create diagrams for each stage
        roll_stage_diagrams = []
        for stage in roll_stages:
            diagram = payoff_generator.generate_payoff_diagram(
                legs=stage['legs'],
                current_underlying_price=trade.underlying_price,
                strategy_type=trade.strategy_type.replace(' (Rolled)', ''),
                x_range=x_range,
                y_range=y_range,
                underlying_symbol=trade.underlying,
                expiration_date=trade.expiration
            )
            roll_stage_diagrams.append({
                'label': stage['label'],
                'figure': diagram,
                'legs': stage['legs'],
                'roll_number': stage['roll_number']
            })

        # For backwards compatibility, set first and last diagrams
        payoff_fig_original = roll_stage_diagrams[0]['figure']
        payoff_fig_current = roll_stage_diagrams[-1]['figure']
    else:
        # Normal single diagram
        payoff_fig = payoff_generator.generate_payoff_diagram(
            legs=trade.legs,
            current_underlying_price=trade.underlying_price,
            strategy_type=trade.strategy_type,
            underlying_symbol=trade.underlying,
            expiration_date=trade.expiration
        )
        payoff_fig_original = None
        payoff_fig_current = None

    # Accordion item ID for persistence
    accordion_item_id = "item-0"

    # Extract order ID from trade_group_id (format: schwab_<orderid>)
    order_id_display = trade.trade_group_id.replace('schwab_', '').replace('unmatched_', '') if trade.trade_group_id else ''

    return html.Div([
        # Card Header
        html.Div([
            html.Div([
                html.Div([
                    html.Span(trade.underlying, className="trade-underlying"),
                    html.Span(strategy_name, className="trade-strategy"),
                    html.Span(f"Exp: {exp_str}", className="trade-expiration"),
                    html.Span(dte_str, className="trade-dte"),
                    html.Span(f"{int(trade.num_contracts)} contracts", className="trade-contracts", style={'marginLeft': '10px', 'color': '#00bfff', 'fontSize': '0.9em', 'fontWeight': 'bold'}),
                    html.Span(order_time_str, className="trade-order-time", style={'marginLeft': '10px', 'color': '#888', 'fontSize': '0.9em'}) if order_time_str else None,
                ], style={'marginBottom': '4px'}),
                html.Div([
                    html.Span(f"Order: {order_id_display}", className="trade-order-id", style={'fontSize': '0.75em', 'color': '#666'})
                ])
            ], className="trade-header-left"),
            html.Div([
                html.Div([
                    html.Span(
                        format_currency(total_pnl),
                        className=f"trade-pnl {pnl_class}"
                    ),
                    html.Span(
                        f"({trade.profit_capture_pct:.0f}%)",
                        className="trade-capture"
                    ),
                ]),
                # Show realized/unrealized breakdown if there's realized P&L
                html.Div([
                    html.Span(
                        f"R: {format_currency(trade.realized_pnl, show_sign=False)}",
                        className="trade-realized-breakdown",
                        style={'fontSize': '0.75em', 'color': '#888', 'marginRight': '8px'}
                    ) if trade.realized_pnl != 0 else None,
                    html.Span(
                        f"U: {format_currency(trade.unrealized_pnl, show_sign=False)}",
                        className="trade-unrealized-breakdown",
                        style={'fontSize': '0.75em', 'color': '#888'}
                    ) if trade.realized_pnl != 0 else None,
                ]) if trade.realized_pnl != 0 else None,
            ], className="trade-header-right"),
        ], className="trade-card-header"),

        # Card Body
        html.Div([
            # For rolled positions: create a row for EACH roll stage
            html.Div([
                # Dynamically create rows for each roll stage
                *[
                    dbc.Row([
                        dbc.Col([
                            html.Label(
                                f"{stage['label']} - Spread Breakdown",
                                className="section-label",
                                style={'fontSize': '0.9em', 'marginBottom': '8px'}
                            ),
                            create_spread_breakdown(
                                legs=stage['legs'],
                                realized_pnl=0,  # No realized P&L in individual views
                                unrealized_pnl=0,
                                num_contracts=trade.num_contracts,
                                commission=trade.commission if stage['roll_number'] == 0 else 0,
                                fees=trade.fees if stage['roll_number'] == 0 else 0,
                                closing_commission=0,
                                closing_fees=0,
                                show_totals=False
                            )
                        ], width=5),
                        dbc.Col([
                            html.Label(
                                f"{stage['label']} - Payoff",
                                className="section-label",
                                style={'fontSize': '0.9em', 'marginBottom': '8px'}
                            ),
                            dcc.Graph(
                                figure=stage['figure'],
                                config={'displayModeBar': False},
                                style={'height': '140px'}
                            ),
                        ], width=7),
                    ], style={'marginBottom': '15px'})
                    for stage in roll_stage_diagrams
                ],

                # Final Row: Overall Totals
                dbc.Row([
                    dbc.Col([
                        html.Label("Overall Totals", className="section-label", style={'fontSize': '0.9em', 'marginBottom': '8px', 'fontWeight': 'bold'}),
                        create_spread_breakdown(
                            legs=trade.legs,
                            realized_pnl=trade.realized_pnl,
                            unrealized_pnl=trade.unrealized_pnl,
                            num_contracts=trade.num_contracts,
                            commission=trade.commission,
                            fees=trade.fees,
                            closing_commission=trade.closing_commission,
                            closing_fees=trade.closing_fees,
                            show_spreads=False,  # Only show totals section
                            show_totals=True
                        )
                    ], width=12),
                ]),
            ]) if is_rolled else dbc.Row([
                # Normal layout for non-rolled positions
                dbc.Col([
                    html.Div([
                        html.Label("Spread Breakdown", className="section-label"),
                        create_spread_breakdown(
                            legs=trade.legs,
                            realized_pnl=trade.realized_pnl,
                            unrealized_pnl=trade.unrealized_pnl,
                            num_contracts=trade.num_contracts,
                            commission=trade.commission,
                            fees=trade.fees,
                            closing_commission=trade.closing_commission,
                            closing_fees=trade.closing_fees
                        )
                    ], style={'marginTop': '10px'}),
                ], width=5),
                dbc.Col([
                    html.Div([
                        html.Label("Payoff at Expiration", className="section-label"),
                        dcc.Graph(
                            figure=payoff_fig,
                            config={'displayModeBar': False},
                            style={'height': '360px'}
                        ),
                    ], className="payoff-section")
                ], width=7),
            ]),

            # Expandable Legs Section - use pattern matching ID for state persistence
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            create_legs_table(trade.legs, trade.trade_group_id),
                            create_close_button(trade.trade_group_id)
                        ],
                        title=f"View Legs ({len(trade.legs)})",
                        item_id=accordion_item_id
                    )
                ],
                id={'type': 'legs-accordion', 'index': trade.trade_group_id},
                active_item=accordion_item_id if is_expanded else None,
                className="legs-accordion"
            ),

        ], className="trade-card-body"),
    ], className=f"trade-card {alert_class}")


def create_order_card(order: Dict) -> Optional[html.Div]:
    """
    Create a card for an open order using the same visual as live positions.

    Args:
        order: Order dictionary from Schwab API

    Returns:
        Dash HTML component or None
    """
    try:
        order_id = order.get('orderId', 'N/A')
        status = order.get('status', 'UNKNOWN')
        order_type = order.get('orderType', 'UNKNOWN')
        duration = order.get('duration', 'DAY')
        entered_time = order.get('enteredTime', '')

        # Extract legs
        order_legs = order.get('orderLegCollection', [])
        if not order_legs:
            return None

        # Determine strategy type and underlying
        underlying = order_legs[0].get('instrument', {}).get('underlyingSymbol', '').replace('$', '')
        if not underlying:
            underlying = order_legs[0].get('instrument', {}).get('symbol', 'UNKNOWN')

        # Determine strategy type
        if len(order_legs) == 4:
            strategy_name = "Iron Condor"
        elif len(order_legs) == 2:
            strategy_name = "Vertical Spread"
        elif len(order_legs) == 1:
            strategy_name = "Single Leg"
        else:
            strategy_name = f"{len(order_legs)}-Leg"

        # Get price and quantity
        price = order.get('price', 0)
        quantity = abs(order_legs[0].get('quantity', 0))

        # Format time
        if entered_time:
            try:
                from datetime import datetime
                import pytz
                dt = datetime.fromisoformat(entered_time.replace('Z', '+00:00'))
                et = pytz.timezone('US/Eastern')
                dt_et = dt.astimezone(et)
                time_str = dt_et.strftime('%I:%M %p ET')
            except:
                time_str = entered_time[:19]
        else:
            time_str = 'N/A'

        # Get expiration from first option leg
        expiration = None
        dte = 0
        for leg in order_legs:
            instrument = leg.get('instrument', {})
            if instrument.get('assetType') == 'OPTION':
                exp_str = instrument.get('expirationDate', '')
                if exp_str:
                    try:
                        exp_date = datetime.strptime(exp_str[:10], '%Y-%m-%d').date()
                        expiration = exp_date
                        dte = (exp_date - date.today()).days
                        break
                    except:
                        pass

        exp_str = expiration.strftime('%m/%d') if expiration else 'N/A'
        dte_str = f"{dte} DTE" if expiration else ""

        # Convert order legs to position-like format for display
        legs = []
        for leg in order_legs:
            instrument = leg.get('instrument', {})
            quantity = leg.get('quantity', 0)
            instruction = leg.get('instruction', '')

            leg_data = {
                'symbol': instrument.get('symbol', ''),
                'underlying': underlying,
                'quantity': int(quantity),
                'entry_price': price / len(order_legs) if len(order_legs) > 1 else price,  # Estimate
                'current_mark': 0,  # Not available for pending orders
                '_is_pending': True,  # Flag for pending orders
            }

            if instrument.get('assetType') == 'OPTION':
                leg_data['option_type'] = instrument.get('putCall', '')
                leg_data['strike'] = instrument.get('strikePrice', 0)

            legs.append(leg_data)

        # Sort legs (puts first, then calls)
        def sort_key(leg):
            strike = leg.get('strike', 0) or 0
            option_type = leg.get('option_type', '')
            type_order = 0 if option_type == 'PUT' else 1
            return (type_order, strike)

        legs = sorted(legs, key=sort_key)

        # Generate payoff diagram
        payoff_fig = payoff_generator.generate_payoff_diagram(
            legs=legs,
            current_underlying_price=None,  # No current price for pending orders
            strategy_type=strategy_name.upper().replace(' ', '_'),
            underlying_symbol=underlying,
            expiration_date=expiration
        )

        return html.Div([
            # Card Header
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(underlying, className="trade-underlying"),
                        html.Span(strategy_name, className="trade-strategy"),
                        html.Span(f"Exp: {exp_str}", className="trade-expiration"),
                        html.Span(dte_str, className="trade-dte"),
                        html.Span(f"{int(quantity)} contracts", className="trade-contracts", style={'marginLeft': '10px', 'color': '#ffcc00', 'fontSize': '0.9em', 'fontWeight': 'bold'}),
                        html.Span(time_str, className="trade-order-time", style={'marginLeft': '10px', 'color': '#888', 'fontSize': '0.9em'}),
                    ], style={'marginBottom': '4px'}),
                    html.Div([
                        html.Span(f"Order: {order_id}", className="trade-order-id", style={'fontSize': '0.75em', 'color': '#666'}),
                        html.Span(f" • {status}", style={'fontSize': '0.75em', 'color': '#ffcc00', 'marginLeft': '8px'}),
                    ])
                ], className="trade-header-left"),
                html.Div([
                    html.Span(
                        f"${price:.2f}",
                        className="trade-pnl",
                        style={'color': '#ffcc00'}  # Yellow for pending
                    ),
                    html.Span(
                        "PENDING",
                        className="trade-capture",
                        style={'color': '#ffcc00'}
                    ),
                ], className="trade-header-right"),
            ], className="trade-card-header"),

            # Card Body
            html.Div([
                dbc.Row([
                    # Left: Spread breakdown (pending status)
                    dbc.Col([
                        html.Div([
                            html.Label("Order Details", className="section-label"),
                            html.Div([
                                html.Div([
                                    html.Span("Limit Price: ", style={'fontSize': '0.85em', 'color': '#888'}),
                                    html.Span(f"${price:.2f}", style={'fontSize': '0.9em', 'color': '#ffcc00', 'fontWeight': 'bold'}),
                                ], style={'marginBottom': '4px'}),
                                html.Div([
                                    html.Span("Quantity: ", style={'fontSize': '0.85em', 'color': '#888'}),
                                    html.Span(f"{quantity} contracts", style={'fontSize': '0.9em', 'color': '#ffcc00', 'fontWeight': 'bold'}),
                                ], style={'marginBottom': '4px'}),
                                html.Div([
                                    html.Span("Type: ", style={'fontSize': '0.85em', 'color': '#888'}),
                                    html.Span(f"{order_type} / {duration}", style={'fontSize': '0.9em', 'color': '#888'}),
                                ], style={'marginBottom': '4px'}),
                                html.Div([
                                    html.Span(
                                        "⏳ AWAITING FILL",
                                        style={
                                            'fontSize': '0.85em',
                                            'color': '#ffcc00',
                                            'fontWeight': 'bold',
                                            'backgroundColor': 'rgba(255, 204, 0, 0.1)',
                                            'padding': '4px 8px',
                                            'borderRadius': '3px',
                                            'marginTop': '8px',
                                            'display': 'inline-block'
                                        }
                                    ),
                                ]),
                            ], style={'padding': '10px', 'backgroundColor': 'rgba(255, 204, 0, 0.05)', 'borderRadius': '4px'}),
                        ], style={'marginTop': '10px'}),
                    ], width=5),

                    # Right: Payoff Diagram
                    dbc.Col([
                        html.Div([
                            html.Label("Payoff at Expiration", className="section-label"),
                            dcc.Graph(
                                figure=payoff_fig,
                                config={'displayModeBar': False},
                                style={'height': '360px'}
                            ),
                        ], className="payoff-section")
                    ], width=7),
                ]),

                # Expandable Legs Section
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [create_legs_table(legs)],
                            title=f"View Legs ({len(legs)})",
                            item_id="item-0"
                        )
                    ],
                    id={'type': 'legs-accordion', 'index': f'order_{order_id}'},
                    className="legs-accordion"
                ),

            ], className="trade-card-body"),
        ], className="trade-card trade-card-neutral", style={'borderColor': 'rgba(255, 204, 0, 0.3)'})

    except Exception as e:
        logger.error(f"Error creating order card: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_empty_state():
    """Create empty state when no trades."""
    return html.Div([
        html.Div([
            html.H4("No Open Trades", className="text-muted"),
            html.P("Place a trade to see it here.", className="text-muted"),
            html.Div([
                html.I(className="empty-icon")
            ])
        ], className="empty-state-content")
    ], className="empty-state")


# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header
    html.Div([
        html.H1("Trade-Nexus", className="dashboard-title"),
        html.P("Portfolio Overview", className="dashboard-subtitle"),
    ], className="dashboard-header"),

    # Tabs
    dbc.Tabs([
        # Portfolio Tab (existing content)
        dbc.Tab(
            label="Portfolio",
            tab_id="portfolio-tab",
            children=[
                # KPI Bar
                create_kpi_bar(),

                # Open Orders Section
                html.Div([
                    html.H3("Open Orders", className="section-title"),
                    html.Div(id="open-orders-container", className="trade-cards-container"),
                ], className="trades-section", id="open-orders-section"),

                # Options Trades Section
                html.Div([
                    html.Div([
                        html.H3("Options Trades", className="section-title", style={'display': 'inline-block', 'marginRight': '20px'}),
                        html.Div([
                            html.Label("View Date: ", style={'marginRight': '10px', 'color': '#888', 'fontSize': '0.9em'}),
                            dcc.DatePickerSingle(
                                id='trade-date-picker',
                                date=date.today(),
                                display_format='MMM DD, YYYY',
                                style={'display': 'inline-block', 'verticalAlign': 'middle'},
                                className='date-picker'
                            ),
                            html.Button("Today", id='today-btn', n_clicks=0,
                                       style={'marginLeft': '10px', 'padding': '5px 15px', 'fontSize': '0.9em',
                                              'backgroundColor': '#00bfff', 'color': 'white', 'border': 'none',
                                              'borderRadius': '4px', 'cursor': 'pointer'}),
                        ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                    ], style={'marginBottom': '15px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),

                    # Daily summary
                    html.Div(id='daily-summary', style={'marginBottom': '15px'}),

                    html.Div(id="options-cards-container", className="trade-cards-container"),
                ], className="trades-section"),

                # Stock Positions Section
                html.Div([
                    html.H3("Stock Positions", className="section-title"),
                    html.Div(id="stocks-cards-container", className="trade-cards-container"),
                ], className="trades-section stocks-section"),
            ],
            style={'padding': '20px'}
        ),

        # History Tab (new)
        dbc.Tab(
            label="History",
            tab_id="history-tab",
            children=create_history_tab_layout()
        ),
    ], id="main-tabs", active_tab="portfolio-tab"),

    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # 5 seconds
        n_intervals=0
    ),

    # Store for accordion state (persists which legs are expanded)
    dcc.Store(id='accordion-state-store', data={}),

    # Store for trade data
    dcc.Store(id='trade-data-store'),

    # Store for selected legs
    dcc.Store(id='selected-legs-store', data={}),

    # Stores for position data (used by order callbacks)
    dcc.Store(id='live-positions-store', data=[]),
    dcc.Store(id='open-orders-store', data=[]),

    # Order placement modals
    create_order_modal(),
    create_confirmation_modal(),

], fluid=True, className="dashboard-container")


# ============================================================================
# ASYNC HELPER - Persistent event loop in dedicated thread
# ============================================================================

import threading
import concurrent.futures

# Global event loop running in a dedicated thread
_async_loop = None
_async_thread = None
_loop_lock = threading.Lock()


def _start_async_loop():
    """Start the persistent async event loop in a dedicated thread."""
    global _async_loop, _async_thread

    with _loop_lock:
        if _async_loop is not None and _async_loop.is_running():
            return

        _async_loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(_async_loop)
            _async_loop.run_forever()

        _async_thread = threading.Thread(target=run_loop, daemon=True)
        _async_thread.start()
        logger.info("Started persistent async event loop thread")


def run_async(coro):
    """Run an async coroutine using the persistent event loop."""
    global _async_loop

    # Ensure the loop is running
    if _async_loop is None or not _async_loop.is_running():
        _start_async_loop()

    # Submit the coroutine to the persistent loop and wait for result
    future = asyncio.run_coroutine_threadsafe(coro, _async_loop)
    try:
        return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        logger.error("Async operation timed out after 30s")
        raise


# Start the async loop immediately on module load
_start_async_loop()


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    [Output('kpi-daily-realized', 'children'),
     Output('kpi-daily-realized', 'className'),
     Output('kpi-options-unrealized', 'children'),
     Output('kpi-options-unrealized', 'className'),
     Output('kpi-stocks-unrealized', 'children'),
     Output('kpi-stocks-unrealized', 'className'),
     Output('kpi-total-unrealized', 'children'),
     Output('kpi-total-unrealized', 'className'),
     Output('kpi-theta-pulse', 'children'),
     Output('kpi-market-time', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_kpi_bar(n):
    """Update global KPI metrics from Schwab portfolio."""
    try:
        # Fetch real metrics from Schwab
        metrics = run_async(schwab_positions.get_global_metrics())

        daily_realized = metrics.get('daily_realized', 0)
        options_unrealized = metrics.get('options_unrealized', 0)
        stocks_unrealized = metrics.get('stocks_unrealized', 0)
        total_unrealized = metrics.get('open_unrealized', 0)
        theta_pulse = metrics.get('theta_pulse', 0)
        market_time = metrics.get('market_time', '--:--:-- ET')

        # Format values
        daily_str = format_currency(daily_realized)
        daily_class = "kpi-value kpi-positive" if daily_realized >= 0 else "kpi-value kpi-negative"

        options_str = format_currency(options_unrealized)
        options_class = "kpi-value kpi-positive" if options_unrealized >= 0 else "kpi-value kpi-negative"

        stocks_str = format_currency(stocks_unrealized)
        stocks_class = "kpi-value kpi-positive" if stocks_unrealized >= 0 else "kpi-value kpi-negative"

        total_str = format_currency(total_unrealized)
        total_class = "kpi-value kpi-positive" if total_unrealized >= 0 else "kpi-value kpi-negative"

        theta_str = f"${theta_pulse:.2f}/hr"

        return (
            daily_str, daily_class,
            options_str, options_class,
            stocks_str, stocks_class,
            total_str, total_class,
            theta_str,
            market_time
        )

    except Exception as e:
        logger.error(f"Error updating KPI bar: {e}")
        import traceback
        traceback.print_exc()
        return (
            "$0", "kpi-value",
            "$0", "kpi-value",
            "$0", "kpi-value",
            "$0", "kpi-value",
            "$0/hr",
            "--:--:-- ET"
        )


@callback(
    Output('open-orders-container', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_open_orders(n):
    """Update open orders section."""
    try:
        # Fetch open orders from Schwab
        open_orders = run_async(schwab_positions.get_open_orders())

        if not open_orders:
            return html.Div("No open orders", className="empty-message")

        # Create order cards
        order_cards = []
        for order in open_orders:
            order_card = create_order_card(order)
            if order_card:
                order_cards.append(order_card)

        if not order_cards:
            return html.Div("No open orders", className="empty-message")

        return html.Div(order_cards, className="trade-cards-grid")

    except Exception as e:
        logger.error(f"Error updating open orders: {e}")
        import traceback
        traceback.print_exc()
        return html.Div("Error loading orders", className="empty-message")


# Callback for "Today" button
@callback(
    Output('trade-date-picker', 'date'),
    Input('today-btn', 'n_clicks'),
    prevent_initial_call=True
)
def set_today(n_clicks):
    """Set date picker to today."""
    if n_clicks:
        return date.today()
    return no_update


@callback(
    [Output('options-cards-container', 'children'),
     Output('stocks-cards-container', 'children'),
     Output('live-positions-store', 'data'),
     Output('open-orders-store', 'data'),
     Output('daily-summary', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('trade-date-picker', 'date')],
    State('accordion-state-store', 'data')
)
def update_trade_cards(n, selected_date, accordion_state):
    """Update trade cards with real Schwab positions, filtered by date."""
    accordion_state = accordion_state or {}

    # Parse selected date
    if isinstance(selected_date, str):
        view_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
    else:
        view_date = selected_date if selected_date else date.today()

    logger.info(f"🔍 update_trade_cards called: selected_date={selected_date}, view_date={view_date}, today={date.today()}")

    try:
        # Check if viewing historical date or current positions
        is_today = view_date == date.today()
        logger.info(f"📅 is_today={is_today}, will fetch {'current positions' if is_today else 'historical trades'}")

        if is_today:
            # Fetch current positions
            options_trades, stock_positions = run_async(schwab_positions.get_aggregated_trades())
            logger.info(f"✅ Fetched current positions: {len(options_trades)} options trades, {len(stock_positions)} stock positions")

            # Calculate today's realized P&L and commissions from current positions
            total_realized = sum(t.realized_pnl for t in options_trades if t.realized_pnl != 0) + sum(t.realized_pnl for t in stock_positions if t.realized_pnl != 0)
            num_closed = len([t for t in options_trades if t.realized_pnl != 0]) + len([t for t in stock_positions if t.realized_pnl != 0])

            # Calculate total commissions for closed trades
            total_commissions = sum(
                t.commission + t.fees + t.closing_commission + t.closing_fees
                for t in options_trades if t.realized_pnl != 0
            ) + sum(
                t.commission + t.fees + t.closing_commission + t.closing_fees
                for t in stock_positions if t.realized_pnl != 0
            )

            # Net P&L after commissions
            net_realized = total_realized - total_commissions

            if total_realized != 0:
                daily_summary = html.Div([
                    html.Div([
                        html.Span(f"📅 Today's Realized P&L", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#00bfff'}),
                        html.Span(f" • {num_closed} trade(s) with realized P&L", style={'marginLeft': '15px', 'color': '#888'}),
                    ], style={'marginBottom': '12px'}),

                    # Gross P&L (before commissions)
                    html.Div([
                        html.Span("Gross P&L: ", style={'fontSize': '0.95em', 'color': '#888'}),
                        html.Span(
                            f"+${total_realized:,.0f}" if total_realized >= 0 else f"-${abs(total_realized):,.0f}",
                            style={
                                'fontSize': '1.1em',
                                'fontWeight': 'bold',
                                'color': '#00ff88' if total_realized >= 0 else '#ff4444'
                            }
                        ),
                    ], style={'marginBottom': '6px'}),

                    # Commissions
                    html.Div([
                        html.Span("Commissions: ", style={'fontSize': '0.95em', 'color': '#888'}),
                        html.Span(
                            f"-${total_commissions:,.2f}",
                            style={
                                'fontSize': '1.1em',
                                'fontWeight': 'normal',
                                'color': '#ff8844'
                            }
                        ),
                    ], style={'marginBottom': '10px', 'paddingBottom': '10px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)'}),

                    # Net P&L (after commissions)
                    html.Div([
                        html.Span("Net P&L: ", style={'fontSize': '1em', 'color': '#aaa', 'fontWeight': 'bold'}),
                        html.Span(
                            f"+${net_realized:,.0f}" if net_realized >= 0 else f"-${abs(net_realized):,.0f}",
                            style={
                                'fontSize': '1.4em',
                                'fontWeight': 'bold',
                                'color': '#00ff88' if net_realized >= 0 else '#ff4444'
                            }
                        ),
                    ]),
                ], style={
                    'padding': '15px',
                    'backgroundColor': 'rgba(0, 191, 255, 0.05)',
                    'borderRadius': '8px',
                    'border': '1px solid rgba(0, 191, 255, 0.2)'
                })
            else:
                daily_summary = None
        else:
            # Fetch trades closed on selected date
            options_trades, stock_positions = run_async(schwab_positions.get_trades_closed_on_date(view_date))
            logger.info(f"✅ Fetched historical trades for {view_date}: {len(options_trades)} options trades, {len(stock_positions)} stock positions")

            # Calculate daily summary with commissions
            total_realized = sum(t.realized_pnl for t in options_trades) + sum(t.realized_pnl for t in stock_positions)
            num_trades = len(options_trades) + len(stock_positions)

            # Calculate total commissions
            total_commissions = sum(
                t.commission + t.fees + t.closing_commission + t.closing_fees
                for t in options_trades
            ) + sum(
                t.commission + t.fees + t.closing_commission + t.closing_fees
                for t in stock_positions
            )

            # Net P&L after commissions
            net_realized = total_realized - total_commissions

            daily_summary = html.Div([
                html.Div([
                    html.Span(f"📅 {view_date.strftime('%B %d, %Y')}", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#00bfff'}),
                    html.Span(f" • {num_trades} trade(s) closed", style={'marginLeft': '15px', 'color': '#888'}),
                ], style={'marginBottom': '12px'}),

                # Gross P&L (before commissions)
                html.Div([
                    html.Span("Gross P&L: ", style={'fontSize': '0.95em', 'color': '#888'}),
                    html.Span(
                        f"+${total_realized:,.0f}" if total_realized >= 0 else f"-${abs(total_realized):,.0f}",
                        style={
                            'fontSize': '1.1em',
                            'fontWeight': 'bold',
                            'color': '#00ff88' if total_realized >= 0 else '#ff4444'
                        }
                    ),
                ], style={'marginBottom': '6px'}),

                # Commissions
                html.Div([
                    html.Span("Commissions: ", style={'fontSize': '0.95em', 'color': '#888'}),
                    html.Span(
                        f"-${total_commissions:,.2f}",
                        style={
                            'fontSize': '1.1em',
                            'fontWeight': 'normal',
                            'color': '#ff8844'
                        }
                    ),
                ], style={'marginBottom': '10px', 'paddingBottom': '10px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)'}),

                # Net P&L (after commissions)
                html.Div([
                    html.Span("Net P&L: ", style={'fontSize': '1em', 'color': '#aaa', 'fontWeight': 'bold'}),
                    html.Span(
                        f"+${net_realized:,.0f}" if net_realized >= 0 else f"-${abs(net_realized):,.0f}",
                        style={
                            'fontSize': '1.4em',
                            'fontWeight': 'bold',
                            'color': '#00ff88' if net_realized >= 0 else '#ff4444'
                        }
                    ),
                ]),
            ], style={
                'padding': '15px',
                'backgroundColor': 'rgba(0, 191, 255, 0.05)',
                'borderRadius': '8px',
                'border': '1px solid rgba(0, 191, 255, 0.2)'
            })

        # Serialize trades to dict for storage (for order callbacks to access leg data)
        def trade_to_dict(trade):
            """Convert AggregatedTradeGroup to dict."""
            return {
                'trade_group_id': trade.trade_group_id,
                'underlying': trade.underlying,
                'strategy_type': trade.strategy_type,
                'expiration': trade.expiration.isoformat() if trade.expiration else None,
                'days_to_expiration': trade.days_to_expiration,
                'status': trade.status,
                'legs': trade.legs  # Already a list of dicts
            }

        # Combine all trades for storage
        # Currently all trades have status="OPEN" (filled positions)
        # TODO: Add support for pending orders (WORKING/QUEUED status)
        all_trades = (options_trades or []) + (stock_positions or [])
        live_positions = [trade_to_dict(trade) for trade in all_trades]
        open_orders = []  # Not yet implemented

        # Create options cards - sort by expiration date (nearest first)
        if options_trades:
            options_sorted = sorted(
                options_trades,
                key=lambda t: (t.expiration if t.expiration else date(2099, 12, 31), t.underlying)
            )
            options_cards = [
                create_trade_card(trade, accordion_state.get(trade.trade_group_id, False))
                for trade in options_sorted
            ]
            options_content = html.Div(options_cards, className="trade-cards-grid")
        else:
            options_content = html.Div([
                html.P("No options trades", className="text-muted empty-message")
            ])

        # Create stock cards - sort by P&L
        if stock_positions:
            stocks_sorted = sorted(stock_positions, key=lambda t: t.unrealized_pnl, reverse=True)
            stocks_cards = [
                create_trade_card(trade, accordion_state.get(trade.trade_group_id, False))
                for trade in stocks_sorted
            ]
            stocks_content = html.Div(stocks_cards, className="trade-cards-grid")
        else:
            stocks_content = html.Div([
                html.P("No stock positions", className="text-muted empty-message")
            ])

        return options_content, stocks_content, live_positions, open_orders, daily_summary

    except Exception as e:
        logger.error(f"Error updating trade cards: {e}")
        import traceback
        traceback.print_exc()
        error_msg = html.Div([
            html.P(f"Error loading trades: {e}", className="text-danger")
        ])
        return error_msg, error_msg, [], [], None


# Callback to persist accordion state when user clicks
@callback(
    Output('accordion-state-store', 'data'),
    Input({'type': 'legs-accordion', 'index': ALL}, 'active_item'),
    State('accordion-state-store', 'data'),
    prevent_initial_call=True
)
def update_accordion_state(active_items, current_state):
    """Persist accordion expansion state across refreshes."""
    current_state = current_state or {}

    # Get the triggered input
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        trade_id = triggered.get('index')
        # Find the corresponding active_item value
        for i, prop in enumerate(ctx.inputs_list[0]):
            if prop.get('id', {}).get('index') == trade_id:
                is_open = active_items[i] is not None
                current_state[trade_id] = is_open
                break

    return current_state


# ============================================================================
# DEMO DATA (for testing without real trades)
# ============================================================================

def create_demo_trades() -> list:
    """Create demo trade data for testing the dashboard."""
    # Demo Iron Condor on SPX
    demo_trades = [
        AggregatedTradeGroup(
            trade_group_id="demo_ic_1",
            underlying="SPX",
            strategy_type="IRON_CONDOR",
            expiration=date(2025, 2, 7),
            days_to_expiration=4,
            entry_credit=350.00,
            entry_debit=0,
            max_profit=350.00,
            max_loss=650.00,
            current_mark=175.00,
            unrealized_pnl=175.00,
            day_pnl=25.00,
            profit_capture_pct=50.0,
            total_delta=-0.05,
            total_theta=12.50,
            total_gamma=0.002,
            total_vega=-8.00,
            theta_per_hour=1.92,
            alert_color=AlertColor.GREEN,
            short_strike_breached=False,
            pnl_history=[100, 120, 150, 140, 160, 175],
            legs=[
                {'symbol': 'SPX', 'option_type': 'PUT', 'strike': 5800, 'quantity': 1, 'entry_price': 2.50, 'current_mark': 1.00, 'delta': 0.05},
                {'symbol': 'SPX', 'option_type': 'PUT', 'strike': 5850, 'quantity': -1, 'entry_price': 4.00, 'current_mark': 1.75, 'delta': -0.10},
                {'symbol': 'SPX', 'option_type': 'CALL', 'strike': 6100, 'quantity': -1, 'entry_price': 3.50, 'current_mark': 1.50, 'delta': -0.08},
                {'symbol': 'SPX', 'option_type': 'CALL', 'strike': 6150, 'quantity': 1, 'entry_price': 2.00, 'current_mark': 0.75, 'delta': 0.03},
            ],
            opened_at=datetime(2025, 2, 3, 10, 30),
            status="OPEN",
            strategy_name="iron_condor"
        ),
        AggregatedTradeGroup(
            trade_group_id="demo_vs_1",
            underlying="AAPL",
            strategy_type="VERTICAL_SPREAD",
            expiration=date(2025, 2, 14),
            days_to_expiration=11,
            entry_credit=125.00,
            entry_debit=0,
            max_profit=125.00,
            max_loss=375.00,
            current_mark=85.00,
            unrealized_pnl=40.00,
            day_pnl=10.00,
            profit_capture_pct=32.0,
            total_delta=-0.12,
            total_theta=3.50,
            total_gamma=0.005,
            total_vega=-2.00,
            theta_per_hour=0.54,
            alert_color=AlertColor.NONE,
            short_strike_breached=False,
            pnl_history=[20, 25, 30, 35, 38, 40],
            legs=[
                {'symbol': 'AAPL', 'option_type': 'PUT', 'strike': 225, 'quantity': -1, 'entry_price': 2.75, 'current_mark': 1.90, 'delta': -0.18},
                {'symbol': 'AAPL', 'option_type': 'PUT', 'strike': 220, 'quantity': 1, 'entry_price': 1.50, 'current_mark': 1.05, 'delta': 0.06},
            ],
            opened_at=datetime(2025, 2, 1, 14, 15),
            status="OPEN",
            strategy_name="credit_spreads"
        ),
        AggregatedTradeGroup(
            trade_group_id="demo_csp_1",
            underlying="NVDA",
            strategy_type="CASH_SECURED_PUT",
            expiration=date(2025, 2, 21),
            days_to_expiration=18,
            entry_credit=450.00,
            entry_debit=0,
            max_profit=450.00,
            max_loss=12550.00,
            current_mark=520.00,
            unrealized_pnl=-70.00,
            day_pnl=-15.00,
            profit_capture_pct=-15.6,
            total_delta=-0.28,
            total_theta=8.00,
            total_gamma=0.008,
            total_vega=-15.00,
            theta_per_hour=1.23,
            alert_color=AlertColor.ORANGE,
            short_strike_breached=False,
            pnl_history=[50, 40, 20, 0, -30, -70],
            legs=[
                {'symbol': 'NVDA', 'option_type': 'PUT', 'strike': 130, 'quantity': -1, 'entry_price': 4.50, 'current_mark': 5.20, 'delta': -0.28},
            ],
            opened_at=datetime(2025, 1, 28, 11, 45),
            status="OPEN",
            strategy_name="wheel_strategy"
        ),
    ]

    return demo_trades


# Register order placement callbacks
register_order_callbacks(app)


# ============================================================================
# HISTORY TAB CALLBACKS (defined in history_callbacks.py)
# ============================================================================
# Callbacks are automatically registered by importing history_callbacks module


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Trade-Nexus Dashboard...")
    logger.info("=" * 60)
    logger.info("Dashboard URL: http://127.0.0.1:8051")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=8051)
