"""
History Tab - Intraday P&L Visualization

Displays intraday P&L time-series chart with 30-minute ticks showing:
- Realized P&L (solid line)
- Unrealized P&L (dashed line)
- Total P&L (bold line)
- Event markers for opens/closes/rolls
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import date, timedelta

from monitoring.dashboards.intraday_pnl_chart import (
    create_intraday_pnl_chart,
    create_daily_summary_chart,
    create_summary_stats as calc_summary_stats
)
from monitoring.services.intraday_pnl_calculator import IntradayPnLCalculator
from monitoring.dashboards.components.minute_pnl_chart import create_match_selector_options


def create_summary_stats_display(stats: dict) -> html.Div:
    """
    Create summary statistics KPI cards.

    Args:
        stats: Dictionary with keys: total_pnl, num_trading_days, win_rate,
               best_day, worst_day, avg_daily_pnl

    Returns:
        Div with KPI cards
    """
    total_pnl = stats.get('total_pnl', 0)
    num_days = stats.get('num_trading_days', 0)
    win_rate = stats.get('win_rate', 0)
    best_day = stats.get('best_day', 0)
    worst_day = stats.get('worst_day', 0)
    avg_daily = stats.get('avg_daily_pnl', 0)

    # Format values
    pnl_color = '#00ff88' if total_pnl >= 0 else '#ff4444'
    pnl_text = f"${total_pnl:,.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.0f}"

    return html.Div([
        dbc.Row([
            # Total P&L
            dbc.Col([
                html.Div([
                    html.Div("Total P&L", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        pnl_text,
                        className="kpi-value",
                        style={'color': pnl_color, 'fontSize': '2em', 'fontWeight': 'bold'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),

            # Trading Days
            dbc.Col([
                html.Div([
                    html.Div("Trading Days", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        str(num_days),
                        className="kpi-value",
                        style={'fontSize': '1.8em', 'color': '#00bfff'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),

            # Win Rate
            dbc.Col([
                html.Div([
                    html.Div("Win Rate", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        f"{win_rate:.1f}%",
                        className="kpi-value",
                        style={'fontSize': '1.8em', 'color': '#00bfff'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),

            # Best Day
            dbc.Col([
                html.Div([
                    html.Div("Best Day", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        f"${best_day:,.0f}",
                        className="kpi-value",
                        style={'fontSize': '1.8em', 'color': '#00ff88'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),

            # Worst Day
            dbc.Col([
                html.Div([
                    html.Div("Worst Day", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        f"${worst_day:,.0f}",
                        className="kpi-value",
                        style={'fontSize': '1.8em', 'color': '#ff4444'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),

            # Avg Daily
            dbc.Col([
                html.Div([
                    html.Div("Avg Daily", className="kpi-label", style={'fontSize': '0.9em', 'color': '#888'}),
                    html.Div(
                        f"${avg_daily:,.0f}",
                        className="kpi-value",
                        style={'fontSize': '1.8em', 'color': '#00bfff'}
                    ),
                ], className="kpi-item", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px'})
            ], width=2),
        ])
    ], style={'marginBottom': '30px'})


def create_history_tab_layout() -> html.Div:
    """
    Create the main History tab layout with intraday P&L chart.

    Returns:
        Div containing all history tab components
    """
    return html.Div([
        # Header with refresh button
        html.Div([
            html.H2("Intraday P&L - Last 14 Days", style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Button(
                "🔄 Refresh Data",
                id='history-refresh-btn',
                n_clicks=0,
                className='btn btn-primary',
                style={'marginLeft': '20px', 'padding': '8px 20px'}
            ),
            html.Button(
                "⚙️ Backfill Snapshots",
                id='history-backfill-btn',
                n_clicks=0,
                className='btn btn-secondary',
                style={'marginLeft': '10px', 'padding': '8px 20px'}
            ),
            html.Span(
                id='history-last-updated',
                style={'marginLeft': '20px', 'color': '#888', 'fontSize': '0.9em'}
            ),
        ], style={'marginBottom': '30px'}),

        # Summary stats
        html.Div(id='history-summary-stats'),

        # Main intraday P&L chart
        html.Div([
            html.H4("Intraday P&L Time Series", style={'marginBottom': '15px'}),
            html.P(
                "30-minute intervals showing realized (solid) vs unrealized (dashed) P&L. "
                "Event markers indicate trade opens/closes/rolls.",
                style={'color': '#888', 'fontSize': '0.9em', 'marginBottom': '15px'}
            ),
            dcc.Graph(id='intraday-pnl-chart', config={'displayModeBar': True})
        ], style={'marginBottom': '40px'}),

        # Daily summary chart
        html.Div([
            html.H4("Daily P&L Summary", style={'marginBottom': '15px'}),
            dcc.Graph(id='daily-summary-chart', config={'displayModeBar': False})
        ], style={'marginBottom': '40px'}),

        # Minute-level P&L section
        html.Div([
            html.Hr(style={'borderColor': '#444', 'marginTop': '40px', 'marginBottom': '40px'}),
            html.H3("Minute-Level P&L Analysis", style={'marginBottom': '20px'}),
            html.P(
                "View detailed minute-by-minute P&L for individual trades. Shows all 5 lines: 4 individual legs + total P&L.",
                style={'color': '#888', 'fontSize': '0.9em', 'marginBottom': '20px'}
            ),

            # Date and trade filters
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Date:", style={'marginBottom': '5px', 'color': '#888'}),
                    dcc.DatePickerSingle(
                        id='minute-pnl-date-picker',
                        date=date(2026, 2, 12),  # Default to date with data
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Select Trade:", style={'marginBottom': '5px', 'color': '#888'}),
                    dcc.Dropdown(
                        id='minute-pnl-trade-selector',
                        placeholder='Select a trade or "All trades this day"...',
                        style={'backgroundColor': '#1e1e1e', 'color': '#fff'}
                    ),
                ], width=5),
                dbc.Col([
                    html.Div([
                        html.Button(
                            "📊 View Chart",
                            id='minute-pnl-view-btn',
                            n_clicks=0,
                            className='btn btn-primary',
                            style={'marginRight': '10px', 'padding': '8px 20px', 'marginTop': '28px'}
                        ),
                        html.Button(
                            "💾 Export CSV",
                            id='minute-pnl-export-btn',
                            n_clicks=0,
                            className='btn btn-secondary',
                            style={'padding': '8px 20px', 'marginTop': '28px'}
                        ),
                    ]),
                ], width=4),
            ], style={'marginBottom': '20px'}),

            # Info display
            html.Div(id='minute-pnl-info', style={'marginBottom': '15px', 'color': '#888', 'fontSize': '0.9em'}),

            # Download link for CSV
            html.Div(id='minute-pnl-download-link', style={'marginBottom': '20px'}),

            # Chart display area
            html.Div(id='minute-pnl-chart-container'),

        ], style={'marginTop': '40px'}),

        # Hidden store for data
        dcc.Store(id='history-data-store'),
        dcc.Store(id='history-snapshots-store'),
        dcc.Store(id='minute-pnl-selected-match'),

    ], style={'padding': '20px'})


def get_snapshots_for_display(lookback_days: int = 14):
    """
    Fetch intraday P&L snapshots from database.

    Args:
        lookback_days: Number of days to fetch

    Returns:
        List of IntradaySnapshot objects
    """
    calculator = IntradayPnLCalculator()

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    snapshots = calculator.get_snapshots_for_date_range(start_date, end_date)

    return snapshots
