"""
History Tab Callbacks

Callbacks for the intraday P&L visualization dashboard.
"""

from dash import callback, Output, Input, html, ctx
from datetime import datetime
import pandas as pd
import plotly.graph_objs as go
from loguru import logger

from monitoring.dashboards.history_tab import (
    create_summary_stats_display,
    get_snapshots_for_display
)
from monitoring.dashboards.intraday_pnl_chart import (
    create_intraday_pnl_chart,
    create_daily_summary_chart,
    create_summary_stats as calc_summary_stats
)
from monitoring.services.intraday_pnl_calculator import IntradayPnLCalculator, IntradaySnapshot
from monitoring.dashboards.components.minute_pnl_chart import create_minute_pnl_chart
from monitoring.services.minute_pnl_csv_exporter import MinutePnLCSVExporter
from dash import dcc


def run_async(coro):
    """Helper to run async code in callbacks."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@callback(
    [
        Output('history-snapshots-store', 'data'),
        Output('history-last-updated', 'children')
    ],
    [
        Input('history-refresh-btn', 'n_clicks'),
        Input('history-backfill-btn', 'n_clicks')
    ],
    prevent_initial_call=False
)
def refresh_history_data(refresh_clicks, backfill_clicks):
    """
    Load intraday P&L snapshots from database.
    If backfill button clicked, trigger async backfill first.
    """
    try:
        # Check which button was clicked
        triggered_id = ctx.triggered_id if ctx.triggered else None

        if triggered_id == 'history-backfill-btn' and backfill_clicks:
            # Run backfill asynchronously
            logger.info("Starting intraday P&L backfill...")
            calculator = IntradayPnLCalculator()
            run_async(calculator.backfill_snapshots(lookback_days=14))
            status_msg = f"Backfill complete: {datetime.now().strftime('%I:%M:%S %p')}"
        else:
            status_msg = f"Loaded: {datetime.now().strftime('%I:%M:%S %p')}"

        # Get snapshots from database
        snapshots = get_snapshots_for_display(lookback_days=14)

        logger.info(f"Loaded {len(snapshots)} intraday P&L snapshots")

        # Convert to serializable format
        data = [
            {
                'timestamp': s.timestamp.isoformat(),
                'realized_pnl': s.realized_pnl,
                'unrealized_pnl': s.unrealized_pnl,
                'total_pnl': s.total_pnl,
                'num_open': s.num_open_positions,
                'num_closed': s.num_closed_positions,
                'events': s.events
            }
            for s in snapshots
        ]

        return data, status_msg

    except Exception as e:
        logger.error(f"Failed to refresh history data: {e}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"


@callback(
    Output('history-summary-stats', 'children'),
    Input('history-snapshots-store', 'data')
)
def update_summary_stats(data):
    """
    Update summary statistics KPIs.
    """
    if not data:
        return create_summary_stats_display({
            'total_pnl': 0,
            'num_trading_days': 0,
            'win_rate': 0,
            'best_day': 0,
            'worst_day': 0,
            'avg_daily_pnl': 0
        })

    try:
        # Reconstruct snapshots from data
        snapshots = [
            IntradaySnapshot(
                timestamp=pd.to_datetime(s['timestamp']),
                realized_pnl=s['realized_pnl'],
                unrealized_pnl=s['unrealized_pnl'],
                total_pnl=s['total_pnl'],
                num_open_positions=s['num_open'],
                num_closed_positions=s['num_closed'],
                events=s['events']
            )
            for s in data
        ]

        # Calculate stats
        stats = calc_summary_stats(snapshots)

        return create_summary_stats_display(stats)

    except Exception as e:
        logger.error(f"Failed to update summary stats: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading stats: {str(e)}", style={'color': '#ff4444'})


@callback(
    Output('intraday-pnl-chart', 'figure'),
    Input('history-snapshots-store', 'data')
)
def update_intraday_pnl_chart(data):
    """
    Update intraday P&L time-series chart.
    """
    if not data:
        return create_intraday_pnl_chart([])

    try:
        # Reconstruct snapshots from data
        snapshots = [
            IntradaySnapshot(
                timestamp=pd.to_datetime(s['timestamp']),
                realized_pnl=s['realized_pnl'],
                unrealized_pnl=s['unrealized_pnl'],
                total_pnl=s['total_pnl'],
                num_open_positions=s['num_open'],
                num_closed_positions=s['num_closed'],
                events=s['events']
            )
            for s in data
        ]

        return create_intraday_pnl_chart(snapshots)

    except Exception as e:
        logger.error(f"Failed to update intraday P&L chart: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()


@callback(
    Output('daily-summary-chart', 'figure'),
    Input('history-snapshots-store', 'data')
)
def update_daily_summary_chart(data):
    """
    Update daily summary bar chart.
    """
    if not data:
        return create_daily_summary_chart([])

    try:
        # Reconstruct snapshots from data
        snapshots = [
            IntradaySnapshot(
                timestamp=pd.to_datetime(s['timestamp']),
                realized_pnl=s['realized_pnl'],
                unrealized_pnl=s['unrealized_pnl'],
                total_pnl=s['total_pnl'],
                num_open_positions=s['num_open'],
                num_closed_positions=s['num_closed'],
                events=s['events']
            )
            for s in data
        ]

        return create_daily_summary_chart(snapshots)

    except Exception as e:
        logger.error(f"Failed to update daily summary chart: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()


# =============================================================================
# Minute-Level P&L Callbacks
# =============================================================================

@callback(
    [
        Output('minute-pnl-trade-selector', 'options'),
        Output('minute-pnl-info', 'children')
    ],
    Input('minute-pnl-date-picker', 'date'),
    prevent_initial_call=False
)
def update_trade_selector(selected_date):
    """Update trade selector options based on selected date."""
    if not selected_date:
        return [], ""

    try:
        from monitoring.services.minute_pnl_transformer import MinutePnLTransformer
        from datetime import datetime

        transformer = MinutePnLTransformer()
        filter_date = datetime.fromisoformat(selected_date).date()

        matches = transformer.get_matches_by_date(filter_date)

        if not matches:
            return [], f"No trades with minute-level data found for {filter_date}"

        # Create options with "All trades" as first option
        options = [{'label': f'📊 All {len(matches)} trades this day', 'value': 'ALL'}]

        for match in matches:
            options.append({
                'label': match['label'],
                'value': match['match_id']
            })

        info_text = f"Found {len(matches)} trade(s) on {filter_date}"
        return options, info_text

    except Exception as e:
        logger.error(f"Failed to update trade selector: {e}")
        return [], f"Error: {str(e)}"


@callback(
    Output('minute-pnl-selected-match', 'data'),
    Input('minute-pnl-trade-selector', 'value')
)
def store_selected_match(selected_match):
    """Store the selected trade match."""
    return selected_match


@callback(
    Output('minute-pnl-chart-container', 'children'),
    [
        Input('minute-pnl-view-btn', 'n_clicks'),
        Input('minute-pnl-selected-match', 'data'),
        Input('minute-pnl-date-picker', 'date')
    ],
    prevent_initial_call=True
)
def display_minute_pnl_chart(n_clicks, match_selection, selected_date):
    """Display minute-level P&L chart for selected trade(s)."""
    if not match_selection:
        return html.Div(
            "Please select a trade to view the chart.",
            style={'color': '#888', 'padding': '20px', 'textAlign': 'center'}
        )

    try:
        from monitoring.services.minute_pnl_transformer import MinutePnLTransformer
        from datetime import datetime

        transformer = MinutePnLTransformer()

        # Handle "ALL" selection - show all trades for the day
        if match_selection == 'ALL':
            filter_date = datetime.fromisoformat(selected_date).date()
            matches = transformer.get_matches_by_date(filter_date)

            if not matches:
                return html.Div(
                    "No trades found for this date.",
                    style={'color': '#888', 'padding': '20px', 'textAlign': 'center'}
                )

            # Create a chart for each match
            charts = []
            for match in matches:
                logger.info(f"Creating chart for match {match['match_id']}")
                fig = create_minute_pnl_chart(match['match_id'])

                if fig:
                    charts.append(html.Div([
                        html.H5(
                            f"{match['label']}",
                            style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#00bfff'}
                        ),
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': True},
                            style={'height': '500px'}
                        )
                    ]))

            if not charts:
                return html.Div(
                    "No chart data available for these trades.",
                    style={'color': '#ff4444', 'padding': '20px', 'textAlign': 'center'}
                )

            return html.Div(charts)

        # Single trade selection
        logger.info(f"Creating minute-level P&L chart for match {match_selection}")
        fig = create_minute_pnl_chart(match_selection)

        if fig is None:
            return html.Div(
                f"No minute-level data available for this trade.",
                style={'color': '#ff4444', 'padding': '20px', 'textAlign': 'center'}
            )

        return dcc.Graph(
            figure=fig,
            config={'displayModeBar': True},
            style={'height': '600px'}
        )

    except Exception as e:
        logger.error(f"Failed to create minute P&L chart: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(
            f"Error creating chart: {str(e)}",
            style={'color': '#ff4444', 'padding': '20px'}
        )


@callback(
    Output('minute-pnl-download-link', 'children'),
    Input('minute-pnl-export-btn', 'n_clicks'),
    Input('minute-pnl-selected-match', 'data'),
    prevent_initial_call=True
)
def export_minute_pnl_csv(n_clicks, match_id):
    """Export minute-level P&L data to CSV."""
    logger.info(f"Export CSV clicked. Match selection: {match_id}")

    if not match_id:
        return html.Div(
            "⚠️ Please select a trade from the dropdown first.",
            style={'color': '#ffaa00', 'padding': '10px', 'fontSize': '1.1em'}
        )

    # Don't allow exporting "ALL" - only individual trades
    if match_id == 'ALL':
        return html.Div(
            "⚠️ Please select an individual trade (not 'All trades') to export CSV.",
            style={'color': '#ffaa00', 'padding': '10px', 'fontSize': '1.1em'}
        )

    try:
        logger.info(f"Exporting minute-level P&L CSV for match {match_id}")

        exporter = MinutePnLCSVExporter()
        csv_path = exporter.export_match_csv(match_id)

        if not csv_path:
            logger.error(f"CSV export returned empty path for match {match_id}")
            return html.Div(
                "❌ No data available to export. Check logs for details.",
                style={'color': '#ff4444', 'padding': '10px'}
            )

        # Create download link
        return html.Div([
            html.P(
                f"✓ CSV exported successfully!",
                style={'color': '#00ff88', 'marginBottom': '10px'}
            ),
            html.A(
                "📥 Download CSV",
                href=f"/download/{csv_path.split('/')[-1]}",
                download=csv_path.split('/')[-1],
                className='btn btn-success',
                style={'padding': '8px 20px'}
            ),
            html.P(
                f"File: {csv_path.split('/')[-1]}",
                style={'color': '#888', 'fontSize': '0.9em', 'marginTop': '10px'}
            )
        ])

    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(
            f"Error exporting CSV: {str(e)}",
            style={'color': '#ff4444', 'padding': '10px'}
        )
