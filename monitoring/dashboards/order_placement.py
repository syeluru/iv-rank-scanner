"""Order placement components for the dashboard."""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_order_modal():
    """Create a modal for configuring and placing orders."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Close Selected Legs")),
        dbc.ModalBody([
            # Selected legs summary
            html.Div(id='order-legs-summary', style={'marginBottom': '20px'}),

            # Order type
            html.Div([
                html.Label("Order Type", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
                dcc.RadioItems(
                    id='order-type-input',
                    options=[
                        {'label': ' Market (immediate execution)', 'value': 'MARKET'},
                        {'label': ' Limit (specify price)', 'value': 'LIMIT'},
                    ],
                    value='LIMIT',
                    style={'marginBottom': '16px'}
                ),
            ]),

            # Limit price (shown only for LIMIT orders)
            html.Div([
                html.Label("Limit Price (per contract)", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
                dbc.Input(
                    id='limit-price-input',
                    type='number',
                    step=0.01,
                    placeholder='Enter limit price...',
                    style={'width': '200px'}
                ),
                html.Small("Suggested: ", style={'color': '#888', 'marginLeft': '12px'}),
                html.Small(id='suggested-price', style={'color': '#00bfff', 'fontWeight': 'bold'}),
            ], id='limit-price-section', style={'marginBottom': '16px'}),

            # Duration
            html.Div([
                html.Label("Duration", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
                dcc.RadioItems(
                    id='order-duration-input',
                    options=[
                        {'label': ' Day (expires at market close)', 'value': 'DAY'},
                        {'label': ' GTC (good until cancelled)', 'value': 'GOOD_TILL_CANCEL'},
                    ],
                    value='DAY',
                    style={'marginBottom': '16px'}
                ),
            ]),

            # Order summary
            html.Div([
                html.Hr(),
                html.H6("Order Summary", style={'marginBottom': '12px'}),
                html.Div(id='order-summary-text'),
            ], style={'marginTop': '20px', 'padding': '12px', 'backgroundColor': 'rgba(0, 191, 255, 0.1)', 'borderRadius': '4px'}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="order-cancel-btn", color="secondary", n_clicks=0),
            dbc.Button("Place Order", id="order-submit-btn", color="primary", n_clicks=0),
        ]),
    ], id="order-modal", is_open=False, size="lg")


def create_confirmation_modal():
    """Create a modal for order confirmation."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id='confirmation-title')),
        dbc.ModalBody([
            html.Div(id='confirmation-message'),
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="confirmation-close-btn", color="primary", n_clicks=0),
        ]),
    ], id="confirmation-modal", is_open=False)


def create_close_button(trade_id: str):
    """Create a 'Close Selected Legs' button for a trade card."""
    return html.Div([
        dbc.Button(
            "Close Selected Legs",
            id={'type': 'close-legs-btn', 'index': trade_id},
            color="danger",
            size="sm",
            n_clicks=0,
            style={'marginTop': '8px'},
            disabled=True  # Initially disabled until legs are selected
        ),
    ], style={'marginTop': '8px'})
