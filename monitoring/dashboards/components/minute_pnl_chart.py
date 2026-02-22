"""
Minute-level P&L chart component.

Creates interactive Plotly charts showing:
- Individual leg P&L over time (4 traces for iron condor)
- Total position P&L (1 bold trace)
- Roll event markers (vertical lines, stars, annotations)
- Realized vs unrealized distinction (line style)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional
import logging

from monitoring.services.minute_pnl_transformer import MinutePnLTransformer

logger = logging.getLogger(__name__)


def create_minute_pnl_chart(match_id: str) -> Optional[go.Figure]:
    """
    Create minute-level P&L chart for a trade match.

    Shows 5 traces:
    - 4 individual leg traces (colored by leg type)
    - 1 total P&L trace (green/red, bold)

    Plus roll markers:
    - Vertical dotted lines (amber)
    - Star markers on total P&L line
    - Annotations above chart

    Args:
        match_id: Trade match ID

    Returns:
        Plotly Figure object or None if no data
    """
    try:
        transformer = MinutePnLTransformer()

        # Get data
        leg_df = transformer.get_match_timeseries(match_id)
        total_df = transformer.get_match_total_pnl(match_id)
        metadata = transformer.get_match_metadata(match_id)
        roll_events = transformer.get_roll_events(match_id)

        if leg_df.empty or total_df.empty:
            logger.warning(f"No data available for match {match_id}")
            return None

        # Create figure
        fig = go.Figure()

        # Add individual leg traces
        for leg_id in leg_df['leg_id'].unique():
            leg_data = leg_df[leg_df['leg_id'] == leg_id].sort_values('timestamp')

            if leg_data.empty:
                continue

            # Get leg details
            leg_symbol = leg_data.iloc[0]['leg_symbol']
            leg_type = leg_data.iloc[0]['leg_type']
            color = leg_data.iloc[0]['color']
            strike = leg_data.iloc[0]['strike']

            # Prepare P&L values (unrealized or realized)
            pnl_values = leg_data.apply(
                lambda row: row['realized_pnl'] if row['is_realized'] else row['unrealized_pnl'],
                axis=1
            )

            # Determine line style (solid for unrealized, dashed for realized)
            # Split into two traces if there's a transition
            unrealized_mask = ~leg_data['is_realized']

            if unrealized_mask.any():
                # Unrealized portion (solid line)
                unrealized_data = leg_data[unrealized_mask]
                fig.add_trace(go.Scatter(
                    x=unrealized_data['timestamp'],
                    y=unrealized_data['unrealized_pnl'],
                    mode='lines',
                    name=f"{leg_type.replace('_', ' ').title()} ${strike:.0f}",
                    line=dict(color=color, width=2, dash='solid'),
                    hovertemplate='%{y:$,.2f}<br>%{x|%H:%M:%S}<extra></extra>',
                    legendgroup=leg_id,
                ))

            if (~unrealized_mask).any():
                # Realized portion (dashed line)
                realized_data = leg_data[~unrealized_mask]
                fig.add_trace(go.Scatter(
                    x=realized_data['timestamp'],
                    y=realized_data['realized_pnl'],
                    mode='lines',
                    name=f"{leg_type.replace('_', ' ').title()} ${strike:.0f} (realized)",
                    line=dict(color=color, width=2, dash='dash'),
                    hovertemplate='%{y:$,.2f}<br>%{x|%H:%M:%S}<extra></extra>',
                    legendgroup=leg_id,
                    showlegend=False,  # Don't show separate legend entry for realized portion
                ))

        # Add total P&L trace (always solid, bold)
        total_color = '#00ff88' if total_df['total_pnl'].iloc[-1] >= 0 else '#ff4444'

        fig.add_trace(go.Scatter(
            x=total_df['timestamp'],
            y=total_df['total_pnl'],
            mode='lines',
            name='Total P&L',
            line=dict(color=total_color, width=4, dash='solid'),
            hovertemplate='Total: %{y:$,.2f}<br>%{x|%H:%M:%S}<extra></extra>',
        ))

        # Add roll event markers
        for i, event in enumerate(roll_events):
            roll_time = event['timestamp']

            # Vertical dotted line
            fig.add_vline(
                x=roll_time,
                line=dict(color='#ffcc00', width=2, dash='dot'),
                opacity=0.6,
            )

            # Find total P&L value at this time
            total_at_roll = total_df[total_df['timestamp'] >= roll_time]['total_pnl'].iloc[0] if not total_df[total_df['timestamp'] >= roll_time].empty else 0

            # Star marker on total P&L line
            fig.add_trace(go.Scatter(
                x=[roll_time],
                y=[total_at_roll],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='yellow',
                    line=dict(color='white', width=2)
                ),
                hovertext=event['description'],
                hovertemplate='<b>ROLL</b><br>%{hovertext}<extra></extra>',
                showlegend=False,
            ))

            # Annotation above chart
            fig.add_annotation(
                x=roll_time,
                y=1.05,
                yref='paper',
                text='🔄 Roll',
                showarrow=False,
                font=dict(size=10, color='#ffcc00'),
                yanchor='bottom',
            )

        # Update layout
        title = f"{metadata.get('underlying', 'SPX')} Iron Condor - "
        title += f"{metadata.get('expiration', '')} (Minute-Level P&L)"

        fig.update_layout(
            title=title,
            xaxis=dict(
                title='Time (ET)',
                tickformat='%H:%M',  # Show only time
                dtick=60000,  # 1 minute intervals (in milliseconds)
                gridcolor='#333',
            ),
            yaxis=dict(
                title='P&L ($)',
                tickformat='$,.0f',
                zeroline=True,
                zerolinecolor='#666',
                zerolinewidth=1,
                gridcolor='#333',
            ),
            template='plotly_dark',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.02,
                xanchor='left',
                x=0,
            ),
            margin=dict(t=100, b=60, l=60, r=30),
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating chart for match {match_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_match_selector_options(transformer: Optional[MinutePnLTransformer] = None) -> list:
    """
    Get list of matches available for minute-level P&L charts.

    Returns:
        List of dicts with 'label' and 'value' for dropdown options
    """
    if transformer is None:
        transformer = MinutePnLTransformer()

    try:
        matches = transformer.get_all_matches()

        options = []
        for match in matches:
            option = {
                'label': match['label'],
                'value': match['match_id']
            }
            options.append(option)

        return options

    except Exception as e:
        logger.error(f"Error getting match options: {e}")
        return []
