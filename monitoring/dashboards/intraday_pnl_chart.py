#!/usr/bin/env python3
"""
Intraday P&L Chart Visualization

Creates interactive Plotly charts showing realized vs unrealized P&L
with event markers for trade opens, closes, and rolls.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from monitoring.services.intraday_pnl_calculator import IntradaySnapshot


def create_intraday_pnl_chart(
    snapshots: List[IntradaySnapshot],
    title: Optional[str] = None,
    height: int = 600
) -> go.Figure:
    """
    Create intraday P&L chart with dual lines and event markers.

    Features:
    - Realized P&L (solid green/red line)
    - Unrealized P&L (dashed blue/orange line)
    - Total P&L (bold line)
    - Event markers for opens/closes/rolls
    - 30-minute tick marks on x-axis

    Args:
        snapshots: List of IntradaySnapshot objects
        title: Optional chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    if not snapshots:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Run backfill_intraday_pnl.py to populate.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='gray')
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=height
        )
        return fig

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([
        {
            'timestamp': s.timestamp,
            'realized_pnl': s.realized_pnl,
            'unrealized_pnl': s.unrealized_pnl,
            'total_pnl': s.total_pnl,
            'events': '<br>'.join(s.events) if s.events else None
        }
        for s in snapshots
    ])

    # Create figure
    fig = go.Figure()

    # Add realized P&L line (solid)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['realized_pnl'],
        mode='lines',
        name='Realized P&L',
        line=dict(
            color='#00ff88',
            width=2
        ),
        hovertemplate='<b>Realized P&L</b><br>%{y:$,.2f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'
    ))

    # Add unrealized P&L line (dashed)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['unrealized_pnl'],
        mode='lines',
        name='Unrealized P&L',
        line=dict(
            color='#ff9500',
            width=2,
            dash='dash'
        ),
        hovertemplate='<b>Unrealized P&L</b><br>%{y:$,.2f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'
    ))

    # Add total P&L line (bold)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_pnl'],
        mode='lines',
        name='Total P&L',
        line=dict(
            color='#00bfff',
            width=3
        ),
        hovertemplate='<b>Total P&L</b><br>%{y:$,.2f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'
    ))

    # Add event markers
    events_df = df[df['events'].notna()].copy()
    if not events_df.empty:
        fig.add_trace(go.Scatter(
            x=events_df['timestamp'],
            y=events_df['total_pnl'],
            mode='markers',
            name='Events',
            marker=dict(
                size=10,
                color='#ffcc00',
                symbol='star',
                line=dict(color='white', width=1)
            ),
            text=events_df['events'],
            hovertemplate='<b>Events</b><br>%{text}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'
        ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title or "Intraday P&L - Last 14 Days",
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            tickformat='%m/%d %H:%M',
            dtick=1800000,  # 30 minutes in milliseconds
        ),
        yaxis=dict(
            title="P&L ($)",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            tickformat='$,.0f'
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=60, l=60, r=30)
    )

    return fig


def create_daily_summary_chart(
    snapshots: List[IntradaySnapshot],
    height: int = 300
) -> go.Figure:
    """
    Create daily summary bar chart showing net P&L per day.

    Args:
        snapshots: List of IntradaySnapshot objects
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    if not snapshots:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='gray')
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=height
        )
        return fig

    # Get end-of-day P&L for each trading day
    df = pd.DataFrame([
        {
            'date': s.timestamp.date(),
            'total_pnl': s.total_pnl,
            'timestamp': s.timestamp
        }
        for s in snapshots
    ])

    # Get last snapshot of each day
    daily_df = df.sort_values('timestamp').groupby('date').last().reset_index()

    # Calculate daily change
    daily_df['daily_pnl'] = daily_df['total_pnl'].diff().fillna(daily_df['total_pnl'])

    # Color bars based on profit/loss
    colors = ['#00ff88' if pnl >= 0 else '#ff4444' for pnl in daily_df['daily_pnl']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily_df['date'],
        y=daily_df['daily_pnl'],
        marker_color=colors,
        name='Daily P&L',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>P&L: %{y:$,.2f}<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )

    fig.update_layout(
        title=dict(
            text="Daily P&L Summary",
            font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=False
        ),
        yaxis=dict(
            title="P&L ($)",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            tickformat='$,.0f'
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=30)
    )

    return fig


def create_summary_stats(snapshots: List[IntradaySnapshot]) -> dict:
    """
    Calculate summary statistics for display.

    Args:
        snapshots: List of IntradaySnapshot objects

    Returns:
        Dictionary with summary stats
    """
    if not snapshots:
        return {
            'total_pnl': 0,
            'best_day': 0,
            'worst_day': 0,
            'avg_daily_pnl': 0,
            'win_rate': 0,
            'num_trading_days': 0
        }

    # Get end-of-day snapshots
    df = pd.DataFrame([
        {
            'date': s.timestamp.date(),
            'total_pnl': s.total_pnl,
            'timestamp': s.timestamp
        }
        for s in snapshots
    ])

    daily_df = df.sort_values('timestamp').groupby('date').last().reset_index()
    daily_df['daily_pnl'] = daily_df['total_pnl'].diff().fillna(daily_df['total_pnl'])

    # Calculate stats
    total_pnl = daily_df['total_pnl'].iloc[-1] if not daily_df.empty else 0
    best_day = daily_df['daily_pnl'].max() if not daily_df.empty else 0
    worst_day = daily_df['daily_pnl'].min() if not daily_df.empty else 0
    avg_daily = daily_df['daily_pnl'].mean() if not daily_df.empty else 0

    winning_days = (daily_df['daily_pnl'] > 0).sum()
    total_days = len(daily_df)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

    return {
        'total_pnl': total_pnl,
        'best_day': best_day,
        'worst_day': worst_day,
        'avg_daily_pnl': avg_daily,
        'win_rate': win_rate,
        'num_trading_days': total_days
    }
