"""
Payoff Diagram Generator

Generates Plotly "hockey stick" payoff diagrams for option strategies.
Shows profit/loss zones, breakeven points, and current price marker.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date
import numpy as np
import plotly.graph_objs as go
from loguru import logger

from execution.broker_api.schwab_client import schwab_client


@dataclass
class PayoffPoint:
    """Single point on payoff diagram."""
    underlying_price: float
    profit_loss: float


class PayoffDiagramGenerator:
    """
    Generates compact payoff diagrams for embedding in trade cards.

    Features:
    - Hockey stick visualization
    - Green fill for profit zone
    - Red fill for loss zone
    - Current underlying price marker
    - Breakeven point markers
    - Compact 150px height
    """

    # Chart styling - Enhanced for better visual hierarchy
    CHART_HEIGHT = 360  # 2x taller for better readability
    PROFIT_COLOR = '#00ff88'
    LOSS_COLOR = '#ff4444'
    BREAKEVEN_COLOR = '#ffcc00'
    CURRENT_PRICE_COLOR = '#00bfff'
    MIDPOINT_COLOR = '#ff4444'
    SHORT_STRIKE_COLOR = '#ff9900'  # Orange for short strikes
    DELTA_10_COLOR = '#ff66ff'  # Magenta for 10 delta
    DELTA_20_COLOR = '#9966ff'  # Purple for 20 delta
    BACKGROUND_COLOR = 'rgba(0,0,0,0)'
    GRID_COLOR = 'rgba(255,255,255,0.1)'

    def __init__(self):
        """Initialize payoff diagram generator."""
        logger.info("PayoffDiagramGenerator initialized")

    def generate_payoff_diagram(
        self,
        legs: List[Dict],
        current_underlying_price: Optional[float] = None,
        strategy_type: str = "UNKNOWN",
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        underlying_symbol: Optional[str] = None,
        expiration_date: Optional[date] = None
    ) -> go.Figure:
        """
        Generate a payoff diagram for a trade.

        Args:
            legs: List of leg dictionaries with strike, option_type, quantity, entry_price
            current_underlying_price: Current price of underlying (for marker)
            strategy_type: Strategy type for title
            x_range: Optional fixed x-axis range (min, max) for consistent scaling
            y_range: Optional fixed y-axis range (min, max) for consistent scaling

        Returns:
            Plotly Figure object
        """
        if not legs:
            return self._create_empty_figure()

        # Calculate price range
        strikes = [leg.get('strike') for leg in legs if leg.get('strike')]
        if not strikes:
            return self._create_empty_figure()

        # Use provided x_range if available, otherwise calculate from strikes
        if x_range:
            lower_bound, upper_bound = x_range
        else:
            min_strike = min(strikes)
            max_strike = max(strikes)

            # Extend range 20% beyond strikes
            price_range = max_strike - min_strike
            if price_range == 0:
                price_range = min_strike * 0.2

            lower_bound = min_strike - price_range * 0.3
            upper_bound = max_strike + price_range * 0.3

        # Generate price points
        prices = np.linspace(lower_bound, upper_bound, 100)

        # Calculate payoff at each price
        payoffs = [self._calculate_payoff_at_price(price, legs) for price in prices]

        # Find breakeven points
        breakevens = self._find_breakevens(prices, payoffs)

        # Calculate midpoint for iron condors / 4-leg trades
        midpoint = None
        short_strikes = []

        # Find short strikes (negative quantity) - only for OPEN positions (risk boundaries)
        short_strikes = [
            leg.get('strike')
            for leg in legs
            if leg.get('quantity', 0) < 0
            and leg.get('strike')
            and not leg.get('_is_closed', False)  # Exclude closed legs
        ]

        # Calculate midpoint if we have multiple short strikes
        if len(short_strikes) >= 2:
            midpoint = sum(short_strikes) / len(short_strikes)

        # Fetch delta strikes if underlying and expiration provided
        delta_strikes = None
        if underlying_symbol and expiration_date:
            delta_strikes = self._find_delta_strikes_sync(underlying_symbol, expiration_date)

        # Create figure with short strikes and delta markers
        fig = self._create_figure(
            prices,
            payoffs,
            breakevens,
            current_underlying_price,
            midpoint,
            short_strikes,
            y_range,
            delta_strikes
        )

        return fig

    def _calculate_payoff_at_price(self, underlying_price: float, legs: List[Dict]) -> float:
        """
        Calculate total profit/loss at a given underlying price at expiration.

        Args:
            underlying_price: Price to calculate payoff at
            legs: List of position legs

        Returns:
            Net profit/loss
        """
        total_payoff = 0.0

        for leg in legs:
            option_type = leg.get('option_type')
            strike = leg.get('strike')
            quantity = leg.get('quantity', 0)
            entry_price = leg.get('entry_price', 0)

            if not option_type:
                # Stock position
                # Long stock: profit = current - entry
                # Short stock: profit = entry - current
                if quantity > 0:
                    payoff = (underlying_price - entry_price) * quantity
                else:
                    payoff = (entry_price - underlying_price) * abs(quantity)
                total_payoff += payoff
                continue

            # Options
            multiplier = 100  # Options multiplier

            if option_type == 'CALL':
                # Call value at expiration = max(0, underlying - strike)
                intrinsic = max(0, underlying_price - strike)
            else:  # PUT
                # Put value at expiration = max(0, strike - underlying)
                intrinsic = max(0, strike - underlying_price)

            # For long positions (positive quantity): profit = intrinsic - premium_paid
            # For short positions (negative quantity): profit = premium_received - intrinsic
            if quantity > 0:  # Long
                payoff = (intrinsic - entry_price) * quantity * multiplier
            else:  # Short
                payoff = (entry_price - intrinsic) * abs(quantity) * multiplier

            total_payoff += payoff

        return total_payoff

    def _find_breakevens(
        self,
        prices: np.ndarray,
        payoffs: List[float]
    ) -> List[float]:
        """
        Find breakeven points where payoff crosses zero.

        Args:
            prices: Array of underlying prices
            payoffs: List of corresponding payoffs

        Returns:
            List of breakeven prices
        """
        breakevens = []

        for i in range(len(payoffs) - 1):
            # Check if payoff crosses zero between this point and next
            if (payoffs[i] <= 0 and payoffs[i + 1] > 0) or \
               (payoffs[i] >= 0 and payoffs[i + 1] < 0):
                # Linear interpolation to find exact breakeven
                ratio = abs(payoffs[i]) / (abs(payoffs[i]) + abs(payoffs[i + 1]))
                breakeven = prices[i] + ratio * (prices[i + 1] - prices[i])
                breakevens.append(breakeven)

        return breakevens

    def _find_delta_strikes_sync(self, symbol: str, expiration: date) -> Optional[Dict]:
        """
        Find strikes closest to 10Δ and 20Δ for both calls and puts.

        Runs async schwab_client call in sync context.

        Args:
            symbol: Underlying symbol (e.g., 'SPX', 'SPXW')
            expiration: Expiration date to filter option chain

        Returns:
            Dict with keys: 'put_10d', 'put_20d', 'call_10d', 'call_20d' (strike prices)
            or None if unable to fetch
        """
        import asyncio

        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run async function in sync context
            result = loop.run_until_complete(self._find_delta_strikes(symbol, expiration))
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch delta strikes for {symbol}: {e}")
            return None

    async def _find_delta_strikes(self, symbol: str, expiration: date) -> Optional[Dict]:
        """
        Find strikes closest to 10Δ and 20Δ for both calls and puts.

        Args:
            symbol: Underlying symbol (e.g., 'SPX', 'SPXW')
            expiration: Expiration date to filter option chain

        Returns:
            Dict with keys: 'put_10d', 'put_20d', 'call_10d', 'call_20d' (strike prices)
        """
        try:
            # Normalize symbol for option chain lookup
            # SPXW -> $SPX (weekly SPX options trade on SPX index)
            if symbol in ('SPXW', 'SPX'):
                chain_symbol = '$SPX'
            elif symbol == 'NDX' or symbol == 'NDXP':
                chain_symbol = '$NDX'
            elif symbol == 'RUT' or symbol == 'RUTW':
                chain_symbol = '$RUT'
            else:
                # For stocks/ETFs, use symbol as-is
                chain_symbol = symbol.replace('$', '')

            logger.info(f"Fetching option chain for {chain_symbol} expiring {expiration}")

            # Fetch option chain for specific expiration
            # Use high strike_count to ensure far OTM strikes (10-delta) are included
            chain_data = await schwab_client.get_option_chain(
                symbol=chain_symbol,
                from_date=expiration,
                to_date=expiration,
                include_underlying_quote=True,
                strike_count=100
            )

            if not chain_data:
                logger.warning(f"No option chain data for {chain_symbol}")
                return None

            # Extract call and put maps
            call_exp_map = chain_data.get('callExpDateMap', {})
            put_exp_map = chain_data.get('putExpDateMap', {})

            if not call_exp_map or not put_exp_map:
                logger.warning(f"Empty option chain for {chain_symbol}")
                return None

            # Get the expiration key (format: "2026-02-12:0" where :N is days to expiration)
            exp_str = expiration.strftime('%Y-%m-%d')
            exp_keys = [k for k in call_exp_map.keys() if k.startswith(exp_str)]

            if not exp_keys:
                logger.warning(f"No options found for expiration {expiration}")
                logger.debug(f"Available expirations: {list(call_exp_map.keys())[:5]}")
                return None

            exp_key = exp_keys[0]  # Use first matching expiration
            logger.debug(f"Using expiration key: {exp_key}")

            # Extract strikes and deltas
            call_strikes = call_exp_map.get(exp_key, {})
            put_strikes = put_exp_map.get(exp_key, {})

            # Find closest strikes to target deltas
            delta_strikes = {}

            # Find call deltas (positive: 0.10 and 0.20)
            call_10d = self._find_closest_delta_strike(call_strikes, 0.10, 'call')
            call_20d = self._find_closest_delta_strike(call_strikes, 0.20, 'call')

            # Find put deltas (negative: -0.10 and -0.20, but we look for absolute value)
            put_10d = self._find_closest_delta_strike(put_strikes, -0.10, 'put')
            put_20d = self._find_closest_delta_strike(put_strikes, -0.20, 'put')

            logger.info(f"Delta strike lookup results: call_10d={call_10d}, call_20d={call_20d}, put_10d={put_10d}, put_20d={put_20d}")
            logger.info(f"  Total strikes available: {len(call_strikes)} calls, {len(put_strikes)} puts")

            if call_10d:
                delta_strikes['call_10d'] = call_10d
            else:
                logger.warning(f"  ⚠️ Could not find call 10-delta strike (chain may be too narrow)")
            if call_20d:
                delta_strikes['call_20d'] = call_20d
            if put_10d:
                delta_strikes['put_10d'] = put_10d
            else:
                logger.warning(f"  ⚠️ Could not find put 10-delta strike (chain may be too narrow)")
            if put_20d:
                delta_strikes['put_20d'] = put_20d

            logger.info(f"Found delta strikes: {delta_strikes}")
            return delta_strikes if delta_strikes else None

        except Exception as e:
            logger.error(f"Error finding delta strikes: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_closest_delta_strike(
        self,
        strike_map: Dict,
        target_delta: float,
        option_type: str
    ) -> Optional[float]:
        """
        Find the strike price closest to the target delta.

        Args:
            strike_map: Dict of strikes from option chain (strike -> option data)
            target_delta: Target delta value (e.g., 0.10, -0.10)
            option_type: 'call' or 'put'

        Returns:
            Strike price closest to target delta, or None
        """
        best_strike = None
        best_diff = float('inf')

        for strike_str, contracts in strike_map.items():
            strike = float(strike_str)

            # contracts is a list with one element (the option contract)
            if not contracts:
                continue

            contract = contracts[0]
            delta = contract.get('delta')

            if delta is None:
                continue

            # Calculate difference from target
            diff = abs(delta - target_delta)

            if diff < best_diff:
                best_diff = diff
                best_strike = strike

        return best_strike

    def _create_figure(
        self,
        prices: np.ndarray,
        payoffs: List[float],
        breakevens: List[float],
        current_price: Optional[float],
        midpoint: Optional[float] = None,
        short_strikes: List[float] = None,
        y_range: Optional[Tuple[float, float]] = None,
        delta_strikes: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create the Plotly figure.

        Args:
            prices: Array of underlying prices
            payoffs: List of payoffs
            breakevens: List of breakeven prices
            current_price: Current underlying price
            midpoint: Midpoint price for iron condors (optional)
            short_strikes: List of short strike prices (optional)
            y_range: Optional fixed y-axis range (min, max) for consistent scaling
            delta_strikes: Dict with 10Δ and 20Δ strikes for calls/puts (optional)

        Returns:
            Plotly Figure
        """
        if short_strikes is None:
            short_strikes = []
        fig = go.Figure()

        # Split payoffs into profit and loss regions for different colors
        profit_payoffs = [p if p >= 0 else 0 for p in payoffs]
        loss_payoffs = [p if p < 0 else 0 for p in payoffs]

        # Profit zone (green fill) - subtle but clear
        fig.add_trace(go.Scatter(
            x=prices,
            y=profit_payoffs,
            mode='lines',
            line=dict(color=self.PROFIT_COLOR, width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)',  # Slightly more subtle
            name='Profit',
            hovertemplate='Price: $%{x:.0f}<br>P&L: $%{y:.0f}<extra></extra>'
        ))

        # Loss zone (red fill) - subtle but clear
        fig.add_trace(go.Scatter(
            x=prices,
            y=loss_payoffs,
            mode='lines',
            line=dict(color=self.LOSS_COLOR, width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)',  # Slightly more subtle
            name='Loss',
            hovertemplate='Price: $%{x:.0f}<br>P&L: $%{y:.0f}<extra></extra>'
        ))

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="rgba(255,255,255,0.5)",
            line_width=1
        )

        # Midpoint line for iron condors (before breakevens so it's behind)
        if midpoint:
            fig.add_vline(
                x=midpoint,
                line_dash="dot",
                line_color=self.MIDPOINT_COLOR,
                line_width=1.5,
                opacity=0.6
            )
            # Add annotation below the x-axis - tertiary info, subtle
            fig.add_annotation(
                x=midpoint,
                y=-0.18,  # Adjusted to stay within margin
                yref="paper",
                text=f"${midpoint:,.0f}",  # No decimals for cleaner look
                showarrow=False,
                font=dict(size=10, color=self.MIDPOINT_COLOR),  # Smaller, lighter
                opacity=0.7,
                yanchor="top"
            )

        # Short strike markers (risk boundaries)
        for strike in short_strikes:
            fig.add_vline(
                x=strike,
                line_dash="dot",
                line_color=self.SHORT_STRIKE_COLOR,
                line_width=1.5,
                opacity=0.8
            )
            # Add annotation below the x-axis - secondary info
            fig.add_annotation(
                x=strike,
                y=-0.11,  # Adjusted to stay within margin
                yref="paper",
                text=f"${strike:,.0f}",  # No decimals for cleaner look
                showarrow=False,
                font=dict(size=11, color=self.SHORT_STRIKE_COLOR),
                opacity=0.85,
                yanchor="top"
            )

        # Delta strike markers (10Δ and 20Δ)
        if delta_strikes:
            # 10 delta markers (magenta, high priority - critical risk zone)
            if 'put_10d' in delta_strikes:
                strike = delta_strikes['put_10d']
                fig.add_vline(
                    x=strike,
                    line_dash="dash",
                    line_color=self.DELTA_10_COLOR,
                    line_width=1.5,
                    opacity=0.5
                )
                fig.add_annotation(
                    x=strike,
                    y=1.30,  # Higher above chart - top priority
                    yref="paper",
                    text=f"10Δ",
                    showarrow=False,
                    font=dict(size=10, color=self.DELTA_10_COLOR, weight='bold'),
                    opacity=0.9,
                    yanchor="bottom"
                )

            if 'call_10d' in delta_strikes:
                strike = delta_strikes['call_10d']
                fig.add_vline(
                    x=strike,
                    line_dash="dash",
                    line_color=self.DELTA_10_COLOR,
                    line_width=1.5,
                    opacity=0.5
                )
                fig.add_annotation(
                    x=strike,
                    y=1.30,  # Higher above chart - top priority
                    yref="paper",
                    text=f"10Δ",
                    showarrow=False,
                    font=dict(size=10, color=self.DELTA_10_COLOR, weight='bold'),
                    opacity=0.9,
                    yanchor="bottom"
                )

            # 20 delta markers (purple, medium priority - early warning zone)
            if 'put_20d' in delta_strikes:
                strike = delta_strikes['put_20d']
                fig.add_vline(
                    x=strike,
                    line_dash="dot",
                    line_color=self.DELTA_20_COLOR,
                    line_width=1,
                    opacity=0.4
                )
                # Label above chart - medium priority
                fig.add_annotation(
                    x=strike,
                    y=1.20,  # Between 10Δ and underlying - clear spacing
                    yref="paper",
                    text=f"20Δ",
                    showarrow=False,
                    font=dict(size=9, color=self.DELTA_20_COLOR),
                    opacity=0.8,
                    yanchor="bottom"
                )
                # Strike price below midpoint line - tertiary info
                fig.add_annotation(
                    x=strike,
                    y=-0.25,  # Adjusted to stay within margin
                    yref="paper",
                    text=f"${strike:,.0f}",  # No decimals
                    showarrow=False,
                    font=dict(size=10, color=self.DELTA_20_COLOR),
                    opacity=0.6,
                    yanchor="top"
                )

            if 'call_20d' in delta_strikes:
                strike = delta_strikes['call_20d']
                fig.add_vline(
                    x=strike,
                    line_dash="dot",
                    line_color=self.DELTA_20_COLOR,
                    line_width=1,
                    opacity=0.4
                )
                # Label above chart - medium priority
                fig.add_annotation(
                    x=strike,
                    y=1.20,  # Between 10Δ and underlying - clear spacing
                    yref="paper",
                    text=f"20Δ",
                    showarrow=False,
                    font=dict(size=9, color=self.DELTA_20_COLOR),
                    opacity=0.8,
                    yanchor="bottom"
                )
                # Strike price below midpoint line - tertiary info
                fig.add_annotation(
                    x=strike,
                    y=-0.25,  # Adjusted to stay within margin
                    yref="paper",
                    text=f"${strike:,.0f}",  # No decimals
                    showarrow=False,
                    font=dict(size=10, color=self.DELTA_20_COLOR),
                    opacity=0.6,
                    yanchor="top"
                )

        # Breakeven markers - subtle but clear
        for be in breakevens:
            fig.add_vline(
                x=be,
                line_dash="dash",
                line_color=self.BREAKEVEN_COLOR,
                line_width=1,
                opacity=0.6
            )
            # Add breakeven label inside chart area
            fig.add_annotation(
                x=be,
                y=1.0,
                yref="paper",
                text=f"${be:,.0f}",  # Cleaner without "BE:" prefix
                showarrow=False,
                font=dict(size=10, color=self.BREAKEVEN_COLOR),
                opacity=0.8,
                yanchor="top",
                xanchor="center"
            )

        # Find max/min profit for Y-axis labeling
        # Use provided y_range if available, otherwise calculate from payoffs
        if y_range:
            min_payoff, max_payoff = y_range
        else:
            max_payoff = max(payoffs)
            min_payoff = min(payoffs)

        # Determine Y-axis tick values (show min, 0, max)
        y_tickvals = []
        y_ticktext = []

        if min_payoff < -10:  # Show min if it's significant
            y_tickvals.append(min_payoff)
            y_ticktext.append(f'${min_payoff:,.0f}')

        y_tickvals.append(0)
        y_ticktext.append('$0')

        if max_payoff > 10:  # Show max if it's significant
            y_tickvals.append(max_payoff)
            y_ticktext.append(f'${max_payoff:,.0f}')

        # Current price marker
        if current_price:
            # Find payoff at current price
            idx = np.argmin(np.abs(prices - current_price))
            current_payoff = payoffs[idx]

            fig.add_trace(go.Scatter(
                x=[current_price],
                y=[current_payoff],
                mode='markers',
                marker=dict(
                    size=12,  # Slightly larger for prominence
                    color=self.CURRENT_PRICE_COLOR,
                    symbol='diamond',
                    line=dict(color='white', width=1.5),
                    opacity=1.0
                ),
                name='Current',
                hovertemplate=f'Price: ${current_price:,.0f}<br>P&L: ${current_payoff:,.0f}<extra></extra>'
            ))

            # Vertical line at current price
            fig.add_vline(
                x=current_price,
                line_dash="solid",
                line_color=self.CURRENT_PRICE_COLOR,
                line_width=1.5,
                opacity=0.8
            )

            # Add annotation above chart - primary info, prominent
            fig.add_annotation(
                x=current_price,
                y=1.10,  # Below delta markers, clear spacing
                yref="paper",
                text=f"${current_price:,.0f}",  # No decimals for cleaner look
                showarrow=False,
                font=dict(size=12, color=self.CURRENT_PRICE_COLOR, weight='bold'),
                opacity=1.0,
                yanchor="bottom"
            )

            # Horizontal line at current P&L
            fig.add_hline(
                y=current_payoff,
                line_dash="dot",
                line_color=self.CURRENT_PRICE_COLOR,
                line_width=1
            )

        # Layout - Enhanced margins for better breathing room
        fig.update_layout(
            height=self.CHART_HEIGHT,
            margin=dict(l=45, r=15, t=50, b=80),  # Extra bottom margin for below-chart labels
            paper_bgcolor=self.BACKGROUND_COLOR,
            plot_bgcolor=self.BACKGROUND_COLOR,
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.06)',  # More subtle grid
                gridwidth=0.5,
                tickformat='$,.0f',
                tickfont=dict(size=11, color='rgba(255,255,255,0.6)'),  # Lighter, smaller
                title=None,
                showline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.06)',  # More subtle grid
                gridwidth=0.5,
                tickfont=dict(size=11, color='rgba(255,255,255,0.6)'),  # Lighter, smaller
                title=None,
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.3)',  # More subtle zero line
                zerolinewidth=1,
                showline=False,
                # Show max and min on the axis
                tickmode='array',
                tickvals=y_tickvals,
                ticktext=y_ticktext,
                range=[min_payoff, max_payoff] if y_range else None
            ),
            hovermode='x unified'
        )

        return fig

    def _create_empty_figure(self) -> go.Figure:
        """Create an empty placeholder figure."""
        fig = go.Figure()

        fig.add_annotation(
            x=0.5, y=0.5,
            text="No payoff data",
            showarrow=False,
            font=dict(size=12, color='rgba(255,255,255,0.5)'),
            xref='paper', yref='paper'
        )

        fig.update_layout(
            height=self.CHART_HEIGHT,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=self.BACKGROUND_COLOR,
            plot_bgcolor=self.BACKGROUND_COLOR,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        return fig

    def generate_strategy_payoff(
        self,
        strategy_type: str,
        entry_credit: float,
        entry_debit: float,
        max_profit: float,
        max_loss: float,
        short_strikes: List[float],
        long_strikes: List[float],
        current_price: Optional[float] = None
    ) -> go.Figure:
        """
        Generate payoff diagram from strategy parameters.

        Simplified version that doesn't require full leg details.

        Args:
            strategy_type: Type of strategy
            entry_credit: Net credit received
            entry_debit: Net debit paid
            max_profit: Maximum profit
            max_loss: Maximum loss
            short_strikes: List of short option strikes
            long_strikes: List of long option strikes
            current_price: Current underlying price

        Returns:
            Plotly Figure
        """
        all_strikes = short_strikes + long_strikes
        if not all_strikes:
            return self._create_empty_figure()

        min_strike = min(all_strikes)
        max_strike = max(all_strikes)
        strike_range = max_strike - min_strike

        if strike_range == 0:
            strike_range = min_strike * 0.2

        lower = min_strike - strike_range * 0.5
        upper = max_strike + strike_range * 0.5

        prices = np.linspace(lower, upper, 100)

        # Generate payoff based on strategy type
        if strategy_type == 'IRON_CONDOR':
            payoffs = self._iron_condor_payoff(
                prices, sorted(short_strikes), sorted(long_strikes), entry_credit
            )
        elif strategy_type == 'VERTICAL_SPREAD':
            payoffs = self._vertical_spread_payoff(
                prices, short_strikes, long_strikes, entry_credit, entry_debit
            )
        elif strategy_type in ['STRADDLE', 'STRANGLE']:
            payoffs = self._straddle_strangle_payoff(
                prices, short_strikes, entry_credit, entry_debit
            )
        else:
            # Generic payoff
            payoffs = [0] * len(prices)

        breakevens = self._find_breakevens(prices, payoffs)

        # Calculate midpoint for iron condors
        midpoint = None
        if strategy_type == 'IRON_CONDOR' and len(short_strikes) >= 2:
            midpoint = sum(short_strikes) / len(short_strikes)

        return self._create_figure(prices, payoffs, breakevens, current_price, midpoint, short_strikes)

    def _iron_condor_payoff(
        self,
        prices: np.ndarray,
        short_strikes: List[float],
        long_strikes: List[float],
        credit: float
    ) -> List[float]:
        """Calculate iron condor payoff at expiration."""
        # Assume: long put < short put < short call < long call
        long_put = min(long_strikes)
        short_put = min(short_strikes)
        short_call = max(short_strikes)
        long_call = max(long_strikes)

        payoffs = []
        for price in prices:
            if price <= long_put:
                # Max loss on put side
                payoff = credit - (short_put - long_put) * 100
            elif price <= short_put:
                # In put spread
                payoff = credit - (short_put - price) * 100
            elif price <= short_call:
                # Between short strikes - max profit
                payoff = credit
            elif price <= long_call:
                # In call spread
                payoff = credit - (price - short_call) * 100
            else:
                # Max loss on call side
                payoff = credit - (long_call - short_call) * 100

            payoffs.append(payoff)

        return payoffs

    def _vertical_spread_payoff(
        self,
        prices: np.ndarray,
        short_strikes: List[float],
        long_strikes: List[float],
        credit: float,
        debit: float
    ) -> List[float]:
        """Calculate vertical spread payoff."""
        if not short_strikes or not long_strikes:
            return [0] * len(prices)

        short_strike = short_strikes[0]
        long_strike = long_strikes[0]
        net_premium = credit - debit

        # Determine if it's a call or put spread based on strike relationship
        is_bull_spread = long_strike < short_strike

        payoffs = []
        for price in prices:
            if is_bull_spread:
                # Bull put spread (credit)
                if price >= short_strike:
                    payoff = net_premium
                elif price <= long_strike:
                    payoff = net_premium - (short_strike - long_strike) * 100
                else:
                    payoff = net_premium - (short_strike - price) * 100
            else:
                # Bear call spread (credit)
                if price <= short_strike:
                    payoff = net_premium
                elif price >= long_strike:
                    payoff = net_premium - (long_strike - short_strike) * 100
                else:
                    payoff = net_premium - (price - short_strike) * 100

            payoffs.append(payoff)

        return payoffs

    def _straddle_strangle_payoff(
        self,
        prices: np.ndarray,
        strikes: List[float],
        credit: float,
        debit: float
    ) -> List[float]:
        """Calculate straddle/strangle payoff."""
        if not strikes:
            return [0] * len(prices)

        net_premium = credit - debit
        is_short = credit > debit

        # For straddle: one strike, for strangle: two strikes
        if len(strikes) == 1:
            strike = strikes[0]
            payoffs = []
            for price in prices:
                move = abs(price - strike) * 100
                if is_short:
                    payoff = net_premium - move
                else:
                    payoff = move - abs(net_premium)
                payoffs.append(payoff)
        else:
            put_strike = min(strikes)
            call_strike = max(strikes)
            payoffs = []
            for price in prices:
                if price <= put_strike:
                    move = (put_strike - price) * 100
                elif price >= call_strike:
                    move = (price - call_strike) * 100
                else:
                    move = 0

                if is_short:
                    payoff = net_premium - move
                else:
                    payoff = move - abs(net_premium)
                payoffs.append(payoff)

        return payoffs


# Global instance
payoff_generator = PayoffDiagramGenerator()
