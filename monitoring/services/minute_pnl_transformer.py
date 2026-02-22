"""
Data transformer for minute-level P&L visualization.

Fetches and transforms minute-level P&L data into formats suitable for:
- Plotly chart rendering (5 traces per iron condor)
- CSV export
- Roll event detection
"""

import duckdb
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from monitoring.services.minute_pnl_utils import localize_eastern

logger = logging.getLogger(__name__)


class MinutePnLTransformer:
    """Transforms minute-level P&L data for visualization and export."""

    # Color scheme for iron condor legs (matches existing dashboard colors)
    LEG_COLORS = {
        'put_short': '#ff9500',   # Orange
        'put_long': '#9966ff',    # Purple
        'call_short': '#ff66ff',  # Magenta
        'call_long': '#00bfff',   # Blue
    }

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize transformer.

        Args:
            db_path: Path to DuckDB database (defaults to data_store/trades.duckdb)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data_store" / "trades.duckdb"
        self.db_path = db_path

    def get_match_timeseries(self, match_id: str) -> pd.DataFrame:
        """
        Get minute-by-minute timeseries data for all legs in a match.

        Returns DataFrame with columns:
        - timestamp
        - leg_id
        - leg_symbol
        - leg_type (put_short, put_long, call_short, call_long)
        - strike
        - unrealized_pnl
        - realized_pnl
        - is_realized
        - color (assigned color for this leg)

        Args:
            match_id: Trade match ID

        Returns:
            DataFrame with minute-level data for all legs
        """
        with duckdb.connect(str(self.db_path)) as conn:
            # Get leg details with P&L data
            query = """
                SELECT
                    p.timestamp,
                    p.leg_id,
                    l.symbol,
                    l.option_type,
                    l.side,
                    l.strike,
                    p.unrealized_pnl,
                    p.realized_pnl,
                    p.is_realized
                FROM minute_leg_pnl p
                JOIN trade_legs l ON p.leg_id = l.leg_id
                WHERE l.match_id = ?
                ORDER BY p.timestamp, l.strike, l.option_type
            """

            results = conn.execute(query, [match_id]).fetchall()

            if not results:
                logger.warning(f"No P&L data found for match {match_id}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(results, columns=[
                'timestamp', 'leg_id', 'leg_symbol', 'option_type', 'side',
                'strike', 'unrealized_pnl', 'realized_pnl', 'is_realized'
            ])

            # Convert timestamps
            df['timestamp'] = df['timestamp'].apply(localize_eastern)

            # Convert Decimal to float
            df['strike'] = df['strike'].astype(float)
            df['unrealized_pnl'] = df['unrealized_pnl'].astype(float)
            df['realized_pnl'] = df['realized_pnl'].astype(float)

            # Assign leg types for color coding
            df['leg_type'] = df.apply(self._get_leg_type, axis=1)

            # Assign colors
            df['color'] = df['leg_type'].map(self.LEG_COLORS)

            return df

    def _get_leg_type(self, row) -> str:
        """
        Determine leg type for color assignment.

        Args:
            row: DataFrame row with option_type and side

        Returns:
            Leg type string (put_short, put_long, call_short, call_long)
        """
        option_type = row['option_type'].upper()
        side = row['side'].upper()

        if option_type == 'PUT' and side == 'SHORT':
            return 'put_short'
        elif option_type == 'PUT' and side == 'LONG':
            return 'put_long'
        elif option_type == 'CALL' and side == 'SHORT':
            return 'call_short'
        elif option_type == 'CALL' and side == 'LONG':
            return 'call_long'
        else:
            return 'unknown'

    def get_match_total_pnl(self, match_id: str) -> pd.DataFrame:
        """
        Get total P&L for the entire match at each minute.

        Returns DataFrame with columns:
        - timestamp
        - total_pnl
        - total_unrealized_pnl
        - total_realized_pnl
        - num_open_legs
        - num_closed_legs

        Args:
            match_id: Trade match ID

        Returns:
            DataFrame with aggregated P&L
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = """
                SELECT
                    timestamp,
                    total_pnl,
                    total_unrealized_pnl,
                    total_realized_pnl,
                    num_open_legs,
                    num_closed_legs
                FROM minute_match_pnl
                WHERE match_id = ?
                ORDER BY timestamp
            """

            results = conn.execute(query, [match_id]).fetchall()

            if not results:
                logger.warning(f"No match-level P&L data found for match {match_id}")
                return pd.DataFrame()

            df = pd.DataFrame(results, columns=[
                'timestamp', 'total_pnl', 'total_unrealized_pnl',
                'total_realized_pnl', 'num_open_legs', 'num_closed_legs'
            ])

            # Convert timestamps
            df['timestamp'] = df['timestamp'].apply(localize_eastern)

            # Convert Decimal to float
            df['total_pnl'] = df['total_pnl'].astype(float)
            df['total_unrealized_pnl'] = df['total_unrealized_pnl'].astype(float)
            df['total_realized_pnl'] = df['total_realized_pnl'].astype(float)

            return df

    def get_roll_events(self, match_id: str) -> List[Dict]:
        """
        Get roll/adjustment events for a match.

        Returns list of dicts with:
        - timestamp: When the roll occurred
        - closed_legs: List of closed leg IDs
        - opened_legs: List of newly opened leg IDs
        - description: Human-readable description

        Args:
            match_id: Trade match ID

        Returns:
            List of roll event dictionaries
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = """
                SELECT
                    roll_time,
                    closed_leg_symbols,
                    opened_leg_symbols
                FROM trade_rolls
                WHERE match_id = ?
                ORDER BY roll_time
            """

            results = conn.execute(query, [match_id]).fetchall()

            roll_events = []
            for roll_time, closed_symbols, opened_symbols in results:
                roll_time = localize_eastern(roll_time)

                closed_list = closed_symbols.split(',') if closed_symbols else []
                opened_list = opened_symbols.split(',') if opened_symbols else []

                event = {
                    'timestamp': roll_time,
                    'closed_legs': closed_list,
                    'opened_legs': opened_list,
                    'description': f"Rolled {len(closed_list)} legs → {len(opened_list)} legs"
                }
                roll_events.append(event)

            return roll_events

    def get_match_metadata(self, match_id: str) -> Dict:
        """
        Get metadata about a trade match.

        Args:
            match_id: Trade match ID

        Returns:
            Dictionary with match metadata
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = """
                SELECT
                    underlying,
                    expiration,
                    opened_at,
                    closed_at,
                    realized_pnl,
                    num_rolls,
                    status
                FROM trade_matches
                WHERE match_id = ?
            """

            result = conn.execute(query, [match_id]).fetchone()

            if not result:
                logger.warning(f"Match {match_id} not found")
                return {}

            metadata = {
                'match_id': match_id,
                'underlying': result[0],
                'expiration': result[1],
                'opened_at': localize_eastern(result[2]),
                'closed_at': localize_eastern(result[3]),
                'realized_pnl': float(result[4]) if result[4] else 0.0,
                'num_rolls': result[5] or 0,
                'status': result[6]
            }

            return metadata

    def assign_leg_colors(self, legs: List[Dict]) -> Dict[str, str]:
        """
        Assign consistent colors to legs based on their role in the iron condor.

        For a 4-leg iron condor:
        - Put short: Orange
        - Put long: Purple
        - Call short: Magenta
        - Call long: Blue

        Args:
            legs: List of leg dictionaries with 'option_type' and 'side'

        Returns:
            Dictionary mapping leg_id to color hex code
        """
        color_map = {}

        for leg in legs:
            leg_type = self._get_leg_type(leg)
            color = self.LEG_COLORS.get(leg_type, '#ffffff')
            color_map[leg['leg_id']] = color

        return color_map

    def get_all_matches(self) -> List[Dict]:
        """
        Get list of all trade matches with minute-level data.

        Returns:
            List of match metadata dictionaries
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = """
                SELECT DISTINCT m.match_id, m.underlying, m.expiration, m.opened_at, m.status
                FROM trade_matches m
                WHERE EXISTS (
                    SELECT 1 FROM minute_match_pnl p WHERE p.match_id = m.match_id
                )
                ORDER BY m.opened_at DESC
            """

            results = conn.execute(query).fetchall()

            matches = []
            for row in results:
                match = {
                    'match_id': row[0],
                    'underlying': row[1],
                    'expiration': row[2],
                    'opened_at': localize_eastern(row[3]),
                    'status': row[4],
                    'label': f"{row[1]} {row[2]} ({row[4]})"
                }
                matches.append(match)

            return matches

    def get_matches_by_date(self, filter_date: date) -> List[Dict]:
        """
        Get trade matches for a specific date.

        Args:
            filter_date: Date to filter by

        Returns:
            List of match metadata dictionaries for that date
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = """
                SELECT DISTINCT m.match_id, m.underlying, m.expiration, m.opened_at, m.status
                FROM trade_matches m
                WHERE EXISTS (
                    SELECT 1 FROM minute_match_pnl p WHERE p.match_id = m.match_id
                )
                AND DATE(m.opened_at) = ?
                ORDER BY m.opened_at DESC
            """

            results = conn.execute(query, [filter_date]).fetchall()

            matches = []
            for row in results:
                opened_at = localize_eastern(row[3])
                match = {
                    'match_id': row[0],
                    'underlying': row[1],
                    'expiration': row[2],
                    'opened_at': opened_at,
                    'status': row[4],
                    'label': f"{row[1]} {row[2]} @ {opened_at.strftime('%I:%M %p')} ({row[4]})"
                }
                matches.append(match)

            return matches
