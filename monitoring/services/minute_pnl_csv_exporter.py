"""
CSV exporter for minute-level P&L data.

Exports detailed minute-by-minute P&L data in a 23-column format:
- Position metadata (timestamp, match_id, underlying, spx_price, total_pnl)
- Per-leg data (symbol, strike, type, side, price, unrealized_pnl, realized_pnl, status) × 4 legs
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from monitoring.services.minute_pnl_transformer import MinutePnLTransformer

logger = logging.getLogger(__name__)


class MinutePnLCSVExporter:
    """Exports minute-level P&L data to CSV format."""

    def __init__(self):
        """Initialize CSV exporter."""
        self.transformer = MinutePnLTransformer()

    def export_match_csv(self, match_id: str, output_path: Optional[str] = None) -> str:
        """
        Export minute-level P&L data for a match to CSV.

        CSV format includes:
        - Metadata headers (position type, dates, P&L, etc.)
        - Column headers (23 columns)
        - Minute-by-minute data rows

        Args:
            match_id: Trade match ID
            output_path: Optional output file path (defaults to temp file)

        Returns:
            Path to generated CSV file
        """
        try:
            # Get data
            leg_df = self.transformer.get_match_timeseries(match_id)
            total_df = self.transformer.get_match_total_pnl(match_id)
            metadata = self.transformer.get_match_metadata(match_id)
            roll_events = self.transformer.get_roll_events(match_id)

            if leg_df.empty or total_df.empty:
                logger.warning(f"No data available for match {match_id}")
                return ""

            # Determine output path
            if output_path is None:
                output_path = f"/tmp/minute_pnl_{match_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Write CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write metadata headers
                writer.writerow([f"# Position Type: Iron Condor"])
                writer.writerow([f"# Underlying: {metadata.get('underlying', 'SPX')}"])
                writer.writerow([f"# Expiration: {metadata.get('expiration', '')}"])
                writer.writerow([f"# Open Date: {metadata.get('opened_at', '')}"])
                writer.writerow([f"# Close Date: {metadata.get('closed_at', '')}"])
                writer.writerow([f"# Number of Rolls: {len(roll_events)}"])
                writer.writerow([f"# Final P&L: ${metadata.get('realized_pnl', 0):.2f}"])
                writer.writerow([])  # Blank line

                # Pivot data to get legs as columns
                pivoted_data = self._pivot_leg_data(leg_df, total_df)

                # Write column headers
                headers = self._get_csv_headers(pivoted_data)
                writer.writerow(headers)

                # Write data rows
                for row in pivoted_data:
                    writer.writerow(row)

            logger.info(f"Exported CSV to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting CSV for match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _pivot_leg_data(self, leg_df, total_df):
        """
        Pivot leg data from long format to wide format (legs as columns).

        Returns list of row lists.
        """
        # Get unique timestamps
        timestamps = sorted(total_df['timestamp'].unique())

        rows = []

        for timestamp in timestamps:
            # Get total P&L at this timestamp
            total_row = total_df[total_df['timestamp'] == timestamp]
            if total_row.empty:
                continue

            total_pnl = total_row.iloc[0]['total_pnl']

            # Get all legs at this timestamp
            legs_at_time = leg_df[leg_df['timestamp'] == timestamp]

            # Build row
            row = [
                timestamp.strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
                total_pnl,  # total_pnl
            ]

            # Add up to 4 legs (pad with empty if fewer)
            leg_count = 0
            for _, leg in legs_at_time.iterrows():
                if leg_count >= 4:
                    break

                symbol = leg['leg_symbol']
                strike = leg['strike']
                option_type = leg['option_type']
                side = leg['side']
                unrealized_pnl = leg['unrealized_pnl']
                realized_pnl = leg['realized_pnl']
                status = 'realized' if leg['is_realized'] else 'unrealized'

                row.extend([
                    symbol,              # leg_N_symbol
                    strike,              # leg_N_strike
                    option_type,         # leg_N_type
                    side,                # leg_N_side
                    unrealized_pnl,      # leg_N_unrealized_pnl
                    realized_pnl,        # leg_N_realized_pnl
                    status,              # leg_N_status
                ])

                leg_count += 1

            # Pad remaining legs
            while leg_count < 4:
                row.extend(['', '', '', '', '', '', ''])  # 7 empty columns per leg
                leg_count += 1

            rows.append(row)

        return rows

    def _get_csv_headers(self, pivoted_data):
        """Get column headers for CSV."""
        headers = ['timestamp', 'total_pnl']

        for i in range(1, 5):
            headers.extend([
                f'leg_{i}_symbol',
                f'leg_{i}_strike',
                f'leg_{i}_type',
                f'leg_{i}_side',
                f'leg_{i}_unrealized_pnl',
                f'leg_{i}_realized_pnl',
                f'leg_{i}_status',
            ])

        return headers
