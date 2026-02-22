#!/usr/bin/env python3
"""
Backfill Historical 1-Minute Candles

Fetches historical intraday data from ThetaData and stores in DuckDB.
Supports SPX, VIX, and other symbols with 1-minute resolution.

Usage:
    # Backfill last 30 days
    python scripts/backfill_candles.py --symbols SPX VIX --days 30

    # Backfill specific date range
    python scripts/backfill_candles.py --symbols SPX --start 2024-01-01 --end 2024-12-31

    # Backfill in smaller batches (slower but safer)
    python scripts/backfill_candles.py --symbols SPX --days 90 --batch-days 10
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from data.providers.theta_provider import ThetaDataProvider
from ml.data.candle_loader import CandleLoader
from config.settings import settings

console = Console()


class CandleBackfiller:
    """Backfill historical 1-minute candles using ThetaData."""

    def __init__(self, candle_loader: CandleLoader):
        """
        Initialize backfiller.

        Args:
            candle_loader: CandleLoader instance for database operations
        """
        self.candle_loader = candle_loader
        self.stats = {
            'symbols_processed': 0,
            'total_candles': 0,
            'errors': 0
        }

    def backfill_symbol(
        self,
        theta_provider: ThetaDataProvider,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        batch_days: int = 30
    ) -> int:
        """
        Backfill candles for a single symbol.

        Args:
            theta_provider: Connected ThetaData provider
            symbol: Symbol to backfill (e.g., 'SPX', 'VIX')
            start_date: Start date
            end_date: End date
            batch_days: Days per batch request

        Returns:
            Number of candles inserted
        """
        try:
            # Check existing data
            existing_range = self.candle_loader.get_available_date_range(symbol)

            if existing_range:
                first_ts, last_ts = existing_range
                console.print(
                    f"[yellow]Existing data for {symbol}: "
                    f"{first_ts.date()} to {last_ts.date()}[/yellow]"
                )

            # Fetch data from ThetaData
            console.print(f"\n[cyan]Fetching {symbol} from ThetaData...[/cyan]")

            df = theta_provider.get_hist_bars_batch(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval_seconds=60,  # 1-minute bars
                batch_days=batch_days
            )

            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return 0

            # Display sample
            console.print(f"\n[green]Retrieved {len(df)} candles[/green]")
            console.print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            console.print(f"\nSample (first 3 rows):")
            console.print(df.head(3).to_string())

            # Insert into database
            console.print(f"\n[cyan]Inserting into database...[/cyan]")
            inserted_count = self.candle_loader.insert_candles(symbol, df)

            console.print(f"[green]✓ Inserted {inserted_count} candles for {symbol}[/green]")

            return inserted_count

        except Exception as e:
            logger.error(f"Failed to backfill {symbol}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return 0

    def display_summary(self, symbols: List[str]):
        """Display summary table of backfilled data."""
        console.print("\n[bold cyan]Backfill Summary[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan")
        table.add_column("Candles", justify="right", style="green")
        table.add_column("Date Range", style="yellow")
        table.add_column("First Timestamp", style="dim")
        table.add_column("Last Timestamp", style="dim")

        for symbol in symbols:
            count = self.candle_loader.get_candle_count(symbol)
            date_range = self.candle_loader.get_available_date_range(symbol)

            if date_range:
                first_ts, last_ts = date_range
                range_str = f"{first_ts.date()} to {last_ts.date()}"
                first_str = first_ts.strftime("%Y-%m-%d %H:%M")
                last_str = last_ts.strftime("%Y-%m-%d %H:%M")
            else:
                range_str = "No data"
                first_str = "-"
                last_str = "-"

            table.add_row(
                symbol,
                f"{count:,}",
                range_str,
                first_str,
                last_str
            )

        console.print(table)

        # Overall stats
        console.print(f"\n[bold]Overall Statistics:[/bold]")
        console.print(f"Symbols processed: {self.stats['symbols_processed']}")
        console.print(f"Total candles inserted: {self.stats['total_candles']:,}")
        console.print(f"Errors: {self.stats['errors']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill historical 1-minute candles from ThetaData"
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SPX', 'VIX'],
        help='Symbols to backfill (default: SPX VIX)'
    )

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        '--days',
        type=int,
        help='Backfill last N days'
    )
    date_group.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD) - requires --end'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD) - requires --start'
    )

    parser.add_argument(
        '--batch-days',
        type=int,
        default=30,
        help='Days per batch request (default: 30)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    console.print("\n[bold cyan]Historical Candle Backfiller[/bold cyan]")
    console.print("[cyan]Powered by ThetaData[/cyan]\n")

    # Determine date range
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        if not args.end:
            console.print("[red]Error: --end required when using --start[/red]")
            sys.exit(1)

        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')

    console.print(f"[yellow]Date Range:[/yellow] {start_date.date()} to {end_date.date()}")
    console.print(f"[yellow]Symbols:[/yellow] {', '.join(args.symbols)}")
    console.print(f"[yellow]Batch Size:[/yellow] {args.batch_days} days\n")

    # Check credentials
    if not settings.THETADATA_USERNAME or not settings.THETADATA_PASSWORD:
        console.print("[red]Error: ThetaData credentials not configured![/red]")
        console.print("Set THETADATA_USERNAME and THETADATA_PASSWORD in .env file")
        sys.exit(1)

    # Initialize components
    candle_loader = CandleLoader()
    backfiller = CandleBackfiller(candle_loader)

    # Connect to ThetaData
    try:
        with ThetaDataProvider() as theta:
            console.print("[green]✓ Connected to ThetaData[/green]\n")

            # Process each symbol
            for symbol in args.symbols:
                console.print(f"\n[bold cyan]Processing {symbol}...[/bold cyan]")

                inserted = backfiller.backfill_symbol(
                    theta_provider=theta,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    batch_days=args.batch_days
                )

                backfiller.stats['symbols_processed'] += 1
                backfiller.stats['total_candles'] += inserted

            # Display summary
            backfiller.display_summary(args.symbols)

            console.print("\n[green]✓ Backfill complete![/green]\n")

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        console.print(f"\n[red]✗ Backfill failed: {e}[/red]\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
