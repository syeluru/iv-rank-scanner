#!/usr/bin/env python3
"""
Ingest 1-minute historical candles for ML features.

Backfills intraday candle data from Schwab API for SPX and VIX.
Saves to both DuckDB and parquet for redundancy.

Usage:
    python scripts/ingest_candles.py --symbol SPX --date 2025-02-13
    python scripts/ingest_candles.py --symbol VIX --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/ingest_candles.py --backfill-days 30  # Backfill last 30 days for all symbols
"""

import asyncio
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from data.storage.database import DatabaseManager
from config.settings import settings


async def fetch_candles_from_schwab(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch 1-minute candles from Schwab API.

    Args:
        symbol: Symbol to fetch (SPX, VIX)
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # TODO: Implement actual Schwab API integration
    # For now, return empty DataFrame
    logger.warning(
        f"Schwab API integration not implemented yet. "
        f"Would fetch {symbol} candles from {start_date} to {end_date}"
    )

    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


async def save_candles_to_db(
    db: DatabaseManager,
    symbol: str,
    df: pd.DataFrame
) -> int:
    """
    Save candles to DuckDB.

    Args:
        db: DatabaseManager instance
        symbol: Symbol name
        df: DataFrame with candle data

    Returns:
        Number of rows inserted
    """
    if df.empty:
        return 0

    # Prepare data for insertion
    records = []
    for _, row in df.iterrows():
        records.append({
            'symbol': symbol,
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        })

    # Insert using batch insert
    query = """
    INSERT OR REPLACE INTO intraday_candles
    (symbol, timestamp, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    for record in records:
        try:
            db.execute(
                query,
                (
                    record['symbol'],
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                )
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Failed to insert candle: {e}")

    return inserted


async def save_candles_to_parquet(
    symbol: str,
    df: pd.DataFrame,
    year: int
) -> Path:
    """
    Save candles to parquet file.

    Args:
        symbol: Symbol name
        df: DataFrame with candle data
        year: Year for filename

    Returns:
        Path to saved file
    """
    parquet_dir = settings.DATA_STORE_PATH / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_1min_{year}.parquet"
    filepath = parquet_dir / filename

    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(df)} candles to {filepath}")

    return filepath


async def ingest_symbol_date_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime
):
    """
    Ingest candles for a symbol and date range.

    Args:
        symbol: Symbol to ingest
        start_date: Start date
        end_date: End date
    """
    logger.info(f"Ingesting {symbol} from {start_date.date()} to {end_date.date()}")

    # Initialize database
    db = DatabaseManager(str(settings.ml_data_db_path))

    # Fetch candles
    df = await fetch_candles_from_schwab(symbol, start_date, end_date)

    if df.empty:
        logger.warning(f"No candles fetched for {symbol}")
        return

    # Save to database
    inserted = await save_candles_to_db(db, symbol, df)
    logger.info(f"Inserted {inserted} candles to database")

    # Save to parquet (group by year)
    years = df['timestamp'].dt.year.unique()
    for year in years:
        year_df = df[df['timestamp'].dt.year == year]
        await save_candles_to_parquet(symbol, year_df, year)


async def backfill_recent_days(days: int):
    """
    Backfill recent N days for all symbols.

    Args:
        days: Number of days to backfill
    """
    symbols = ['SPX', 'VIX']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for symbol in symbols:
        await ingest_symbol_date_range(symbol, start_date, end_date)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest 1-minute historical candles for ML"
    )

    parser.add_argument(
        '--symbol',
        type=str,
        choices=['SPX', 'VIX', 'all'],
        help='Symbol to ingest'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Single date to ingest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for range (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for range (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--backfill-days',
        type=int,
        help='Backfill last N days for all symbols'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.info("Starting candle ingestion")

    if args.backfill_days:
        await backfill_recent_days(args.backfill_days)

    elif args.symbol and args.date:
        # Single date
        date = datetime.strptime(args.date, '%Y-%m-%d')
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)

        if args.symbol == 'all':
            for sym in ['SPX', 'VIX']:
                await ingest_symbol_date_range(sym, start, end)
        else:
            await ingest_symbol_date_range(args.symbol, start, end)

    elif args.symbol and args.start_date and args.end_date:
        # Date range
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')

        if args.symbol == 'all':
            for sym in ['SPX', 'VIX']:
                await ingest_symbol_date_range(sym, start, end)
        else:
            await ingest_symbol_date_range(args.symbol, start, end)

    else:
        logger.error("Invalid arguments. Use --help for usage.")
        sys.exit(1)

    logger.info("Candle ingestion complete")


if __name__ == '__main__':
    asyncio.run(main())
