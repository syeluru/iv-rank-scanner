#!/usr/bin/env python3
"""
Ingest macro economic data for ML features.

Fetches external data from FRED and other sources:
- HY OAS (High Yield Option-Adjusted Spread)
- 10Y-2Y yield curve spread
- Put/call ratio

Usage:
    python scripts/ingest_macro_data.py --update-latest
    python scripts/ingest_macro_data.py --backfill-days 365
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from data.storage.database import DatabaseManager
from config.settings import settings


async def fetch_fred_data(series_id: str, start_date: datetime, end_date: datetime):
    """
    Fetch data from FRED API.

    Args:
        series_id: FRED series ID (e.g., 'BAMLH0A0HYM2')
        start_date: Start date
        end_date: End date

    Returns:
        List of (date, value) tuples
    """
    if not settings.FRED_API_KEY:
        logger.warning("FRED_API_KEY not set in settings. Skipping FRED data.")
        return []

    try:
        from fredapi import Fred
        fred = Fred(api_key=settings.FRED_API_KEY)

        series = fred.get_series(
            series_id,
            observation_start=start_date.strftime('%Y-%m-%d'),
            observation_end=end_date.strftime('%Y-%m-%d')
        )

        # Convert to list of tuples
        data = [(date.to_pydatetime(), float(value)) for date, value in series.items()]
        return data

    except ImportError:
        logger.error("fredapi package not installed. Run: pip install fredapi")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch FRED series {series_id}: {e}")
        return []


async def save_macro_data(db: DatabaseManager, metric_name: str, data: list) -> int:
    """
    Save macro data to database.

    Args:
        db: DatabaseManager instance
        metric_name: Metric name
        data: List of (date, value) tuples

    Returns:
        Number of rows inserted
    """
    if not data:
        return 0

    query = """
    INSERT OR REPLACE INTO macro_data
    (date, metric_name, value)
    VALUES (?, ?, ?)
    """

    inserted = 0
    for date, value in data:
        try:
            db.execute(query, (date.date(), metric_name, value))
            inserted += 1
        except Exception as e:
            logger.error(f"Failed to insert macro data: {e}")

    return inserted


async def ingest_hy_oas(db: DatabaseManager, start_date: datetime, end_date: datetime):
    """
    Ingest High Yield Option-Adjusted Spread.

    FRED series: BAMLH0A0HYM2
    """
    logger.info("Fetching HY OAS from FRED...")
    data = await fetch_fred_data('BAMLH0A0HYM2', start_date, end_date)

    inserted = await save_macro_data(db, 'hy_oas', data)
    logger.info(f"Inserted {inserted} HY OAS data points")


async def ingest_yield_curve(db: DatabaseManager, start_date: datetime, end_date: datetime):
    """
    Ingest 10Y-2Y yield curve spread.

    FRED series: DGS10 (10-year), DGS2 (2-year)
    """
    logger.info("Fetching 10Y-2Y yield curve from FRED...")

    # Fetch both series
    dgs10_data = await fetch_fred_data('DGS10', start_date, end_date)
    dgs2_data = await fetch_fred_data('DGS2', start_date, end_date)

    # Convert to dicts for easier lookup
    dgs10_dict = {date: value for date, value in dgs10_data}
    dgs2_dict = {date: value for date, value in dgs2_data}

    # Calculate spread for dates where both exist
    spread_data = []
    for date in dgs10_dict:
        if date in dgs2_dict:
            spread = dgs10_dict[date] - dgs2_dict[date]
            spread_data.append((date, spread))

    inserted = await save_macro_data(db, 'yield_curve_10y_2y', spread_data)
    logger.info(f"Inserted {inserted} yield curve data points")


async def ingest_put_call_ratio(db: DatabaseManager, start_date: datetime, end_date: datetime):
    """
    Ingest equity put/call ratio.

    TODO: Implement CBOE data source or alternative
    """
    logger.warning("Put/call ratio ingestion not implemented yet")


async def update_all_metrics(days: int):
    """
    Update all macro metrics for recent N days.

    Args:
        days: Number of days to backfill
    """
    db = DatabaseManager(str(settings.ml_data_db_path))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Updating macro data from {start_date.date()} to {end_date.date()}")

    # Ingest each metric
    await ingest_hy_oas(db, start_date, end_date)
    await ingest_yield_curve(db, start_date, end_date)
    await ingest_put_call_ratio(db, start_date, end_date)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest macro economic data for ML"
    )

    parser.add_argument(
        '--update-latest',
        action='store_true',
        help='Update latest data (last 7 days)'
    )

    parser.add_argument(
        '--backfill-days',
        type=int,
        help='Backfill last N days'
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

    logger.info("Starting macro data ingestion")

    if args.update_latest:
        await update_all_metrics(days=7)
    elif args.backfill_days:
        await update_all_metrics(days=args.backfill_days)
    else:
        logger.error("Use --update-latest or --backfill-days N")
        sys.exit(1)

    logger.info("Macro data ingestion complete")


if __name__ == '__main__':
    asyncio.run(main())
