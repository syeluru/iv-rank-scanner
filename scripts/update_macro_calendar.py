#!/usr/bin/env python3
"""
Update macro event calendar for ML features.

Maintains a calendar of major economic events:
- FOMC meetings
- CPI releases
- NFP (Non-farm payrolls)
- GDP releases

Usage:
    python scripts/update_macro_calendar.py --year 2025
    python scripts/update_macro_calendar.py --update-current
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from data.storage.database import DatabaseManager
from config.settings import settings


# Hardcoded event calendars
# TODO: Replace with API integration (e.g., https://www.federalreserve.gov/json/ne-calendar.json)

FOMC_MEETINGS_2025 = [
    ('2025-01-28', '2025-01-29'),  # January
    ('2025-03-18', '2025-03-19'),  # March
    ('2025-04-29', '2025-04-30'),  # April
    ('2025-06-17', '2025-06-18'),  # June
    ('2025-07-29', '2025-07-30'),  # July
    ('2025-09-16', '2025-09-17'),  # September
    ('2025-10-28', '2025-10-29'),  # October
    ('2025-12-09', '2025-12-10'),  # December
]

FOMC_MEETINGS_2026 = [
    ('2026-01-27', '2026-01-28'),
    ('2026-03-17', '2026-03-18'),
    ('2026-04-28', '2026-04-29'),
    ('2026-06-16', '2026-06-17'),
    ('2026-07-28', '2026-07-29'),
    ('2026-09-15', '2026-09-16'),
    ('2026-10-27', '2026-10-28'),
    ('2026-12-15', '2026-12-16'),
]


async def save_event(db: DatabaseManager, date: str, event_type: str, description: str):
    """
    Save a macro event to database.

    Args:
        db: DatabaseManager instance
        date: Event date (YYYY-MM-DD)
        event_type: Event type (FOMC, CPI, NFP, GDP)
        description: Event description
    """
    query = """
    INSERT OR REPLACE INTO macro_events
    (date, event_type, description)
    VALUES (?, ?, ?)
    """

    try:
        db.execute(query, (date, event_type, description))
    except Exception as e:
        logger.error(f"Failed to insert event: {e}")


async def update_fomc_calendar(db: DatabaseManager, year: int):
    """
    Update FOMC meeting calendar.

    Args:
        db: DatabaseManager instance
        year: Year to update
    """
    logger.info(f"Updating FOMC calendar for {year}")

    if year == 2025:
        meetings = FOMC_MEETINGS_2025
    elif year == 2026:
        meetings = FOMC_MEETINGS_2026
    else:
        logger.warning(f"No FOMC calendar data for {year}")
        return

    count = 0
    for start_date, end_date in meetings:
        # Add both days of the meeting
        await save_event(db, start_date, 'FOMC', f'FOMC Meeting Day 1')
        await save_event(db, end_date, 'FOMC', f'FOMC Meeting Day 2 (Decision)')
        count += 2

    logger.info(f"Inserted {count} FOMC events")


async def update_cpi_calendar(db: DatabaseManager, year: int):
    """
    Update CPI release calendar.

    CPI is typically released mid-month (around 13th-15th).
    TODO: Get actual dates from BLS or economic calendar API.

    Args:
        db: DatabaseManager instance
        year: Year to update
    """
    logger.warning(f"CPI calendar not implemented yet for {year}")
    # Placeholder - would need actual CPI release dates


async def update_nfp_calendar(db: DatabaseManager, year: int):
    """
    Update Non-Farm Payrolls calendar.

    NFP is released first Friday of each month.
    TODO: Calculate actual first Fridays or use API.

    Args:
        db: DatabaseManager instance
        year: Year to update
    """
    logger.warning(f"NFP calendar not implemented yet for {year}")
    # Placeholder - would need actual NFP release dates


async def update_all_events(year: int):
    """
    Update all macro events for a year.

    Args:
        year: Year to update
    """
    db = DatabaseManager(str(settings.ml_data_db_path))

    logger.info(f"Updating macro calendar for {year}")

    await update_fomc_calendar(db, year)
    await update_cpi_calendar(db, year)
    await update_nfp_calendar(db, year)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update macro event calendar"
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Year to update calendar for'
    )

    parser.add_argument(
        '--update-current',
        action='store_true',
        help='Update current and next year'
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

    logger.info("Starting macro calendar update")

    if args.update_current:
        current_year = datetime.now().year
        await update_all_events(current_year)
        await update_all_events(current_year + 1)
    elif args.year:
        await update_all_events(args.year)
    else:
        logger.error("Use --year YYYY or --update-current")
        sys.exit(1)

    logger.info("Macro calendar update complete")


if __name__ == '__main__':
    asyncio.run(main())
