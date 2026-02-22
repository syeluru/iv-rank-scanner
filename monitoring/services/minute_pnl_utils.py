"""
Utility functions for minute-level P&L tracking.

Provides helper functions for OCC symbol parsing, timezone handling,
and timestamp generation.
"""

import re
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import pytz


def parse_occ_symbol(symbol: str) -> Dict[str, any]:
    """
    Parse OCC option symbol into components.

    Format: ROOT YYMMDD C/P STRIKE
    Example: SPX   260207 P 05800000 -> root=SPX, exp=2026-02-07, right=P, strike=5800

    Args:
        symbol: OCC option symbol (e.g., "SPX   260207P05800000")

    Returns:
        Dict with keys: root, expiration, strike, right
    """
    # OCC format: 6 chars root + 6 chars date + 1 char C/P + 8 chars strike
    if len(symbol) < 21:
        raise ValueError(f"Invalid OCC symbol length: {symbol}")

    root = symbol[:6].strip()
    date_str = symbol[6:12]
    right = symbol[12]
    strike_str = symbol[13:21]

    # Parse date: YYMMDD
    year = int("20" + date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    expiration = datetime(year, month, day).date()

    # Parse strike: 8 digits, last 3 are decimals
    strike = int(strike_str) / 1000.0

    return {
        "root": root,
        "expiration": expiration,
        "strike": strike,
        "right": right
    }


def localize_eastern(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure datetime has Eastern timezone.

    DuckDB returns timezone-naive datetimes stored in UTC. This function:
    - If naive: treats as UTC and converts to ET
    - If aware: converts to ET
    - If None: returns None

    Args:
        dt: Datetime to localize (naive datetimes are assumed to be UTC)

    Returns:
        Timezone-aware datetime in Eastern time
    """
    if dt is None:
        return None

    eastern = pytz.timezone('US/Eastern')

    if dt.tzinfo is None:
        # DuckDB stores timestamps in UTC as naive datetimes
        # Localize as UTC first, then convert to Eastern
        utc_aware = pytz.utc.localize(dt)
        return utc_aware.astimezone(eastern)
    else:
        # Convert to ET if already has timezone
        return dt.astimezone(eastern)


def generate_minute_timestamps(
    entry_time: datetime,
    exit_time: datetime,
    market_hours_only: bool = True
) -> List[datetime]:
    """
    Generate 1-minute timestamps between entry and exit times.

    Args:
        entry_time: Start time (inclusive)
        exit_time: End time (inclusive)
        market_hours_only: If True, only include 9:30 AM - 4:00 PM ET

    Returns:
        List of datetime objects at 1-minute intervals
    """
    # Ensure both times are timezone-aware
    entry_time = localize_eastern(entry_time)
    exit_time = localize_eastern(exit_time)

    # Market hours in ET
    market_open = time(9, 30)
    market_close = time(16, 0)

    timestamps = []
    current = entry_time

    while current <= exit_time:
        if market_hours_only:
            current_time = current.time()
            if market_open <= current_time <= market_close:
                timestamps.append(current)
        else:
            timestamps.append(current)

        current += timedelta(minutes=1)

    return timestamps


def format_timestamp_for_id(dt: datetime) -> str:
    """
    Format datetime as string for use in IDs.

    Args:
        dt: Datetime to format

    Returns:
        String in format: YYYYMMDD_HHMMSS
    """
    return dt.strftime("%Y%m%d_%H%M%S")


def is_market_hours(dt: datetime) -> bool:
    """
    Check if datetime falls within market hours (9:30 AM - 4:00 PM ET).

    Args:
        dt: Datetime to check

    Returns:
        True if within market hours
    """
    dt = localize_eastern(dt)
    t = dt.time()
    return time(9, 30) <= t <= time(16, 0)


def round_to_nearest_minute(dt: datetime) -> datetime:
    """
    Round datetime to nearest minute.

    Args:
        dt: Datetime to round

    Returns:
        Datetime rounded to nearest minute
    """
    # Round seconds: 0-29 -> 0, 30-59 -> 60
    if dt.second >= 30:
        dt = dt + timedelta(minutes=1)

    return dt.replace(second=0, microsecond=0)
