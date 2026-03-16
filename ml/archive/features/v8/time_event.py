"""
Time, calendar, and event features for 0DTE iron condor models (v8).

All functions are pure numpy/pandas. Returns np.nan when data is unavailable.
Entry time must be timezone-aware (US/Eastern).
"""

import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# FOMC announcement dates (Federal Reserve published schedule)
# ---------------------------------------------------------------------------
FOMC_DATES = {
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}

# ---------------------------------------------------------------------------
# CPI release dates (Bureau of Labor Statistics published schedule)
# ---------------------------------------------------------------------------
CPI_DATES = {
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-16", "2026-10-13", "2026-11-12", "2026-12-09",
}


def _third_friday(year: int, month: int) -> datetime:
    """Return the third Friday of the given month."""
    # First day of month
    first = datetime(year, month, 1)
    # Day of week: Monday=0, Friday=4
    dow = first.weekday()
    # First Friday
    first_friday = 1 + (4 - dow) % 7
    # Third Friday
    third_friday_day = first_friday + 14
    return datetime(year, month, third_friday_day)


def _first_friday(year: int, month: int) -> datetime:
    """Return the first Friday of the given month (NFP release day)."""
    first = datetime(year, month, 1)
    dow = first.weekday()
    first_friday_day = 1 + (4 - dow) % 7
    return datetime(year, month, first_friday_day)


def _is_nfp_day(dt: datetime) -> bool:
    """NFP is typically released on the first Friday of each month."""
    ff = _first_friday(dt.year, dt.month)
    return dt.date() == ff.date()


def _week_bounds(dt: datetime) -> tuple:
    """Return (monday, friday) dates for the week containing dt."""
    monday = dt.date() - timedelta(days=dt.weekday())
    friday = monday + timedelta(days=4)
    return monday, friday


def _is_opex_week(dt: datetime) -> bool:
    """Check if the 3rd Friday of the month falls within this week."""
    tf = _third_friday(dt.year, dt.month)
    monday, friday = _week_bounds(dt)
    return monday <= tf.date() <= friday


def _trading_days_to_opex(dt: datetime) -> int:
    """
    Count trading days (weekdays) until the next 3rd Friday.

    If today is past this month's opex, look at next month.
    """
    tf = _third_friday(dt.year, dt.month)
    if dt.date() > tf.date():
        # Next month
        if dt.month == 12:
            tf = _third_friday(dt.year + 1, 1)
        else:
            tf = _third_friday(dt.year, dt.month + 1)

    count = 0
    current = dt.date() + timedelta(days=1)
    target = tf.date()
    while current <= target:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def _is_quarter_end(dt: datetime) -> bool:
    """True if within last 3 trading days of a quarter."""
    year = dt.year
    month = dt.month
    # Quarter-end months: 3, 6, 9, 12
    if month not in (3, 6, 9, 12):
        return False

    # Find last trading day of month
    if month == 12:
        next_month_1st = datetime(year + 1, 1, 1)
    else:
        next_month_1st = datetime(year, month + 1, 1)

    last_day = next_month_1st - timedelta(days=1)
    # Walk back to find last 3 trading days
    trading_days = []
    d = last_day.date()
    while len(trading_days) < 3:
        if d.weekday() < 5:
            trading_days.append(d)
        d -= timedelta(days=1)

    return dt.date() in trading_days


def _is_month_end(dt: datetime) -> bool:
    """True if within last 3 trading days of the month."""
    year = dt.year
    month = dt.month

    if month == 12:
        next_month_1st = datetime(year + 1, 1, 1)
    else:
        next_month_1st = datetime(year, month + 1, 1)

    last_day = next_month_1st - timedelta(days=1)
    trading_days = []
    d = last_day.date()
    while len(trading_days) < 3:
        if d.weekday() < 5:
            trading_days.append(d)
        d -= timedelta(days=1)

    return dt.date() in trading_days


def _hours_to_next_macro(dt: datetime) -> float:
    """
    Hours until the next FOMC/CPI/NFP event.

    Looks ahead up to 60 days. Returns np.nan if none found.
    """
    today = dt.date()

    # Collect all upcoming macro dates
    candidates = []

    for date_str in FOMC_DATES:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if d >= today:
            candidates.append(d)

    for date_str in CPI_DATES:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if d >= today:
            candidates.append(d)

    # NFP: first Friday of upcoming months
    for m_offset in range(3):
        year = dt.year
        month = dt.month + m_offset
        if month > 12:
            year += 1
            month -= 12
        nfp = _first_friday(year, month).date()
        if nfp >= today:
            candidates.append(nfp)

    if not candidates:
        return np.nan

    next_event = min(candidates)

    # Assume events at 14:00 ET for FOMC, 8:30 ET for CPI/NFP
    date_str = next_event.strftime("%Y-%m-%d")
    if date_str in FOMC_DATES:
        event_hour = 14.0
    else:
        event_hour = 8.5  # 8:30 AM

    # Compute hours from current time
    event_dt = datetime(next_event.year, next_event.month, next_event.day)
    if dt.tzinfo is not None:
        import pytz
        eastern = pytz.timezone("US/Eastern")
        event_dt = eastern.localize(
            event_dt.replace(hour=int(event_hour), minute=int((event_hour % 1) * 60))
        )
        delta = (event_dt - dt).total_seconds() / 3600.0
    else:
        event_dt = event_dt.replace(
            hour=int(event_hour), minute=int((event_hour % 1) * 60)
        )
        delta = (event_dt - dt).total_seconds() / 3600.0

    return float(max(delta, 0.0))


def _is_fomc_week(dt: datetime) -> bool:
    """Check if any FOMC date falls within this week."""
    monday, friday = _week_bounds(dt)
    for date_str in FOMC_DATES:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if monday <= d <= friday:
            return True
    return False


def compute_time_event_features(
    entry_time: datetime, prior_ic_results: list = None
) -> dict:
    """
    Compute ~25 time, calendar, and event features.

    Args:
        entry_time: Current timestamp (ET timezone-aware).
        prior_ic_results: Optional list of recent IC P&L results (most recent first).

    Returns:
        dict of feature_name -> float
    """
    if entry_time is None:
        return {k: np.nan for k in [
            "minutes_since_open", "minutes_to_close", "time_sin", "time_cos",
            "session_phase", "theta_acceleration_zone", "gamma_intensification_zone",
            "day_of_week", "is_monday", "is_friday", "is_opex_week", "days_to_opex",
            "is_quarter_end", "is_month_end", "is_fomc_day", "is_fomc_week",
            "is_cpi_day", "is_nfp_day", "hours_to_next_macro", "post_event_30min",
            "prior_day_ic_result", "ic_5day_win_rate", "ic_streak",
            "day_of_month", "week_of_year",
        ]}

    if prior_ic_results is None:
        prior_ic_results = []

    result = {}
    dt = entry_time

    # Market open/close in minutes since midnight
    market_open_min = 9 * 60 + 30   # 9:30 ET
    market_close_min = 16 * 60       # 16:00 ET
    current_min = dt.hour * 60 + dt.minute

    # --- 1. minutes_since_open ---
    minutes_since_open = current_min - market_open_min
    result["minutes_since_open"] = float(max(minutes_since_open, 0))

    # --- 2. minutes_to_close ---
    minutes_to_close = market_close_min - current_min
    result["minutes_to_close"] = float(max(minutes_to_close, 0))

    # --- 3-4. time cyclical encoding ---
    total_session = 390.0  # 6.5 hours in minutes
    frac = max(minutes_since_open, 0) / total_session
    result["time_sin"] = float(np.sin(2 * np.pi * frac))
    result["time_cos"] = float(np.cos(2 * np.pi * frac))

    # --- 5. session_phase ---
    if current_min < market_open_min + 30:
        phase = 0  # open: 9:30-10:00
    elif current_min < market_open_min + 120:
        phase = 1  # morning: 10:00-11:30
    elif current_min < market_open_min + 240:
        phase = 2  # midday: 11:30-13:30
    elif current_min < market_open_min + 330:
        phase = 3  # afternoon: 13:30-15:00
    else:
        phase = 4  # close: 15:00-16:00
    result["session_phase"] = float(phase)

    # --- 6. theta_acceleration_zone ---
    result["theta_acceleration_zone"] = 1.0 if minutes_to_close <= 120 else 0.0

    # --- 7. gamma_intensification_zone ---
    result["gamma_intensification_zone"] = 1.0 if minutes_to_close <= 60 else 0.0

    # --- 8-10. day of week ---
    dow = dt.weekday()  # 0=Mon, 4=Fri
    result["day_of_week"] = float(dow)
    result["is_monday"] = 1.0 if dow == 0 else 0.0
    result["is_friday"] = 1.0 if dow == 4 else 0.0

    # --- 11. is_opex_week ---
    result["is_opex_week"] = 1.0 if _is_opex_week(dt) else 0.0

    # --- 12. days_to_opex ---
    result["days_to_opex"] = float(_trading_days_to_opex(dt))

    # --- 13. is_quarter_end ---
    result["is_quarter_end"] = 1.0 if _is_quarter_end(dt) else 0.0

    # --- 14. is_month_end ---
    result["is_month_end"] = 1.0 if _is_month_end(dt) else 0.0

    # --- 15. is_fomc_day ---
    date_str = dt.strftime("%Y-%m-%d")
    result["is_fomc_day"] = 1.0 if date_str in FOMC_DATES else 0.0

    # --- 16. is_fomc_week ---
    result["is_fomc_week"] = 1.0 if _is_fomc_week(dt) else 0.0

    # --- 17. is_cpi_day ---
    result["is_cpi_day"] = 1.0 if date_str in CPI_DATES else 0.0

    # --- 18. is_nfp_day ---
    result["is_nfp_day"] = 1.0 if _is_nfp_day(dt) else 0.0

    # --- 19. hours_to_next_macro ---
    result["hours_to_next_macro"] = _hours_to_next_macro(dt)

    # --- 20. post_event_30min ---
    # Within 30 min after a morning macro release at 8:30 ET
    is_macro_morning = date_str in CPI_DATES or _is_nfp_day(dt)
    macro_release_min = 8 * 60 + 30
    result["post_event_30min"] = (
        1.0 if is_macro_morning and macro_release_min <= current_min <= macro_release_min + 30
        else 0.0
    )

    # --- 21. prior_day_ic_result ---
    result["prior_day_ic_result"] = (
        float(prior_ic_results[0]) if prior_ic_results else 0.0
    )

    # --- 22. ic_5day_win_rate ---
    if prior_ic_results and len(prior_ic_results) >= 1:
        recent = prior_ic_results[:5]
        wins = sum(1 for r in recent if r > 0)
        result["ic_5day_win_rate"] = float(wins / len(recent))
    else:
        result["ic_5day_win_rate"] = np.nan

    # --- 23. ic_streak ---
    if prior_ic_results:
        streak = 0
        direction = None
        for r in prior_ic_results:
            if direction is None:
                direction = 1 if r > 0 else -1
                streak = direction
            elif (r > 0 and direction > 0) or (r <= 0 and direction < 0):
                streak += direction
            else:
                break
        result["ic_streak"] = float(streak)
    else:
        result["ic_streak"] = 0.0

    # --- 24. day_of_month ---
    result["day_of_month"] = float(dt.day)

    # --- 25. week_of_year ---
    result["week_of_year"] = float(dt.isocalendar()[1])

    # --- 26. theta_decay_proxy: 1/sqrt(minutes_to_close) ---
    if minutes_to_close > 0:
        result["theta_decay_proxy"] = float(1.0 / np.sqrt(minutes_to_close))
    else:
        result["theta_decay_proxy"] = float(1.0 / np.sqrt(1))  # cap at 1 min

    # --- 27. session_fraction_squared ---
    session_frac = max(minutes_since_open, 0) / 390.0
    result["session_fraction_squared"] = float(session_frac ** 2)

    # --- 28-30. Gaussian proximity kernels for FOMC/CPI/NFP ---
    # exp(-0.5*(hours/24)^2) — peaks at event day, decays over days
    hours_to_fomc = _hours_to_specific_event(dt, FOMC_DATES)
    hours_to_cpi = _hours_to_specific_event(dt, CPI_DATES)
    hours_to_nfp = _hours_to_nfp_event(dt)

    result["fomc_proximity_kernel"] = float(
        np.exp(-0.5 * (hours_to_fomc / 24.0) ** 2)
    ) if np.isfinite(hours_to_fomc) else 0.0

    result["cpi_proximity_kernel"] = float(
        np.exp(-0.5 * (hours_to_cpi / 24.0) ** 2)
    ) if np.isfinite(hours_to_cpi) else 0.0

    result["nfp_proximity_kernel"] = float(
        np.exp(-0.5 * (hours_to_nfp / 24.0) ** 2)
    ) if np.isfinite(hours_to_nfp) else 0.0

    # --- 31-33. Post-event hours (hours since last FOMC/CPI/NFP) ---
    result["post_fomc_hours"] = _hours_since_event(dt, FOMC_DATES)
    result["post_cpi_hours"] = _hours_since_event(dt, CPI_DATES)
    result["post_nfp_hours"] = _hours_since_nfp(dt)

    # --- 34. Pre-event compression flag ---
    # Within 24h before any macro event
    min_hours_ahead = min(
        h for h in [hours_to_fomc, hours_to_cpi, hours_to_nfp]
        if np.isfinite(h)
    ) if any(np.isfinite(h) for h in [hours_to_fomc, hours_to_cpi, hours_to_nfp]) else np.nan
    result["pre_event_compression"] = 1.0 if (np.isfinite(min_hours_ahead) and 0 < min_hours_ahead <= 24) else 0.0

    # --- 35. Event cluster density (number of macro events within 5 trading days) ---
    cluster_count = 0
    today = dt.date()
    for offset in range(-5, 6):
        check_date = today + timedelta(days=offset)
        date_str_check = check_date.strftime("%Y-%m-%d")
        if date_str_check in FOMC_DATES or date_str_check in CPI_DATES or _is_nfp_day(
            datetime(check_date.year, check_date.month, check_date.day)
        ):
            cluster_count += 1
    result["event_cluster_density"] = float(cluster_count)

    return result


def _hours_to_specific_event(dt: datetime, event_dates: set) -> float:
    """Hours until next event from the given set."""
    today = dt.date()
    candidates = []
    for ds in event_dates:
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        if d >= today:
            candidates.append(d)
    if not candidates:
        return np.nan
    next_event = min(candidates)
    # FOMC at 14:00, CPI/NFP at 8:30
    next_str = next_event.strftime("%Y-%m-%d")
    event_hour = 14 if next_str in FOMC_DATES else 8
    event_minute = 0 if next_str in FOMC_DATES else 30
    event_dt = datetime(next_event.year, next_event.month, next_event.day, event_hour, event_minute)
    naive_dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
    delta = (event_dt - naive_dt).total_seconds() / 3600.0
    return float(max(delta, 0.0))


def _hours_to_nfp_event(dt: datetime) -> float:
    """Hours until next NFP (first Friday of month)."""
    today = dt.date()
    for m_offset in range(3):
        year = dt.year
        month = dt.month + m_offset
        if month > 12:
            year += 1
            month -= 12
        nfp = _first_friday(year, month).date()
        if nfp >= today:
            event_dt = datetime(nfp.year, nfp.month, nfp.day, 8, 30)
            naive_dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
            delta = (event_dt - naive_dt).total_seconds() / 3600.0
            return float(max(delta, 0.0))
    return np.nan


def _hours_since_event(dt: datetime, event_dates: set) -> float:
    """Hours since most recent past event."""
    today = dt.date()
    candidates = []
    for date_str in event_dates:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if d <= today:
            candidates.append(d)
    if not candidates:
        return np.nan
    last_event = max(candidates)
    event_dt = datetime(last_event.year, last_event.month, last_event.day, 14, 0)
    naive_dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
    delta = (naive_dt - event_dt).total_seconds() / 3600.0
    return float(max(delta, 0.0))


def _hours_since_nfp(dt: datetime) -> float:
    """Hours since most recent NFP."""
    today = dt.date()
    for m_offset in range(3):
        year = dt.year
        month = dt.month - m_offset
        if month < 1:
            year -= 1
            month += 12
        nfp = _first_friday(year, month).date()
        if nfp <= today:
            event_dt = datetime(nfp.year, nfp.month, nfp.day, 8, 30)
            naive_dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
            delta = (naive_dt - event_dt).total_seconds() / 3600.0
            return float(max(delta, 0.0))
    return np.nan
