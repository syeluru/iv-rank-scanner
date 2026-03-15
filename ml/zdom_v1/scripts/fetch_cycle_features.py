"""
Fetch presidential cycle and business cycle features.

Calculates/fetches:
  - Presidential cycle year (1-4) and month (0-47)
    Trump 2nd term: inaugurated Jan 20, 2025 → Year 2 starts Jan 20, 2026
  - Pre-election year flag (Year 3 — historically strongest)
  - Election year flag (Year 4)
  - ISM Manufacturing PMI from FRED (NAPM series)
  - PMI expansion/contraction flag (>50 = expansion)
  - PMI 3-month change + crossing-50 signal
  - Conference Board LEI (USSLIND — Leading Index for US)
  - LEI direction (rising = expansion, falling = contraction)
  - Business cycle phase estimate (expansion / slowdown / contraction / recovery)
  - NBER recession indicator (USREC — 0/1 monthly)
  - Week of year, day of month, days into quarter (calendar features)
  - OpEx week, triple witching, special calendar flags

Saves to: data/presidential_cycles.parquet

Usage:
  python3 scripts/fetch_cycle_features.py
  python3 scripts/fetch_cycle_features.py --start 2018-01-01
  python3 scripts/fetch_cycle_features.py --no-cache
"""

import argparse
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# Load .env
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env.development"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE   = DATA_DIR / "presidential_cycles.parquet"
FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "2018-01-01"

# ── Presidential term history ──────────────────────────────────────────────────
# (inauguration_date, president, party, term_number)
PRESIDENTIAL_TERMS = [
    (datetime(1993, 1, 20), "Clinton",  "D", 1),
    (datetime(1997, 1, 20), "Clinton",  "D", 2),
    (datetime(2001, 1, 20), "Bush",     "R", 1),
    (datetime(2005, 1, 20), "Bush",     "R", 2),
    (datetime(2009, 1, 20), "Obama",    "D", 1),
    (datetime(2013, 1, 20), "Obama",    "D", 2),
    (datetime(2017, 1, 20), "Trump",    "R", 1),
    (datetime(2021, 1, 20), "Biden",    "D", 1),
    (datetime(2025, 1, 20), "Trump",    "R", 2),  # Current — Year 2 as of Jan 2026
]

# ── Options expiration calendar helpers ───────────────────────────────────────
# Standard monthly OpEx = 3rd Friday of each month

def get_third_friday(year: int, month: int) -> datetime:
    """Return 3rd Friday of given month/year."""
    first_day = datetime(year, month, 1)
    # Find first Friday
    days_to_friday = (4 - first_day.weekday()) % 7  # 4 = Friday
    first_friday = first_day + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def build_opex_flags(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build OpEx and triple witching flags for each date in the index.
    Triple witching = OpEx in March, June, September, December.
    """
    years  = sorted(set(date_index.year))
    months = range(1, 13)

    opex_dates   = set()
    triple_dates = set()

    for y in years:
        for m in months:
            third_fri = get_third_friday(y, m)
            opex_dates.add(pd.Timestamp(third_fri).normalize())
            if m in (3, 6, 9, 12):
                triple_dates.add(pd.Timestamp(third_fri).normalize())

    flags = pd.DataFrame(index=date_index)
    flags["opex_week"]         = flags.index.normalize().map(
        lambda d: int(any(abs((d - o).days) <= 4 for o in opex_dates))
    )
    flags["triple_witching"]   = flags.index.normalize().map(
        lambda d: int(d in triple_dates)
    )
    flags["opex_day"]          = flags.index.normalize().map(
        lambda d: int(d in opex_dates)
    )
    return flags


# ── Presidential cycle calculations ───────────────────────────────────────────

def presidential_cycle_for_date(dt: datetime) -> dict:
    """
    Return presidential cycle metadata for a given date.

    Returns:
      cycle_year: 1-4 (year within 4-year presidential term)
      cycle_month: 0-47 (month within term, 0 = inauguration month)
      president: name
      party: D or R
      term_num: 1 or 2
      pre_election_year: bool (cycle_year == 3)
      election_year: bool (cycle_year == 4)
    """
    dt = pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None)

    # Find the current term
    current_term = None
    for i, (inag, president, party, term_num) in enumerate(PRESIDENTIAL_TERMS):
        next_inag = (PRESIDENTIAL_TERMS[i + 1][0]
                     if i + 1 < len(PRESIDENTIAL_TERMS)
                     else datetime(inag.year + 4, inag.month, inag.day))
        if inag <= dt < next_inag:
            current_term = (inag, president, party, term_num)
            break

    if current_term is None:
        # Before first tracked term or future — use most recent
        current_term = PRESIDENTIAL_TERMS[-1]

    inag, president, party, term_num = current_term

    # Months since inauguration
    months_since = (dt.year - inag.year) * 12 + (dt.month - inag.month)
    months_since = max(0, months_since)

    cycle_year  = (months_since // 12) + 1
    cycle_month = months_since % 48  # cap at 47

    return {
        "president":          president,
        "party":              party,
        "term_num":           term_num,
        "cycle_year":         min(cycle_year, 4),
        "cycle_month":        min(cycle_month, 47),
        "pre_election_year":  int(cycle_year == 3),
        "election_year":      int(cycle_year == 4),
        "midterm_year":       int(cycle_year == 2),
        "first_year":         int(cycle_year == 1),
    }


# ── FRED fetch ─────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str, end: str,
                      api_key: str, frequency: str = "m") -> pd.Series:
    """Fetch a FRED series at given frequency, return daily-indexed Series."""
    params = {
        "series_id":    series_id,
        "observation_start": start,
        "observation_end":   end,
        "file_type":    "json",
    }
    if api_key:
        params["api_key"] = api_key
    if frequency != "default":
        params["frequency"] = frequency
        params["aggregation_method"] = "avg"

    for attempt in range(3):
        try:
            resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "observations" not in data:
                return pd.Series(dtype=float)

            rows = [
                (obs["date"], float(obs["value"]))
                for obs in data["observations"]
                if obs["value"] not in (".", "")
            ]
            if not rows:
                return pd.Series(dtype=float)

            dates, values = zip(*rows)
            return pd.Series(values, index=pd.to_datetime(dates), name=series_id, dtype=float)

        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", 0)
            if status == 400:
                print(f"  [warn] {series_id}: HTTP 400 — skipping")
                return pd.Series(dtype=float)
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] {series_id}: failed after 3 attempts: {e}")
                return pd.Series(dtype=float)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] {series_id}: {e}")
                return pd.Series(dtype=float)


# ── PMI derived features ───────────────────────────────────────────────────────

def add_pmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISM PMI derived signals."""
    if "ism_pmi" not in df.columns:
        return df

    pmi = df["ism_pmi"]

    df["pmi_expanding"]      = (pmi >= 50).astype(int)
    df["pmi_contracting"]    = (pmi < 50).astype(int)
    df["pmi_chg_1m"]         = pmi.diff(21)   # approx 1 month lag
    df["pmi_chg_3m"]         = pmi.diff(63)   # approx 3 months
    df["pmi_above_52"]       = (pmi >= 52).astype(int)   # solid expansion
    df["pmi_below_48"]       = (pmi <= 48).astype(int)   # solid contraction

    # PMI crossing 50 signal (+1 = crossed above, -1 = crossed below, 0 = no cross)
    pmi_prev = pmi.shift(1)
    df["pmi_cross_50"] = np.where(
        (pmi >= 50) & (pmi_prev < 50),  +1,
        np.where((pmi < 50) & (pmi_prev >= 50), -1, 0)
    )

    return df


def add_lei_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Conference Board LEI signals."""
    if "lei" not in df.columns:
        return df

    lei = df["lei"]

    df["lei_chg_1m"]         = lei.diff(21)
    df["lei_chg_3m"]         = lei.diff(63)
    df["lei_rising"]         = (df["lei_chg_1m"] > 0).astype(int)
    df["lei_falling"]        = (df["lei_chg_1m"] < 0).astype(int)

    # LEI trend: 3-month direction
    df["lei_3m_direction"] = np.where(
        df["lei_chg_3m"] > 0, 1,
        np.where(df["lei_chg_3m"] < 0, -1, 0)
    )

    return df


def add_business_cycle_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate business cycle phase from PMI + LEI signals.

    Phase encoding:
      3 = Expansion (PMI > 52, LEI rising)
      2 = Slowdown  (PMI 48-52 or PMI falling)
      1 = Contraction (PMI < 48, LEI falling)
      0 = Recovery  (PMI rising from below 50, LEI turning up)
    """
    if "ism_pmi" not in df.columns:
        return df

    pmi = df["ism_pmi"]
    pmi_chg = df.get("pmi_chg_3m", pmi.diff(63))
    lei_chg = df.get("lei_chg_3m", pd.Series(0, index=df.index))

    phase = np.where(
        (pmi >= 52) & (lei_chg >= 0),   3,   # Expansion
        np.where(
            (pmi >= 48) & (pmi < 52),   2,   # Slowdown
            np.where(
                (pmi < 48) & (pmi_chg > 0) & (lei_chg >= 0),   0,   # Recovery
                1                                                      # Contraction
            )
        )
    )
    df["business_cycle_phase"] = phase
    # Phase labels for logging (not a model feature — too many cardinality issues)
    # df["business_cycle_label"] = pd.Categorical.from_codes(
    #     phase, ["Contraction", "Recovery", "Slowdown", "Expansion"])

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar cycle features useful for trading patterns."""
    dates = pd.to_datetime(df["date"])

    df["week_of_year"]       = dates.dt.isocalendar().week.astype(int)
    df["day_of_month"]       = dates.dt.day
    df["month"]              = dates.dt.month
    df["quarter"]            = dates.dt.quarter

    # Days since quarter start
    q_start = dates.apply(lambda d: pd.Timestamp(d.year, (d.quarter - 1) * 3 + 1, 1))
    df["days_into_quarter"]  = (dates - q_start).dt.days

    # Seasonal flags
    df["sell_in_may"]        = dates.dt.month.between(5, 10).astype(int)  # May–Oct
    df["q4_rally"]           = dates.dt.month.between(10, 12).astype(int)  # Oct–Dec
    df["january_effect"]     = (dates.dt.month == 1).astype(int)

    # Month-end proximity
    month_end = dates + pd.tseries.offsets.MonthEnd(0)
    df["days_to_month_end"]  = (month_end - dates).dt.days
    df["near_month_end"]     = (df["days_to_month_end"] <= 3).astype(int)

    # Quarter-end proximity
    quarter_end = dates + pd.tseries.offsets.QuarterEnd(0)
    df["days_to_qtr_end"]    = (quarter_end - dates).dt.days
    df["near_qtr_end"]       = (df["days_to_qtr_end"] <= 5).astype(int)

    # Thanksgiving week (4th week of November, US market tends to drift)
    def is_thanksgiving_week(d):
        if d.month != 11:
            return 0
        import calendar
        cal = calendar.monthcalendar(d.year, 11)
        # 4th Thursday: find weeks with a Thursday (index 3), pick the 4th one
        thursdays = [week[3] for week in cal if week[3] != 0]
        if len(thursdays) < 4:
            return 0
        thanksgiving_day = thursdays[3]
        thanksgiving = d.replace(month=11, day=thanksgiving_day)
        return int(abs((d - thanksgiving).days) <= 4)

    df["thanksgiving_week"]  = dates.apply(is_thanksgiving_week)

    # Christmas week (Dec 24-31)
    df["christmas_week"]     = (
        (dates.dt.month == 12) & (dates.dt.day >= 24)
    ).astype(int)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch presidential cycle + business cycle features"
    )
    parser.add_argument("--start",    default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force full refresh")
    args = parser.parse_args()

    end_date = str(date.today())

    if args.no_cache or not OUTPUT_FILE.exists():
        start_date  = args.start or DEFAULT_START
        existing_df = None
        print(f"Full fetch from {start_date} → {end_date}")
    else:
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        last_date   = existing_df["date"].max()
        start_date  = args.start or str((last_date - timedelta(days=35)).date())
        print(f"Incremental fetch from {start_date} → {end_date}")
        print(f"  Existing: {len(existing_df)} rows through {last_date.date()}")

    if not FRED_API_KEY:
        print("\n[warn] FRED_API_KEY not set in .env.development")
        print("  PMI and LEI data requires a free FRED API key.")
        print("  Get one at: https://fred.stlouisfed.org/docs/api/api_key.html\n")

    # ── Build business day date index ──
    date_idx = pd.bdate_range(start=start_date, end=end_date)
    df = pd.DataFrame({"date": date_idx})
    df["date"] = pd.to_datetime(df["date"])

    # ── Presidential cycle features (pure calculation) ──
    print("Building presidential cycle features...")
    cycle_rows = [presidential_cycle_for_date(d) for d in df["date"]]
    cycle_df   = pd.DataFrame(cycle_rows, index=df.index)
    for col in cycle_df.columns:
        if col not in ("president", "party"):  # skip string cols for now
            df[col] = cycle_df[col]
    # Keep president/party as categorical
    df["president"] = cycle_df["president"].astype("category")
    df["party_dem"] = (cycle_df["party"] == "D").astype(int)

    # Print cycle summary
    latest = df.tail(1).iloc[0]
    print(f"  Current cycle: Year {latest['cycle_year']}, Month {latest['cycle_month']}")
    print(f"  President: {cycle_df.tail(1).iloc[0]['president']} "
          f"(term {latest['term_num']})")

    # ── FRED: ISM Manufacturing PMI ──
    FRED_CYCLE_SERIES = {
        "ism_pmi":   ("NAPM",    "m"),   # ISM Manufacturing PMI (monthly)
        "lei":       ("USSLIND", "m"),   # Conference Board LEI (monthly)
        "recession": ("USREC",   "m"),   # NBER recession indicator (0/1)
    }

    for col_name, (series_id, freq) in FRED_CYCLE_SERIES.items():
        print(f"  Fetching {series_id:15s} ({col_name})...")
        s = fetch_fred_series(series_id, start_date, end_date, FRED_API_KEY, frequency=freq)
        if not s.empty:
            s.index = pd.to_datetime(s.index).normalize()
            s = s[~s.index.duplicated(keep="last")]
            # Reindex to business days and forward-fill (monthly data → daily)
            aligned = s.reindex(date_idx)
            aligned = aligned.fillna(method="ffill", limit=31)
            df[col_name] = aligned.values
            print(f"    Got {s.notna().sum()} monthly observations → forward-filled daily")
        else:
            print(f"    [empty — FRED_API_KEY may be needed]")
            df[col_name] = np.nan

    # ── Derived features ──
    print("\nBuilding derived features...")
    df = df.set_index("date")
    df = add_pmi_features(df)
    df = add_lei_features(df)
    df = add_business_cycle_phase(df)
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})

    # Calendar features
    df = add_calendar_features(df)

    # OpEx / triple witching flags
    print("Building options expiration calendar flags...")
    opex_flags = build_opex_flags(pd.DatetimeIndex(df["date"]))
    opex_flags = opex_flags.reset_index(drop=True)
    for col in opex_flags.columns:
        df[col] = opex_flags[col].values

    # ── Merge with existing ──
    if existing_df is not None and not args.no_cache:
        existing_df = existing_df[existing_df["date"] < df["date"].min()]
        # Align columns
        all_cols = sorted(set(existing_df.columns) | set(df.columns))
        for c in all_cols:
            if c not in existing_df.columns:
                existing_df[c] = np.nan
            if c not in df.columns:
                df[c] = np.nan
        # Drop category columns in existing (they cause concat issues)
        for c in existing_df.select_dtypes(include="category").columns:
            existing_df[c] = existing_df[c].astype(str)
        for c in df.select_dtypes(include="category").columns:
            df[c] = df[c].astype(str)
        df_final = pd.concat([existing_df[all_cols], df[all_cols]], ignore_index=True)
    else:
        # Convert category to string for parquet
        for c in df.select_dtypes(include="category").columns:
            df[c] = df[c].astype(str)
        df_final = df

    df_final = (df_final
                .sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True))

    # ── Save ──
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_final)} rows to {OUTPUT_FILE}")
    print(f"Date range: {df_final['date'].min().date()} → {df_final['date'].max().date()}")
    print(f"Features: {len(df_final.columns) - 1}")

    # Latest snapshot
    print(f"\nLatest cycle snapshot ({df_final['date'].max().date()}):")
    latest = df_final.tail(1).iloc[0]
    print_fields = [
        ("cycle_year",          "Cycle Year"),
        ("cycle_month",         "Cycle Month"),
        ("pre_election_year",   "Pre-Election Year"),
        ("midterm_year",        "Midterm Year"),
        ("ism_pmi",             "ISM PMI"),
        ("pmi_expanding",       "PMI Expanding"),
        ("lei",                 "LEI"),
        ("recession",           "NBER Recession"),
        ("business_cycle_phase","Business Cycle Phase"),
        ("week_of_year",        "Week of Year"),
        ("sell_in_may",         "Sell in May Period"),
    ]
    for col, label in print_fields:
        if col in latest and not pd.isna(latest[col]):
            print(f"  {label:25s}: {latest[col]}")


if __name__ == "__main__":
    main()
