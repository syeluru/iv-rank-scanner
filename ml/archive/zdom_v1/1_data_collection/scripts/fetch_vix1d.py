"""
Fetch VIX1D (CBOE 1-Day Volatility Index) 1-minute intraday bars from ThetaData.
VIX1D measures expected volatility over the current trading day — the ideal
volatility metric for 0DTE options. Launched by CBOE in April 2023.

Saves to data/vix1d_1min.parquet and data/vix1d_daily.parquet

Requires Theta Terminal running on localhost:25503 with Index tier.
Supports incremental updates — only fetches dates not already in the file.
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "http://127.0.0.1:25503"
OUTFILE = DATA_DIR / "vix1d_1min.parquet"
DAILY_OUTFILE = DATA_DIR / "vix1d_daily.parquet"

# VIX1D launched April 2023
START_DATE = "2023-04-01"


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def fetch_1min(date_str: str) -> list[dict]:
    resp = requests.get(f"{BASE_URL}/v3/index/history/ohlc", params={
        "symbol": "VIX1D",
        "start_date": date_str,
        "end_date": date_str,
        "interval": "1m",
        "format": "json",
    }, timeout=60)
    resp.raise_for_status()

    data = resp.json().get("response", [])
    if not data:
        return []

    rows = []
    for tick in data:
        ts = tick.get("timestamp")
        if ts is None or tick.get("close", 0) == 0:
            continue
        rows.append({
            "datetime": ts,
            "vix1d_open": tick.get("open"),
            "vix1d_high": tick.get("high"),
            "vix1d_low": tick.get("low"),
            "vix1d_close": tick.get("close"),
        })
    return rows


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date").agg(
        vix1d_open=("vix1d_open", "first"),
        vix1d_high=("vix1d_high", "max"),
        vix1d_low=("vix1d_low", "min"),
        vix1d_close=("vix1d_close", "last"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def _save(all_rows):
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df.to_parquet(OUTFILE, index=False)
    return df


def main():
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching VIX1D 1-min bars: {START_DATE} to {end_date}")

    existing_dates = set()
    all_rows = []
    if OUTFILE.exists():
        existing_df = pd.read_parquet(OUTFILE)
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])
        existing_dates = set(existing_df["datetime"].dt.strftime("%Y-%m-%d").unique())
        all_rows = existing_df.to_dict("records")
        print(f"  Existing: {len(existing_dates)} dates ({len(all_rows):,} rows)")

    candidates = get_trading_days(START_DATE, end_date)
    remaining = [d for d in candidates if d not in existing_dates]
    print(f"  Remaining: {len(remaining)} dates to fetch\n")

    if not remaining:
        print("All dates already fetched!")
        return

    fetched = 0
    errors = []

    for i, date_str in enumerate(remaining):
        pct = (i + 1) / len(remaining) * 100
        sys.stdout.write(f"\r[{i+1}/{len(remaining)}] ({pct:.1f}%) {date_str}...")
        sys.stdout.flush()

        try:
            rows = fetch_1min(date_str)
            if rows:
                all_rows.extend(rows)
                fetched += 1
                sys.stdout.write(f" {len(rows)} bars\n")
            else:
                sys.stdout.write(f" (holiday/no data)\n")
        except Exception as e:
            errors.append((date_str, str(e)))
            sys.stdout.write(f" ERROR: {e}\n")

        if (i + 1) % 20 == 0:
            _save(all_rows)

        time.sleep(0.25)

    df = _save(all_rows)

    print(f"\n{'='*60}")
    print(f"Fetched {fetched} new dates ({len(errors)} errors)")
    print(f"Total: {len(df):,} rows, {df['datetime'].dt.date.nunique()} dates")
    print(f"Range: {df['datetime'].min()} -> {df['datetime'].max()}")

    if errors:
        print(f"\nErrors:")
        for date_str, err in errors[:10]:
            print(f"  {date_str}: {err}")

    daily = build_daily(df)
    daily.to_parquet(DAILY_OUTFILE, index=False)
    print(f"\nDaily OHLC: {len(daily)} days -> {DAILY_OUTFILE}")


if __name__ == "__main__":
    main()
