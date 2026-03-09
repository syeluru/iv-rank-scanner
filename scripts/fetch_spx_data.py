"""
Fetch 3 years of SPY 1-minute intraday data from Twelve Data.
SPY is used as a proxy for SPX (tracks ~1:1, just 1/10th the price).
Saves to data/spy_1min.parquet
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

BASE_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "SPY"
INTERVAL = "1min"
MAX_OUTPUT = 5000  # max data points per request

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "spy_1min.parquet"

# Free tier: 8 requests/min, 800/day
RATE_LIMIT_DELAY = 8  # seconds between calls


def fetch_chunk(start_date, end_date):
    """Fetch 1-min bars for a date range. Returns up to 5000 bars."""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_date": start_date,
        "end_date": end_date,
        "outputsize": MAX_OUTPUT,
        "timezone": "America/New_York",
        "apikey": API_KEY,
    }

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "code" in data and data["code"] != 200:
        return [], data.get("message", "Unknown error")

    values = data.get("values", [])
    return values, None


def generate_chunks(start_date, end_date, chunk_days=16):
    """Split date range into ~2-week chunks to stay under 5000 bars per call.
    ~12 trading days * 390 bars = ~4680, safely under 5000."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


def main():
    if not API_KEY:
        print("ERROR: TWELVE_DATA_API_KEY not found in .env")
        return

    # 3 years back from today
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print(f"Fetching SPY 1-min data: {start_date} to {end_date}")
    print(f"API: {BASE_URL}")
    print(f"Symbol: {SYMBOL}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    chunks = generate_chunks(start_date, end_date, chunk_days=16)
    print(f"Split into {len(chunks)} chunks (~2 weeks each)")
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between calls (8 req/min)")
    est_time = len(chunks) * RATE_LIMIT_DELAY
    print(f"Estimated time: ~{est_time // 60}m {est_time % 60}s")
    print(f"API credits used: ~{len(chunks)} of 800/day")
    print()

    all_rows = []
    failed_chunks = []
    empty_chunks = 0

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk_start} to {chunk_end} ...", end=" ", flush=True)
        try:
            values, error = fetch_chunk(chunk_start, chunk_end)
            if error:
                failed_chunks.append((chunk_start, chunk_end, error))
                print(f"ERROR: {error}")
            elif values:
                all_rows.extend(values)
                print(f"{len(values):,} bars")
            else:
                empty_chunks += 1
                print("empty")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited — waiting 60s...")
                time.sleep(60)
                try:
                    values, error = fetch_chunk(chunk_start, chunk_end)
                    if values and not error:
                        all_rows.extend(values)
                        print(f"  {len(values):,} bars (retry)")
                    else:
                        failed_chunks.append((chunk_start, chunk_end, error))
                        print(f"  FAILED on retry")
                except Exception:
                    failed_chunks.append((chunk_start, chunk_end, str(e)))
            else:
                failed_chunks.append((chunk_start, chunk_end, str(e)))
                print(f"HTTP {e.response.status_code}")
        except Exception as e:
            failed_chunks.append((chunk_start, chunk_end, str(e)))
            print(f"ERROR: {e}")

        # Respect rate limit
        if i < len(chunks) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone fetching.")
    print(f"  Total bars: {len(all_rows):,}")
    print(f"  Empty chunks: {empty_chunks}")
    print(f"  Failed chunks: {len(failed_chunks)}")

    if failed_chunks:
        for cs, ce, err in failed_chunks[:5]:
            print(f"    {cs} to {ce}: {err}")

    if not all_rows:
        print("No data fetched. Check your API key.")
        return

    # Build DataFrame — Twelve Data returns: datetime, open, high, low, close, volume
    df = pd.DataFrame(all_rows)

    # Convert types
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime").reset_index(drop=True)

    # Drop duplicates (overlapping chunks)
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique trading days: {df['date'].nunique()}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nLast 5 rows:")
    print(df.tail().to_string())

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
