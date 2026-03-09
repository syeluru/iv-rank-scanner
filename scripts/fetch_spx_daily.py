"""
Fetch SPX daily OHLC from ThetaData REST API (Index VALUE tier).
Saves to data/spx_daily.parquet

Requires Theta Terminal running on localhost:25503.
"""

import time
import requests
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "http://127.0.0.1:25503"


def fetch_spx_eod(start_date, end_date):
    """Fetch SPX index EOD data for a date range (max 365 days)."""
    resp = requests.get(f"{BASE_URL}/v3/index/history/eod", params={
        "symbol": "SPX",
        "start_date": start_date,
        "end_date": end_date,
        "format": "json",
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    # Fetch in yearly chunks (max 365 days per request)
    chunks = [
        ("2023-02-27", "2024-02-26"),
        ("2024-02-27", "2025-02-26"),
        ("2025-02-27", "2026-02-27"),
    ]

    all_rows = []
    for start, end in chunks:
        print(f"Fetching SPX EOD: {start} to {end}...")
        data = fetch_spx_eod(start, end)
        print(f"  Got {len(data)} days")
        for row in data:
            all_rows.append({
                "date": row["last_trade"][:10],
                "spx_open": row["open"],
                "spx_high": row["high"],
                "spx_low": row["low"],
                "spx_close": row["close"],
            })
        time.sleep(1)  # be nice to the API

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    print(f"\nSPX daily OHLC: {len(df)} trading days")
    print(f"Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nSample:")
    print(df.tail(5).to_string(index=False))

    outfile = DATA_DIR / "spx_daily.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
