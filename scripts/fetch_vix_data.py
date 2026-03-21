"""
Fetch 3 years of VIX daily data via Yahoo Finance (direct HTTP, no library needed).
VIX approximates 30-day implied volatility for SPX.
Saves to data/vix_daily.parquet
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "vix_daily.parquet"


def fetch_yahoo_daily(symbol, start_date, end_date):
    """Fetch daily OHLCV from Yahoo Finance via chart API."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s").date,
        "vix_open": quotes["open"],
        "vix_high": quotes["high"],
        "vix_low": quotes["low"],
        "vix_close": quotes["close"],
    })

    return df


def main():
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print(f"Fetching VIX daily data: {start_date} to {end_date}")

    df = fetch_yahoo_daily("^VIX", start_date, end_date)
    df = df.dropna().sort_values("date").reset_index(drop=True)

    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nLast 5 rows:")
    print(df.tail().to_string())

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
