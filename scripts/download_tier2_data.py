#!/usr/bin/env python3
"""
Download Tier 2 cross-asset data from ThetaData for v8 features.

Pulls SPY (with volume), VIX3M, VIX9D, VVIX, TLT, GLD as 1-min bars.
Saves to data/<symbol>_1min.parquet with same schema as spx_1min.parquet.

Usage:
    python -m scripts.download_tier2_data
    python -m scripts.download_tier2_data --symbol SPY --start 2023-03-01 --end 2026-03-13
"""

import argparse
import logging
import sys
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_tier2")

DATA_DIR = Path("data")
THETA_HOST = "http://127.0.0.1:25503"

# Symbols and their ThetaData endpoint types
SYMBOLS = {
    "SPY": {"type": "stock", "has_volume": True},
    "VIX3M": {"type": "index", "has_volume": False},
    "VIX9D": {"type": "index", "has_volume": False},
    "VVIX": {"type": "index", "has_volume": False},
    "TLT": {"type": "stock", "has_volume": True},
    "GLD": {"type": "stock", "has_volume": True},
}


def fetch_ohlc(symbol: str, sym_config: dict, start: date, end: date) -> pd.DataFrame:
    """
    Fetch 1-min OHLC from ThetaData API.

    Returns DataFrame with [datetime, open, high, low, close] + optional [volume].
    """
    import urllib.request
    import json

    asset_type = sym_config["type"]
    endpoint = f"{THETA_HOST}/v3/{asset_type}/history/ohlc"

    all_rows = []
    current_start = start

    # Process in monthly chunks to avoid API limits
    while current_start < end:
        chunk_end = min(current_start + timedelta(days=31), end)

        url = (
            f"{endpoint}?symbol={symbol}"
            f"&interval=1m"
            f"&start_date={current_start.strftime('%Y%m%d')}"
            f"&end_date={chunk_end.strftime('%Y%m%d')}"
            f"&format=json"
        )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"  Failed {symbol} {current_start}-{chunk_end}: {e}")
            current_start = chunk_end + timedelta(days=1)
            continue

        # Parse response
        if isinstance(data, dict) and "response" in data:
            rows = data["response"]
        elif isinstance(data, list):
            rows = data
        else:
            logger.warning(f"  Unexpected response format for {symbol}")
            current_start = chunk_end + timedelta(days=1)
            continue

        for row in rows:
            if isinstance(row, dict):
                parsed = {
                    "datetime": row.get("datetime") or row.get("time_ms"),
                    "open": row.get("open", 0),
                    "high": row.get("high", 0),
                    "low": row.get("low", 0),
                    "close": row.get("close", 0),
                }
                if sym_config["has_volume"]:
                    parsed["volume"] = row.get("volume", 0)
                all_rows.append(parsed)
            elif isinstance(row, list) and len(row) >= 5:
                # CSV-style response: [timestamp, open, high, low, close, volume?]
                parsed = {
                    "datetime": row[0],
                    "open": row[1] / 100 if isinstance(row[1], int) and row[1] > 100000 else row[1],
                    "high": row[2] / 100 if isinstance(row[2], int) and row[2] > 100000 else row[2],
                    "low": row[3] / 100 if isinstance(row[3], int) and row[3] > 100000 else row[3],
                    "close": row[4] / 100 if isinstance(row[4], int) and row[4] > 100000 else row[4],
                }
                if sym_config["has_volume"] and len(row) >= 6:
                    parsed["volume"] = row[5]
                all_rows.append(parsed)

        n_chunk = len(rows)
        logger.info(f"  {symbol} {current_start} - {chunk_end}: {n_chunk} bars")

        current_start = chunk_end + timedelta(days=1)
        _time.sleep(0.5)  # Rate limit

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Parse datetime
    if df["datetime"].dtype == np.int64 or df["datetime"].dtype == np.float64:
        # Millisecond timestamps
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter weekends
    df = df[df["datetime"].dt.weekday < 5]

    # Sort and deduplicate
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Tier 2 data from ThetaData")
    parser.add_argument("--symbol", type=str, help="Single symbol to download")
    parser.add_argument("--start", type=str, default="2023-03-01", help="Start date")
    parser.add_argument("--end", type=str, default="2026-03-13", help="End date")
    parser.add_argument("--test", action="store_true", help="Test with 1 month only")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.test:
        end = start + timedelta(days=31)
        logger.info(f"Test mode: {start} to {end}")

    symbols = {args.symbol: SYMBOLS[args.symbol]} if args.symbol else SYMBOLS

    for symbol, config in symbols.items():
        output_file = DATA_DIR / f"{symbol.lower()}_1min.parquet"

        if output_file.exists():
            existing = pd.read_parquet(output_file)
            logger.info(f"{symbol}: existing file has {len(existing):,} rows, skipping")
            continue

        logger.info(f"Downloading {symbol} 1-min bars ({start} to {end})...")
        df = fetch_ohlc(symbol, config, start, end)

        if df.empty:
            logger.warning(f"  No data for {symbol}")
            continue

        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved: {output_file} ({len(df):,} rows)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
