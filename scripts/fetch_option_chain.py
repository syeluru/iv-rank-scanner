"""
Fetch SPXW 0DTE option chain EOD + Open Interest from ThetaData.
For each trading day with a SPXW expiration, pulls:
  - All strikes (calls + puts) EOD data: volume, bid/ask, OHLC
  - All strikes (calls + puts) open interest

Saves to:
  data/spxw_0dte_eod.parquet    — option chain EOD (volume, bid/ask, OHLC)
  data/spxw_0dte_oi.parquet     — open interest per strike

Requires Theta Terminal running on localhost:25503.
Rate limit: 2 max concurrent requests. We pace at ~1 req/sec.
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "http://127.0.0.1:25503"
DELAY = 0.6  # seconds between requests


def get_0dte_expirations(start_date, end_date):
    """Get all SPXW expiration dates in range (these are our 0DTE days)."""
    resp = requests.get(f"{BASE_URL}/v3/option/list/expirations", params={
        "symbol": "SPXW", "format": "json"
    }, timeout=30)
    resp.raise_for_status()
    exps = [e["expiration"] for e in resp.json()["response"]]
    return [e for e in exps if start_date <= e <= end_date]


def fetch_chain_eod(expiration, right):
    """Fetch EOD data for all strikes of a given expiration and right (call/put)."""
    resp = requests.get(f"{BASE_URL}/v3/option/history/eod", params={
        "symbol": "SPXW",
        "expiration": expiration,
        "strike": "*",
        "right": right,
        "start_date": expiration,
        "end_date": expiration,
        "format": "json",
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


def fetch_chain_oi(expiration, right):
    """Fetch open interest for all strikes of a given expiration and right."""
    resp = requests.get(f"{BASE_URL}/v3/option/history/open_interest", params={
        "symbol": "SPXW",
        "expiration": expiration,
        "strike": "*",
        "right": right,
        "start_date": expiration,
        "end_date": expiration,
        "format": "json",
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_eod_response(contracts, expiration, right):
    """Parse EOD response into flat rows."""
    rows = []
    for contract in contracts:
        strike = contract["contract"]["strike"]
        for d in contract["data"]:
            rows.append({
                "date": expiration,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "volume": d["volume"],
                "bid": d["bid"],
                "ask": d["ask"],
                "bid_size": d["bid_size"],
                "ask_size": d["ask_size"],
            })
    return rows


def parse_oi_response(contracts, expiration, right):
    """Parse OI response into flat rows."""
    rows = []
    for contract in contracts:
        strike = contract["contract"]["strike"]
        for d in contract["data"]:
            rows.append({
                "date": expiration,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "open_interest": d["open_interest"],
            })
    return rows


def main():
    print("Getting SPXW 0DTE expiration dates...")
    expirations = get_0dte_expirations("2023-02-27", "2026-02-27")
    print(f"  {len(expirations)} 0DTE trading days")

    # Check for existing progress (resume support)
    eod_file = DATA_DIR / "spxw_0dte_eod.parquet"
    oi_file = DATA_DIR / "spxw_0dte_oi.parquet"

    existing_dates = set()
    eod_rows = []
    oi_rows = []

    if eod_file.exists():
        print("  Found existing EOD file, loading for resume...")
        existing_df = pd.read_parquet(eod_file)
        existing_dates = set(existing_df["date"].astype(str).unique())
        eod_rows = existing_df.to_dict("records")
        print(f"  Already have {len(existing_dates)} dates")

    if oi_file.exists():
        existing_oi = pd.read_parquet(oi_file)
        oi_rows = existing_oi.to_dict("records")

    remaining = [e for e in expirations if e not in existing_dates]
    print(f"  Remaining to fetch: {len(remaining)} days")

    if not remaining:
        print("All dates already fetched!")
        return

    total = len(remaining)
    errors = []
    save_every = 25  # save checkpoint every N days

    for i, exp in enumerate(remaining):
        pct = (i + 1) / total * 100
        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) Fetching {exp}...")
        sys.stdout.flush()

        try:
            # Fetch calls + puts EOD
            calls_eod = fetch_chain_eod(exp, "call")
            time.sleep(DELAY)
            puts_eod = fetch_chain_eod(exp, "put")
            time.sleep(DELAY)

            eod_rows.extend(parse_eod_response(calls_eod, exp, "call"))
            eod_rows.extend(parse_eod_response(puts_eod, exp, "put"))

            # Fetch calls + puts OI
            calls_oi = fetch_chain_oi(exp, "call")
            time.sleep(DELAY)
            puts_oi = fetch_chain_oi(exp, "put")
            time.sleep(DELAY)

            oi_rows.extend(parse_oi_response(calls_oi, exp, "call"))
            oi_rows.extend(parse_oi_response(puts_oi, exp, "put"))

            n_contracts = len(calls_eod) + len(puts_eod)
            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {exp}: {n_contracts} contracts    ")

        except Exception as e:
            errors.append((exp, str(e)))
            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {exp}: ERROR - {str(e)[:60]}    ")
            time.sleep(2)  # back off on error

        # Checkpoint save
        if (i + 1) % save_every == 0 or (i + 1) == total:
            df_eod = pd.DataFrame(eod_rows)
            df_eod["date"] = pd.to_datetime(df_eod["date"])
            df_eod.to_parquet(eod_file, index=False)

            df_oi = pd.DataFrame(oi_rows)
            df_oi["date"] = pd.to_datetime(df_oi["date"])
            df_oi.to_parquet(oi_file, index=False)

            sys.stdout.write(f" [saved]")

    print(f"\n\n{'='*60}")
    df_eod = pd.DataFrame(eod_rows)
    df_oi = pd.DataFrame(oi_rows)

    print(f"EOD data: {len(df_eod)} rows, {df_eod['date'].nunique()} dates")
    print(f"OI data:  {len(df_oi)} rows, {df_oi['date'].nunique()} dates")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for exp, err in errors:
            print(f"  {exp}: {err}")

    print(f"\nSaved to:")
    print(f"  {eod_file} ({eod_file.stat().st_size / 1e6:.1f} MB)")
    print(f"  {oi_file} ({oi_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
