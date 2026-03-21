"""
Fetch SPXW 0DTE tick quotes and resample to 1-minute bid/ask bars.
For each trading day, fetches all strikes within ATM +/- 250 pts.

Output: data/spxw_0dte_minute_quotes.parquet
Columns: date, strike, right, minute, bid_open, bid_high, bid_low, bid_close,
         ask_open, ask_high, ask_low, ask_close, mid_close

Requires Theta Terminal running on localhost:25503 (OPTION.PRO tier).
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

BASE_URL = "http://127.0.0.1:25503"
MAX_WORKERS = 8
SAVE_EVERY = 10  # save checkpoint every N days


def fetch_ticks(date_str, strike, right):
    """Fetch tick quotes for one contract on one day."""
    resp = requests.get(f"{BASE_URL}/v3/option/history/quote", params={
        "symbol": "SPXW",
        "expiration": date_str,
        "strike": str(int(strike)),
        "right": right,
        "start_date": date_str,
        "end_date": date_str,
        "format": "json",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["response"][0]["data"]


def ticks_to_minute_bars(ticks, date_str, strike, right):
    """Resample tick quotes to 1-minute OHLC bid/ask bars."""
    if not ticks:
        return []

    df = pd.DataFrame(ticks)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("ts").sort_index()

    # Keep only market hours
    df = df.between_time("09:30", "16:00")
    if df.empty:
        return []

    # Resample bid and ask to 1-minute OHLC
    bid = df["bid"].resample("1min").ohlc().add_prefix("bid_")
    ask = df["ask"].resample("1min").ohlc().add_prefix("ask_")
    bars = pd.concat([bid, ask], axis=1).dropna(how="all")
    bars["mid_close"] = (bars["bid_close"] + bars["ask_close"]) / 2
    bars["date"] = date_str
    bars["strike"] = strike
    bars["right"] = right
    bars.index.name = "minute"
    return bars.reset_index().to_dict("records")


def main():
    # Load EOD data to get strikes per day
    eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")

    eod["date"] = pd.to_datetime(eod["date"])
    spx["date"] = pd.to_datetime(spx["date"])
    eod = eod.merge(spx[["date", "spx_open"]], on="date", how="left")

    # Build per-day contract list (strikes within ATM +/- 150)
    trading_days = sorted(eod["date"].dt.strftime("%Y-%m-%d").unique())

    # Check for resume
    outfile = DATA_DIR / "spxw_0dte_minute_quotes.parquet"
    existing_dates = set()
    all_rows = []

    if outfile.exists():
        print("Found existing file, loading for resume...")
        existing_df = pd.read_parquet(outfile)
        existing_dates = set(existing_df["date"].astype(str).unique())
        all_rows = existing_df.to_dict("records")
        print(f"  Already have {len(existing_dates)} dates ({len(all_rows):,} rows)")

    remaining = [d for d in trading_days if d not in existing_dates]
    print(f"  Remaining: {len(remaining)} days\n")

    if not remaining:
        print("All dates already fetched!")
        return

    total = len(remaining)
    errors = []

    for i, trade_date in enumerate(remaining):
        pct = (i + 1) / total * 100

        # Get strikes for this day
        day_eod = eod[eod["date"].dt.strftime("%Y-%m-%d") == trade_date]
        if day_eod.empty:
            continue

        spx_open = day_eod["spx_open"].iloc[0]
        atm = round(spx_open / 5) * 5
        nearby = day_eod[
            (day_eod["strike"] >= atm - 250) &
            (day_eod["strike"] <= atm + 250)
        ]
        strikes = sorted(nearby["strike"].unique())
        contracts = [(trade_date, s, r) for s in strikes for r in ["call", "put"]]

        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date} | ATM={atm} | {len(contracts)} contracts...")
        sys.stdout.flush()

        day_rows = 0
        day_errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch_ticks, date_str, strike, right): (date_str, strike, right)
                for date_str, strike, right in contracts
            }
            for future in as_completed(futures):
                date_str, strike, right = futures[future]
                try:
                    ticks = future.result()
                    bars = ticks_to_minute_bars(ticks, date_str, strike, right)
                    all_rows.extend(bars)
                    day_rows += len(bars)
                except Exception as e:
                    errors.append((date_str, strike, right, str(e)))
                    day_errors += 1

        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date} | ATM={atm} | {day_rows} rows ({day_errors} errors)    \n")
        sys.stdout.flush()

        # Checkpoint
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
            df = pd.DataFrame(all_rows)
            if len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                df["minute"] = pd.to_datetime(df["minute"])
                df.to_parquet(outfile, index=False)
                print(f"  [saved {len(df):,} rows to {outfile.name}]")

    print(f"\n{'='*60}")
    df = pd.DataFrame(all_rows)
    print(f"Final: {len(df):,} rows, {df['date'].nunique()} dates")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
