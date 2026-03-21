#!/usr/bin/env python3
"""
ZDOM V1.2 Live Intraday Data Collector

Runs from 9:30 AM to 4:00 PM ET every trading day.
Pulls SPX, VIX, VIX1D 1-min bars and SPXW 0DTE greeks every completed minute.
Appends to the v1_2 raw parquets in real-time.

This is the data collection loop ONLY — no scoring or trading.
The live_pipeline_v1_2.py handles scoring separately.

Usage:
  python3 live_intraday_collector_v1_2.py              # run until market close
  python3 live_intraday_collector_v1_2.py --dry-run    # pull but don't save

Schedule: Start via LaunchAgent at 9:25 AM ET Mon-Fri.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

THETA = "http://127.0.0.1:25503"

# Market hours (ET)
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 16, 0
POLL_INTERVAL_SEC = 10  # check every 10 seconds if a new minute completed


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def is_market_hours():
    now = datetime.now()
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= now <= market_close


def wait_for_market_open():
    now = datetime.now()
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    if now < market_open:
        wait = (market_open - now).total_seconds()
        log(f"Waiting {wait:.0f}s for market open at {MARKET_OPEN_HOUR}:{MARKET_OPEN_MIN:02d}...")
        time.sleep(wait)


def check_theta():
    try:
        resp = requests.get(f"{THETA}/", timeout=3)
        return True
    except:
        return False


def pull_index_bars(symbol, today_str):
    """Pull all 1-min bars for today for an index symbol."""
    try:
        resp = requests.get(f"{THETA}/v3/index/history/ohlc", params={
            "symbol": symbol, "start_date": today_str,
            "end_date": today_str, "interval": "1m",
        }, timeout=10)
        df = pd.read_csv(StringIO(resp.text))
        if len(df) == 0:
            return pd.DataFrame()
        df = df.rename(columns={"timestamp": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Filter market hours
        df = df[(df["datetime"].dt.time >= pd.Timestamp("09:30").time()) &
                (df["datetime"].dt.time <= pd.Timestamp("16:00").time())]
        return df
    except Exception as e:
        log(f"  [warn] {symbol} pull failed: {e}")
        return pd.DataFrame()


def pull_greeks(today_str, atm_strike, strike_range=250):
    """Pull first_order greeks for all strikes within ATM +/- range."""
    strikes = list(range(int(atm_strike - strike_range), int(atm_strike + strike_range + 5), 5))
    all_rows = []

    for strike in strikes:
        for right in ["call", "put"]:
            try:
                resp = requests.get(f"{THETA}/v3/option/history/greeks/first_order", params={
                    "symbol": "SPXW", "expiration": today_str,
                    "strike": str(strike), "right": right,
                    "start_date": today_str, "end_date": today_str,
                    "interval": "1m",
                }, timeout=10)
                df = pd.read_csv(StringIO(resp.text))
                if len(df) > 0:
                    df["strike"] = strike
                    df["right"] = right
                    df["date"] = today_str
                    all_rows.append(df)
            except:
                pass

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def append_to_parquet(new_df, filepath, dedup_col="datetime"):
    """Append new rows to existing parquet, dedup by key column."""
    if new_df.empty:
        return 0

    if filepath.exists():
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[dedup_col], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(dedup_col).reset_index(drop=True)
    combined.to_parquet(filepath, index=False)
    return len(new_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Pull but don't save")
    args = parser.parse_args()

    log("ZDOM V1.2 Live Intraday Collector")
    log(f"  Raw dir: {RAW_DIR}")
    log(f"  Dry run: {args.dry_run}")

    if not check_theta():
        log("ERROR: ThetaData not running on localhost:25503")
        sys.exit(1)

    log("ThetaData connected")

    # Wait for market open
    wait_for_market_open()

    today_str = datetime.now().strftime("%Y%m%d")
    last_minute_pulled = None
    pull_count = 0
    greeks_pull_count = 0

    log(f"Starting collection for {today_str}")

    while True:
        now = datetime.now()

        # Check market close
        if now.hour > MARKET_CLOSE_HOUR or (now.hour == MARKET_CLOSE_HOUR and now.minute > MARKET_CLOSE_MIN):
            log("Market closed. Stopping collector.")
            break

        # Wait for market open if before 9:30
        if now.hour < MARKET_OPEN_HOUR or (now.hour == MARKET_OPEN_HOUR and now.minute < MARKET_OPEN_MIN):
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # Check if a new minute has completed
        current_minute = now.replace(second=0, microsecond=0)
        if current_minute == last_minute_pulled:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # New minute — pull data
        # Wait 2 seconds past the boundary for data to settle
        if now.second < 2:
            time.sleep(2 - now.second)

        # Pull SPX bars
        spx = pull_index_bars("SPX", today_str)
        if not spx.empty and not args.dry_run:
            spx_renamed = spx[["datetime", "open", "high", "low", "close"]].copy()
            n = append_to_parquet(spx_renamed, RAW_DIR / "spx_1min.parquet")

        # Pull VIX bars
        vix = pull_index_bars("VIX", today_str)
        if not vix.empty and not args.dry_run:
            vix_renamed = vix.rename(columns={"open": "vix_open", "high": "vix_high",
                                               "low": "vix_low", "close": "vix_close"})
            vix_renamed = vix_renamed[["datetime", "vix_open", "vix_high", "vix_low", "vix_close"]]
            append_to_parquet(vix_renamed, RAW_DIR / "vix_1min.parquet")

        # Pull VIX1D bars
        vix1d = pull_index_bars("VIX1D", today_str)
        if not vix1d.empty and not args.dry_run:
            v1d_renamed = vix1d.rename(columns={"open": "vix1d_open", "high": "vix1d_high",
                                                 "low": "vix1d_low", "close": "vix1d_close"})
            v1d_renamed = v1d_renamed[["datetime", "vix1d_open", "vix1d_high", "vix1d_low", "vix1d_close"]]
            append_to_parquet(v1d_renamed, RAW_DIR / "vix1d_1min.parquet")

        spx_close = spx.iloc[-2]["close"] if len(spx) >= 2 else 0
        n_bars = len(spx) if not spx.empty else 0

        pull_count += 1
        last_minute_pulled = current_minute

        # Pull greeks every 5 minutes (heavy operation)
        if pull_count % 5 == 1 and spx_close > 0:
            atm = round(spx_close / 5) * 5
            log(f"  Pulling greeks (ATM={atm})...")
            greeks = pull_greeks(today_str, atm, strike_range=200)
            if not greeks.empty and not args.dry_run:
                greeks["timestamp"] = pd.to_datetime(greeks["timestamp"])
                greeks["date"] = pd.to_datetime(greeks["date"])
                # Always compute mid from bid/ask (ThetaData returns mid column but it's NaN)
                if "bid" in greeks.columns and "ask" in greeks.columns:
                    greeks["mid"] = (greeks["bid"] + greeks["ask"]) / 2
                # Save to daily chunk file (avoid rewriting 56M row main file)
                greeks_daily_dir = RAW_DIR / "greeks_daily_chunks"
                greeks_daily_dir.mkdir(exist_ok=True)
                chunk_file = greeks_daily_dir / f"greeks_{today_str}.parquet"
                if chunk_file.exists():
                    existing_chunk = pd.read_parquet(chunk_file)
                    combined = pd.concat([existing_chunk, greeks], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date", "strike", "right", "timestamp"], keep="last")
                    combined.to_parquet(chunk_file, index=False)
                else:
                    greeks.to_parquet(chunk_file, index=False)
                greeks_pull_count += 1
                log(f"  Greeks: {len(greeks)} rows appended (pull #{greeks_pull_count})")

            log(f"Pull #{pull_count}: SPX={spx_close:.2f}, {n_bars} bars, "
                f"VIX={vix.iloc[-2]['close'] if len(vix) >= 2 else '?'}")
        else:
            log(f"Pull #{pull_count}: SPX={spx_close:.2f}, {n_bars} bars")

    log(f"\nCollection complete: {pull_count} pulls, {greeks_pull_count} greeks pulls")


if __name__ == "__main__":
    main()
