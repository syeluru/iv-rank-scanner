"""
Fetch SPXW 0DTE per-minute option quotes (bid/ask OHLC) from ThetaData.

For each trading day with a SPXW 0DTE expiration, fetches 1-minute
bid OHLC, ask OHLC, and computes mid_close for all strikes within ATM +/- 250 pts.

Used by build_target_v1.py to walk forward option prices for SL/TP simulation.

Output: data/spxw_0dte_minute_quotes.parquet
Columns: minute, bid_open, bid_high, bid_low, bid_close,
         ask_open, ask_high, ask_low, ask_close, mid_close,
         date, strike, right

Requires Theta Terminal running on localhost:25503 (OPTION.PRO tier).
"""

import gc
import sys
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_DIR = DATA_DIR / "quotes_chunks"

BASE_URL = "http://127.0.0.1:25503"
MAX_WORKERS = 10
SAVE_EVERY = 5  # save checkpoint every N days


def fetch_ohlc(date_str, strike, right):
    """Fetch 1-minute option OHLC for one contract on one day."""
    resp = requests.get(f"{BASE_URL}/v3/option/history/ohlc", params={
        "symbol": "SPXW",
        "expiration": date_str,
        "strike": str(int(strike)),
        "right": right,
        "start_date": date_str,
        "end_date": date_str,
        "interval": "1m",
        "format": "json",
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("response"):
        return []
    return data["response"][0]["data"]


def parse_ohlc(rows, date_str, strike, right):
    """Parse raw OHLC response into clean records with bid/ask OHLC."""
    if not rows:
        return []
    out = []
    for r in rows:
        ts = r.get("timestamp") or r.get("datetime")
        if not ts:
            continue
        t = pd.Timestamp(ts)
        # Only keep market hours
        if t.time() < pd.Timestamp("09:30").time() or t.time() > pd.Timestamp("16:00").time():
            continue

        bid_open = r.get("bid_open", 0.0) or 0.0
        bid_high = r.get("bid_high", 0.0) or 0.0
        bid_low = r.get("bid_low", 0.0) or 0.0
        bid_close = r.get("bid_close", 0.0) or 0.0
        ask_open = r.get("ask_open", 0.0) or 0.0
        ask_high = r.get("ask_high", 0.0) or 0.0
        ask_low = r.get("ask_low", 0.0) or 0.0
        ask_close = r.get("ask_close", 0.0) or 0.0

        mid_close = (bid_close + ask_close) / 2

        out.append({
            "minute": ts,
            "date": date_str,
            "strike": strike,
            "right": right,
            "bid_open": bid_open,
            "bid_high": bid_high,
            "bid_low": bid_low,
            "bid_close": bid_close,
            "ask_open": ask_open,
            "ask_high": ask_high,
            "ask_low": ask_low,
            "ask_close": ask_close,
            "mid_close": mid_close,
        })
    return out


def get_existing_dates():
    """Get all dates already fetched from the main file + chunk files."""
    existing_dates = set()
    outfile = DATA_DIR / "spxw_0dte_minute_quotes.parquet"
    if outfile.exists():
        meta = pq.read_table(outfile, columns=["date"])
        existing_dates.update(meta.column("date").to_pandas().astype(str).unique())
        row_count = meta.num_rows
        del meta
        gc.collect()
        print(f"  Main file: {len(existing_dates)} dates ({row_count:,} rows)")

    if CHUNKS_DIR.exists():
        for chunk_file in sorted(CHUNKS_DIR.glob("*.parquet")):
            chunk_meta = pq.read_table(chunk_file, columns=["date"])
            chunk_dates = set(chunk_meta.column("date").to_pandas().astype(str).unique())
            existing_dates.update(chunk_dates)
            del chunk_meta
        chunk_count = len(list(CHUNKS_DIR.glob("*.parquet")))
        if chunk_count:
            print(f"  Chunk files: {chunk_count} pending merge")

    return existing_dates


def merge_chunks():
    """Merge all chunk files into the main parquet file, then delete chunks."""
    outfile = DATA_DIR / "spxw_0dte_minute_quotes.parquet"
    chunk_files = sorted(CHUNKS_DIR.glob("*.parquet")) if CHUNKS_DIR.exists() else []
    if not chunk_files:
        return

    print(f"\nMerging {len(chunk_files)} chunk files into main file...")
    tables = []
    if outfile.exists():
        tables.append(pq.read_table(outfile))
    for cf in chunk_files:
        tables.append(pq.read_table(cf))

    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, outfile)
    total_rows = combined.num_rows
    del tables, combined
    gc.collect()

    for cf in chunk_files:
        cf.unlink()
    print(f"  Merged: {total_rows:,} total rows")


def main():
    eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")

    eod["date"] = pd.to_datetime(eod["date"])
    spx["date"] = pd.to_datetime(spx["date"])
    eod = eod.merge(spx[["date", "spx_open"]], on="date", how="left")

    trading_days = sorted(eod["date"].dt.strftime("%Y-%m-%d").unique())
    print(f"Fetching SPXW 0DTE minute quotes (bid/ask OHLC)")
    print(f"  Trading days available: {len(trading_days)}")

    CHUNKS_DIR.mkdir(exist_ok=True)

    print("Checking existing data...")
    existing_dates = get_existing_dates()

    remaining = [d for d in trading_days if d not in existing_dates]
    print(f"  Remaining: {len(remaining)} days\n")

    if not remaining:
        print("All dates already fetched!")
        merge_chunks()
        return

    total = len(remaining)
    errors = []
    batch_rows = []
    total_new = 0
    chunk_num = 0

    for i, trade_date in enumerate(remaining):
        pct = (i + 1) / total * 100

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
                executor.submit(fetch_ohlc, date_str, strike, right): (date_str, strike, right)
                for date_str, strike, right in contracts
            }
            for future in as_completed(futures):
                date_str, strike, right = futures[future]
                try:
                    raw = future.result()
                    records = parse_ohlc(raw, date_str, strike, right)
                    batch_rows.extend(records)
                    day_rows += len(records)
                except Exception as e:
                    errors.append((date_str, strike, right, str(e)))
                    day_errors += 1

        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date} | ATM={atm} | {day_rows} rows ({day_errors} errors)    \n")
        sys.stdout.flush()

        # Save batch to chunk file periodically
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
            if batch_rows:
                new_df = pd.DataFrame(batch_rows)
                new_df["date"] = pd.to_datetime(new_df["date"])
                new_df["minute"] = pd.to_datetime(new_df["minute"])
                chunk_file = CHUNKS_DIR / f"chunk_{chunk_num:04d}.parquet"
                new_df.to_parquet(chunk_file, index=False)
                total_new += len(batch_rows)
                print(f"  [chunk saved: {len(batch_rows):,} rows — {total_new:,} new this run]")
                del new_df
                batch_rows.clear()
                chunk_num += 1
                gc.collect()

    # Final merge
    merge_chunks()

    outfile = DATA_DIR / "spxw_0dte_minute_quotes.parquet"
    print(f"\n{'='*60}")
    final_meta = pq.read_table(outfile, columns=["date"])
    print(f"Final: {final_meta.num_rows:,} rows, {len(final_meta.column('date').to_pandas().unique())} dates")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
