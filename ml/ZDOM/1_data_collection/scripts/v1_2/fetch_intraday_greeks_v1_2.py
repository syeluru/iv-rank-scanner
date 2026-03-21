"""
Fetch SPXW 0DTE intraday greeks at 1-minute intervals for v1_2.

For each trading day, fetches delta, gamma, theta, vega, IV, bid, ask
for all strikes within ATM +/- 250 pts.

Uses the second_order endpoint to get TRUE gamma (not lambda/leverage).

Source: ThetaData localhost:25503
Output: raw/v1_2/spxw_0dte_intraday_greeks.parquet

Columns: date, strike, right, timestamp, underlying_price,
         bid, ask, mid, delta, gamma, theta, vega, rho,
         implied_vol, iv_error

Run modes:
  python fetch_intraday_greeks_v1_2.py              # fetch all missing dates
  python fetch_intraday_greeks_v1_2.py --days 5     # only last N days
"""

import gc
import sys
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).resolve().parents[2]
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CHUNKS_DIR = RAW_V1_2 / "greeks_fetch_chunks"

BASE_URL = "http://127.0.0.1:25503"
MAX_WORKERS = 10
SAVE_EVERY = 5  # save checkpoint every N days


def fetch_greeks(date_str, strike, right):
    """Fetch 1-minute greeks for one contract on one day.
    Uses second_order endpoint which includes true gamma."""
    resp = requests.get(f"{BASE_URL}/v3/option/history/greeks/second_order", params={
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


def parse_greeks(rows, date_str, strike, right):
    """Parse raw greeks response into clean records."""
    if not rows:
        return []
    out = []
    for r in rows:
        ts = r.get("timestamp")
        if not ts:
            continue
        t = pd.Timestamp(ts)
        if t.time() < pd.Timestamp("09:30").time() or t.time() > pd.Timestamp("16:00").time():
            continue
        bid = r.get("bid", 0.0)
        ask = r.get("ask", 0.0)
        out.append({
            "date": date_str,
            "strike": strike,
            "right": right,
            "timestamp": ts,
            "underlying_price": r.get("underlying_price", 0.0),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "delta": r.get("delta", 0.0),
            "gamma": r.get("gamma", 0.0),
            "theta": r.get("theta", 0.0),
            "vega": r.get("vega", 0.0),
            "rho": r.get("rho", 0.0),
            "implied_vol": r.get("implied_vol", 0.0),
            "iv_error": r.get("iv_error", 0.0),
        })
    return out


def get_existing_dates():
    """Get all dates already fetched from the main file + chunk files."""
    existing_dates = set()
    outfile = RAW_V1_2 / "spxw_0dte_intraday_greeks.parquet"
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
    outfile = RAW_V1_2 / "spxw_0dte_intraday_greeks.parquet"
    chunk_files = sorted(CHUNKS_DIR.glob("*.parquet")) if CHUNKS_DIR.exists() else []
    if not chunk_files:
        return

    print(f"\nMerging {len(chunk_files)} chunk files into main file...")
    tables = []
    if outfile.exists():
        tables.append(pq.read_table(outfile))
    for cf in chunk_files:
        tables.append(pq.read_table(cf))

    # Unify schemas
    all_cols = set()
    for t in tables:
        all_cols.update(t.schema.names)
    unified = []
    for t in tables:
        for col in all_cols - set(t.schema.names):
            t = t.append_column(col, pa.nulls(t.num_rows, type=pa.float64()))
        unified.append(t)
    combined = pa.concat_tables(unified, promote_options="default")
    pq.write_table(combined, outfile)
    total_rows = combined.num_rows
    del tables, combined
    gc.collect()

    for cf in chunk_files:
        cf.unlink()
    print(f"  Merged: {total_rows:,} total rows")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=0, help="Only fetch last N days (0=all missing)")
    args = parser.parse_args()

    # Use v1_2 SPX daily for price lookups and trading day list
    spx_file = DATA_V1_2 / "spx_daily_v1_2.parquet"
    if not spx_file.exists():
        print(f"ERROR: {spx_file} not found")
        sys.exit(1)
    spx = pd.read_parquet(spx_file)
    spx["date"] = pd.to_datetime(spx["date"])
    spx["date_str"] = spx["date"].dt.strftime("%Y-%m-%d")
    price_map = dict(zip(spx["date_str"], spx["spx_close"]))

    # Get trading days from SPXW expirations
    print("Getting SPXW expirations for trading day list...")
    resp = requests.get(f"{BASE_URL}/v3/option/list/expirations", params={
        "symbol": "SPXW", "format": "json"
    }, timeout=30)
    resp.raise_for_status()
    all_exps = [e["expiration"] for e in resp.json()["response"]]
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    trading_days = [e for e in all_exps if "2023-02-27" <= e <= today]
    print(f"  {len(trading_days)} trading days")

    CHUNKS_DIR.mkdir(exist_ok=True)

    print("Checking existing data...")
    existing_dates = get_existing_dates()

    remaining = [d for d in trading_days if d not in existing_dates]

    if args.days > 0:
        remaining = remaining[-args.days:]

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

        spx_close = price_map.get(trade_date)
        if spx_close is None:
            prev_prices = [(d, p) for d, p in price_map.items() if d <= trade_date]
            if prev_prices:
                spx_close = sorted(prev_prices)[-1][1]
            else:
                errors.append((trade_date, "No SPX price"))
                continue

        atm = round(spx_close / 5) * 5
        strikes = list(range(int(atm - 250), int(atm + 255), 5))
        contracts = [(trade_date, s, r) for s in strikes for r in ["call", "put"]]

        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date} | ATM={atm} | {len(contracts)} contracts...")
        sys.stdout.flush()

        day_rows = 0
        day_errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch_greeks, date_str, strike, right): (date_str, strike, right)
                for date_str, strike, right in contracts
            }
            for future in as_completed(futures):
                date_str, strike, right = futures[future]
                try:
                    raw = future.result()
                    records = parse_greeks(raw, date_str, strike, right)
                    batch_rows.extend(records)
                    day_rows += len(records)
                except Exception as e:
                    errors.append((date_str, strike, right, str(e)))
                    day_errors += 1

        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date} | ATM={atm} | {day_rows} rows ({day_errors} errors)    \n")
        sys.stdout.flush()

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
            if batch_rows:
                new_df = pd.DataFrame(batch_rows)
                new_df["date"] = pd.to_datetime(new_df["date"])
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                chunk_file = CHUNKS_DIR / f"chunk_{chunk_num:04d}.parquet"
                new_df.to_parquet(chunk_file, index=False)
                total_new += len(batch_rows)
                print(f"  [chunk saved: {len(batch_rows):,} rows — {total_new:,} new this run]")
                del new_df
                batch_rows.clear()
                chunk_num += 1
                gc.collect()

    merge_chunks()

    outfile = RAW_V1_2 / "spxw_0dte_intraday_greeks.parquet"
    print(f"\n{'='*60}")
    final_meta = pq.read_table(outfile, columns=["date"])
    print(f"Final: {final_meta.num_rows:,} rows, {len(final_meta.column('date').to_pandas().unique())} dates")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
