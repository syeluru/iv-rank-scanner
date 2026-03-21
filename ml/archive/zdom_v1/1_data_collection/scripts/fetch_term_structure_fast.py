"""
Fast parallel version of fetch_term_structure.py.
Uses ThreadPoolExecutor to fire multiple requests concurrently.
Resumes from existing spxw_term_structure.parquet.
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

BASE_URL = "http://127.0.0.1:25503"
MAX_WORKERS = 12  # concurrent requests to ThetaData
STRIKE_OFFSETS = [-250, -200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150, 200, 250]


def round_strike(price, step=5):
    return round(price / step) * step


def fetch_one(strike, right, date):
    """Fetch EOD across all expirations for a single strike+right+date."""
    try:
        resp = requests.get(f"{BASE_URL}/v3/option/history/eod", params={
            "symbol": "SPXW",
            "expiration": "*",
            "strike": str(int(strike)),
            "right": right,
            "start_date": date,
            "end_date": date,
            "format": "json",
        }, timeout=60)
        resp.raise_for_status()
        return (strike, right, date, resp.json()["response"], None)
    except Exception as e:
        return (strike, right, date, None, str(e))


def process_day(trade_date, spx_close, existing_strikes):
    """Fetch all needed strikes for one day using thread pool."""
    atm = round_strike(spx_close)
    strikes = [atm + offset for offset in STRIKE_OFFSETS]
    needed = [s for s in strikes if s not in existing_strikes]

    if not needed:
        return atm, 0, []

    # Build all (strike, right) pairs
    tasks = [(s, r, trade_date) for s in needed for r in ["call", "put"]]

    rows = []
    errors = []
    trade_ts = pd.Timestamp(trade_date)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, s, r, d): (s, r) for s, r, d in tasks}
        for future in as_completed(futures):
            strike, right, date, contracts, err = future.result()
            if err:
                errors.append(err)
                continue
            if contracts is None:
                continue
            for contract in contracts:
                exp = contract["contract"]["expiration"]
                exp_ts = pd.Timestamp(exp)
                dte = (exp_ts - trade_ts).days
                if dte < 0:
                    continue
                for d in contract["data"]:
                    rows.append({
                        "date": trade_date,
                        "expiration": exp,
                        "dte": dte,
                        "strike": strike,
                        "right": right,
                        "spx_close": spx_close,
                        "atm_strike": atm,
                        "moneyness": (strike - spx_close) / spx_close,
                        "bid": d["bid"],
                        "ask": d["ask"],
                        "mid": (d["bid"] + d["ask"]) / 2,
                        "volume": d["volume"],
                    })

    return atm, len(needed), rows


def main():
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    spx["date_str"] = spx["date"].dt.strftime("%Y-%m-%d")
    price_map = dict(zip(spx["date_str"], spx["spx_close"]))

    print("Getting SPXW expirations for trading day list...")
    resp = requests.get(f"{BASE_URL}/v3/option/list/expirations", params={
        "symbol": "SPXW", "format": "json"
    }, timeout=30)
    resp.raise_for_status()
    all_exps = [e["expiration"] for e in resp.json()["response"]]
    today = datetime.now().strftime("%Y-%m-%d")
    trading_days = [e for e in all_exps if "2023-02-27" <= e <= today]
    print(f"  {len(trading_days)} trading days")

    outfile = DATA_DIR / "spxw_term_structure.parquet"
    all_rows = []
    complete_dates = set()
    existing_strikes_by_date = {}

    if outfile.exists():
        print("  Loading existing file for resume...")
        existing_df = pd.read_parquet(outfile)
        all_rows = existing_df.to_dict("records")

        for dt, grp in existing_df.groupby(existing_df["date"].astype(str)):
            strikes = set(grp["strike"].unique())
            existing_strikes_by_date[dt] = strikes
            if len(strikes) >= 15:
                complete_dates.add(dt)

        print(f"  {len(complete_dates)} complete, {len(existing_strikes_by_date) - len(complete_dates)} partial")

    remaining = [d for d in trading_days if d not in complete_dates]
    print(f"  Remaining: {len(remaining)} days")
    print(f"  Using {MAX_WORKERS} parallel workers\n")

    if not remaining:
        print("All done!")
        return

    total = len(remaining)
    errors = []
    save_every = 15
    t0 = time.time()

    for i, trade_date in enumerate(remaining):
        spx_close = price_map.get(trade_date)
        if spx_close is None:
            prev_prices = [(d, p) for d, p in price_map.items() if d <= trade_date]
            if prev_prices:
                spx_close = sorted(prev_prices)[-1][1]
            else:
                errors.append((trade_date, "No SPX price"))
                continue

        existing = existing_strikes_by_date.get(trade_date, set())

        try:
            atm, n_needed, day_rows = process_day(trade_date, spx_close, existing)
            all_rows.extend(day_rows)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta_min = (total - i - 1) / rate if rate > 0 else 0

            pct = (i + 1) / total * 100
            sys.stdout.write(
                f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date}: ATM={atm}, "
                f"{n_needed} strikes, {len(day_rows)} rows  "
                f"[{rate:.1f} days/min, ETA {eta_min:.0f}m]    "
            )

        except Exception as e:
            errors.append((trade_date, str(e)))
            sys.stdout.write(f"\r[{i+1}/{total}] {trade_date}: ERROR - {str(e)[:60]}    ")
            time.sleep(2)

        # Checkpoint
        if (i + 1) % save_every == 0 or (i + 1) == total:
            df = pd.DataFrame(all_rows)
            if len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                df = df.drop_duplicates(subset=["date", "expiration", "strike", "right"])
                df.to_parquet(outfile, index=False)
                sys.stdout.write(f" [saved {len(df):,} rows]")
            sys.stdout.write("\n")

    print(f"\n{'='*60}")
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["date", "expiration", "strike", "right"])
    print(f"Term structure: {len(df):,} rows, {df['date'].nunique()} dates")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for d, err in errors[:10]:
            print(f"  {d}: {err}")

    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
