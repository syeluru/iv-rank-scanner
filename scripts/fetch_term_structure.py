"""
Fetch SPXW option EOD at multiple expirations for IV term structure.
For each trading day, finds ATM strike and nearby strikes, then pulls
all expirations at once using wildcard expiration (1 request per strike+right).

Strategy: For 3 strikes (ATM-25, ATM, ATM+25) × 2 rights = 6 requests/day.
Each request returns all expirations, giving us the full term structure.

Saves to: data/spxw_term_structure.parquet

Requires Theta Terminal running on localhost:25503.
Must run AFTER fetch_spx_daily.py (needs SPX close prices for ATM strike).
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

BASE_URL = "http://127.0.0.1:25503"
DELAY = 0.8  # slightly more conservative


def round_strike(price, step=5):
    """Round price to nearest strike increment."""
    return round(price / step) * step


def fetch_all_expirations_for_strike(strike, right, date):
    """Fetch EOD across ALL expirations for a single strike+right.
    Uses wildcard expiration — one request returns ~30-40 expirations."""
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
    return resp.json()["response"]


def main():
    # Load SPX daily prices for ATM strike reference
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    spx["date_str"] = spx["date"].dt.strftime("%Y-%m-%d")
    price_map = dict(zip(spx["date_str"], spx["spx_close"]))

    print("Getting SPXW expirations for trading day list...")
    resp = requests.get(f"{BASE_URL}/v3/option/list/expirations", params={
        "symbol": "SPXW", "format": "json"
    }, timeout=30)
    resp.raise_for_status()
    all_exps = [e["expiration"] for e in resp.json()["response"]]
    trading_days = [e for e in all_exps if "2023-02-27" <= e <= "2026-02-27"]
    print(f"  {len(trading_days)} trading days")

    # Check for resume
    outfile = DATA_DIR / "spxw_term_structure.parquet"
    existing_dates = set()
    all_rows = []

    if outfile.exists():
        print("  Found existing file, loading for resume...")
        existing_df = pd.read_parquet(outfile)
        existing_dates = set(existing_df["date"].astype(str).unique())
        all_rows = existing_df.to_dict("records")
        print(f"  Already have {len(existing_dates)} dates")

    remaining = [d for d in trading_days if d not in existing_dates]
    print(f"  Remaining: {len(remaining)} days\n")

    if not remaining:
        print("All dates already fetched!")
        return

    total = len(remaining)
    errors = []
    save_every = 25

    for i, trade_date in enumerate(remaining):
        pct = (i + 1) / total * 100
        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date}...")
        sys.stdout.flush()

        # Get SPX close for ATM reference
        spx_close = price_map.get(trade_date)
        if spx_close is None:
            prev_prices = [(d, p) for d, p in price_map.items() if d <= trade_date]
            if prev_prices:
                spx_close = sorted(prev_prices)[-1][1]
            else:
                errors.append((trade_date, "No SPX price"))
                continue

        atm = round_strike(spx_close)
        strikes = [atm - 25, atm, atm + 25]

        try:
            day_rows = 0
            for strike in strikes:
                for right in ["call", "put"]:
                    contracts = fetch_all_expirations_for_strike(strike, right, trade_date)
                    time.sleep(DELAY)

                    trade_ts = pd.Timestamp(trade_date)
                    for contract in contracts:
                        exp = contract["contract"]["expiration"]
                        exp_ts = pd.Timestamp(exp)
                        dte = (exp_ts - trade_ts).days
                        if dte < 0:
                            continue

                        for d in contract["data"]:
                            all_rows.append({
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
                            day_rows += 1

            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date}: ATM={atm}, {day_rows} rows    ")

        except Exception as e:
            errors.append((trade_date, str(e)))
            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {trade_date}: ERROR - {str(e)[:50]}    ")
            time.sleep(3)

        # Checkpoint
        if (i + 1) % save_every == 0 or (i + 1) == total:
            df = pd.DataFrame(all_rows)
            if len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                df.to_parquet(outfile, index=False)
                sys.stdout.write(f" [saved {len(df)} rows]")

    print(f"\n\n{'='*60}")
    df = pd.DataFrame(all_rows)
    print(f"Term structure data: {len(df):,} rows, {df['date'].nunique()} dates")
    print(f"DTE range: {df['dte'].min()} to {df['dte'].max()}")
    print(f"Strikes: {sorted(df['strike'].unique())[:10]}...")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for exp, err in errors[:10]:
            print(f"  {exp}: {err}")

    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
