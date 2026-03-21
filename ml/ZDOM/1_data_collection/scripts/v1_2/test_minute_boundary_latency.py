"""
Empirical test: ThetaData minute-boundary data availability for v1_2 intraday design.

Tests whether the completed prior-minute row is available immediately at the next
minute boundary, or whether there is a delay.

Polls ThetaData at T+0s, T+1s, T+2s, T+3s, T+5s, T+10s, T+15s, T+20s, T+30s
after each target minute boundary.

Tests these endpoints:
  - option/history/greeks/second_order?interval=1m  (greeks + bid/ask)
  - option/history/quote?interval=1m                (NBBO quote snapshot)
  - index/history/ohlc?interval=1m                  (SPX bars)

Writes results to /tmp/thetadata_latency_test.csv

Run with: python3 test_minute_boundary_latency.py
"""

import csv
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

BASE_URL = "http://127.0.0.1:25503"
TODAY = datetime.now().strftime("%Y%m%d")
OUTFILE = Path("/tmp/thetadata_latency_test.csv")

# Test windows: minute boundaries we want to probe
# Format: (target_minute_ET, description)
TEST_MINUTES = [
    ("09:31", "first bar availability (09:30 completed)"),
    ("09:32", "second bar (09:31 completed)"),
    ("09:35", "early session (09:34 completed)"),
    ("09:45", "mid warmup (09:44 completed)"),
    ("10:00", "entry window start (09:59 completed)"),
    ("10:01", "entry window (10:00 completed)"),
    ("10:15", "mid morning (10:14 completed)"),
    ("10:30", "late morning (10:29 completed)"),
]

# Poll offsets after the minute boundary (seconds)
POLL_OFFSETS = [0, 1, 2, 3, 5, 10, 15, 20, 30]


def get_atm_strike():
    """Get today's ATM strike from SPX opening price."""
    resp = requests.get(f"{BASE_URL}/v3/index/history/ohlc", params={
        "symbol": "SPX", "start_date": TODAY, "end_date": TODAY, "interval": "1m",
    }, timeout=10)
    df = pd.read_csv(StringIO(resp.text))
    if len(df) == 0:
        return 5800  # fallback
    first_open = df.iloc[0]["open"]
    if first_open == 0:
        return 5800
    return round(first_open / 5) * 5


def probe_greeks(strike, right, target_minute):
    """Query greeks endpoint and check if target_minute row exists."""
    try:
        resp = requests.get(f"{BASE_URL}/v3/option/history/greeks/second_order", params={
            "symbol": "SPXW", "expiration": TODAY, "strike": str(int(strike)),
            "right": right, "start_date": TODAY, "end_date": TODAY, "interval": "1m",
        }, timeout=5)
        if resp.status_code != 200:
            return {"available": False, "error": f"HTTP {resp.status_code}", "rows": 0}
        df = pd.read_csv(StringIO(resp.text))
        if len(df) == 0:
            return {"available": False, "error": "empty", "rows": 0}

        # Check if the target minute row exists
        target_ts = f"{TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:8]}T{target_minute}:00.000"
        has_row = any(str(t).startswith(target_ts[:19]) for t in df["timestamp"])
        last_ts = df["timestamp"].iloc[-1] if len(df) > 0 else "none"

        # Get the target row values if present
        row_data = {}
        if has_row:
            match = df[df["timestamp"].str.startswith(target_ts[:19])]
            if len(match) > 0:
                r = match.iloc[0]
                row_data = {
                    "bid": r.get("bid", None),
                    "ask": r.get("ask", None),
                    "delta": r.get("delta", None),
                    "implied_vol": r.get("implied_vol", None),
                }

        return {
            "available": has_row,
            "last_ts": str(last_ts),
            "total_rows": len(df),
            **row_data,
        }
    except Exception as e:
        return {"available": False, "error": str(e)[:80], "rows": 0}


def probe_quote(strike, right, target_minute):
    """Query quote endpoint and check if target_minute row exists."""
    try:
        resp = requests.get(f"{BASE_URL}/v3/option/history/quote", params={
            "symbol": "SPXW", "expiration": TODAY, "strike": str(int(strike)),
            "right": right, "start_date": TODAY, "end_date": TODAY, "interval": "1m",
        }, timeout=5)
        if resp.status_code != 200:
            return {"available": False, "error": f"HTTP {resp.status_code}"}
        df = pd.read_csv(StringIO(resp.text))
        if len(df) == 0:
            return {"available": False, "error": "empty"}

        target_ts = f"{TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:8]}T{target_minute}:00.000"
        has_row = any(str(t).startswith(target_ts[:19]) for t in df["timestamp"])
        last_ts = df["timestamp"].iloc[-1]

        row_data = {}
        if has_row:
            match = df[df["timestamp"].str.startswith(target_ts[:19])]
            if len(match) > 0:
                r = match.iloc[0]
                row_data = {"bid": r.get("bid", None), "ask": r.get("ask", None)}

        return {"available": has_row, "last_ts": str(last_ts), "total_rows": len(df), **row_data}
    except Exception as e:
        return {"available": False, "error": str(e)[:80]}


def probe_spx(target_minute):
    """Query SPX index bars and check if target_minute row exists."""
    try:
        resp = requests.get(f"{BASE_URL}/v3/index/history/ohlc", params={
            "symbol": "SPX", "start_date": TODAY, "end_date": TODAY, "interval": "1m",
        }, timeout=5)
        df = pd.read_csv(StringIO(resp.text))
        if len(df) == 0:
            return {"available": False, "error": "empty"}

        target_ts = f"{TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:8]}T{target_minute}:00.000"
        has_row = any(str(t).startswith(target_ts[:19]) for t in df["timestamp"])
        last_ts = df["timestamp"].iloc[-1]
        last_close = df["close"].iloc[-1]

        return {"available": has_row, "last_ts": str(last_ts), "last_close": last_close, "total_rows": len(df)}
    except Exception as e:
        return {"available": False, "error": str(e)[:80]}


def wait_until(target_dt):
    """Sleep until target datetime."""
    now = datetime.now()
    delta = (target_dt - now).total_seconds()
    if delta > 0:
        print(f"  Waiting {delta:.1f}s until {target_dt.strftime('%H:%M:%S')}...")
        time.sleep(delta)


def run_test():
    print("=" * 70)
    print("ThetaData Minute-Boundary Latency Test")
    print(f"Date: {TODAY}")
    print(f"Local time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    # Wait for market to be open enough to get ATM
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=30, microsecond=0)
    if now < market_open:
        print(f"\nMarket not open yet. Waiting until 09:30:30...")
        wait_until(market_open)

    atm = get_atm_strike()
    print(f"\nATM strike: {atm}")
    print(f"Testing call at {atm}")
    print()

    results = []

    for target_minute, desc in TEST_MINUTES:
        hour, minute = map(int, target_minute.split(":"))
        boundary_dt = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

        # The completed bar we're looking for
        prior_minute = (boundary_dt - timedelta(minutes=1)).strftime("%H:%M")

        # Skip if this boundary already passed
        if datetime.now() > boundary_dt + timedelta(seconds=35):
            print(f"[{target_minute}] Already passed, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"[{target_minute}] Testing: {desc}")
        print(f"  Looking for completed {prior_minute}:00 row")
        print(f"  Boundary: {boundary_dt.strftime('%H:%M:%S')}")

        for offset in POLL_OFFSETS:
            poll_dt = boundary_dt + timedelta(seconds=offset)
            wait_until(poll_dt)

            poll_time = datetime.now()
            poll_ts = poll_time.strftime("%H:%M:%S.%f")[:12]

            # Probe all three endpoints
            greeks_result = probe_greeks(atm, "call", prior_minute)
            quote_result = probe_quote(atm, "call", prior_minute)
            spx_result = probe_spx(prior_minute)

            elapsed = (datetime.now() - poll_time).total_seconds()

            row = {
                "target_boundary": target_minute,
                "prior_minute": prior_minute,
                "poll_offset_s": offset,
                "poll_local_time": poll_ts,
                "greeks_available": greeks_result.get("available", False),
                "greeks_last_ts": greeks_result.get("last_ts", ""),
                "greeks_bid": greeks_result.get("bid", ""),
                "greeks_ask": greeks_result.get("ask", ""),
                "greeks_delta": greeks_result.get("delta", ""),
                "greeks_iv": greeks_result.get("implied_vol", ""),
                "quote_available": quote_result.get("available", False),
                "quote_last_ts": quote_result.get("last_ts", ""),
                "quote_bid": quote_result.get("bid", ""),
                "quote_ask": quote_result.get("ask", ""),
                "spx_available": spx_result.get("available", False),
                "spx_last_ts": spx_result.get("last_ts", ""),
                "spx_last_close": spx_result.get("last_close", ""),
                "query_time_s": round(elapsed, 3),
            }
            results.append(row)

            g = "✓" if greeks_result["available"] else "✗"
            q = "✓" if quote_result["available"] else "✗"
            s = "✓" if spx_result["available"] else "✗"

            print(f"  T+{offset:2d}s [{poll_ts}]  greeks={g}  quote={q}  spx={s}  "
                  f"last_greeks={greeks_result.get('last_ts','')[-12:]}")

    # Write results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTFILE, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to {OUTFILE}")
        print(f"Total probes: {len(results)}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY: First offset where prior-minute row is available")
        print(f"{'Boundary':<12} {'Prior Min':<12} {'Greeks':>8} {'Quote':>8} {'SPX':>8}")
        print("-" * 52)

        for target_minute, desc in TEST_MINUTES:
            sub = df[df["target_boundary"] == target_minute]
            if sub.empty:
                continue

            def first_avail(col):
                avail = sub[sub[col] == True]
                if avail.empty:
                    return "never"
                return f"T+{int(avail.iloc[0]['poll_offset_s'])}s"

            print(f"{target_minute:<12} {sub.iloc[0]['prior_minute']:<12} "
                  f"{first_avail('greeks_available'):>8} "
                  f"{first_avail('quote_available'):>8} "
                  f"{first_avail('spx_available'):>8}")

        # Check for value revisions
        print(f"\nVALUE STABILITY CHECK:")
        for target_minute, desc in TEST_MINUTES:
            sub = df[(df["target_boundary"] == target_minute) & (df["greeks_available"] == True)]
            if len(sub) < 2:
                continue
            bids = sub["greeks_bid"].unique()
            asks = sub["greeks_ask"].unique()
            deltas = sub["greeks_delta"].unique()
            changed = len(bids) > 1 or len(asks) > 1 or len(deltas) > 1
            status = "REVISED" if changed else "STABLE"
            print(f"  {target_minute}: {status} (bids={list(bids)}, deltas={list(deltas)[:3]})")


if __name__ == "__main__":
    run_test()
