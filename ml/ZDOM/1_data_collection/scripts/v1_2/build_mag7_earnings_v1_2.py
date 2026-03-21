"""
Build mag7_earnings_v1_2.parquet — Mag7 per-ticker earnings day flags.

Source: Bootstrapped from v1_1 static earnings table:
  - raw/v1_1/mag7_earnings.parquet (symbol, date, time_of_day)

This is a hardcoded/curated earnings calendar. If future earnings dates
are added, update the v1_1 source file first, then re-run this script.

Output: data/v1_2/mag7_earnings_v1_2.parquet
  One row per business day, keyed by date.
  Coverage: 2022-06-06 through 2026-03-13.

Columns:
  date, is_earnings_aapl, is_earnings_amzn, is_earnings_googl,
  is_earnings_meta, is_earnings_msft, is_earnings_nvda, is_earnings_tsla,
  mag7_earnings_count, is_mag7_earnings_day

Join lag: same-day (calendar features describe the decision date itself).
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
DATA_V1_2 = BASE / "data" / "v1_2"

START = pd.Timestamp("2022-06-06")
CLIP = pd.Timestamp("2026-03-13")

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


def main():
    DATA_V1_2.mkdir(parents=True, exist_ok=True)

    mag = pd.read_parquet(RAW_V1_1 / "mag7_earnings.parquet")
    mag["date"] = pd.to_datetime(mag["date"])

    earn_dates = {}
    for t in TICKERS:
        earn_dates[t] = set(mag[mag["symbol"] == t]["date"])

    all_bdays = pd.bdate_range(START, CLIP)
    rows = []
    for d in all_bdays:
        row = {"date": d}
        count = 0
        for t in TICKERS:
            is_earn = int(d in earn_dates[t])
            row[f"is_earnings_{t.lower()}"] = is_earn
            count += is_earn
        row["mag7_earnings_count"] = count
        row["is_mag7_earnings_day"] = int(count > 0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / "mag7_earnings_v1_2.parquet"
    df.to_parquet(out, index=False)

    print(f"mag7_earnings_v1_2.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Earnings days: {df['is_mag7_earnings_day'].sum()}")
    for t in TICKERS:
        print(f"    {t}: {df[f'is_earnings_{t.lower()}'].sum()}")
    print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
