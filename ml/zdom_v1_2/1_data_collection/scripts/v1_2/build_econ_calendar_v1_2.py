"""
Build econ_calendar_v1_2.parquet — economic event day flags + days-to-next features.

Source: Bootstrapped from v1_1 static event tables:
  - raw/v1_1/econ_calendar.parquet (CPI, NFP, GDP, PPI dates)
  - raw/v1_1/fomc_dates.parquet (FOMC meeting dates)

These are hardcoded/curated event calendars, not API-fetched.
If the underlying event dates change or extend, update the v1_1 source
files first, then re-run this script.

Output: data/v1_2/econ_calendar_v1_2.parquet
  One row per business day, keyed by date.
  Coverage: 2022-06-06 through 2026-03-13.

Columns:
  date, is_cpi_day, is_nfp_day, is_gdp_day, is_ppi_day,
  days_to_next_cpi, days_to_next_nfp, days_to_next_gdp,
  days_to_next_ppi, days_to_next_fomc, days_to_next_any_econ

Join lag: same-day (calendar features describe the decision date itself).
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
DATA_V1_2 = BASE / "data" / "v1_2"

START = pd.Timestamp("2022-06-06")
CLIP = pd.Timestamp("2026-03-13")


def main():
    DATA_V1_2.mkdir(parents=True, exist_ok=True)

    econ = pd.read_parquet(RAW_V1_1 / "econ_calendar.parquet")
    econ["date"] = pd.to_datetime(econ["date"])

    fomc = pd.read_parquet(RAW_V1_1 / "fomc_dates.parquet")
    fomc["date"] = pd.to_datetime(fomc["date"])

    cpi_dates = set(econ[econ["event"] == "CPI"]["date"])
    nfp_dates = set(econ[econ["event"] == "NFP"]["date"])
    gdp_dates = set(econ[econ["event"] == "GDP"]["date"])
    ppi_dates = set(econ[econ["event"] == "PPI"]["date"])
    fomc_dates = set(fomc["date"])
    all_econ = cpi_dates | nfp_dates | gdp_dates | ppi_dates | fomc_dates

    event_sets = [
        ("cpi", cpi_dates),
        ("nfp", nfp_dates),
        ("gdp", gdp_dates),
        ("ppi", ppi_dates),
        ("fomc", fomc_dates),
        ("any_econ", all_econ),
    ]

    all_bdays = pd.bdate_range(START, CLIP)
    rows = []
    for d in all_bdays:
        row = {
            "date": d,
            "is_cpi_day": int(d in cpi_dates),
            "is_nfp_day": int(d in nfp_dates),
            "is_gdp_day": int(d in gdp_dates),
            "is_ppi_day": int(d in ppi_dates),
        }
        for name, date_set in event_sets:
            future = [x for x in date_set if x > d]
            row[f"days_to_next_{name}"] = (min(future) - d).days if future else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / "econ_calendar_v1_2.parquet"
    df.to_parquet(out, index=False)

    print(f"econ_calendar_v1_2.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Nulls: {dict(df.isna().sum())}")
    print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
