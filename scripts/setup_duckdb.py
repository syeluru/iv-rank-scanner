"""
setup_duckdb.py — Initialize DuckDB from all parquet files.

Creates: data/zero_dte.duckdb
Tables:
  - spxw_intraday_greeks  (main, ~25MB+ and growing)
  - spxw_eod              (EOD option chain)
  - spxw_oi               (open interest)
  - spxw_term_structure   (IV term structure)
  - spx_daily             (SPX OHLCV)
  - vix_daily             (VIX)
  - spy_1min              (SPY intraday)
  - econ_calendar         (economic events calendar)
  - econ_events           (FOMC, CPI, etc.)
  - fomc_dates
  - mag7_earnings
  - options_features      (engineered features)
  - spy_features
  - model_table           (ML-ready table)
  - target                (labels)

Run again at any time to refresh all tables from latest parquets.
"""

import duckdb
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH  = DATA_DIR / "zero_dte.duckdb"

TABLES = {
    "spxw_intraday_greeks": "spxw_0dte_intraday_greeks.parquet",
    "spxw_eod":             "spxw_0dte_eod.parquet",
    "spxw_oi":              "spxw_0dte_oi.parquet",
    "spxw_term_structure":  "spxw_term_structure.parquet",
    "spx_daily":            "spx_daily.parquet",
    "vix_daily":            "vix_daily.parquet",
    "spy_1min":             "spy_1min.parquet",
    "econ_calendar":        "econ_calendar.parquet",
    "econ_events":          "econ_events.parquet",
    "fomc_dates":           "fomc_dates.parquet",
    "mag7_earnings":        "mag7_earnings.parquet",
    "options_features":     "options_features.parquet",
    "spy_features":         "spy_features.parquet",
    "model_table":          "model_table.parquet",
    "target":               "target.parquet",
}

def main():
    print(f"Connecting to {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    # Validate table names are safe identifiers (alphanumeric + underscore only)
    _valid_id = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    for table in TABLES:
        if not _valid_id.match(table):
            raise ValueError(f"Invalid table name: {table!r}")

    for table, parquet_file in TABLES.items():
        path = DATA_DIR / parquet_file
        if not path.exists():
            print(f"  SKIP  {table} — {parquet_file} not found")
            continue

        print(f"  LOAD  {table} <- {parquet_file} ({path.stat().st_size // 1024}KB)")
        con.execute(f'DROP TABLE IF EXISTS "{table}"')
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_parquet(?)', [str(path)])
        count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"         {count:,} rows")

    print("\nDone. Summary:")
    for row in con.execute("SHOW TABLES").fetchall():
        tname = row[0]
        count = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
        print(f"  {tname:<30} {count:>12,} rows")

    con.close()
    size_mb = DB_PATH.stat().st_size / 1_000_000
    print(f"\nDatabase: {DB_PATH}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
