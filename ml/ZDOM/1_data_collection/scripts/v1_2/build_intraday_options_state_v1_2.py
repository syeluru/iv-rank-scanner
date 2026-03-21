"""
Build the canonical intraday decision-state options table (v1.2).

Row semantics:
  A row labeled decision_datetime=10:00:00 contains the completed 09:59:00
  minute-bucket state that would be known at the start of the 10:00 minute.

Pricing semantics:
  The v1.2 option pricing source is the minute-level greeks snapshot table.
  We use its bid, ask, and mid snapshot values as the option close-state
  pricing fields for the completed minute bucket.

Inputs:
  - raw/v1_2/spxw_0dte_intraday_greeks.parquet
  - raw/v1_2/spx_1min.parquet
  - raw/v1_2/vix_1min.parquet
  - raw/v1_2/vix1d_1min.parquet

Output:
  - data/v1_2/intraday_decision_state_v1_2.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "raw" / "v1_2"
OUT_DIR = PROJECT_DIR / "data" / "v1_2"

GREEKS_FILE = RAW_DIR / "spxw_0dte_intraday_greeks.parquet"
SPX_FILE = RAW_DIR / "spx_1min.parquet"
VIX_FILE = RAW_DIR / "vix_1min.parquet"
VIX1D_FILE = RAW_DIR / "vix1d_1min.parquet"
OUT_FILE = OUT_DIR / "intraday_decision_state_v1_2.parquet"

GREEK_COLS = [
    "date",
    "timestamp",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "implied_vol",
    "iv_error",
]


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def drop_weekends(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[dt_col])
    return df.loc[dt.dt.dayofweek < 5].copy()


def drop_all_zero_ohlc_rows(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    mask = (df[value_cols].fillna(0) == 0).all(axis=1)
    return df.loc[~mask].copy()


def load_index_table(path: Path, rename_map: dict[str, str], value_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["state_datetime", *rename_map.values()])

    df = pd.read_parquet(path)
    df = drop_weekends(df, "datetime")
    df = drop_all_zero_ohlc_rows(df, value_cols)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns=rename_map)
    df = df.rename(columns={"datetime": "state_datetime"})
    df = df.sort_values("state_datetime")
    return df


def list_greek_dates() -> list[pd.Timestamp]:
    dates = pd.read_parquet(GREEKS_FILE, columns=["date"])["date"]
    dates = pd.to_datetime(dates).dt.normalize().drop_duplicates().sort_values()
    return list(dates)


def load_greeks_for_date(trade_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(GREEKS_FILE, columns=GREEK_COLS, filters=[("date", "==", trade_date)])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = drop_weekends(df, "timestamp")
    df = df.loc[df["timestamp"].dt.time < pd.Timestamp("16:00:00").time()].copy()
    return df


def pivot_contract_sides(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in {"date", "state_datetime", "strike", "right"}]
    pivoted = (
        df.set_index(["date", "state_datetime", "strike", "right"])[value_cols]
        .unstack("right")
        .sort_index(axis=1)
    )
    pivoted.columns = [f"{side}_{field}" for field, side in pivoted.columns]
    pivoted = pivoted.reset_index()
    return pivoted


def build_day(trade_date: pd.Timestamp, spx: pd.DataFrame, vix: pd.DataFrame, vix1d: pd.DataFrame) -> pd.DataFrame:
    greeks = load_greeks_for_date(trade_date)
    if greeks.empty:
        return pd.DataFrame()

    greeks = greeks.rename(
        columns={
            "timestamp": "state_datetime",
            "bid": "bid_close",
            "ask": "ask_close",
            "mid": "mid_close",
        }
    )
    day = pivot_contract_sides(greeks)

    day = day.merge(spx, on="state_datetime", how="left")
    if not vix.empty:
        day = day.merge(vix, on="state_datetime", how="left")
    if not vix1d.empty:
        day = day.merge(vix1d, on="state_datetime", how="left")

    day["decision_datetime"] = day["state_datetime"] + pd.Timedelta(minutes=1)
    day["date"] = day["decision_datetime"].dt.normalize()

    keep = [
        "date",
        "decision_datetime",
        "state_datetime",
        "strike",
        "call_bid_close",
        "call_ask_close",
        "call_mid_close",
        "put_bid_close",
        "put_ask_close",
        "put_mid_close",
        "call_delta",
        "call_gamma",
        "call_theta",
        "call_vega",
        "call_rho",
        "call_implied_vol",
        "call_iv_error",
        "put_delta",
        "put_gamma",
        "put_theta",
        "put_vega",
        "put_rho",
        "put_implied_vol",
        "put_iv_error",
        "spx_open",
        "spx_high",
        "spx_low",
        "spx_close",
        "vix_open",
        "vix_high",
        "vix_low",
        "vix_close",
        "vix1d_open",
        "vix1d_high",
        "vix1d_low",
        "vix1d_close",
    ]

    for col in keep:
        if col not in day.columns:
            day[col] = pd.NA

    day = day[keep].sort_values(["decision_datetime", "strike"]).reset_index(drop=True)
    numeric_cols = [c for c in keep if c not in {"date", "decision_datetime", "state_datetime"}]
    for col in numeric_cols:
        day[col] = pd.to_numeric(day[col], errors="coerce")
    return day


def write_days(days: list[pd.Timestamp], spx: pd.DataFrame, vix: pd.DataFrame, vix1d: pd.DataFrame) -> None:
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    if OUT_FILE.exists():
        OUT_FILE.unlink()

    for i, trade_date in enumerate(days, start=1):
        day = build_day(trade_date, spx, vix, vix1d)
        if day.empty:
            print(f"[{i}/{len(days)}] {trade_date.date()} empty")
            continue

        table = pa.Table.from_pandas(day, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_FILE, table.schema)
        writer.write_table(table)
        total_rows += len(day)

        if i % 25 == 0 or i == len(days):
            print(f"[{i}/{len(days)}] {trade_date.date()} rows={len(day):,} total={total_rows:,}")

    if writer is not None:
        writer.close()

    print(f"\nWrote {total_rows:,} rows to {OUT_FILE}")


def main() -> None:
    for path in [GREEKS_FILE, SPX_FILE]:
        require_exists(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spx = load_index_table(
        SPX_FILE,
        {"open": "spx_open", "high": "spx_high", "low": "spx_low", "close": "spx_close"},
        ["open", "high", "low", "close"],
    )
    vix = load_index_table(
        VIX_FILE,
        {},
        ["vix_open", "vix_high", "vix_low", "vix_close"],
    )
    vix1d = load_index_table(
        VIX1D_FILE,
        {},
        ["vix1d_open", "vix1d_high", "vix1d_low", "vix1d_close"],
    )

    days = list_greek_dates()

    print("v1.2 canonical intraday decision-state build")
    print(f"  Greeks:  {GREEKS_FILE}")
    print(f"  SPX:     {SPX_FILE}")
    print(f"  VIX:     {VIX_FILE} {'(loaded)' if not vix.empty else '(missing/empty)'}")
    print(f"  VIX1D:   {VIX1D_FILE} {'(loaded)' if not vix1d.empty else '(missing/empty)'}")
    print(f"  Output:  {OUT_FILE}")
    print(f"  Trade dates: {len(days)}")
    print()

    write_days(days, spx, vix, vix1d)


if __name__ == "__main__":
    main()
