"""
Build a single merged table: SPY 1-min as the base,
left joined with VIX daily, econ calendar, and MAG7 earnings on date.

Output: data/spy_merged.parquet
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def main():
    # ── Load all data ──
    print("Loading data...")
    spy = pd.read_parquet(DATA_DIR / "spy_1min.parquet")
    vix = pd.read_parquet(DATA_DIR / "vix_daily.parquet")
    econ = pd.read_parquet(DATA_DIR / "econ_calendar.parquet")
    mag7 = pd.read_parquet(DATA_DIR / "mag7_earnings.parquet")

    print(f"  SPY 1-min:      {spy.shape}")
    print(f"  VIX daily:      {vix.shape}")
    print(f"  Econ calendar:  {econ.shape}")
    print(f"  MAG7 earnings:  {mag7.shape}")

    # ── Ensure date columns are the same type ──
    spy["date"] = pd.to_datetime(spy["date"])
    vix["date"] = pd.to_datetime(vix["date"])
    econ["date"] = pd.to_datetime(econ["date"])
    mag7["date"] = pd.to_datetime(mag7["date"])

    # ── VIX: straight left join on date ──
    print("\nJoining VIX daily...")
    merged = spy.merge(vix, on="date", how="left")
    print(f"  After VIX join: {merged.shape}")

    # ── Econ calendar: pivot to one-hot flags per event type ──
    print("Joining econ calendar...")
    econ_pivot = econ.assign(flag=1).pivot_table(
        index="date", columns="event", values="flag", fill_value=0
    ).reset_index()
    econ_pivot.columns = ["date"] + [f"is_{col.lower()}_day" for col in econ_pivot.columns[1:]]

    merged = merged.merge(econ_pivot, on="date", how="left")
    # Fill NaN flags with 0 (days with no events)
    flag_cols = [c for c in merged.columns if c.startswith("is_") and c.endswith("_day")]
    merged[flag_cols] = merged[flag_cols].fillna(0).astype(int)
    print(f"  After econ join: {merged.shape}")

    # ── MAG7 earnings: count + per-stock binary flags ──
    print("Joining MAG7 earnings...")
    mag7_daily = mag7.groupby("date").agg(
        mag7_earnings_count=("symbol", "count"),
        mag7_earnings_symbols=("symbol", lambda x: ",".join(sorted(x))),
    ).reset_index()
    mag7_daily["is_mag7_earnings_day"] = 1

    # Per-stock binary flags
    mag7_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    mag7_ohe = mag7.assign(flag=1).pivot_table(
        index="date", columns="symbol", values="flag", fill_value=0
    ).reset_index()
    mag7_ohe.columns = ["date"] + [f"is_earnings_{col.lower()}" for col in mag7_ohe.columns[1:]]

    mag7_daily = mag7_daily.merge(mag7_ohe, on="date", how="left")

    merged = merged.merge(mag7_daily, on="date", how="left")
    merged["is_mag7_earnings_day"] = merged["is_mag7_earnings_day"].fillna(0).astype(int)
    merged["mag7_earnings_count"] = merged["mag7_earnings_count"].fillna(0).astype(int)
    merged["mag7_earnings_symbols"] = merged["mag7_earnings_symbols"].fillna("")
    for stock in mag7_stocks:
        col = f"is_earnings_{stock.lower()}"
        merged[col] = merged[col].fillna(0).astype(int)
    print(f"  After MAG7 join: {merged.shape}")

    # ── Days to next event (resets to 0 on event day) ──
    print("Computing days-to-next-event features...")
    # Load FOMC dates too
    fomc = pd.read_parquet(DATA_DIR / "fomc_dates.parquet")
    fomc["date"] = pd.to_datetime(fomc["date"])

    trading_dates = merged["date"].drop_duplicates().sort_values().reset_index(drop=True)

    event_sources = {
        "cpi": pd.to_datetime(econ[econ["event"] == "CPI"]["date"]).values,
        "ppi": pd.to_datetime(econ[econ["event"] == "PPI"]["date"]).values,
        "nfp": pd.to_datetime(econ[econ["event"] == "NFP"]["date"]).values,
        "gdp": pd.to_datetime(econ[econ["event"] == "GDP"]["date"]).values,
        "fomc": pd.to_datetime(fomc["date"]).values,
    }

    days_to_dfs = []
    for event_name, event_dates in event_sources.items():
        col_name = f"days_to_next_{event_name}"
        event_dates_sorted = sorted(event_dates)
        vals = []
        for td in trading_dates:
            future = [e for e in event_dates_sorted if e >= td]
            if future:
                delta = (pd.Timestamp(future[0]) - pd.Timestamp(td)).days
            else:
                delta = -1  # no future event in our data
            vals.append(delta)
        days_to_dfs.append(pd.DataFrame({"date": trading_dates, col_name: vals}))

    # days_to_next_any_econ: nearest of any type
    all_econ_dates = sorted(set(
        list(event_sources["cpi"]) + list(event_sources["ppi"]) +
        list(event_sources["nfp"]) + list(event_sources["gdp"]) +
        list(event_sources["fomc"])
    ))
    vals = []
    for td in trading_dates:
        future = [e for e in all_econ_dates if e >= td]
        if future:
            delta = (pd.Timestamp(future[0]) - pd.Timestamp(td)).days
        else:
            delta = -1
        vals.append(delta)
    days_to_dfs.append(pd.DataFrame({"date": trading_dates, "days_to_next_any_econ": vals}))

    # Merge all days-to columns
    for dtdf in days_to_dfs:
        merged = merged.merge(dtdf, on="date", how="left")

    print(f"  After days-to-event: {merged.shape}")

    # ── Sort and save ──
    merged = merged.sort_values("datetime").reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Final merged table: {merged.shape}")
    print(f"Columns ({len(merged.columns)}):")
    for col in merged.columns:
        null_pct = merged[col].isna().mean() * 100
        dtype = merged[col].dtype
        print(f"  {col:30s} {str(dtype):15s} nulls: {null_pct:.1f}%")

    print(f"\nSample row (first row):")
    print(merged.iloc[0].to_string())

    print(f"\nEvent day counts:")
    for col in flag_cols + ["is_mag7_earnings_day"]:
        days = merged.groupby("date")[col].first().sum()
        print(f"  {col:25s} {int(days):3d} trading days")

    outfile = DATA_DIR / "spy_merged.parquet"
    merged.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")
    print(f"File size: {outfile.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
