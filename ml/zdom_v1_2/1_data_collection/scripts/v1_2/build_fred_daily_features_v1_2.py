"""
Build fred_daily_features_v1_2.parquet
Sources: FRED API + yfinance (gold, DXY fallback)
Series: DFF, DGS2, DGS10, BAMLC0A0CM, BAMLH0A0HYM2, DCOILWTICO, DTWEXBGS
Also: GC=F from yfinance for gold
Computed: yield_spread_2y10y, yield_curve_inverted, credit_stress, fed_policy_direction
Rolling: _chg_5d, _chg_20d, _chg_21d, _chg_63d, _ma20, _trend
Coverage: 2022-06-06 to 2026-03-13 (clipped), start fetch from 2022-01-01 for lookback
"""
import pandas as pd
import numpy as np
from pathlib import Path
from fredapi import Fred
import yfinance as yf

BASE = Path(__file__).resolve().parents[2]
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")
START_DATE = "2022-01-01"

FRED_API_KEY = "dfb3f177682f9fd569755b3c0b86201d"
fred = Fred(api_key=FRED_API_KEY)

# -------------------------------------------------------------------
# Step 1: Fetch FRED series
# -------------------------------------------------------------------
fred_series = {
    "fed_funds_rate": "DFF",
    "yield_2y": "DGS2",
    "yield_10y": "DGS10",
    "ig_spread": "BAMLC0A0CM",
    "hy_spread": "BAMLH0A0HYM2",
    "wti_crude": "DCOILWTICO",
    "dxy_broad": "DTWEXBGS",
}

dfs = {}
for name, series_id in fred_series.items():
    try:
        s = fred.get_series(series_id, observation_start=START_DATE)
        s = s.dropna()
        s.name = name
        dfs[name] = s
        print(f"  FRED {series_id} ({name}): {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
    except Exception as e:
        print(f"  FRED {series_id} FAILED: {e}")

# -------------------------------------------------------------------
# Step 2: Fetch gold from yfinance
# -------------------------------------------------------------------
print("\nFetching gold (GC=F) from yfinance...")
gold_df = yf.download("GC=F", start=START_DATE, auto_adjust=True, progress=False)
if isinstance(gold_df.columns, pd.MultiIndex):
    gold_df.columns = [col[0] for col in gold_df.columns]
gold_s = gold_df["Close"].dropna()
gold_s.name = "gold_fix_am"
dfs["gold_fix_am"] = gold_s
print(f"  Gold: {len(gold_s)} obs, {gold_s.index.min().date()} to {gold_s.index.max().date()}")

# -------------------------------------------------------------------
# Step 3: Fetch DXY fallback from yfinance if FRED DTWEXBGS is sparse
# -------------------------------------------------------------------
if "dxy_broad" not in dfs or len(dfs["dxy_broad"]) < 100:
    print("\nFetching DXY fallback (DX-Y.NYB) from yfinance...")
    dxy_df = yf.download("DX-Y.NYB", start=START_DATE, auto_adjust=True, progress=False)
    if isinstance(dxy_df.columns, pd.MultiIndex):
        dxy_df.columns = [col[0] for col in dxy_df.columns]
    dxy_s = dxy_df["Close"].dropna()
    dxy_s.name = "dxy_broad"
    dfs["dxy_broad"] = dxy_s
    print(f"  DXY fallback: {len(dxy_s)} obs")

# -------------------------------------------------------------------
# Step 4: Merge into single daily DataFrame
# -------------------------------------------------------------------
# Create date index from trading days
all_dates = pd.bdate_range(start=START_DATE, end=CLIP_DATE.strftime("%Y-%m-%d"))
df = pd.DataFrame(index=all_dates)
df.index.name = "date"

for name, s in dfs.items():
    s.index = pd.to_datetime(s.index)
    df[name] = s.reindex(df.index)
    # Forward fill FRED series (they skip weekends/holidays)
    df[name] = df[name].ffill()

# -------------------------------------------------------------------
# Step 5: Computed fields
# -------------------------------------------------------------------
df["yield_spread_2y10y"] = df["yield_10y"] - df["yield_2y"]
df["yield_curve_inverted"] = (df["yield_spread_2y10y"] < 0).astype(int)
df["credit_stress"] = df["hy_spread"] - df["ig_spread"]

# Fed policy direction: 21-day change in fed funds
df["fed_policy_direction"] = df["fed_funds_rate"].diff(21).apply(
    lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
)

# -------------------------------------------------------------------
# Step 6: Rolling features
# -------------------------------------------------------------------
rolling_bases = [
    "fed_funds_rate", "yield_2y", "yield_10y", "yield_spread_2y10y",
    "ig_spread", "hy_spread", "wti_crude", "gold_fix_am", "dxy_broad",
]

for col in rolling_bases:
    if col not in df.columns:
        continue
    df[f"{col}_chg_5d"] = df[col].pct_change(5)
    df[f"{col}_chg_20d"] = df[col].pct_change(20)

    # For rates that can be zero/near-zero, use diff instead of pct_change
    if col in ("fed_funds_rate", "yield_spread_2y10y"):
        df[f"{col}_chg_5d"] = df[col].diff(5)
        df[f"{col}_chg_20d"] = df[col].diff(20)
        df[f"{col}_chg_21d"] = df[col].diff(21)
        df[f"{col}_chg_63d"] = df[col].diff(63)
    else:
        df[f"{col}_chg_21d"] = df[col].pct_change(21)
        df[f"{col}_chg_63d"] = df[col].pct_change(63)

    df[f"{col}_ma20"] = df[col].rolling(20).mean()
    df[f"{col}_trend"] = (df[col] > df[f"{col}_ma20"]).astype(int)

# -------------------------------------------------------------------
# Step 7: Clip to coverage window and save raw
# -------------------------------------------------------------------
df = df[df.index >= "2022-06-06"]
df = df[df.index <= CLIP_DATE]
df = df.reset_index()
df.rename(columns={"index": "date"}, inplace=True)
if "date" not in df.columns and df.columns[0] != "date":
    df = df.rename(columns={df.columns[0]: "date"})

# Save raw fetch
raw_out = RAW_V1_2 / "fred_daily_raw_v1_2.parquet"
df.to_parquet(raw_out, index=False)
print(f"\nSaved raw: {raw_out}")

# Save processed
out = DATA_V1_2 / "fred_daily_features_v1_2.parquet"
df.to_parquet(out, index=False)

print(f"\nfred_daily_features_v1_2.parquet")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"  Saved to: {out}")
