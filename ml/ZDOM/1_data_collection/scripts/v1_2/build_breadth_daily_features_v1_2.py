"""
Build breadth_daily_features_v1_2.parquet
Source: yfinance (RSP, SPY, ^VIX, ^VIX3M, XLF, XLK, XLE, IWM)
Features: vix_vxv_ratio, vix_vxv_zscore_21d, vix_term_structure_slope,
          rsp_spy_ratio + chg/zscore, iwm_spy_ret_spread_20d,
          sector spreads (xlf/xlk/xle), sector_divergence_count/flag,
          breadth_thrust + cumulative_5d
Coverage: 2022-06-06 to 2026-03-13 (clipped), fetch from 2022-01-01
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

BASE = Path(__file__).resolve().parents[2]
DATA_V1_2 = BASE / "data" / "v1_2"
RAW_V1_2 = BASE / "raw" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")
START_DATE = "2022-01-01"

# -------------------------------------------------------------------
# Step 1: Download data
# -------------------------------------------------------------------
tickers = ["RSP", "SPY", "^VIX", "^VIX3M", "XLF", "XLK", "XLE", "IWM"]
print(f"Downloading {tickers} from yfinance...")
raw = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
else:
    close = raw

# Rename VIX columns
close = close.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})

print(f"Downloaded: {close.shape[0]} rows, {close.index.min().date()} to {close.index.max().date()}")
for t in close.columns:
    non_null = close[t].notna().sum()
    print(f"  {t}: {non_null} non-null")

# Save raw
raw_out = RAW_V1_2 / "breadth_raw_v1_2.parquet"
close.to_parquet(raw_out)

# -------------------------------------------------------------------
# Step 2: Build features
# -------------------------------------------------------------------
df = pd.DataFrame(index=close.index)

# VIX/VXV (VIX3M) ratio
df["vix_vxv_ratio"] = close["VIX"] / close["VIX3M"]
df["vix_vxv_zscore_21d"] = (
    (df["vix_vxv_ratio"] - df["vix_vxv_ratio"].rolling(21).mean())
    / df["vix_vxv_ratio"].rolling(21).std()
)
df["vix_term_structure_slope"] = close["VIX3M"] - close["VIX"]

# RSP/SPY ratio (equal-weight vs cap-weight = breadth measure)
df["rsp_spy_ratio"] = close["RSP"] / close["SPY"]
df["rsp_spy_ratio_chg_5d"] = df["rsp_spy_ratio"].pct_change(5)
df["rsp_spy_ratio_chg_20d"] = df["rsp_spy_ratio"].pct_change(20)
df["rsp_spy_ratio_zscore_21d"] = (
    (df["rsp_spy_ratio"] - df["rsp_spy_ratio"].rolling(21).mean())
    / df["rsp_spy_ratio"].rolling(21).std()
)

# IWM-SPY return spread
iwm_ret_20d = close["IWM"].pct_change(20)
spy_ret_20d = close["SPY"].pct_change(20)
df["iwm_spy_ret_spread_20d"] = iwm_ret_20d - spy_ret_20d

# Sector spreads (20d return vs SPY)
for sector, ticker in [("xlf", "XLF"), ("xlk", "XLK"), ("xle", "XLE")]:
    sector_ret = close[ticker].pct_change(20)
    df[f"{sector}_spy_spread_20d"] = sector_ret - spy_ret_20d

# Sector divergence: count sectors with opposite sign to SPY
spy_sign = spy_ret_20d.apply(lambda x: 1 if x > 0 else -1)
divergence_count = 0
for ticker in ["XLF", "XLK", "XLE"]:
    sec_sign = close[ticker].pct_change(20).apply(lambda x: 1 if x > 0 else -1)
    divergence_count = divergence_count + (sec_sign != spy_sign).astype(int)
df["sector_divergence_count"] = divergence_count
df["sector_divergence_flag"] = (df["sector_divergence_count"] >= 2).astype(int)

# Breadth thrust: RSP 5-day return momentum
rsp_ret_1d = close["RSP"].pct_change()
df["breadth_thrust"] = rsp_ret_1d.rolling(5).mean()
df["breadth_thrust_cumulative_5d"] = rsp_ret_1d.rolling(5).sum()

# -------------------------------------------------------------------
# Step 3: Clip and save
# -------------------------------------------------------------------
df = df[df.index >= "2022-06-06"]
df = df[df.index <= CLIP_DATE]
df = df.reset_index()
df.rename(columns={"Date": "date", "index": "date"}, inplace=True)
if "date" not in df.columns:
    df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"])

out = DATA_V1_2 / "breadth_daily_features_v1_2.parquet"
df.to_parquet(out, index=False)

print(f"\nbreadth_daily_features_v1_2.parquet")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"  Saved to: {out}")
