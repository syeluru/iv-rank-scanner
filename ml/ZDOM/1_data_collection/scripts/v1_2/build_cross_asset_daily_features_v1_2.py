"""
Build cross_asset_daily_features_v1_2.parquet
Source: yfinance (SPY, QQQ, IWM, EEM, TLT)
Features: spx_ret_5d/20d/60d, spx_iwm_ratio + chg_5d/20d, qqq_spx_ratio + chg,
          iwm_spx_ratio + chg, eem_spx_ratio + chg, tlt_ret_5d/20d,
          tlt_spx_ratio_chg_5d/20d, tlt_spx_corr_20d
Coverage: 2022-06-06 to 2026-03-13 (clipped), fetch from 2022-01-01 for lookback
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
tickers = ["SPY", "QQQ", "IWM", "EEM", "TLT"]
print(f"Downloading {tickers} from yfinance...")
raw = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    # Extract Close prices
    close = raw["Close"]
else:
    close = raw

# Ensure we have all tickers
for t in tickers:
    if t not in close.columns:
        print(f"WARNING: {t} missing from download")
        close[t] = np.nan

close = close.dropna(how="all")
print(f"Downloaded: {close.shape[0]} rows, {close.index.min().date()} to {close.index.max().date()}")

# Save raw
raw_out = RAW_V1_2 / "cross_asset_raw_v1_2.parquet"
close.to_parquet(raw_out)
print(f"Saved raw: {raw_out}")

# -------------------------------------------------------------------
# Step 2: Build features
# -------------------------------------------------------------------
df = pd.DataFrame(index=close.index)

# SPX (SPY proxy) returns
df["spx_ret_5d"] = close["SPY"].pct_change(5)
df["spx_ret_20d"] = close["SPY"].pct_change(20)
df["spx_ret_60d"] = close["SPY"].pct_change(60)

# Ratios
df["qqq_spx_ratio"] = close["QQQ"] / close["SPY"]
df["iwm_spx_ratio"] = close["IWM"] / close["SPY"]
df["eem_spx_ratio"] = close["EEM"] / close["SPY"]
df["tlt_spx_ratio"] = close["TLT"] / close["SPY"]
df["spx_iwm_ratio"] = close["SPY"] / close["IWM"]

# Ratio changes
for ratio_col in ["qqq_spx_ratio", "iwm_spx_ratio", "eem_spx_ratio", "spx_iwm_ratio"]:
    df[f"{ratio_col}_chg_5d"] = df[ratio_col].pct_change(5)
    df[f"{ratio_col}_chg_20d"] = df[ratio_col].pct_change(20)

# TLT returns
df["tlt_ret_5d"] = close["TLT"].pct_change(5)
df["tlt_ret_20d"] = close["TLT"].pct_change(20)

# TLT/SPX ratio changes
df["tlt_spx_ratio_chg_5d"] = df["tlt_spx_ratio"].pct_change(5)
df["tlt_spx_ratio_chg_20d"] = df["tlt_spx_ratio"].pct_change(20)

# TLT-SPX correlation (20-day rolling)
spy_ret = close["SPY"].pct_change()
tlt_ret = close["TLT"].pct_change()
df["tlt_spx_corr_20d"] = spy_ret.rolling(20).corr(tlt_ret)

# Drop the raw ratio columns (keep only the changes)
df = df.drop(columns=["tlt_spx_ratio"], errors="ignore")

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

out = DATA_V1_2 / "cross_asset_daily_features_v1_2.parquet"
df.to_parquet(out, index=False)

print(f"\ncross_asset_daily_features_v1_2.parquet")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"  Saved to: {out}")
