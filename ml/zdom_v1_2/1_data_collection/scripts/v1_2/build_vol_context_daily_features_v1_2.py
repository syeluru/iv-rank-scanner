"""
Build vol_context_daily_features_v1_2.parquet
Source: yfinance (^VVIX, ^VIX), SPX daily from v1_2 raw spx_1min, FOMC dates from v1_1
Features: vvix_close, vvix_ma20, vvix_zscore_21d, vvix_vs_vix_ratio,
          iv_crush_post_fomc, rv_iv_spread_10d/20d, vol_regime_persistence
Coverage: 2022-06-06 to 2026-03-13 (clipped)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")
START_DATE = "2022-01-01"

# -------------------------------------------------------------------
# Step 1: Download VVIX and VIX from yfinance
# -------------------------------------------------------------------
tickers = ["^VVIX", "^VIX"]
print(f"Downloading {tickers} from yfinance...")
raw = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
else:
    close = raw

close = close.rename(columns={"^VVIX": "VVIX", "^VIX": "VIX"})
print(f"Downloaded: {close.shape[0]} rows, {close.index.min().date()} to {close.index.max().date()}")
for t in close.columns:
    print(f"  {t}: {close[t].notna().sum()} non-null")

# Save raw
raw_out = RAW_V1_2 / "vol_context_raw_v1_2.parquet"
close.to_parquet(raw_out)

# -------------------------------------------------------------------
# Step 2: Build SPX daily from 1-min bars (for realized vol)
# -------------------------------------------------------------------
print("\nBuilding SPX daily from 1-min bars...")
spx_1min = pd.read_parquet(RAW_V1_2 / "spx_1min.parquet")
spx_1min["datetime"] = pd.to_datetime(spx_1min["datetime"])
spx_1min["date"] = spx_1min["datetime"].dt.date

# Daily OHLC
spx_daily = spx_1min.groupby("date").agg(
    spx_open=("open", "first"),
    spx_high=("high", "max"),
    spx_low=("low", "min"),
    spx_close=("close", "last"),
).reset_index()
spx_daily["date"] = pd.to_datetime(spx_daily["date"])
spx_daily = spx_daily.set_index("date")

# Also get SPX daily from yfinance for broader coverage
print("Downloading SPY from yfinance for lookback coverage...")
spy_raw = yf.download("SPY", start=START_DATE, auto_adjust=True, progress=False)
if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_raw.columns = [col[0] for col in spy_raw.columns]
spy_daily = spy_raw[["Close"]].rename(columns={"Close": "spy_close"})

# -------------------------------------------------------------------
# Step 3: Load FOMC dates
# -------------------------------------------------------------------
fomc_path = RAW_V1_1 / "fomc_dates.parquet"
if fomc_path.exists():
    fomc = pd.read_parquet(fomc_path)
    fomc["date"] = pd.to_datetime(fomc["date"])
    fomc_dates = set(fomc["date"].dt.date)
    print(f"Loaded {len(fomc_dates)} FOMC dates")
else:
    fomc_dates = set()
    print("WARNING: No FOMC dates found")

# -------------------------------------------------------------------
# Step 4: Build features
# -------------------------------------------------------------------
df = pd.DataFrame(index=close.index)

# VVIX features
df["vvix_close"] = close["VVIX"]
df["vvix_ma20"] = close["VVIX"].rolling(20).mean()
df["vvix_zscore_21d"] = (
    (close["VVIX"] - close["VVIX"].rolling(21).mean())
    / close["VVIX"].rolling(21).std()
)
df["vvix_vs_vix_ratio"] = close["VVIX"] / close["VIX"]

# Realized vol from SPX (using SPY for broader coverage)
spy_ret = spy_daily["spy_close"].pct_change()
rv_10d = spy_ret.rolling(10).std() * np.sqrt(252) * 100  # annualized
rv_20d = spy_ret.rolling(20).std() * np.sqrt(252) * 100

# Align to df index
rv_10d_aligned = rv_10d.reindex(df.index)
rv_20d_aligned = rv_20d.reindex(df.index)

# RV-IV spread
df["rv_iv_spread_10d"] = rv_10d_aligned - close["VIX"]
df["rv_iv_spread_20d"] = rv_20d_aligned - close["VIX"]

# IV crush post FOMC: VIX change on the day after FOMC
vix_chg = close["VIX"].pct_change()
df["iv_crush_post_fomc"] = 0.0
for idx in df.index:
    dt = idx.date() if hasattr(idx, 'date') else idx
    if dt in fomc_dates:
        # Next trading day
        next_idx = df.index[df.index > idx]
        if len(next_idx) > 0:
            next_day = next_idx[0]
            if next_day in vix_chg.index:
                df.loc[next_day, "iv_crush_post_fomc"] = vix_chg.loc[next_day]

# Vol regime persistence: consecutive days VIX above/below 20
vix_above_20 = (close["VIX"] > 20).astype(int)
persistence = []
count = 0
prev_state = None
for val in vix_above_20:
    if val == prev_state:
        count += 1
    else:
        count = 1
        prev_state = val
    persistence.append(count if val == 1 else -count)
df["vol_regime_persistence"] = persistence

# -------------------------------------------------------------------
# Step 5: Clip and save
# -------------------------------------------------------------------
df = df[df.index >= "2022-06-06"]
df = df[df.index <= CLIP_DATE]
df = df.reset_index()
df.rename(columns={"Date": "date", "index": "date"}, inplace=True)
if "date" not in df.columns:
    df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"])

out = DATA_V1_2 / "vol_context_daily_features_v1_2.parquet"
df.to_parquet(out, index=False)

print(f"\nvol_context_daily_features_v1_2.parquet")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"  Saved to: {out}")
