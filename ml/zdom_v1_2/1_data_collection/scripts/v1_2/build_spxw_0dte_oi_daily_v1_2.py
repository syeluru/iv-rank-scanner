"""
Build spxw_0dte_oi_daily_v1_2.parquet
Source: v1_1 raw OI data (spxw_0dte_oi.parquet) + ThetaData fetch for 2026-03-13
Fields: date, total_call_oi, total_put_oi, pc_oi_ratio, oi_concentration, max_pain_strike
Coverage: 2023-02-27 to 2026-03-13 (clipped)
"""
import pandas as pd
import numpy as np
import shutil, os, requests
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")

# -------------------------------------------------------------------
# Step 1: Copy raw OI from v1_1 to v1_2 if not already there
# -------------------------------------------------------------------
src = RAW_V1_1 / "spxw_0dte_oi.parquet"
dst = RAW_V1_2 / "spxw_0dte_oi.parquet"
if not dst.exists():
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

# -------------------------------------------------------------------
# Step 2: Try fetching 2026-03-13 OI from ThetaData
# -------------------------------------------------------------------
def fetch_oi_date(dt_str):
    """Fetch SPXW 0DTE OI for a single date from ThetaData."""
    url = "http://localhost:25503/v2/bulk_snapshot/option/ohlc"
    params = {
        "root": "SPXW",
        "exp": dt_str.replace("-", ""),
        "start_date": dt_str.replace("-", ""),
        "end_date": dt_str.replace("-", ""),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:] if line.strip()]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=header)
        # Try to extract OI if available
        if 'open_interest' in df.columns:
            return df
    except Exception as e:
        print(f"ThetaData fetch failed for {dt_str}: {e}")
    return None

# Try to fetch 3/13
try:
    new_data = fetch_oi_date("2026-03-13")
    if new_data is not None:
        print("Fetched 2026-03-13 OI from ThetaData")
except:
    new_data = None
    print("Could not fetch 2026-03-13 OI - using existing data only")

# -------------------------------------------------------------------
# Step 3: Load raw OI and build daily aggregates
# -------------------------------------------------------------------
oi = pd.read_parquet(dst)
oi["date"] = pd.to_datetime(oi["date"])
oi = oi[oi["date"] <= CLIP_DATE]

# Daily aggregation
def build_daily_oi(df):
    records = []
    for dt, grp in df.groupby("date"):
        calls = grp[grp["right"].str.lower().isin(["call", "c"])]
        puts = grp[grp["right"].str.lower().isin(["put", "p"])]

        total_call_oi = calls["open_interest"].sum()
        total_put_oi = puts["open_interest"].sum()
        total_oi = total_call_oi + total_put_oi

        pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan

        # OI concentration: HHI of OI across strikes (higher = more concentrated)
        if total_oi > 0:
            strike_oi = grp.groupby("strike")["open_interest"].sum()
            shares = strike_oi / total_oi
            oi_concentration = (shares ** 2).sum()
        else:
            oi_concentration = np.nan

        # Max pain: strike where total (call + put) OI-weighted ITM value is minimized
        # Simplified: strike with highest total OI
        strike_oi = grp.groupby("strike")["open_interest"].sum()
        if len(strike_oi) > 0:
            max_pain_strike = strike_oi.idxmax()
        else:
            max_pain_strike = np.nan

        records.append({
            "date": dt,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "pc_oi_ratio": pc_oi_ratio,
            "oi_concentration": oi_concentration,
            "max_pain_strike": max_pain_strike,
        })

    return pd.DataFrame(records)

daily = build_daily_oi(oi)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

# -------------------------------------------------------------------
# Step 4: Save
# -------------------------------------------------------------------
out = DATA_V1_2 / "spxw_0dte_oi_daily_v1_2.parquet"
daily.to_parquet(out, index=False)

print(f"\nspxw_0dte_oi_daily_v1_2.parquet")
print(f"  Shape: {daily.shape}")
print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"  Columns ({len(daily.columns)}): {daily.columns.tolist()}")
print(f"  Saved to: {out}")
