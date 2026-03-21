"""
Build spxw_term_structure_daily_v1_2.parquet
Source: v1_1 raw term structure (spxw_term_structure.parquet)
Fields: date, ts_0dte_atm_mid, ts_7dte_atm_mid, ts_30dte_atm_mid, ts_slope_0_30, ts_slope_0_7
Coverage: 2023-02-27 to 2026-03-13 (clipped)
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")

# -------------------------------------------------------------------
# Step 1: Copy raw term structure from v1_1 to v1_2 if needed
# -------------------------------------------------------------------
src = RAW_V1_1 / "spxw_term_structure.parquet"
dst = RAW_V1_2 / "spxw_term_structure.parquet"
if not dst.exists():
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

# -------------------------------------------------------------------
# Step 2: Load and process
# -------------------------------------------------------------------
ts = pd.read_parquet(dst)
ts["date"] = pd.to_datetime(ts["date"])
ts = ts[ts["date"] <= CLIP_DATE]

# Compute mid price
ts["mid"] = (ts["bid"] + ts["ask"]) / 2

# For ATM: moneyness closest to 0 (or strike closest to spx_close)
# Group by date+dte, find ATM mid
def get_atm_mid(grp):
    """Get ATM mid for a group (same date, same dte)."""
    # ATM = closest moneyness to 0 for calls (or use atm_strike field)
    atm = grp[grp["strike"] == grp["atm_strike"].iloc[0]]
    if len(atm) == 0:
        # fallback: closest strike to spx_close
        atm = grp.iloc[(grp["strike"] - grp["spx_close"].iloc[0]).abs().argsort()[:2]]

    # Average call and put mids for ATM
    call_mid = atm[atm["right"].str.lower().isin(["call", "c"])]["mid"].mean()
    put_mid = atm[atm["right"].str.lower().isin(["put", "p"])]["mid"].mean()

    # ATM straddle mid or just average
    if pd.notna(call_mid) and pd.notna(put_mid):
        return (call_mid + put_mid) / 2
    elif pd.notna(call_mid):
        return call_mid
    elif pd.notna(put_mid):
        return put_mid
    return np.nan

records = []
for dt, day_grp in ts.groupby("date"):
    row = {"date": dt}

    for target_dte, col_name in [(0, "ts_0dte_atm_mid"), (7, "ts_7dte_atm_mid"), (30, "ts_30dte_atm_mid")]:
        dte_grp = day_grp[day_grp["dte"] == target_dte]
        if len(dte_grp) == 0:
            # Find closest DTE
            available_dtes = day_grp["dte"].unique()
            if len(available_dtes) == 0:
                row[col_name] = np.nan
                continue
            closest_dte = available_dtes[np.argmin(np.abs(available_dtes - target_dte))]
            if abs(closest_dte - target_dte) > 5:
                row[col_name] = np.nan
                continue
            dte_grp = day_grp[day_grp["dte"] == closest_dte]

        row[col_name] = get_atm_mid(dte_grp)

    # Slopes
    if pd.notna(row.get("ts_0dte_atm_mid")) and pd.notna(row.get("ts_30dte_atm_mid")):
        row["ts_slope_0_30"] = row["ts_30dte_atm_mid"] - row["ts_0dte_atm_mid"]
    else:
        row["ts_slope_0_30"] = np.nan

    if pd.notna(row.get("ts_0dte_atm_mid")) and pd.notna(row.get("ts_7dte_atm_mid")):
        row["ts_slope_0_7"] = row["ts_7dte_atm_mid"] - row["ts_0dte_atm_mid"]
    else:
        row["ts_slope_0_7"] = np.nan

    records.append(row)

daily = pd.DataFrame(records)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

# -------------------------------------------------------------------
# Step 3: Save
# -------------------------------------------------------------------
out = DATA_V1_2 / "spxw_term_structure_daily_v1_2.parquet"
daily.to_parquet(out, index=False)

print(f"\nspxw_term_structure_daily_v1_2.parquet")
print(f"  Shape: {daily.shape}")
print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"  Columns ({len(daily.columns)}): {daily.columns.tolist()}")
print(f"  Saved to: {out}")
