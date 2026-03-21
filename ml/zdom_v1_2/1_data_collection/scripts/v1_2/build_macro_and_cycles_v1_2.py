"""
Build macro_regime_v1_2.parquet and presidential_cycles_v1_2.parquet
Source: Copy from v1_1 and clip to <= 2026-03-13
"""
import pandas as pd
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_1 = BASE / "raw" / "v1_1"
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"
CLIP_DATE = pd.Timestamp("2026-03-13")

for name in ["macro_regime", "presidential_cycles"]:
    src = RAW_V1_1 / f"{name}.parquet"
    raw_dst = RAW_V1_2 / f"{name}.parquet"

    # Copy raw to v1_2
    if not raw_dst.exists():
        shutil.copy2(src, raw_dst)
        print(f"Copied {src} -> {raw_dst}")

    # Load, clip, save processed
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CLIP_DATE]
    df = df.sort_values("date").reset_index(drop=True)

    out = DATA_V1_2 / f"{name}_v1_2.parquet"
    df.to_parquet(out, index=False)

    print(f"\n{name}_v1_2.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"  Saved to: {out}")
