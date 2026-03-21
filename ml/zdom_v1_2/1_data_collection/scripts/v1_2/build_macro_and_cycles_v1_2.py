"""
Build macro_regime_v1_2.parquet and presidential_cycles_v1_2.parquet

macro_regime: reads raw/v1_2/macro_regime.parquet, drops stale computed
columns, recomputes price-level changes, trends, and diff-based features,
then saves to data/v1_2/macro_regime_v1_2.parquet.

presidential_cycles: simple copy with optional clip.
"""
import argparse
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_V1_2 = BASE / "raw" / "v1_2"
DATA_V1_2 = BASE / "data" / "v1_2"

STALE_COLUMNS = [
    "dxy_chg_5d", "dxy_chg_20d",
    "gold_chg_5d", "gold_chg_20d",
    "oil_chg_5d", "oil_chg_20d",
    "dxy_ma20", "dxy_trend",
    "gold_ma20", "gold_trend",
    "oil_ma20", "oil_trend",
    "credit_stress_chg_20d",
    "fed_funds_chg_21d",
]


def build_macro_regime(clip_date: pd.Timestamp | None) -> None:
    src = RAW_V1_2 / "macro_regime.parquet"
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Drop stale computed columns (ignore missing)
    df = df.drop(columns=[c for c in STALE_COLUMNS if c in df.columns])

    # --- Price-level pct_change features ---
    for prefix, raw_col in [("dxy", "dxy_broad"), ("gold", "gold_fix_am"), ("oil", "wti_crude")]:
        df[f"{prefix}_chg_5d"] = df[raw_col].pct_change(5)
        df[f"{prefix}_chg_20d"] = df[raw_col].pct_change(20)
        df[f"{prefix}_ma20"] = df[raw_col].rolling(20).mean()
        df[f"{prefix}_trend"] = df[raw_col] / df[f"{prefix}_ma20"] - 1

    # --- Diff-based features ---
    df["credit_stress_chg_20d"] = df["credit_stress"].diff(20)
    df["fed_funds_chg_21d"] = df["fed_funds_rate"].diff(21)

    if clip_date is not None:
        df = df[df["date"] <= clip_date]

    df = df.sort_values("date").reset_index(drop=True)
    out = DATA_V1_2 / "macro_regime_v1_2.parquet"
    df.to_parquet(out, index=False)

    print(f"\nmacro_regime_v1_2.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"  Saved to: {out}")


def build_presidential_cycles(clip_date: pd.Timestamp | None) -> None:
    src = RAW_V1_2 / "presidential_cycles.parquet"
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if clip_date is not None:
        df = df[df["date"] <= clip_date]

    df = df.sort_values("date").reset_index(drop=True)
    out = DATA_V1_2 / "presidential_cycles_v1_2.parquet"
    df.to_parquet(out, index=False)

    print(f"\npresidential_cycles_v1_2.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"  Saved to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build macro & cycles v1.2 features")
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip date for training mode (e.g. 2026-03-13)")
    args = parser.parse_args()

    clip_date = pd.Timestamp(args.clip) if args.clip else None

    build_macro_regime(clip_date)
    build_presidential_cycles(clip_date)


if __name__ == "__main__":
    main()
