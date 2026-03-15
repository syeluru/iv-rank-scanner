"""
Build the final model table from V2 target + daily feature tables.

The V2 target has one row per strategy per 5-min interval per trading day.
All daily features are broadcast (joined by date) to every row.

Inputs:
  data/target_v1.parquet               — V2 IC simulation (the spine)
  data/spx_features.parquet            — SPX technicals
  data/options_features.parquet        — options-derived features
  data/iv_surface_features.parquet     — IV surface: skew, term structure
  data/regime_features.parquet         — HMM regime state + probabilities
  data/gex_regime_features.parquet     — GEX-based HMM regime
  data/vanna_charm_features.parquet    — Vanna/charm greeks features
  data/momentum_features.parquet       — Intraday momentum
  data/vix1d_daily.parquet             — VIX1D data
  data/macro_regime.parquet            — FRED macro features
  data/presidential_cycles.parquet     — Cycle + calendar features
  data/breadth_features.parquet        — Market breadth
  data/cross_asset_features.parquet    — Cross-asset momentum
  data/vol_expansion_features.parquet  — VVIX, vol surface
  data/microstructure_features.parquet — Prev IC outcome, gap fill
  data/intraday_features.parquet       — Per-minute SPX intraday features (ORB, RV, squeeze, etc.)

Output:
  data/model_table_v1.parquet — one row per strategy per 5-min slot, all features + targets
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Columns to drop from spx_features (lookahead or non-stationary)
SPX_DROP = [
    "open", "high", "low", "close",
    "d_open", "d_high", "d_low", "d_close",
    "prev_close",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_5", "ema_20",
    "bb_upper", "bb_lower",
    "atr_14",
    "macd_line", "macd_signal",
    "datetime",
    "vix_sma_5", "vix_sma_10", "vix_sma_20",
    "vol_sma_20",
    "mag7_earnings_symbols",
]

# Target and metadata columns — NOT features
_TP_META = []
for _pct in range(10, 55, 5):
    _k = f"tp{_pct}"
    _TP_META += [f"{_k}_target", f"{_k}_exit_reason", f"{_k}_exit_time",
                 f"{_k}_exit_debit", f"{_k}_pnl"]

META_COLS = [
    "datetime", "date", "strategy",
    "spx_at_entry",
    "short_call", "short_put", "long_call", "long_put",
    "call_wing_width", "put_wing_width",
    "sc_delta", "sp_delta", "sc_iv", "sp_iv",
    "credit",
    "time_to_close_min",
] + _TP_META


def load_optional(path, name):
    """Load a parquet file if it exists, return None otherwise."""
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  {name}: {df.shape}")
        return df
    print(f"  {name}: NOT FOUND")
    return None


def main():
    print("Loading V2 target (strategy x 5-min spine)...")
    target = pd.read_parquet(DATA_DIR / "target_v1.parquet")
    target["datetime"] = pd.to_datetime(target["datetime"])
    target["date"] = pd.to_datetime(target["date"])
    print(f"  target_v1: {target.shape} ({target['date'].nunique()} days, "
          f"{target['strategy'].nunique()} strategies)")

    # Filter to 10:00+ entries only (9:30-9:59 used for feature warmup, not training)
    before_filter = len(target)
    target = target[target["datetime"].dt.hour * 100 + target["datetime"].dt.minute >= 1000].copy()
    print(f"  Filtered to 10:00+ entries: {before_filter:,} -> {len(target):,} rows")

    # Extract intraday timing features from target datetime
    target["entry_hour"] = target["datetime"].dt.hour
    target["entry_minute"] = target["datetime"].dt.minute
    target["minutes_since_open"] = (
        (target["datetime"].dt.hour - 9) * 60 +
        target["datetime"].dt.minute - 30
    )
    target["minutes_to_close"] = (
        15 * 60 - (target["datetime"].dt.hour * 60 + target["datetime"].dt.minute)
    )
    total_mins = 390.0
    target["time_sin"] = np.sin(2 * np.pi * target["minutes_since_open"] / total_mins)
    target["time_cos"] = np.cos(2 * np.pi * target["minutes_since_open"] / total_mins)

    # One-hot encode strategy (so tree models treat each delta bucket individually)
    strategy_dummies = pd.get_dummies(target["strategy"], prefix="strat", dtype=int)
    target = pd.concat([target, strategy_dummies], axis=1)
    print(f"  One-hot encoded strategy: {strategy_dummies.shape[1]} columns")

    print("\nLoading daily feature tables...")

    # SPX features
    spx_daily_path = DATA_DIR / "spx_features_daily.parquet"
    if spx_daily_path.exists():
        spx_daily = pd.read_parquet(spx_daily_path)
        spx_daily["date"] = pd.to_datetime(spx_daily["date"])
    else:
        spx = pd.read_parquet(DATA_DIR / "spx_features.parquet")
        spx["date"] = pd.to_datetime(spx["date"])
        intraday_cols = ["datetime", "open", "high", "low", "close",
                         "time_sin", "time_cos", "minutes_since_open", "minutes_to_close"]
        daily_cols = [c for c in spx.columns if c not in intraday_cols and c != "date"]
        spx_daily = spx.groupby("date")[daily_cols].first().reset_index()
        del spx
    spx_drop_actual = [c for c in SPX_DROP if c in spx_daily.columns]
    spx_daily = spx_daily.drop(columns=spx_drop_actual, errors="ignore")
    print(f"  spx_features (daily): {spx_daily.shape}")

    # All other feature tables
    opts = load_optional(DATA_DIR / "options_features.parquet", "options_features")
    iv_surface = load_optional(DATA_DIR / "iv_surface_features.parquet", "iv_surface_features")
    regime = load_optional(DATA_DIR / "regime_features.parquet", "regime_features")
    gex_regime = load_optional(DATA_DIR / "gex_regime_features.parquet", "gex_regime_features")
    vanna_charm = load_optional(DATA_DIR / "vanna_charm_features.parquet", "vanna_charm_features")
    momentum = load_optional(DATA_DIR / "momentum_features.parquet", "momentum_features")
    macro = load_optional(DATA_DIR / "macro_regime.parquet", "macro_regime")
    breadth = load_optional(DATA_DIR / "breadth_features.parquet", "breadth_features")
    cross_asset = load_optional(DATA_DIR / "cross_asset_features.parquet", "cross_asset_features")
    vol_expansion = load_optional(DATA_DIR / "vol_expansion_features.parquet", "vol_expansion_features")
    microstructure = load_optional(DATA_DIR / "microstructure_features.parquet", "microstructure_features")
    if microstructure is not None and len(microstructure) > microstructure["date"].nunique():
        microstructure = microstructure.groupby("date").first().reset_index()
        print(f"    (collapsed to daily: {microstructure.shape})")

    vix1d = load_optional(DATA_DIR / "vix1d_daily.parquet", "vix1d_daily")
    if vix1d is not None:
        vix1d["vix1d_sma_5"] = vix1d["vix1d_close"].rolling(5, min_periods=3).mean()
        vix1d["vix1d_sma_20"] = vix1d["vix1d_close"].rolling(20, min_periods=10).mean()
        vix1d["vix1d_vs_sma_5"] = (
            (vix1d["vix1d_close"] - vix1d["vix1d_sma_5"]) / vix1d["vix1d_sma_5"] * 100
        )
        vix1d["vix1d_vs_sma_20"] = (
            (vix1d["vix1d_close"] - vix1d["vix1d_sma_20"]) / vix1d["vix1d_sma_20"] * 100
        )
        vix1d_std = vix1d["vix1d_close"].rolling(21, min_periods=10).std()
        vix1d_mean = vix1d["vix1d_close"].rolling(21, min_periods=10).mean()
        vix1d["vix1d_zscore_21d"] = (vix1d["vix1d_close"] - vix1d_mean) / vix1d_std.replace(0, np.nan)
        vix1d = vix1d.drop(columns=["vix1d_open", "vix1d_high", "vix1d_low",
                                      "vix1d_sma_5", "vix1d_sma_20"], errors="ignore")

    cycle_path = DATA_DIR / "presidential_cycles.parquet"
    cycles = None
    if cycle_path.exists():
        cycles = pd.read_parquet(cycle_path)
        cycles["date"] = pd.to_datetime(cycles["date"])
        str_cols = cycles.select_dtypes(include=["object", "category"]).columns.tolist()
        cycles = cycles.drop(columns=str_cols, errors="ignore")
        if "month" in cycles.columns:
            cycles = cycles.rename(columns={"month": "cycle_month_of_year"})
        if "quarter" in cycles.columns:
            cycles = cycles.rename(columns={"quarter": "cycle_quarter"})
        print(f"  cycle_features: {cycles.shape}")

    # ── Join everything onto the V2 target ──
    print("\nJoining features onto V2 target...")
    df = target.copy()

    daily_tables = [
        (spx_daily, "spx"),
        (opts, "opts"),
        (iv_surface, "iv_surface"),
        (regime, "regime"),
        (gex_regime, "gex_regime"),
        (vanna_charm, "vanna_charm"),
        (momentum, "momentum"),
        (vix1d, "vix1d"),
        (macro, "macro"),
        (cycles, "cycles"),
        (breadth, "breadth"),
        (cross_asset, "cross_asset"),
        (vol_expansion, "vol_expansion"),
        (microstructure, "microstructure"),
    ]

    for table, name in daily_tables:
        if table is not None:
            before = len(df.columns)
            df = df.merge(table, on="date", how="left", suffixes=("", f"_{name}"))
            added = len(df.columns) - before
            print(f"  + {name}: {added} columns")

    if "vix_close" in df.columns and "vix1d_close" in df.columns:
        df["vix1d_vs_vix"] = df["vix1d_close"] / df["vix_close"]

    # ── Intraday features (join on datetime, not date) ──
    intraday_path = DATA_DIR / "intraday_features.parquet"
    if intraday_path.exists():
        intraday = pd.read_parquet(intraday_path)
        intraday["datetime"] = pd.to_datetime(intraday["datetime"])
        intraday = intraday.drop(columns=["date"], errors="ignore")
        before = len(df.columns)
        df = df.merge(intraday, on="datetime", how="left", suffixes=("", "_intraday"))
        added = len(df.columns) - before
        print(f"  + intraday_features: {added} columns (joined on datetime)")
        # Drop any duplicate columns from suffix
        intraday_dup = [c for c in df.columns if c.endswith("_intraday")]
        if intraday_dup:
            df = df.drop(columns=intraday_dup, errors="ignore")
            print(f"    Dropped {len(intraday_dup)} duplicate intraday columns")
        del intraday
    else:
        print("  intraday_features: NOT FOUND")

    # ── Cleanup ──
    dup_cols = [c for c in df.columns if any(c.endswith(f"_{name}") for _, name in daily_tables)]
    if dup_cols:
        df = df.drop(columns=dup_cols, errors="ignore")
        print(f"  Dropped {len(dup_cols)} duplicate columns")

    dead_features = [
        "skew_25", "skew_50", "skew_100",
        "call_mid_25otm", "put_mid_25otm",
        "call_mid_50otm", "put_mid_50otm",
        "call_mid_100otm", "put_mid_100otm",
        "first_30m_vol_ratio", "ism_pmi", "lei", "lei_chg_1m", "lei_chg_3m",
        "pmi_chg_1m", "pmi_chg_3m",
        "yz_hv_60d",
        "recession", "pmi_expanding", "pmi_contracting", "pmi_above_52",
        "pmi_below_48", "pmi_cross_50", "lei_rising", "lei_falling",
        "lei_3m_direction", "business_cycle_phase",
        # Known leaky features from V1
        "consecutive_wins", "consecutive_losses", "win_streak", "loss_streak",
        # Previous IC outcome — autocorrelation artifact, not genuine signal
        "prev_ic_outcome", "prev_ic_pnl_pct",
        # Intraday windows > 30 min — not available at 10:00 entry
        "first_60m_return_pct",
    ]
    dead_actual = [c for c in dead_features if c in df.columns]
    if dead_actual:
        df = df.drop(columns=dead_actual)
        print(f"  Dropped {len(dead_actual)} dead/leaky features")

    inf_cols = ["gap_pct", "close_to_close_pct", "atr_pct"]
    for col in inf_cols:
        if col in df.columns:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                print(f"  Replaced {n_inf:,} inf values in {col}")

    dedup_drop = [
        "minutes_to_close",
        "cycle_month_of_year",
        "regime_prob_lowvol",
        "overnight_range_pct",
        "gap_fill_rate_5d",
        "gap_fill_rate_20d",
        "close_to_close_pct",
        "days_to_month_end",
        "days_to_qtr_end",
        "iwm_spx_ratio",
        "term_num",
        "net_gex_normalized",
        "breadth_thrust",
    ]
    dedup_actual = [c for c in dedup_drop if c in df.columns]
    if dedup_actual:
        df = df.drop(columns=dedup_actual)
        print(f"  Dropped {len(dedup_actual)} near-duplicate features")

    # Drop rows with no target
    before = len(df)
    df = df.dropna(subset=["tp10_target"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing target")

    df = df.sort_values(["datetime", "strategy"]).reset_index(drop=True)

    # ── Summary ──
    feature_cols = [c for c in df.columns if c not in META_COLS]

    print(f"\n{'='*60}")
    print(f"Model table V2: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Trading days: {df['date'].nunique()}")
    print(f"Strategies: {df['strategy'].nunique()}")
    print(f"Avg rows/day: {len(df) / df['date'].nunique():.0f}")

    print(f"\nTarget distribution:")
    for pct in range(10, 55, 5):
        key = f"tp{pct}_target"
        if key in df.columns:
            print(f"  {key}:  {df[key].mean():.1%}")

    print(f"\nPer strategy:")
    for strat in sorted(df["strategy"].unique()):
        s = df[df["strategy"] == strat]
        print(f"  {strat:15s}  n={len(s):7,}  "
              f"tp10={s['tp10_target'].mean():.1%}  "
              f"tp25={s['tp25_target'].mean():.1%}  "
              f"tp50={s['tp50_target'].mean():.1%}")

    null_rates = df[feature_cols].isna().mean()
    high_null = null_rates[null_rates > 0.30].sort_values(ascending=False)
    if len(high_null) > 0:
        print(f"\nHigh null features (>30%):")
        for col, rate in high_null.items():
            print(f"  {col:40s} {rate:.1%}")

    outfile = DATA_DIR / "model_table_v1.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")

    with open(PROJECT_DIR / "feature_list_v2.txt", "w") as f:
        f.write(f"# V2 Features ({len(feature_cols)} total)\n\n")
        for col in sorted(feature_cols):
            f.write(f"{col}\n")
    print(f"Feature list saved to feature_list_v2.txt")


if __name__ == "__main__":
    main()
