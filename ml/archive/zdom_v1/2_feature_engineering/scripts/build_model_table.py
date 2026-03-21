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

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
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

    # ═══════════════════════════════════════════════════════════════════════
    # LEAKAGE FIX: Shift daily features to T-1 and rename with temporal prefix
    # ═══════════════════════════════════════════════════════════════════════
    # All daily features that use same-day close/high/low are shifted by 1 day
    # so each row gets the PRIOR day's values. This matches production where
    # we only have T-1 data at scoring time.
    #
    # The shift is done at the daily level: for each date in the model table,
    # we replace the daily feature values with the prior trading day's values.
    print("\n── Leakage fix: shifting daily features to T-1 ──")

    # Features that are SAFE (no shift needed):
    # - Target columns and meta (datetime, date, strategy, strikes, credit, etc.)
    # - Entry timing (entry_hour, entry_minute, minutes_since_open, time_sin, time_cos)
    # - Strategy one-hots (strat_IC_*)
    # - gap_pct (uses today open + prior close — already safe)
    # - spx_open / d_open (today's opening price — known at 9:30)
    # - Calendar/seasonal flags (day_of_week, month, is_monday, etc.)
    # - Intraday features (joined on datetime, computed from 1-min bars — handled separately)
    # - Momentum first_*m features (from 9:30-9:59 bars — known by 10am)
    # - IV surface features from 10am snapshot (atm_iv, skew, etc.) — already T-1 in production
    # - Options OI features (prior day by definition)
    # - Macro FRED features (lagged by publication — separate 2-day shift below)

    safe_prefixes = ["tp", "strat_IC_", "entry_", "minutes_", "time_sin", "time_cos"]
    safe_exact = [
        "datetime", "date", "strategy", "spx_at_entry",
        "short_call", "short_put", "long_call", "long_put",
        "call_wing_width", "put_wing_width", "sc_delta", "sp_delta", "sc_iv", "sp_iv",
        "credit", "time_to_close_min",
        "gap_pct", "spx_open", "d_open",
        "day_of_week", "month", "is_monday", "is_friday", "is_month_end", "is_quarter_end",
        # Momentum (from 9:30-9:59 bars)
        "first_15m_return_pct", "first_30m_return_pct", "first_30m_range_pct",
        "first_30m_direction", "open_drive_strength", "prev_close_gap_reversal",
        "intraday_range_vs_atr",
        # Calendar/cycles
        "day_of_month", "week_of_year", "days_into_quarter",
        "near_month_end", "near_qtr_end", "opex_day", "opex_week", "triple_witching",
        "sell_in_may", "q4_rally", "january_effect", "thanksgiving_week", "christmas_week",
        "cycle_year", "cycle_month", "cycle_quarter",
        "pre_election_year", "election_year", "midterm_year", "first_year", "party_dem",
        # Events
        "is_cpi_day", "is_ppi_day", "is_nfp_day", "is_gdp_day",
        "is_mag7_earnings_day", "mag7_earnings_count",
        "is_earnings_aapl", "is_earnings_amzn", "is_earnings_googl",
        "is_earnings_meta", "is_earnings_msft", "is_earnings_nvda", "is_earnings_tsla",
        "days_to_next_cpi", "days_to_next_ppi", "days_to_next_nfp",
        "days_to_next_gdp", "days_to_next_fomc", "days_to_next_any_econ",
        # IV surface features (from 10am intraday greeks snapshot — already correct timing)
        "atm_iv", "iv_25d_call", "iv_25d_put", "iv_10d_call", "iv_10d_put",
        "skew_25d", "skew_10d", "iv_smile_curve",
        "atm_iv_5d_mean", "atm_iv_5d_std", "atm_iv_21d_mean", "atm_iv_21d_std", "atm_iv_zscore_21d",
        "skew_25d_5d_mean", "skew_25d_5d_std", "skew_25d_21d_mean", "skew_25d_21d_std", "skew_25d_zscore_21d",
        # Options OI (prior day by definition)
        "total_call_oi", "total_put_oi", "pc_oi_ratio", "oi_concentration",
        "max_pain_strike", "max_pain_distance_pts", "max_pain_distance_pct",
        # Options open prices (today's opening — known at 9:30)
        "atm_call_open", "atm_put_open", "atm_straddle_open",
        "ic_credit_50", "ic_credit_100", "ic_credit_150",
        # Term structure mid prices (from prior day term structure)
        "ts_0dte_atm_mid", "ts_7dte_atm_mid", "ts_30dte_atm_mid",
        "ts_slope_0_30", "ts_slope_0_7",
        # IV term structure (from 10am snapshot + prior day)
        "ts_iv_7dte", "ts_iv_14dte", "iv_ts_slope_0_7", "iv_ts_slope_0_14",
        "ts_slope_7_14", "ts_inverted_7_14",
        # iv_crush_post_fomc is a calendar flag
        "iv_crush_post_fomc",
    ]

    # Intraday features (joined on datetime, not date — these are per-minute, not daily)
    intraday_safe = [c for c in df.columns if c.startswith("spx_ret_") and "m" in c.split("_")[-1]]
    intraday_safe += [c for c in df.columns if c.startswith("spx_vs_") or c.startswith("intraday_")
                      or c.startswith("rvol_") or c.startswith("garman_") or c.startswith("range_")
                      or c.startswith("orb_") or c.startswith("ttm_") or c.startswith("bb_squeeze")]
    intraday_safe += ["iv_rank_30d", "iv_rv_ratio"]

    all_safe = set(safe_exact + intraday_safe)
    for prefix in safe_prefixes:
        all_safe.update(c for c in df.columns if c.startswith(prefix))

    # Features to SHIFT by 1 day (use prior day values)
    shift_1day_cols = []
    for col in df.columns:
        if col in all_safe:
            continue
        if col in META_COLS:
            continue
        shift_1day_cols.append(col)

    if shift_1day_cols:
        print(f"  Shifting {len(shift_1day_cols)} features by 1 day (T-1):")
        # Build a daily lookup table: one row per date with shift column values
        daily_vals = df.groupby("date")[shift_1day_cols].first().reset_index()
        daily_vals = daily_vals.sort_values("date")

        # Shift: for each date, assign the PRIOR date's feature values
        # pandas shift(1) moves values DOWN by 1 row = each date gets prior row's values
        for col in shift_1day_cols:
            daily_vals[col] = daily_vals[col].shift(1)

        # Drop the original columns and merge shifted versions
        df = df.drop(columns=shift_1day_cols)
        df = df.merge(daily_vals, on="date", how="left")

        for col in shift_1day_cols[:10]:
            print(f"    {col}")
        if len(shift_1day_cols) > 10:
            print(f"    ... and {len(shift_1day_cols) - 10} more")

    # Features to SHIFT by 2 days (FRED data — publication lag)
    fred_cols = [c for c in df.columns if c in [
        "fed_funds_rate", "fed_funds_chg_21d", "fed_policy_direction",
        "yield_2y", "yield_2y_chg_5d", "yield_2y_chg_21d", "yield_2y_chg_63d",
        "yield_10y", "yield_10y_chg_5d", "yield_10y_chg_21d", "yield_10y_chg_63d",
        "yield_spread_2y10y", "yield_spread_2y10y_chg_5d", "yield_spread_2y10y_chg_21d", "yield_spread_2y10y_chg_63d",
        "yield_curve_inverted", "ig_spread", "hy_spread", "credit_stress", "credit_stress_chg_20d",
        "gold_fix_am", "gold_ma20", "gold_trend", "gold_chg_5d", "gold_chg_20d",
        "wti_crude", "oil_ma20", "oil_trend", "oil_chg_5d", "oil_chg_20d",
        "dxy_broad", "dxy_ma20", "dxy_trend", "dxy_chg_5d", "dxy_chg_20d",
    ]]
    if fred_cols:
        print(f"\n  Applying additional 1-day shift to {len(fred_cols)} FRED features (total 2-day lag):")
        # These already got shifted 1 day above. Shift 1 more day for 2 total.
        daily_vals2 = df.groupby("date")[fred_cols].first().reset_index()
        daily_vals2 = daily_vals2.sort_values("date")
        for col in fred_cols:
            daily_vals2[col] = daily_vals2[col].shift(1)
        df = df.drop(columns=fred_cols)
        df = df.merge(daily_vals2, on="date", how="left")
        print(f"    FRED features now at T-2 lag total")

    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE RENAME: Apply temporal prefix naming convention
    # ═══════════════════════════════════════════════════════════════════════
    # All features are renamed to make their temporal nature explicit:
    #   daily_prior_*  = computed from prior day's data (shifted T-1)
    #   prior_day_*    = data that is inherently from prior day (OI, max pain)
    #   intraday_*     = computed from live 1-min bars at scoring time
    #   spx_intraday_* = SPX-specific intraday features
    #   vix_intraday_* = VIX-specific intraday features
    # Features without prefix are either static (calendar) or entry-specific.
    #
    # See zdom_feature_audit_merged_final.csv for the full mapping and rationale.
    print("\n── Renaming features with temporal prefixes ──")

    FEATURE_RENAME_MAP = {
        "atm_call_open": "intraday_atm_call_mid",
        "atm_iv": "intraday_atm_iv",
        "atm_iv_21d_mean": "daily_prior_atm_iv_21day_mean",
        "atm_iv_21d_std": "daily_prior_atm_iv_21day_std",
        "atm_iv_5d_mean": "daily_prior_atm_iv_5day_mean",
        "atm_iv_5d_std": "daily_prior_atm_iv_5day_std",
        "atm_iv_zscore_21d": "daily_prior_atm_iv_zscore_21day",
        "atm_put_open": "intraday_atm_put_mid",
        "atm_straddle_open": "intraday_atm_straddle_mid",
        "atm_strike": "spx_intraday_atm_strike",
        "atr_pct": "daily_prior_atr_pct",
        "bb_bandwidth": "daily_prior_bb_bandwidth",
        "bb_bandwidth_pctile": "daily_prior_bb_bandwidth_pctile",
        "bb_pct_b": "daily_prior_bb_pct_b",
        "bb_squeeze_duration": "intraday_bb_squeeze_duration",
        "breadth_thrust_cumulative_5d": "daily_prior_breadth_thrust_cumulative_5d",
        "choppiness_14": "daily_prior_choppiness_14day",
        "close_vs_ema_20": "daily_prior_close_ratio_ema_20day",
        "close_vs_ema_5": "daily_prior_close_ratio_ema_5day",
        "close_vs_sma_10": "daily_prior_close_ratio_sma_10day",
        "close_vs_sma_100": "daily_prior_close_ratio_sma_100day",
        "close_vs_sma_20": "daily_prior_close_ratio_sma_20day",
        "close_vs_sma_200": "daily_prior_close_ratio_sma_200day",
        "close_vs_sma_5": "daily_prior_close_ratio_sma_5day",
        "close_vs_sma_50": "daily_prior_close_ratio_sma_50day",
        "credit_stress": "daily_prior_credit_stress",
        "credit_stress_chg_20d": "daily_prior_credit_stress_chg_20d",
        "dxy_broad": "daily_prior_dxy_broad",
        "dxy_chg_20d": "daily_prior_dxy_chg_20d",
        "dxy_chg_5d": "daily_prior_dxy_chg_5d",
        "dxy_ma20": "daily_prior_dxy_ma20",
        "dxy_trend": "daily_prior_dxy_trend",
        "eem_spx_ratio": "daily_prior_eem_spx_ratio",
        "eem_spx_ratio_chg_20d": "daily_prior_eem_spx_ratio_chg_20d",
        "eem_spx_ratio_chg_5d": "daily_prior_eem_spx_ratio_chg_5d",
        "efficiency_ratio": "daily_prior_efficiency_ratio",
        "fed_funds_chg_21d": "daily_prior_fed_funds_chg_21d",
        "fed_funds_rate": "daily_prior_fed_funds_rate",
        "fed_policy_direction": "daily_prior_fed_policy_direction",
        "garman_klass_rv_30m": "intraday_garman_klass_rv_30m",
        "gold_chg_20d": "daily_prior_gold_chg_20d",
        "gold_chg_5d": "daily_prior_gold_chg_5d",
        "gold_fix_am": "daily_prior_gold_fix_am",
        "gold_ma20": "daily_prior_gold_ma20",
        "gold_trend": "daily_prior_gold_trend",
        "hv_iv_ratio_10d": "daily_prior_hv_iv_ratio_10day",
        "hv_iv_ratio_20d": "daily_prior_hv_iv_ratio_20day",
        "hv_iv_ratio_30d": "daily_prior_hv_iv_ratio_30day",
        "hv_iv_ratio_5d": "daily_prior_hv_iv_ratio_5day",
        "hy_spread": "daily_prior_hy_spread",
        "ic_credit_100": "intraday_ic_credit_100pt",
        "ic_credit_150": "intraday_ic_credit_150pt",
        "ic_credit_50": "intraday_ic_credit_50pt",
        "ig_spread": "daily_prior_ig_spread",
        "intraday_range_pct": "daily_prior_range_pct",
        "iv_10d_call": "intraday_iv_10d_call",
        "iv_10d_put": "intraday_iv_10d_put",
        "iv_25d_call": "intraday_iv_25d_call",
        "iv_25d_put": "intraday_iv_25d_put",
        "iv_rank_30d": "daily_prior_iv_rank_30d",
        "iv_rv_ratio": "daily_prior_iv_rv_ratio",
        "iv_smile_curve": "intraday_iv_smile_curve",
        "iv_ts_slope_0_14": "intraday_iv_ts_slope_0dte_14dte",
        "iv_ts_slope_0_7": "intraday_iv_ts_slope_0dte_7dte",
        "iwm_spx_ratio_chg_20d": "daily_prior_iwm_spx_ratio_chg_20d",
        "iwm_spx_ratio_chg_5d": "daily_prior_iwm_spx_ratio_chg_5d",
        "iwm_spy_ret_spread_20d": "daily_prior_iwm_spy_ret_spread_20d",
        "macd_hist": "daily_prior_macd_hist",
        "max_pain_distance_pct": "intraday_max_pain_distance_pct",
        "max_pain_distance_pts": "intraday_max_pain_distance_pts",
        "max_pain_strike": "prior_day_max_pain_strike",
        "oi_concentration": "prior_day_oi_concentration",
        "oil_chg_20d": "daily_prior_oil_chg_20d",
        "oil_chg_5d": "daily_prior_oil_chg_5d",
        "oil_ma20": "daily_prior_oil_ma20",
        "oil_trend": "daily_prior_oil_trend",
        "open_to_close_pct": "daily_prior_open_to_close_pct",
        "orb_breakout_dir": "intraday_orb_breakout_dir",
        "orb_containment": "intraday_orb_containment",
        "orb_range_pct": "intraday_orb_range_pct",
        "pc_oi_ratio": "prior_day_pc_oi_ratio",
        "pc_volume_ratio": "daily_prior_pc_volume_ratio",
        "qqq_spx_ratio": "daily_prior_qqq_spx_ratio",
        "qqq_spx_ratio_chg_20d": "daily_prior_qqq_spx_ratio_chg_20d",
        "qqq_spx_ratio_chg_5d": "daily_prior_qqq_spx_ratio_chg_5d",
        "range_exhaustion": "intraday_range_exhaustion",
        "regime_duration": "daily_prior_regime_duration",
        "regime_prob_highvol": "daily_prior_regime_prob_highvol",
        "regime_state": "daily_prior_regime_state",
        "regime_switch": "daily_prior_regime_switch",
        "rsi_14": "daily_prior_rsi_14day",
        "rsp_spy_ratio": "daily_prior_rsp_spy_ratio",
        "rsp_spy_ratio_20d_chg": "daily_prior_rsp_spy_ratio_20d_chg",
        "rsp_spy_ratio_5d_chg": "daily_prior_rsp_spy_ratio_5d_chg",
        "rsp_spy_ratio_zscore_21d": "daily_prior_rsp_spy_ratio_zscore_21d",
        "rv_iv_spread_10d": "daily_prior_rv_iv_spread_10d",
        "rv_iv_spread_20d": "daily_prior_rv_iv_spread_20d",
        "rvol_10m": "intraday_rvol_10m",
        "rvol_15m": "intraday_rvol_15m",
        "rvol_30m": "intraday_rvol_30m",
        "rvol_5m": "intraday_rvol_5m",
        "rvol_intraday_vs_20d": "intraday_rvol_vs_daily_20d",
        "rvol_intraday_vs_5d": "intraday_rvol_vs_daily_5d",
        "sector_divergence_count": "daily_prior_sector_divergence_count",
        "sector_divergence_flag": "daily_prior_sector_divergence_flag",
        "skew_10d": "intraday_skew_10d",
        "skew_25d": "intraday_skew_25d",
        "skew_25d_21d_mean": "daily_prior_skew_25d_21day_mean",
        "skew_25d_21d_std": "daily_prior_skew_25d_21day_std",
        "skew_25d_5d_mean": "daily_prior_skew_25d_5day_mean",
        "skew_25d_5d_std": "daily_prior_skew_25d_5day_std",
        "skew_25d_zscore_21d": "daily_prior_skew_25d_zscore_21day",
        "spx_iwm_ratio": "daily_prior_spx_iwm_ratio",
        "spx_iwm_ratio_chg_20d": "daily_prior_spx_iwm_ratio_chg_20d",
        "spx_iwm_ratio_chg_5d": "daily_prior_spx_iwm_ratio_chg_5d",
        "spx_open": "spx_intraday_first_bar_open",
        "spx_ret_10m": "intraday_spx_ret_10m",
        "spx_ret_15m": "intraday_spx_ret_15m",
        "spx_ret_20d": "daily_prior_spx_ret_20d",
        "spx_ret_30m": "intraday_spx_ret_30m",
        "spx_ret_5d": "daily_prior_spx_ret_5d",
        "spx_ret_5m": "intraday_spx_ret_5m",
        "spx_ret_60d": "daily_prior_spx_ret_60d",
        "spx_vs_intraday_ma15": "intraday_spx_vs_ma15",
        "spx_vs_intraday_ma30": "intraday_spx_vs_ma30",
        "spx_vs_intraday_ma5": "intraday_spx_vs_ma5",
        "spx_vs_sma_180d": "intraday_spx_vs_sma_180d",
        "spx_vs_sma_20d": "intraday_spx_vs_sma_20d",
        "spx_vs_sma_50d": "intraday_spx_vs_sma_50d",
        "spx_vs_sma_7d": "intraday_spx_vs_sma_7d",
        "spx_vs_vwap": "intraday_spx_vs_vwap",
        "tlt_ret_20d": "daily_prior_tlt_ret_20d",
        "tlt_ret_5d": "daily_prior_tlt_ret_5d",
        "tlt_spx_corr_20d": "daily_prior_tlt_spx_corr_20d",
        "tlt_spx_ratio_chg_20d": "daily_prior_tlt_spx_ratio_chg_20d",
        "tlt_spx_ratio_chg_5d": "daily_prior_tlt_spx_ratio_chg_5d",
        "total_call_oi": "prior_day_total_call_oi",
        "total_call_volume": "daily_prior_total_call_volume",
        "total_put_oi": "prior_day_total_put_oi",
        "total_put_volume": "daily_prior_total_put_volume",
        "ts_30dte_atm_mid": "daily_prior_ts_30dte_atm_mid",
        "ts_7dte_atm_mid": "daily_prior_ts_7dte_atm_mid",
        "ts_inverted_7_14": "daily_prior_ts_iv_inverted_7dte_14dte",
        "ts_iv_14dte": "daily_prior_ts_iv_14dte",
        "ts_iv_7dte": "daily_prior_ts_iv_7dte",
        "ts_slope_0_30": "intraday_ts_slope_0dte_30dte",
        "ts_slope_0_7": "intraday_ts_slope_0dte_7dte",
        "ts_slope_7_14": "daily_prior_ts_iv_slope_7dte_14dte",
        "ttm_squeeze": "intraday_ttm_squeeze",
        "vix1d_close": "daily_prior_vix1d_close",
        "vix1d_vs_sma_20": "daily_prior_vix1d_vs_sma_20",
        "vix1d_vs_sma_5": "daily_prior_vix1d_vs_sma_5",
        "vix1d_vs_vix": "daily_prior_vix1d_vs_vix",
        "vix1d_zscore_21d": "daily_prior_vix1d_zscore_21d",
        "vix_close": "vix_intraday_prior_min_close",
        "vix_high": "vix_intraday_running_high",
        "vix_low": "vix_intraday_running_low",
        "vix_term_structure_slope": "daily_prior_vix_term_structure_slope",
        "vix_vs_sma_10": "daily_prior_vix_ratio_sma_10day",
        "vix_vs_sma_20": "daily_prior_vix_ratio_sma_20day",
        "vix_vxv_ratio": "daily_prior_vix_vxv_ratio",
        "vix_vxv_zscore_21d": "daily_prior_vix_vxv_zscore_21d",
        "vol_regime_persistence": "daily_prior_vol_regime_persistence",
        "vvix_close": "daily_prior_vvix_close",
        "vvix_ma20": "daily_prior_vvix_ma20",
        "vvix_vs_vix_ratio": "daily_prior_vvix_vs_vix_ratio",
        "vvix_zscore_21d": "daily_prior_vvix_zscore_21d",
        "wti_crude": "daily_prior_wti_crude",
        "xle_spy_ret_spread_1d": "daily_prior_xle_spy_ret_spread_1d",
        "xlf_spy_ret_spread_1d": "daily_prior_xlf_spy_ret_spread_1d",
        "xlk_spy_ret_spread_1d": "daily_prior_xlk_spy_ret_spread_1d",
        "yield_10y": "daily_prior_yield_10y",
        "yield_10y_chg_21d": "daily_prior_yield_10y_chg_21d",
        "yield_10y_chg_5d": "daily_prior_yield_10y_chg_5d",
        "yield_10y_chg_63d": "daily_prior_yield_10y_chg_63d",
        "yield_2y": "daily_prior_yield_2y",
        "yield_2y_chg_21d": "daily_prior_yield_2y_chg_21d",
        "yield_2y_chg_5d": "daily_prior_yield_2y_chg_5d",
        "yield_2y_chg_63d": "daily_prior_yield_2y_chg_63d",
        "yield_curve_inverted": "daily_prior_yield_curve_inverted",
        "yield_spread_2y10y": "daily_prior_yield_spread_2y10y",
        "yield_spread_2y10y_chg_21d": "daily_prior_yield_spread_2y10y_chg_21d",
        "yield_spread_2y10y_chg_5d": "daily_prior_yield_spread_2y10y_chg_5d",
        "yield_spread_2y10y_chg_63d": "daily_prior_yield_spread_2y10y_chg_63d",
        "yz_hv_10d": "daily_prior_yz_hv_10day",
        "yz_hv_20d": "daily_prior_yz_hv_20day",
        "yz_hv_30d": "daily_prior_yz_hv_30day",
        "yz_hv_5d": "daily_prior_yz_hv_5day",
    }

    # Apply renames (only for columns that exist in the dataframe)
    actual_renames = {k: v for k, v in FEATURE_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=actual_renames)
    print(f"  Renamed {len(actual_renames)} features with temporal prefixes")

    # ── Drop V2-deferred features ──
    v2_features = [
        "net_gex", "gex_imbalance", "zero_gamma_level", "call_wall", "put_wall",
        "call_wall_distance", "put_wall_distance",
        "gex_regime_prob_pos", "gex_regime_prob_trans", "gex_regime_prob_neg",
        "gex_regime_duration", "gex_regime_switch",
        "net_vanna_exposure", "net_vanna_normalized", "vanna_imbalance",
        "net_charm_exposure", "net_charm_normalized", "charm_imbalance", "vanna_charm_ratio",
    ]
    v2_actual = [c for c in v2_features if c in df.columns]
    if v2_actual:
        df = df.drop(columns=v2_actual)
        print(f"\n  Dropped {len(v2_actual)} V2-deferred features (GEX, vanna/charm)")

    # ── Drop features replaced by intraday versions or redundant ──
    drop_replaced = ["atm_call_close", "atm_put_close", "vol_ratio",
                      "vix_open",         # redundant with vix_intraday_prior_min_close (T-1 rule)
                      "ts_0dte_atm_mid",  # redundant with intraday_atm_straddle_mid
                     ]
    drop_actual = [c for c in drop_replaced if c in df.columns]
    if drop_actual:
        df = df.drop(columns=drop_actual)
        print(f"  Dropped {len(drop_actual)} features replaced by intraday versions")

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
