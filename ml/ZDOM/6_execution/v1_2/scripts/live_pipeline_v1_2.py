"""
Live scoring pipeline for ZDOM v1.2.

Reuses the EXACT same code paths as the training build
(build_master_data_v1_2_final.py) so that feature values match
the model pickles bit-for-bit.

Usage:
    # Live mode (every minute from 10:00 onward):
    python live_pipeline_v1_2.py

    # Validation mode (compare against training build):
    python live_pipeline_v1_2.py --validate
    python live_pipeline_v1_2.py --validate --val-date 2025-11-26 --val-time 10:15
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.iv_surface import extract_snapshot_iv_features, extract_term_structure_iv

# Import the EXACT feature computation functions from the training build.
# We add the build script's directory to sys.path so we can import it.
BUILD_SCRIPT_DIR = PROJECT_DIR / "3_feature_engineering" / "v1_2" / "scripts"
if str(BUILD_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_SCRIPT_DIR))

import build_master_data_v1_2_final as bm

# Raw data paths (for validation mode)
RAW_DIR = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2"
DATA_DIR = PROJECT_DIR / "1_data_collection" / "data" / "v1_2"
MASTER_FINAL_FILE = (
    PROJECT_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final.parquet"
)
MODEL_DIR = PROJECT_DIR / "4_models" / "v1_2"
VALIDATION_OUTPUT = (
    PROJECT_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "_live_validation_master_row.parquet"
)

# ThetaData API
THETA_BASE = "http://localhost:25503"

# Strategy configs (same as training build)
STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]
STRATEGY_DEFS = [
    {"name": f"IC_{d:02d}d_25w", "short_delta": d / 100, "wing_width": 25}
    for d in range(5, 50, 5)
]
ENTRY_OPEN_MINUTE = 9 * 60 + 30
TOTAL_SESSION_MINUTES = 390.0


# ---------------------------------------------------------------------------
# Daily source table loading with correct lag rules
# ---------------------------------------------------------------------------

# Lag conventions for daily tables:
#   same_day: data from date D is available on decision date D
#   t1: data from date D is available on decision date D+1 (i.e., use D-1 data)
#   t2: data from date D is available on decision date D+2 (i.e., use D-2 data)
# These are the SAME 8 tables joined by build_master_data_v1_2_daily_joins.py,
# in the SAME order (first-writer-wins for overlapping columns).
# Tables used only by family 7/9 (econ_calendar, mag7_earnings, regime_features,
# spx_daily, vix_daily, vix1d_daily) are loaded directly by those family functions.
DAILY_JOIN_SPECS = [
    ("spxw_0dte_oi_daily_v1_2.parquet", "same_day"),
    ("spxw_term_structure_daily_v1_2.parquet", "t1"),
    ("fred_daily_features_v1_2.parquet", "t1"),
    ("cross_asset_daily_features_v1_2.parquet", "t1"),
    ("breadth_daily_features_v1_2.parquet", "t1"),
    ("vol_context_daily_features_v1_2.parquet", "t1"),
    ("macro_regime_v1_2.parquet", "t1"),
    ("presidential_cycles_v1_2.parquet", "same_day"),
]


def load_daily_features_for_date(today: pd.Timestamp) -> dict:
    """Load all 11+ daily source parquets and apply lag rules.

    Returns a flat dict of {column_name: value} for the given date.
    """
    # Build a sorted list of all trading dates from the OI table (most complete date index)
    all_dates = set()
    for fname, _ in DAILY_JOIN_SPECS:
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_parquet(path, columns=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)
    date_idx = {d: i for i, d in enumerate(all_dates)}

    daily_values = {}

    for fname, lag in DAILY_JOIN_SPECS:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [WARN] Missing daily table: {path.name}")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)

        # Determine effective source date based on lag rule
        if lag == "same_day":
            source_date = today
        elif lag == "t1":
            # Use data from the previous trading day
            candidates = [d for d in all_dates if d < today]
            source_date = candidates[-1] if candidates else None
        elif lag == "t2":
            # Use data from two trading days ago
            candidates = [d for d in all_dates if d < today]
            source_date = candidates[-2] if len(candidates) >= 2 else None
        else:
            raise ValueError(f"Unknown lag: {lag}")

        if source_date is None:
            continue

        row = df[df["date"] == source_date]
        if row.empty:
            # Try nearest prior date
            prior = df[df["date"] <= source_date]
            if prior.empty:
                continue
            row = prior.iloc[[-1]]

        for col in row.columns:
            if col == "date":
                continue
            # Match training build: skip columns already present (first-writer wins)
            if col not in daily_values:
                daily_values[col] = row[col].iloc[0]

    return daily_values


# ---------------------------------------------------------------------------
# ThetaData API helpers
# ---------------------------------------------------------------------------

def theta_get_csv(endpoint: str, params: dict) -> pd.DataFrame:
    """Call ThetaData REST API with CSV format and return DataFrame."""
    url = f"{THETA_BASE}{endpoint}"
    params["format"] = "csv"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.startswith("error"):
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


def fetch_index_1min(symbol: str, date_str: str) -> pd.DataFrame:
    """Fetch 1-min OHLC for an index from ThetaData."""
    df = theta_get_csv(
        "/v3/index/history/ohlc",
        {"symbol": symbol, "start_date": date_str, "end_date": date_str, "interval": "1m"},
    )
    if df.empty:
        return df
    # ThetaData returns ms_of_day or datetime columns; normalize
    if "ms_of_day" in df.columns and "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(
            df["ms_of_day"], unit="ms"
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_greeks_0dte(date_str: str, spx_price: float) -> pd.DataFrame:
    """Fetch SPXW 0DTE greeks within ATM +/- 250 from ThetaData."""
    atm = int(round(spx_price / 5.0) * 5)
    low_strike = (atm - 250) * 1000
    high_strike = (atm + 250) * 1000
    # Use the exp date = today for 0DTE
    df = theta_get_csv(
        "/v3/option/history/greeks/second_order",
        {
            "root": "SPXW",
            "exp": date_str,
            "start_date": date_str,
            "end_date": date_str,
            "strike_gte": str(low_strike),
            "strike_lte": str(high_strike),
        },
    )
    return df


# ---------------------------------------------------------------------------
# Decision state construction (mirrors build_intraday_options_state_v1_2.py)
# ---------------------------------------------------------------------------

def build_decision_state_from_greeks(
    greeks_df: pd.DataFrame,
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    vix1d_bars: pd.DataFrame,
    decision_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Build a decision-state table for a single decision_datetime.

    This mirrors build_intraday_options_state_v1_2.build_day() but for
    a single minute from accumulated data.
    """
    # Always recompute mid (ThetaData returns NaN mid column)
    if "bid" in greeks_df.columns and "ask" in greeks_df.columns:
        greeks_df = greeks_df.copy()
        greeks_df["mid"] = (greeks_df["bid"] + greeks_df["ask"]) / 2

    # state_datetime is one minute before decision_datetime
    state_dt = decision_dt - pd.Timedelta(minutes=1)

    # Filter greeks to the state_datetime minute bucket
    greeks = greeks_df.copy()
    greeks["timestamp"] = pd.to_datetime(greeks["timestamp"])
    minute_greeks = greeks[greeks["timestamp"] == state_dt].copy()
    if minute_greeks.empty:
        # Try floor to minute
        minute_greeks = greeks[
            greeks["timestamp"].dt.floor("min") == state_dt
        ].copy()
    if minute_greeks.empty:
        return pd.DataFrame()

    minute_greeks = minute_greeks.rename(
        columns={
            "timestamp": "state_datetime",
            "bid": "bid_close",
            "ask": "ask_close",
            "mid": "mid_close",
        }
    )

    # Pivot to call/put columns
    value_cols = [c for c in minute_greeks.columns if c not in {"date", "state_datetime", "strike", "right"}]
    pivoted = (
        minute_greeks.set_index(["date", "state_datetime", "strike", "right"])[value_cols]
        .unstack("right")
        .sort_index(axis=1)
    )
    pivoted.columns = [f"{side}_{field}" for field, side in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Add index bars
    spx_row = spx_bars[spx_bars["datetime"] == state_dt]
    if not spx_row.empty:
        pivoted["spx_open"] = spx_row["open"].iloc[0]
        pivoted["spx_high"] = spx_row["high"].iloc[0]
        pivoted["spx_low"] = spx_row["low"].iloc[0]
        pivoted["spx_close"] = spx_row["close"].iloc[0]
    else:
        for c in ["spx_open", "spx_high", "spx_low", "spx_close"]:
            pivoted[c] = np.nan

    vix_row = vix_bars[vix_bars["datetime"] == state_dt] if not vix_bars.empty else pd.DataFrame()
    if not vix_row.empty:
        pivoted["vix_open"] = vix_row["vix_open"].iloc[0]
        pivoted["vix_high"] = vix_row["vix_high"].iloc[0]
        pivoted["vix_low"] = vix_row["vix_low"].iloc[0]
        pivoted["vix_close"] = vix_row["vix_close"].iloc[0]
    else:
        for c in ["vix_open", "vix_high", "vix_low", "vix_close"]:
            pivoted[c] = np.nan

    vix1d_row = vix1d_bars[vix1d_bars["datetime"] == state_dt] if not vix1d_bars.empty else pd.DataFrame()
    if not vix1d_row.empty:
        pivoted["vix1d_open"] = vix1d_row["vix1d_open"].iloc[0]
        pivoted["vix1d_high"] = vix1d_row["vix1d_high"].iloc[0]
        pivoted["vix1d_low"] = vix1d_row["vix1d_low"].iloc[0]
        pivoted["vix1d_close"] = vix1d_row["vix1d_close"].iloc[0]
    else:
        for c in ["vix1d_open", "vix1d_high", "vix1d_low", "vix1d_close"]:
            pivoted[c] = np.nan

    pivoted["decision_datetime"] = decision_dt
    pivoted["state_datetime"] = state_dt
    pivoted["date"] = decision_dt.normalize()

    return pivoted


# ---------------------------------------------------------------------------
# IC construction (mirrors build_master_data_v1_2.build_rows_for_slice)
# ---------------------------------------------------------------------------

def build_ic_rows(decision_state: pd.DataFrame, decision_dt: pd.Timestamp) -> pd.DataFrame:
    """Build IC strategy rows from decision state, same logic as training."""
    g = decision_state.copy()
    spx_prev_min_close = g["spx_close"].iloc[0]
    if pd.isna(spx_prev_min_close):
        return pd.DataFrame()

    date = decision_dt.normalize()
    strikes = g["strike"].to_numpy(dtype=float)
    call_bid = g["call_bid_close"].to_numpy(dtype=float)
    put_bid = g["put_bid_close"].to_numpy(dtype=float)
    call_mid = g["call_mid_close"].to_numpy(dtype=float)
    put_mid = g["put_mid_close"].to_numpy(dtype=float)
    call_delta = g["call_delta"].to_numpy(dtype=float)
    put_delta = g["put_delta"].to_numpy(dtype=float)
    call_iv = g["call_implied_vol"].to_numpy(dtype=float)
    put_iv = g["put_implied_vol"].to_numpy(dtype=float)

    call_short_valid = (call_delta > 0) & (call_delta < 1) & (call_bid > 0)
    put_short_valid = (put_delta < 0) & (put_delta > -1) & (put_bid > 0)

    if not call_short_valid.any() or not put_short_valid.any():
        return pd.DataFrame()

    rows = []
    for strat in STRATEGY_DEFS:
        target_call_delta = strat["short_delta"]
        target_put_delta = -strat["short_delta"]
        wing = strat["wing_width"]

        call_diff = np.where(call_short_valid, np.abs(call_delta - target_call_delta), np.inf)
        put_diff = np.where(put_short_valid, np.abs(put_delta - target_put_delta), np.inf)

        sc_idx = int(np.argmin(call_diff))
        sp_idx = int(np.argmin(put_diff))

        if not np.isfinite(call_diff[sc_idx]) or not np.isfinite(put_diff[sp_idx]):
            continue

        sc_strike = strikes[sc_idx]
        sp_strike = strikes[sp_idx]
        if not (sp_strike < spx_prev_min_close < sc_strike):
            continue

        # Find long legs
        def pick_long_idx(valid_mask, target_strike):
            valid_idx = np.flatnonzero(valid_mask)
            if len(valid_idx) == 0:
                return -1
            valid_strikes = strikes[valid_idx]
            pos = np.searchsorted(valid_strikes, target_strike)
            candidates = []
            if pos < len(valid_strikes):
                candidates.append(valid_idx[pos])
            if pos > 0:
                candidates.append(valid_idx[pos - 1])
            if not candidates:
                return -1
            best = min(candidates, key=lambda idx: abs(strikes[idx] - target_strike))
            if abs(strikes[best] - target_strike) > 10:
                return -1
            return int(best)

        call_long_valid = call_bid > 0
        put_long_valid = put_bid > 0

        lc_idx = pick_long_idx(call_long_valid, sc_strike + wing)
        lp_idx = pick_long_idx(put_long_valid, sp_strike - wing)
        if lc_idx < 0 or lp_idx < 0:
            continue

        call_wing_width = float(strikes[lc_idx] - sc_strike)
        put_wing_width = float(sp_strike - strikes[lp_idx])
        if call_wing_width <= 0 or put_wing_width <= 0:
            continue

        credit = float(call_mid[sc_idx] + put_mid[sp_idx] - call_mid[lc_idx] - put_mid[lp_idx])
        if not np.isfinite(credit) or credit <= 0:
            continue

        rows.append({
            "decision_datetime": decision_dt,
            "date": date,
            "strategy": strat["name"],
            "spx_prev_min_close": float(spx_prev_min_close),
            "short_call": float(sc_strike),
            "short_put": float(sp_strike),
            "long_call": float(strikes[lc_idx]),
            "long_put": float(strikes[lp_idx]),
            "call_wing_width": call_wing_width,
            "put_wing_width": put_wing_width,
            "sc_delta": float(call_delta[sc_idx]),
            "sp_delta": float(put_delta[sp_idx]),
            "sc_iv": float(call_iv[sc_idx]),
            "sp_iv": float(put_iv[sp_idx]),
            "credit": round(credit, 4),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering (reuses EXACT functions from build_master_data_v1_2_final)
# ---------------------------------------------------------------------------

def build_feature_vector(
    ic_rows: pd.DataFrame,
    daily_values: dict,
    spx_1m_history: pd.DataFrame,
    vix_bars_history: pd.DataFrame,
    vix1d_bars_history: pd.DataFrame,
    decision_state_snapshot: pd.DataFrame,
    decision_dt: pd.Timestamp,
    model_feature_cols: list[str],
) -> pd.DataFrame:
    """Build a feature vector using the same pipeline as the training build.

    Returns a DataFrame with one row per strategy, containing exactly
    the 285 model feature columns.
    """
    df = ic_rows.copy()
    today = decision_dt.normalize()

    # --- Daily values injection ---
    # The training build joins daily parquets onto the master table via
    # the build_master_data_v1_2_daily_joins.py script. We replicate
    # that by injecting daily_values as columns onto each row.
    daily_df = pd.DataFrame({col: [val] * len(df) for col, val in daily_values.items()}, index=df.index)
    df = pd.concat([df, daily_df], axis=1)
    df = df.copy()  # defragment

    # --- Family 1: Strategy/time derived fields ---
    df = bm.add_family_1(df)

    # --- Family 2: Calendar/event pass-through fields ---
    df = bm.add_family_2(df)

    # --- Family 3: Daily prior base projections ---
    df = bm.add_family_3(df)

    # --- Family 4: Daily lookback / rolling projections ---
    df = bm.add_family_4(df)

    # Defragment after mass column injection
    df = df.copy()

    # --- Family 5: Per-minute intraday features ---
    # _per_minute_intraday_features needs full-day SPX 1min bars up to now
    spx_1m = spx_1m_history.copy()
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.normalize()
    spx_1m = spx_1m.sort_values("datetime").reset_index(drop=True)

    per_minute = bm._per_minute_intraday_features(spx_1m)
    fixed_daily = bm._daily_fixed_intraday_features(spx_1m)

    # VIX intraday features - build from accumulated decision state bars
    vix_state = _build_vix_intraday_from_bars(vix_bars_history, decision_dt)

    intraday = per_minute.merge(fixed_daily, on="date", how="left").merge(
        vix_state, on="decision_datetime", how="left"
    )
    # Safe merge: drop overlapping columns from df before merging intraday
    overlap = [c for c in intraday.columns if c not in {"decision_datetime", "date"} and c in df.columns]
    if overlap:
        df = df.drop(columns=overlap, errors="ignore")
    df = df.merge(intraday, on=["decision_datetime", "date"], how="left")
    df = bm.sanitize_numeric(df)

    # --- Family 6: Cross-horizon intraday features ---
    spx_daily = _load_spx_daily_for_live(daily_values, today)
    daily_ctx = bm._daily_spx_context(spx_daily)
    intraday_ranges = bm._intraday_range_points(spx_1m)

    # Safe merge for daily context and intraday ranges
    overlap_ctx = [c for c in daily_ctx.columns if c != "date" and c in df.columns]
    if overlap_ctx:
        df = df.drop(columns=overlap_ctx, errors="ignore")
    df = df.merge(daily_ctx, on="date", how="left")
    overlap_rng = [c for c in intraday_ranges.columns if c != "decision_datetime" and c in df.columns]
    if overlap_rng:
        df = df.drop(columns=overlap_rng, errors="ignore")
    df = df.merge(intraday_ranges, on="decision_datetime", how="left")
    df["spx_intraday_atm_strike"] = np.round(df["spx_prev_min_close"] / 5.0) * 5.0

    for w in (7, 20, 50, 180):
        sma_col = f"sma_{w}day"
        df[f"intraday_spx_vs_sma_{w}day"] = np.where(
            df[sma_col] > 0,
            (df["spx_prev_min_close"] - df[sma_col]) / df[sma_col] * 100.0,
            np.nan,
        )

    ma15 = np.where(
        (1 + df["intraday_spx_vs_ma15"] / 100.0) != 0,
        df["spx_prev_min_close"] / (1 + df["intraday_spx_vs_ma15"] / 100.0),
        np.nan,
    )
    df["intraday_ma15_vs_sma_7day"] = np.where(
        df["sma_7day"] > 0, (ma15 - df["sma_7day"]) / df["sma_7day"] * 100.0, np.nan,
    )
    df["intraday_ma15_vs_sma_20day"] = np.where(
        df["sma_20day"] > 0, (ma15 - df["sma_20day"]) / df["sma_20day"] * 100.0, np.nan,
    )
    df["intraday_rvol_vs_daily_5day"] = np.where(
        df["daily_rv_5day"] > 0, df["intraday_rvol_30m"] / df["daily_rv_5day"], np.nan,
    )
    df["intraday_rvol_vs_daily_20day"] = np.where(
        df["daily_rv_20day"] > 0, df["intraday_rvol_30m"] / df["daily_rv_20day"], np.nan,
    )
    df["intraday_range_exhaustion"] = np.where(
        df["atr_14day"] > 0, df["intraday_running_range_points"] / df["atr_14day"], np.nan,
    )
    df["intraday_range_vs_atr"] = np.where(
        df["atr_14day"] > 0, df["first_30m_range_points"] / df["atr_14day"] * 100.0, np.nan,
    )

    df = df.drop(
        columns=[
            "daily_open", "daily_high", "daily_low", "daily_close",
            "daily_rv_5day", "daily_rv_20day", "atr_14day",
            "sma_7day", "sma_20day", "sma_50day", "sma_180day",
            "intraday_running_range_points", "first_30m_range_points",
        ],
        errors="ignore",
    )
    df = bm.sanitize_numeric(df)

    # --- Family 7: Daily SPX/VIX/VIX1D/IV technicals ---
    # These are all pre-shifted by 1 day in the training build.
    # We already have daily_values injected, but family 7 computes
    # new features from raw daily data (SPX technicals, VIX features, etc.)
    # We need to build these from the SPX daily / VIX 1min / VIX1D 1min histories.
    def _safe_merge(left, right, on="date"):
        """Merge without creating _x/_y columns; new columns from right take precedence."""
        overlap = [c for c in right.columns if c != on and c in left.columns]
        if overlap:
            left = left.drop(columns=overlap, errors="ignore")
        return left.merge(right, on=on, how="left")

    daily_spx_tech = bm._build_daily_spx_technical_features(spx_daily)
    df = _safe_merge(df, daily_spx_tech)

    daily_vix_feats = _build_daily_vix_features_for_live(daily_values, today)
    df = _safe_merge(df, daily_vix_feats)

    daily_vix1d_feats = _build_daily_vix1d_features_for_live(daily_values, today)
    df = _safe_merge(df, daily_vix1d_feats)

    daily_iv = _build_daily_snapshot_iv_features_for_live(decision_state_snapshot, daily_values, today)
    df = _safe_merge(df, daily_iv)

    # Cross features from family 7
    df["daily_prior_vix1d_vs_vix"] = np.where(
        df["daily_prior_vix_close"] > 0,
        df["daily_prior_vix1d_close"] / df["daily_prior_vix_close"],
        np.nan,
    )
    df["daily_prior_iv_rv_ratio"] = np.where(
        df["daily_prior_yz_hv_20day"] > 0,
        df["daily_prior_atm_iv"] / df["daily_prior_yz_hv_20day"],
        np.nan,
    )
    for w in (5, 10, 20, 30):
        hv_col = f"daily_prior_yz_hv_{w}day"
        df[f"daily_prior_hv_iv_ratio_{w}day"] = np.where(
            df[hv_col] > 0, df["daily_prior_atm_iv"] / df[hv_col], np.nan,
        )
    df = bm.sanitize_numeric(df)

    # --- Family 8: Decision state snapshot features ---
    snap_feats = _decision_state_snapshot_features_live(decision_state_snapshot, df, decision_dt)
    overlap_snap = [c for c in snap_feats.columns if c != "decision_datetime" and c in df.columns]
    if overlap_snap:
        df = df.drop(columns=overlap_snap, errors="ignore")
    df = df.merge(snap_feats, on="decision_datetime", how="left")
    df = bm.sanitize_numeric(df)

    # --- Family 9: Term structure IV, options volume, sector spreads, regime, events, earnings ---
    # Most of these come from daily_values already injected. But we need to compute
    # some cross-features and the specific family 9 derived features.
    df = _add_family_9_live(df, daily_values, today)
    df = bm.sanitize_numeric(df)

    # --- Select exactly the model's feature columns ---
    for col in model_feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[model_feature_cols + ["decision_datetime", "date", "strategy"]]


# ---------------------------------------------------------------------------
# Helper: ensure shifted daily DataFrames cover today
# ---------------------------------------------------------------------------

def _ensure_today_coverage(daily: pd.DataFrame, today: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """After shift(1), ensure today is present with valid values.

    Many daily feature functions build a time series, shift(1) for T-1 lag,
    then return. If today isn't in the source data, the shifted result won't
    have a row for today. This helper adds today and forward-fills any NaN
    feature values from the last valid observation.
    """
    if daily.empty:
        return daily
    daily = daily.copy()
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.normalize()
    today_norm = today.normalize()
    if today_norm not in daily[date_col].values:
        today_row = pd.DataFrame([{date_col: today_norm}])
        daily = pd.concat([daily, today_row], ignore_index=True)
        daily = daily.sort_values(date_col).reset_index(drop=True)
    # Forward-fill feature columns so NaN gaps from data quality issues
    # get the last valid value (correct for T-1 shifted features)
    feat_cols = [c for c in daily.columns if c != date_col]
    daily[feat_cols] = daily[feat_cols].ffill()
    return daily


# ---------------------------------------------------------------------------
# Live-adapted helper functions (replicate training-build logic for single-day)
# ---------------------------------------------------------------------------

def _build_vix_intraday_from_bars(vix_bars: pd.DataFrame, decision_dt: pd.Timestamp) -> pd.DataFrame:
    """Build VIX intraday features from accumulated VIX bars."""
    if vix_bars.empty:
        return pd.DataFrame({
            "decision_datetime": [decision_dt],
            "vix_intraday_prior_min_close": [np.nan],
            "vix_intraday_running_high": [np.nan],
            "vix_intraday_running_low": [np.nan],
        })

    vix = vix_bars.copy()
    vix["datetime"] = pd.to_datetime(vix["datetime"])
    today = decision_dt.normalize()
    vix = vix[vix["datetime"].dt.normalize() == today].sort_values("datetime")

    if vix.empty:
        return pd.DataFrame({
            "decision_datetime": [decision_dt],
            "vix_intraday_prior_min_close": [np.nan],
            "vix_intraday_running_high": [np.nan],
            "vix_intraday_running_low": [np.nan],
        })

    # Build running features for all minutes, then filter to decision minutes
    rows = []
    running_high = -np.inf
    running_low = np.inf
    for _, r in vix.iterrows():
        running_high = max(running_high, r["vix_high"])
        running_low = min(running_low, r["vix_low"])
        dt = r["datetime"] + pd.Timedelta(minutes=1)
        rows.append({
            "decision_datetime": dt,
            "vix_intraday_prior_min_close": r["vix_close"],
            "vix_intraday_running_high": running_high,
            "vix_intraday_running_low": running_low,
        })

    return pd.DataFrame(rows)


def _load_spx_daily_for_live(daily_values: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Load SPX daily from parquet for computing technicals.

    The training build uses the full SPX daily history. For live, we load
    the same file.
    """
    spx_daily_file = DATA_DIR / "spx_daily_v1_2.parquet"
    if not spx_daily_file.exists():
        return pd.DataFrame(columns=["date", "spx_open", "spx_high", "spx_low", "spx_close"])
    daily = pd.read_parquet(spx_daily_file)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily.sort_values("date").reset_index(drop=True)


def _build_daily_vix_features_for_live(daily_values: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_daily_vix_features for live use.

    Uses VIX 1min history to build daily VIX features, shifted by 1 day.
    """
    vix_file = RAW_DIR / "vix_1min.parquet"
    if not vix_file.exists():
        return pd.DataFrame({"date": [today]})

    vix = pd.read_parquet(vix_file, columns=["datetime", "vix_close"])
    vix["datetime"] = pd.to_datetime(vix["datetime"])
    vix["date"] = vix["datetime"].dt.normalize()
    daily = (
        vix.groupby("date", sort=True)["vix_close"]
        .last()
        .reset_index()
        .sort_values("date")
    )
    daily["daily_prior_vix_close"] = daily["vix_close"]
    sma10 = daily["vix_close"].rolling(10, min_periods=5).mean()
    sma20 = daily["vix_close"].rolling(20, min_periods=10).mean()
    daily["daily_prior_vix_ratio_sma_10day"] = ((daily["vix_close"] - sma10) / sma10.replace(0, np.nan)) * 100.0
    daily["daily_prior_vix_ratio_sma_20day"] = ((daily["vix_close"] - sma20) / sma20.replace(0, np.nan)) * 100.0
    feature_cols = [c for c in daily.columns if c not in {"date", "vix_close"}]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    return daily[["date", "daily_prior_vix_close", "daily_prior_vix_ratio_sma_10day", "daily_prior_vix_ratio_sma_20day"]]


def _build_daily_vix1d_features_for_live(daily_values: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_daily_vix1d_features for live use."""
    vix1d_file = RAW_DIR / "vix1d_1min.parquet"
    if not vix1d_file.exists():
        return pd.DataFrame({"date": [today]})

    vix1d = pd.read_parquet(vix1d_file, columns=["datetime", "vix1d_close"])
    vix1d["datetime"] = pd.to_datetime(vix1d["datetime"])
    vix1d["date"] = vix1d["datetime"].dt.normalize()
    daily = (
        vix1d.groupby("date", sort=True)["vix1d_close"]
        .last()
        .reset_index()
        .sort_values("date")
    )

    close = daily["vix1d_close"]
    sma5 = close.rolling(5, min_periods=3).mean()
    sma20 = close.rolling(20, min_periods=10).mean()
    mean21 = close.rolling(21, min_periods=10).mean()
    std21 = close.rolling(21, min_periods=10).std()

    daily["daily_prior_vix1d_close"] = close
    daily["daily_prior_vix1d_vs_sma_5"] = ((close - sma5) / sma5.replace(0, np.nan)) * 100.0
    daily["daily_prior_vix1d_vs_sma_5day"] = ((close - sma5) / sma5.replace(0, np.nan)) * 100.0
    daily["daily_prior_vix1d_vs_sma_20day"] = ((close - sma20) / sma20.replace(0, np.nan)) * 100.0
    daily["daily_prior_vix1d_zscore_21day"] = (close - mean21) / std21.replace(0, np.nan)

    feature_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    return daily


def _build_daily_snapshot_iv_features_for_live(
    decision_state_history: pd.DataFrame,
    daily_values: dict,
    today: pd.Timestamp,
) -> pd.DataFrame:
    """Replicate _build_daily_snapshot_iv_features for live use.

    Uses the full decision state history + greeks daily chunks for recent dates.
    """
    decision_file = DATA_DIR / "intraday_decision_state_v1_2.parquet"

    rows = []

    # 1. Process historical decision state
    if decision_file.exists():
        cols = [
            "date", "decision_datetime", "strike",
            "call_delta", "put_delta", "call_implied_vol", "put_implied_vol",
        ]
        ds = pd.read_parquet(decision_file, columns=cols)
        ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
        ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])
        snap = ds[ds["decision_datetime"].dt.time == pd.Timestamp("10:00:00").time()].copy()

        for date, grp in snap.groupby("date", sort=True):
            snapshot_rows = []
            for _, r in grp.iterrows():
                snapshot_rows.append({
                    "strike": r["strike"], "right": "call",
                    "delta": r["call_delta"], "implied_vol": r["call_implied_vol"], "iv_error": 0,
                })
                snapshot_rows.append({
                    "strike": r["strike"], "right": "put",
                    "delta": r["put_delta"], "implied_vol": r["put_implied_vol"], "iv_error": 0,
                })
            snapshot = pd.DataFrame(snapshot_rows)
            feats = extract_snapshot_iv_features(snapshot)
            feats["date"] = date
            rows.append(feats)

    # 2. Process greeks daily chunks for dates not in decision state
    ds_max_date = max(r["date"] for r in rows) if rows else pd.Timestamp("1970-01-01")
    chunks_dir = RAW_DIR / "greeks_daily_chunks"
    if chunks_dir.exists():
        for chunk_file in sorted(chunks_dir.glob("greeks_*.parquet")):
            chunk_date_str = chunk_file.stem.replace("greeks_", "")
            chunk_date = pd.Timestamp(chunk_date_str).normalize()
            if chunk_date <= ds_max_date:
                continue  # already covered by decision state
            chunk = pd.read_parquet(chunk_file)
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
            # Use 10:00 (9:59 state) snapshot — same as training
            snap_time = pd.Timestamp(f"{chunk_date_str[:4]}-{chunk_date_str[4:6]}-{chunk_date_str[6:8]} 09:59:00")
            minute_data = chunk[chunk["timestamp"] == snap_time]
            if minute_data.empty:
                # Try floor
                minute_data = chunk[chunk["timestamp"].dt.floor("min") == snap_time]
            if minute_data.empty:
                continue
            snapshot_rows = []
            for _, r in minute_data.iterrows():
                snapshot_rows.append({
                    "strike": r["strike"], "right": str(r["right"]).lower(),
                    "delta": r["delta"], "implied_vol": r["implied_vol"],
                    "iv_error": r.get("iv_error", 0),
                })
            snapshot = pd.DataFrame(snapshot_rows)
            feats = extract_snapshot_iv_features(snapshot)
            feats["date"] = chunk_date
            rows.append(feats)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame({"date": [today]})

    daily["daily_prior_atm_iv_5day_mean"] = daily["atm_iv"].rolling(5, min_periods=3).mean()
    daily["daily_prior_atm_iv_5day_std"] = daily["atm_iv"].rolling(5, min_periods=3).std()
    daily["daily_prior_atm_iv_21day_mean"] = daily["atm_iv"].rolling(21, min_periods=10).mean()
    daily["daily_prior_atm_iv_21day_std"] = daily["atm_iv"].rolling(21, min_periods=10).std()
    daily["daily_prior_atm_iv_zscore_21day"] = (
        (daily["atm_iv"] - daily["daily_prior_atm_iv_21day_mean"])
        / daily["daily_prior_atm_iv_21day_std"].replace(0, np.nan)
    )

    for w in (5, 21):
        daily[f"daily_prior_skew_25d_{w}day_mean"] = daily["skew_25d"].rolling(w, min_periods=max(3, w // 2)).mean()
        daily[f"daily_prior_skew_25d_{w}day_std"] = daily["skew_25d"].rolling(w, min_periods=max(3, w // 2)).std()
    daily["daily_prior_skew_25d_zscore_21day"] = (
        (daily["skew_25d"] - daily["daily_prior_skew_25d_21day_mean"])
        / daily["daily_prior_skew_25d_21day_std"].replace(0, np.nan)
    )
    daily["daily_prior_iv_rank_30day"] = daily["atm_iv"].rolling(30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False,
    )

    rename_map = {
        "atm_iv": "daily_prior_atm_iv",
        "iv_25d_call": "daily_prior_iv_25d_call",
        "iv_25d_put": "daily_prior_iv_25d_put",
        "skew_25d": "daily_prior_skew_25d",
        "iv_10d_call": "daily_prior_iv_10d_call",
        "iv_10d_put": "daily_prior_iv_10d_put",
        "skew_10d": "daily_prior_skew_10d",
        "iv_smile_curve": "daily_prior_iv_smile_curve",
    }
    daily = daily.rename(columns=rename_map)

    feature_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    return _ensure_today_coverage(daily, today)


def _decision_state_snapshot_features_live(
    decision_state_snapshot: pd.DataFrame,
    master_df: pd.DataFrame,
    decision_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Replicate _decision_state_snapshot_features for a single minute.

    Uses the decision_state_snapshot (for the current decision_datetime) and
    daily values from master_df to compute intraday options snapshot features.
    """
    ds = decision_state_snapshot.copy()
    if "decision_datetime" not in ds.columns:
        ds["decision_datetime"] = decision_dt

    # Get needed metadata from master_df
    unique_master = master_df[["decision_datetime", "max_pain_strike", "ts_7dte_atm_mid", "ts_30dte_atm_mid"]].drop_duplicates("decision_datetime")

    # Ensure decision state columns match expectations
    if "spx_close" not in ds.columns and "spx_prev_min_close" in ds.columns:
        ds = ds.rename(columns={"spx_prev_min_close": "spx_close"})

    cols_needed = ["decision_datetime", "strike", "call_delta", "put_delta",
                   "call_mid_close", "put_mid_close", "call_implied_vol",
                   "put_implied_vol", "spx_close"]
    for c in cols_needed:
        if c not in ds.columns:
            ds[c] = np.nan

    ds = ds[cols_needed].copy()
    ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])

    want_times = set(unique_master["decision_datetime"])
    ds = ds[ds["decision_datetime"].isin(want_times)]

    if ds.empty:
        # Return empty with correct columns
        return pd.DataFrame(columns=["decision_datetime"])

    dt_values = ds["decision_datetime"].to_numpy()
    strikes_all = ds["strike"].to_numpy(dtype=float)
    call_delta_all = ds["call_delta"].to_numpy(dtype=float)
    put_delta_all = ds["put_delta"].to_numpy(dtype=float)
    call_mid_all = ds["call_mid_close"].to_numpy(dtype=float)
    put_mid_all = ds["put_mid_close"].to_numpy(dtype=float)
    call_iv_all = ds["call_implied_vol"].to_numpy(dtype=float)
    put_iv_all = ds["put_implied_vol"].to_numpy(dtype=float)
    spx_all = ds["spx_close"].to_numpy(dtype=float)

    meta_map = unique_master.set_index("decision_datetime").to_dict("index")
    boundaries = np.r_[0, np.flatnonzero(dt_values[1:] != dt_values[:-1]) + 1, len(ds)]
    rows = []

    def nearest_iv(delta_arr, iv_arr, target, tolerance):
        mask = np.isfinite(delta_arr) & np.isfinite(iv_arr) & (iv_arr > 0.01) & (iv_arr < 5.0)
        if not mask.any():
            return np.nan
        d = np.abs(delta_arr[mask] - target)
        idx = np.argmin(d)
        return float(iv_arr[mask][idx]) if d[idx] <= tolerance else np.nan

    def strike_lookup(strikes, values, strike):
        idx = np.searchsorted(strikes, strike)
        if idx < len(strikes) and strikes[idx] == strike:
            val = values[idx]
            return float(val) if np.isfinite(val) and val > 0 else np.nan
        return np.nan

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        dt = dt_values[start]
        strikes = strikes_all[start:end]
        call_delta = call_delta_all[start:end]
        put_delta = put_delta_all[start:end]
        call_mid = call_mid_all[start:end]
        put_mid = put_mid_all[start:end]
        call_iv = call_iv_all[start:end]
        put_iv = put_iv_all[start:end]
        spx = float(spx_all[start])
        atm = bm._round_strike(spx, 5.0)

        atm_call_mid = strike_lookup(strikes, call_mid, atm)
        atm_put_mid = strike_lookup(strikes, put_mid, atm)
        atm_straddle_mid = (
            atm_call_mid + atm_put_mid
            if pd.notna(atm_call_mid) and pd.notna(atm_put_mid)
            else np.nan
        )

        iv_25d_call = nearest_iv(call_delta, call_iv, 0.25, 0.10)
        iv_25d_put = nearest_iv(put_delta, put_iv, -0.25, 0.10)
        iv_10d_call = nearest_iv(call_delta, call_iv, 0.10, 0.10)
        iv_10d_put = nearest_iv(put_delta, put_iv, -0.10, 0.10)
        atm_call_iv = nearest_iv(call_delta, call_iv, 0.50, 0.15)
        atm_put_iv = nearest_iv(put_delta, put_iv, -0.50, 0.15)
        if pd.notna(atm_call_iv) and pd.notna(atm_put_iv):
            atm_iv = (atm_call_iv + atm_put_iv) / 2.0
        elif pd.notna(atm_call_iv):
            atm_iv = atm_call_iv
        else:
            atm_iv = atm_put_iv
        skew_25d = iv_25d_put - iv_25d_call if pd.notna(iv_25d_put) and pd.notna(iv_25d_call) else np.nan
        skew_10d = iv_10d_put - iv_10d_call if pd.notna(iv_10d_put) and pd.notna(iv_10d_call) else np.nan
        iv_smile = ((iv_25d_put + iv_25d_call) / 2.0 - atm_iv) if pd.notna(iv_25d_put) and pd.notna(iv_25d_call) and pd.notna(atm_iv) else np.nan

        def ic_credit(short_offset, wing_offset):
            sc = strike_lookup(strikes, call_mid, atm + short_offset)
            lc = strike_lookup(strikes, call_mid, atm + wing_offset)
            sp = strike_lookup(strikes, put_mid, atm - short_offset)
            lp = strike_lookup(strikes, put_mid, atm - wing_offset)
            vals = [sc, lc, sp, lp]
            if any(pd.isna(v) for v in vals):
                return np.nan
            credit = (sc + sp) - (lc + lp)
            return float(credit) if credit > 0 else np.nan

        meta = meta_map.get(pd.Timestamp(dt), {})
        max_pain = meta.get("max_pain_strike", np.nan)
        ts7 = meta.get("ts_7dte_atm_mid", np.nan)
        ts30 = meta.get("ts_30dte_atm_mid", np.nan)

        rows.append({
            "decision_datetime": dt,
            "intraday_atm_call_mid": atm_call_mid,
            "intraday_atm_put_mid": atm_put_mid,
            "intraday_atm_straddle_mid": atm_straddle_mid,
            "intraday_atm_iv": atm_iv,
            "intraday_iv_25d_call": iv_25d_call,
            "intraday_iv_25d_put": iv_25d_put,
            "intraday_skew_25d": skew_25d,
            "intraday_iv_10d_call": iv_10d_call,
            "intraday_iv_10d_put": iv_10d_put,
            "intraday_skew_10d": skew_10d,
            "intraday_iv_smile_curve": iv_smile,
            "intraday_ic_credit_50pt": ic_credit(50, 100),
            "intraday_ic_credit_100pt": ic_credit(100, 150),
            "intraday_ic_credit_150pt": ic_credit(150, 200),
            "intraday_max_pain_distance_pts": (spx - max_pain) if pd.notna(max_pain) else np.nan,
            "intraday_max_pain_distance_pct": ((spx - max_pain) / spx * 100.0) if pd.notna(max_pain) and spx > 0 else np.nan,
            "intraday_ts_slope_0dte_7dte": np.log(ts7 / atm_straddle_mid) if pd.notna(ts7) and pd.notna(atm_straddle_mid) and ts7 > 0 and atm_straddle_mid > 0 else np.nan,
            "intraday_ts_slope_0dte_30dte": np.log(ts30 / atm_straddle_mid) if pd.notna(ts30) and pd.notna(atm_straddle_mid) and ts30 > 0 and atm_straddle_mid > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def _add_family_9_live(df: pd.DataFrame, daily_values: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Add family 9 features (term structure IV, volume, spreads, regime, events, earnings).

    For live mode, most of these are already in daily_values.
    The training build computes them from raw files, but the daily parquets
    already contain the pre-computed values. We just need the cross-features.
    """
    def _safe_merge_9(left, right, on="date"):
        overlap = [c for c in right.columns if c != on and c in left.columns]
        if overlap:
            left = left.drop(columns=overlap, errors="ignore")
        return left.merge(right, on=on, how="left")

    # Term structure IV features
    ts_iv = _build_daily_term_structure_iv_features_for_live(today)
    df = _safe_merge_9(df, ts_iv)

    # Options volume features
    vol = _build_daily_options_volume_features_for_live(today)
    df = _safe_merge_9(df, vol)

    # Sector spread features
    spreads = _build_daily_sector_spread_features_for_live(today)
    df = _safe_merge_9(df, spreads)

    # Regime features
    regime = _build_regime_features_for_live(today)
    df = _safe_merge_9(df, regime)

    # Event calendar features
    events = _build_event_calendar_features_for_live(today)
    df = _safe_merge_9(df, events)

    # Mag7 earnings features
    earnings = _build_mag7_earnings_features_for_live(today)
    df = _safe_merge_9(df, earnings)

    # Cross-features
    df["intraday_iv_ts_slope_0dte_7dte"] = np.where(
        df["daily_prior_ts_iv_7dte"] > 0,
        df["intraday_atm_iv"] / df["daily_prior_ts_iv_7dte"],
        np.nan,
    )
    df["intraday_iv_ts_slope_0dte_14dte"] = np.where(
        df["daily_prior_ts_iv_14dte"] > 0,
        df["intraday_atm_iv"] / df["daily_prior_ts_iv_14dte"],
        np.nan,
    )
    return df


def _build_daily_term_structure_iv_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_daily_term_structure_iv_features."""
    ts_file = RAW_DIR / "spxw_term_structure.parquet"
    if not ts_file.exists():
        return pd.DataFrame({"date": [today]})

    ts = pd.read_parquet(ts_file, columns=["date", "dte", "strike", "right", "spx_close", "moneyness", "mid"])
    ts["date"] = pd.to_datetime(ts["date"]).dt.normalize()
    ts = ts.sort_values(["date", "dte", "right", "strike"]).reset_index(drop=True)

    rows = []
    for date, grp in ts.groupby("date", sort=True):
        spx_close = grp["spx_close"].dropna().iloc[0] if grp["spx_close"].notna().any() else np.nan
        feats = extract_term_structure_iv(grp[["dte", "moneyness", "right", "mid", "strike"]], spx_close)
        feats["date"] = date
        rows.append(feats)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame({"date": [today]})

    daily["daily_prior_ts_iv_7dte"] = daily["ts_iv_7dte"]
    daily["daily_prior_ts_iv_14dte"] = daily["ts_iv_14dte"]
    daily["daily_prior_ts_iv_slope_7dte_14dte"] = np.where(
        daily["ts_iv_14dte"] > 0, daily["ts_iv_7dte"] / daily["ts_iv_14dte"], np.nan,
    )
    daily["daily_prior_ts_iv_inverted_7dte_14dte"] = np.where(
        daily["daily_prior_ts_iv_slope_7dte_14dte"].notna(),
        (daily["daily_prior_ts_iv_slope_7dte_14dte"] > 1.0).astype(float),
        np.nan,
    )
    feature_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    daily = _ensure_today_coverage(daily, today)
    return daily[["date", "daily_prior_ts_iv_7dte", "daily_prior_ts_iv_14dte",
                   "daily_prior_ts_iv_slope_7dte_14dte", "daily_prior_ts_iv_inverted_7dte_14dte"]]


def _build_daily_options_volume_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_daily_options_volume_features."""
    oi_file = RAW_DIR / "spxw_0dte_oi.parquet"
    if not oi_file.exists():
        return pd.DataFrame({"date": [today]})

    oi = pd.read_parquet(oi_file, columns=["date", "expiration", "strike", "right", "open_interest"])
    oi["date"] = pd.to_datetime(oi["date"]).dt.normalize()
    oi["expiration"] = pd.to_datetime(oi["expiration"]).dt.normalize()
    oi = oi[oi["date"] == oi["expiration"]].copy()
    if oi.empty:
        return pd.DataFrame({"date": [today], "daily_prior_total_call_volume": [np.nan],
                             "daily_prior_total_put_volume": [np.nan], "daily_prior_pc_volume_ratio": [np.nan]})

    daily = (
        oi.pivot_table(index="date", columns="right", values="open_interest", aggfunc="sum")
        .rename(columns={"call": "total_call_volume", "put": "total_put_volume"})
        .reset_index()
        .sort_values("date")
    )
    if "total_call_volume" not in daily.columns:
        daily["total_call_volume"] = np.nan
    if "total_put_volume" not in daily.columns:
        daily["total_put_volume"] = np.nan
    daily["pc_volume_ratio"] = np.where(
        daily["total_call_volume"] > 0, daily["total_put_volume"] / daily["total_call_volume"], np.nan,
    )
    daily["daily_prior_total_call_volume"] = daily["total_call_volume"].shift(1)
    daily["daily_prior_total_put_volume"] = daily["total_put_volume"].shift(1)
    daily["daily_prior_pc_volume_ratio"] = daily["pc_volume_ratio"].shift(1)
    daily = _ensure_today_coverage(daily, today)
    return daily[["date", "daily_prior_total_call_volume", "daily_prior_total_put_volume", "daily_prior_pc_volume_ratio"]]


def _build_daily_sector_spread_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_daily_sector_spread_features."""
    breadth_file = RAW_DIR / "breadth_raw_v1_2.parquet"
    if not breadth_file.exists():
        return pd.DataFrame({"date": [today]})

    raw = pd.read_parquet(breadth_file)
    raw = raw.reset_index().rename(columns={"Date": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.sort_values("date").reset_index(drop=True)
    for col in ("SPY", "XLF", "XLK", "XLE"):
        raw[f"{col.lower()}_ret_1day"] = raw[col].pct_change()
    raw["daily_prior_xlf_spy_ret_spread_1day"] = (raw["xlf_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    raw["daily_prior_xlk_spy_ret_spread_1day"] = (raw["xlk_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    raw["daily_prior_xle_spy_ret_spread_1day"] = (raw["xle_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    return raw[["date", "daily_prior_xlf_spy_ret_spread_1day",
                "daily_prior_xlk_spy_ret_spread_1day", "daily_prior_xle_spy_ret_spread_1day"]]


def _build_regime_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_regime_features."""
    regime_file = DATA_DIR / "regime_features_v1_2.parquet"
    if not regime_file.exists():
        return pd.DataFrame({"date": [today]})

    regime = pd.read_parquet(regime_file)
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    regime = regime.sort_values("date").reset_index(drop=True)
    regime["daily_prior_regime_state"] = regime["regime_state"].shift(1)
    regime["daily_prior_regime_prob_highvol"] = regime["regime_prob_highvol"].shift(1)
    regime["daily_prior_regime_duration"] = regime["regime_duration"].shift(1)
    regime["daily_prior_regime_switch"] = regime["regime_switch"].shift(1)
    return regime[["date", "daily_prior_regime_state", "daily_prior_regime_prob_highvol",
                    "daily_prior_regime_duration", "daily_prior_regime_switch"]]


def _build_event_calendar_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_event_calendar_features."""
    cal_file = DATA_DIR / "econ_calendar_v1_2.parquet"
    if not cal_file.exists():
        return pd.DataFrame({"date": [today]})

    cal = pd.read_parquet(cal_file)
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    return cal[["date", "is_cpi_day", "is_gdp_day", "is_nfp_day", "is_ppi_day",
                "days_to_next_cpi", "days_to_next_gdp", "days_to_next_nfp",
                "days_to_next_ppi", "days_to_next_fomc", "days_to_next_any_econ"]].copy()


def _build_mag7_earnings_features_for_live(today: pd.Timestamp) -> pd.DataFrame:
    """Replicate _build_mag7_earnings_features."""
    earn_file = DATA_DIR / "mag7_earnings_v1_2.parquet"
    if not earn_file.exists():
        return pd.DataFrame({"date": [today]})

    earn = pd.read_parquet(earn_file)
    earn["date"] = pd.to_datetime(earn["date"]).dt.normalize()
    return earn[["date", "is_earnings_aapl", "is_earnings_amzn", "is_earnings_googl",
                 "is_earnings_meta", "is_earnings_msft", "is_earnings_nvda", "is_earnings_tsla",
                 "mag7_earnings_count", "is_mag7_earnings_day"]].copy()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_feature_cols() -> list[str]:
    """Load feature_cols from the model pickle."""
    import pickle
    # Try tp25 first, fall back to any pkl
    for candidate in ["tp25_target_xgb_v1_2.pkl"]:
        path = MODEL_DIR / candidate
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            if isinstance(model, dict) and "feature_cols" in model:
                return model["feature_cols"]
    # Search for any pkl
    for path in MODEL_DIR.glob("*.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        if isinstance(model, dict) and "feature_cols" in model:
            return model["feature_cols"]
    raise FileNotFoundError("No model pickle found with feature_cols")


# ---------------------------------------------------------------------------
# VALIDATION MODE
# ---------------------------------------------------------------------------

def run_validation(val_date: str = "2025-11-26", val_time: str = "10:15") -> None:
    """Run pipeline against historical data and compare to master_data_v1_2_final."""
    print("=" * 70)
    print(f"VALIDATION MODE: {val_date} {val_time}")
    print("=" * 70)

    today = pd.Timestamp(val_date).normalize()
    decision_dt = pd.Timestamp(f"{val_date} {val_time}:00")

    # 1. Load model feature columns
    print("\n[1] Loading model feature columns...")
    feature_cols = load_model_feature_cols()
    print(f"    Model expects {len(feature_cols)} features")

    # 2. Load daily features
    print("\n[2] Loading daily features...")
    daily_values = load_daily_features_for_date(today)
    print(f"    Loaded {len(daily_values)} daily feature values")

    # 3. Load raw intraday data (simulating ThetaData pulls)
    print("\n[3] Loading raw intraday data from parquets...")

    # SPX 1min bars up to decision time
    spx_1m = pd.read_parquet(RAW_DIR / "spx_1min.parquet")
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.normalize()
    # Include prior days for rolling features, plus today up to state_datetime
    state_dt = decision_dt - pd.Timedelta(minutes=1)
    spx_1m = spx_1m[spx_1m["datetime"] <= state_dt].copy()
    spx_1m = spx_1m.sort_values("datetime").reset_index(drop=True)
    print(f"    SPX 1min bars: {len(spx_1m):,} (up to {state_dt})")

    # VIX 1min
    vix_1m = pd.read_parquet(RAW_DIR / "vix_1min.parquet")
    vix_1m["datetime"] = pd.to_datetime(vix_1m["datetime"])
    vix_1m = vix_1m[vix_1m["datetime"] <= state_dt].copy()
    print(f"    VIX 1min bars: {len(vix_1m):,}")

    # VIX1D 1min
    vix1d_1m = pd.read_parquet(RAW_DIR / "vix1d_1min.parquet")
    vix1d_1m["datetime"] = pd.to_datetime(vix1d_1m["datetime"])
    vix1d_1m = vix1d_1m[vix1d_1m["datetime"] <= state_dt].copy()
    print(f"    VIX1D 1min bars: {len(vix1d_1m):,}")

    # 4. Load decision state for this minute
    print("\n[4] Loading decision state...")
    # First try the historical decision state parquet
    ds_file = DATA_DIR / "intraday_decision_state_v1_2.parquet"
    decision_state = pd.DataFrame()

    if ds_file.exists():
        ds = pd.read_parquet(ds_file)
        ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])
        ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
        decision_state = ds[ds["decision_datetime"] == decision_dt].copy()

    # If not in historical, try today's greeks daily chunk (live data)
    if decision_state.empty:
        greeks_chunk = RAW_DIR / "greeks_daily_chunks" / f"greeks_{val_date.replace('-', '')}.parquet"
        if greeks_chunk.exists():
            print(f"    Building decision state from live greeks chunk: {greeks_chunk.name}")
            greeks_df = pd.read_parquet(greeks_chunk)
            greeks_df["timestamp"] = pd.to_datetime(greeks_df["timestamp"])
            greeks_df["date"] = pd.to_datetime(greeks_df["date"])

            # Get SPX/VIX/VIX1D at state_datetime
            spx_at_state = spx_1m[spx_1m["datetime"] == state_dt]
            spx_close = float(spx_at_state.iloc[0]["close"]) if len(spx_at_state) > 0 else 0.0
            vix_row = vix_1m[vix_1m["datetime"] == state_dt]
            vix1d_row = vix1d_1m[vix1d_1m["datetime"] == state_dt]

            decision_state = build_decision_state_from_greeks(
                greeks_df=greeks_df,
                spx_bars=spx_1m,
                vix_bars=vix_1m,
                vix1d_bars=vix1d_1m,
                decision_dt=decision_dt,
            )
        else:
            # Last resort: try fetching directly from ThetaData
            print(f"    No greeks chunk found. Fetching from ThetaData...")
            spx_close = float(spx_1m.iloc[-1]["close"]) if len(spx_1m) > 0 else 0.0
            greeks_df = fetch_greeks_0dte(val_date.replace("-", ""), spx_close)
            if not greeks_df.empty:
                decision_state = build_decision_state_from_greeks(
                    greeks_df=greeks_df,
                    spx_bars=spx_1m,
                    vix_bars=vix_1m,
                    vix1d_bars=vix1d_1m,
                    decision_dt=decision_dt,
                )

    print(f"    Decision state rows: {len(decision_state)} (strikes at {val_time})")

    if decision_state.empty:
        print("    ERROR: No decision state found for this datetime!")
        return

    # 5. Build IC rows
    print("\n[5] Building IC strategy rows...")
    ic_rows = build_ic_rows(decision_state, decision_dt)
    print(f"    IC rows: {len(ic_rows)} strategies")
    if ic_rows.empty:
        print("    ERROR: No IC rows could be built!")
        return

    # 6. Build feature vector
    print("\n[6] Building feature vector...")
    result = build_feature_vector(
        ic_rows=ic_rows,
        daily_values=daily_values,
        spx_1m_history=spx_1m,
        vix_bars_history=vix_1m,
        vix1d_bars_history=vix1d_1m,
        decision_state_snapshot=decision_state,
        decision_dt=decision_dt,
        model_feature_cols=feature_cols,
    )
    print(f"    Result shape: {result.shape}")

    # 7. Save live-built row FIRST (before comparison, so it's always saved)
    print(f"\n[7] Saving live-built row to {VALIDATION_OUTPUT}...")
    import pyarrow as pa
    import pyarrow.parquet as pq
    VALIDATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), VALIDATION_OUTPUT)
    print(f"    Saved {len(result)} rows")

    # 8. Compare against master_data_v1_2_final.parquet (optional — only works for historical dates)
    print("\n[8] Comparing against master_data_v1_2_final.parquet...")
    master = pd.read_parquet(MASTER_FINAL_FILE)
    master["decision_datetime"] = pd.to_datetime(master["decision_datetime"])
    master_row = master[master["decision_datetime"] == decision_dt].copy()

    if master_row.empty:
        print("    No matching row in master — this is a live/future date, skipping comparison.")
        print("    Live vector saved successfully. Use it for scoring.")
        return

    print(f"    Master rows at {val_time}: {len(master_row)} ({master_row['strategy'].unique()})")

    # Pick a strategy that exists in both
    common_strategies = set(result["strategy"]) & set(master_row["strategy"])
    if not common_strategies:
        print("    ERROR: No common strategies between live and master!")
        return

    strat = sorted(common_strategies)[0]
    live_row = result[result["strategy"] == strat].iloc[0]
    ref_row = master_row[master_row["strategy"] == strat].iloc[0]

    total = 0
    matches = 0
    mismatches = []
    max_diff = 0.0
    max_diff_col = ""

    for col in feature_cols:
        total += 1
        live_val = live_row.get(col, np.nan)
        ref_val = ref_row.get(col, np.nan)

        both_nan = pd.isna(live_val) and pd.isna(ref_val)
        if both_nan:
            matches += 1
            continue

        if pd.isna(live_val) or pd.isna(ref_val):
            mismatches.append((col, live_val, ref_val, "NaN mismatch"))
            continue

        try:
            lv = float(live_val)
            rv = float(ref_val)
            if rv == 0 and lv == 0:
                matches += 1
                continue
            denom = max(abs(rv), 1e-10)
            pct_diff = abs(lv - rv) / denom
            abs_diff = abs(lv - rv)

            if abs_diff < 1e-6 or pct_diff < 1e-4:
                matches += 1
            else:
                mismatches.append((col, lv, rv, f"diff={abs_diff:.6f} pct={pct_diff:.4%}"))
                if abs_diff > max_diff:
                    max_diff = abs_diff
                    max_diff_col = col
        except (TypeError, ValueError):
            if live_val == ref_val:
                matches += 1
            else:
                mismatches.append((col, live_val, ref_val, "type mismatch"))

    print(f"\n{'=' * 70}")
    print(f"VALIDATION RESULTS (strategy={strat})")
    print(f"{'=' * 70}")
    print(f"  Total features:  {total}")
    print(f"  Matches:         {matches}")
    print(f"  Mismatches:      {len(mismatches)}")
    print(f"  Match rate:      {matches / total:.1%}")
    if max_diff_col:
        print(f"  Max difference:  {max_diff:.6f} ({max_diff_col})")

    if mismatches:
        print(f"\n  Top mismatches (up to 30):")
        for col, lv, rv, note in mismatches[:30]:
            print(f"    {col:50s} live={lv!s:>20s}  ref={rv!s:>20s}  ({note})")

    # 8. Save live-built row
    print(f"\n[8] Saving live-built row to {VALIDATION_OUTPUT}...")
    VALIDATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), VALIDATION_OUTPUT)
    print(f"    Saved {len(result)} rows")

    print("\nValidation complete.")


# ---------------------------------------------------------------------------
# LIVE MODE (placeholder for real-time operation)
# ---------------------------------------------------------------------------

def run_live() -> None:
    """Run the live scoring pipeline.

    This would be called every minute from 10:00 onward during market hours.
    For now, prints a message indicating live mode is not yet implemented.
    """
    print("Live mode: not yet implemented.")
    print("Use --validate to test against historical data.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ZDOM v1.2 Live Scoring Pipeline")
    parser.add_argument("--validate", action="store_true", help="Run validation against historical data")
    parser.add_argument("--val-date", default="2025-11-26", help="Validation date (YYYY-MM-DD)")
    parser.add_argument("--val-time", default="10:15", help="Validation decision time (HH:MM)")
    args = parser.parse_args()

    if args.validate:
        run_validation(args.val_date, args.val_time)
    else:
        run_live()


if __name__ == "__main__":
    main()
