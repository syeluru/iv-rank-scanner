"""
Build master_data_v1_2_final from the joined master_data_v1_2 base table.

Layer order:
1. Family 1: strategy/time derived fields
2. Family 2: calendar/event pass-through fields
3. Family 3: daily_prior base projections from lag-correct daily fields
4. Family 4: daily lookback / rolling projections already present in master_data_v1_2

Families 5-6 (intraday and cross-horizon features computed from minute history)
will be layered onto this script after the safe families are validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.iv_surface import extract_snapshot_iv_features, extract_term_structure_iv

INPUT_FILE = PROJECT_DIR / "2_data_join" / "master_data_v1_2.parquet"
SPX_1MIN_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "spx_1min.parquet"
SPX_DAILY_FILE = PROJECT_DIR / "1_data_collection" / "data" / "v1_2" / "spx_daily_v1_2.parquet"
VIX_1MIN_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "vix_1min.parquet"
VIX1D_1MIN_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "vix1d_1min.parquet"
TERM_STRUCTURE_RAW_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "spxw_term_structure.parquet"
OI_RAW_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "spxw_0dte_oi.parquet"
BREADTH_RAW_FILE = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2" / "breadth_raw_v1_2.parquet"
ECON_CALENDAR_FILE = PROJECT_DIR / "1_data_collection" / "data" / "v1_2" / "econ_calendar_v1_2.parquet"
MAG7_EARNINGS_FILE = PROJECT_DIR / "1_data_collection" / "data" / "v1_2" / "mag7_earnings_v1_2.parquet"
REGIME_FEATURES_FILE = PROJECT_DIR / "1_data_collection" / "data" / "v1_2" / "regime_features_v1_2.parquet"
DECISION_STATE_FILE = (
    PROJECT_DIR / "1_data_collection" / "data" / "v1_2" / "intraday_decision_state_v1_2.parquet"
)
OUTPUT_FILE = (
    PROJECT_DIR
    / "3_feature_engineering"
    / "v1_2"
    / "outputs"
    / "master_data_v1_2_final.parquet"
)

ENTRY_OPEN_MINUTE = 9 * 60 + 30
TOTAL_SESSION_MINUTES = 390.0

STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]

CALENDAR_PASS_THROUGH = [
    "christmas_week",
    "cycle_month",
    "cycle_year",
    "day_of_month",
    "days_into_quarter",
    "election_year",
    "first_year",
    "january_effect",
    "midterm_year",
    "month",
    "near_month_end",
    "near_qtr_end",
    "opex_day",
    "opex_week",
    "party_dem",
    "pre_election_year",
    "q4_rally",
    "quarter",
    "sell_in_may",
    "thanksgiving_week",
    "triple_witching",
    "week_of_year",
]

DAILY_PRIOR_BASE_MAP = {
    "daily_prior_total_call_oi": "total_call_oi",
    "daily_prior_total_put_oi": "total_put_oi",
    "daily_prior_pc_oi_ratio": "pc_oi_ratio",
    "daily_prior_oi_concentration": "oi_concentration",
    "daily_prior_max_pain_strike": "max_pain_strike",
    "daily_prior_ts_0dte_atm_mid": "ts_0dte_atm_mid",
    "daily_prior_ts_7dte_atm_mid": "ts_7dte_atm_mid",
    "daily_prior_ts_30dte_atm_mid": "ts_30dte_atm_mid",
    "daily_prior_ts_slope_0_30": "ts_slope_0_30",
    "daily_prior_ts_slope_0_7": "ts_slope_0_7",
    "daily_prior_fed_funds_rate": "fed_funds_rate",
    "daily_prior_yield_2y": "yield_2y",
    "daily_prior_yield_10y": "yield_10y",
    "daily_prior_ig_spread": "ig_spread",
    "daily_prior_hy_spread": "hy_spread",
    "daily_prior_wti_crude": "wti_crude",
    "daily_prior_dxy_broad": "dxy_broad",
    "daily_prior_gold_fix_am": "gold_fix_am",
    "daily_prior_yield_spread_2y10y": "yield_spread_2y10y",
    "daily_prior_yield_curve_inverted": "yield_curve_inverted",
    "daily_prior_credit_stress": "credit_stress",
    "daily_prior_fed_policy_direction": "fed_policy_direction",
    "daily_prior_spx_ret_5day": "spx_ret_5d",
    "daily_prior_spx_ret_20day": "spx_ret_20d",
    "daily_prior_spx_ret_60day": "spx_ret_60d",
    "daily_prior_qqq_spx_ratio": "qqq_spx_ratio",
    "daily_prior_iwm_spx_ratio": "iwm_spx_ratio",
    "daily_prior_eem_spx_ratio": "eem_spx_ratio",
    "daily_prior_spx_iwm_ratio": "spx_iwm_ratio",
    "daily_prior_tlt_ret_5day": "tlt_ret_5d",
    "daily_prior_tlt_ret_20day": "tlt_ret_20d",
    "daily_prior_rsp_spy_ratio": "rsp_spy_ratio",
    "daily_prior_vix_vxv_ratio": "vix_vxv_ratio",
    "daily_prior_vix_term_structure_slope": "vix_term_structure_slope",
    "daily_prior_vvix_close": "vvix_close",
    "daily_prior_vvix_ma20": "vvix_ma20",
    "daily_prior_vvix_vs_vix_ratio": "vvix_vs_vix_ratio",
    "daily_prior_vix1d_close": "vix1d_close",
    "daily_prior_vol_regime_persistence": "vol_regime_persistence",
}

DAILY_LOOKBACK_MAP = {
    "daily_prior_breadth_thrust_cumulative_5day": "breadth_thrust_cumulative_5d",
    "daily_prior_credit_stress_chg_20day": "credit_stress_chg_20d",
    "daily_prior_dxy_chg_5day": "dxy_chg_5d",
    "daily_prior_dxy_chg_20day": "dxy_chg_20d",
    "daily_prior_dxy_ma20": "dxy_ma20",
    "daily_prior_dxy_trend": "dxy_trend",
    "daily_prior_eem_spx_ratio_chg_5day": "eem_spx_ratio_chg_5d",
    "daily_prior_eem_spx_ratio_chg_20day": "eem_spx_ratio_chg_20d",
    "daily_prior_fed_funds_chg_21day": "fed_funds_chg_21d",
    "daily_prior_gold_chg_5day": "gold_chg_5d",
    "daily_prior_gold_chg_20day": "gold_chg_20d",
    "daily_prior_gold_ma20": "gold_ma20",
    "daily_prior_gold_trend": "gold_trend",
    "daily_prior_hv_iv_ratio_10day": "daily_prior_hv_iv_ratio_10day",
    "daily_prior_hv_iv_ratio_20day": "daily_prior_hv_iv_ratio_20day",
    "daily_prior_hv_iv_ratio_30day": "daily_prior_hv_iv_ratio_30day",
    "daily_prior_hv_iv_ratio_5day": "daily_prior_hv_iv_ratio_5day",
    "daily_prior_iwm_spx_ratio_chg_5day": "iwm_spx_ratio_chg_5d",
    "daily_prior_iwm_spx_ratio_chg_20day": "iwm_spx_ratio_chg_20d",
    "daily_prior_iwm_spy_ret_spread_20day": "iwm_spy_ret_spread_20d",
    "daily_prior_oil_chg_5day": "oil_chg_5d",
    "daily_prior_oil_chg_20day": "oil_chg_20d",
    "daily_prior_oil_ma20": "oil_ma20",
    "daily_prior_oil_trend": "oil_trend",
    "daily_prior_qqq_spx_ratio_chg_5day": "qqq_spx_ratio_chg_5d",
    "daily_prior_qqq_spx_ratio_chg_20day": "qqq_spx_ratio_chg_20d",
    "daily_prior_rsp_spy_ratio_5day_chg": "rsp_spy_ratio_chg_5d",
    "daily_prior_rsp_spy_ratio_20day_chg": "rsp_spy_ratio_chg_20d",
    "daily_prior_rsp_spy_ratio_zscore_21day": "rsp_spy_ratio_zscore_21d",
    "daily_prior_rv_iv_spread_10day": "rv_iv_spread_10d",
    "daily_prior_rv_iv_spread_20day": "rv_iv_spread_20d",
    "daily_prior_sector_divergence_count": "sector_divergence_count",
    "daily_prior_sector_divergence_flag": "sector_divergence_flag",
    "daily_prior_spx_iwm_ratio_chg_5day": "spx_iwm_ratio_chg_5d",
    "daily_prior_spx_iwm_ratio_chg_20day": "spx_iwm_ratio_chg_20d",
    "daily_prior_tlt_spx_ratio_chg_5day": "tlt_spx_ratio_chg_5d",
    "daily_prior_tlt_spx_ratio_chg_20day": "tlt_spx_ratio_chg_20d",
    "daily_prior_tlt_spx_corr_20day": "tlt_spx_corr_20d",
    "daily_prior_vix_vxv_zscore_21day": "vix_vxv_zscore_21d",
    "daily_prior_vvix_zscore_21day": "vvix_zscore_21d",
    "daily_prior_yield_2y_chg_5day": "yield_2y_chg_5d",
    "daily_prior_yield_2y_chg_21day": "yield_2y_chg_21d",
    "daily_prior_yield_2y_chg_63day": "yield_2y_chg_63d",
    "daily_prior_yield_10y_chg_5day": "yield_10y_chg_5d",
    "daily_prior_yield_10y_chg_21day": "yield_10y_chg_21d",
    "daily_prior_yield_10y_chg_63day": "yield_10y_chg_63d",
    "daily_prior_yield_spread_2y10y_chg_5day": "yield_spread_2y10y_chg_5d",
    "daily_prior_yield_spread_2y10y_chg_21day": "yield_spread_2y10y_chg_21d",
    "daily_prior_yield_spread_2y10y_chg_63day": "yield_spread_2y10y_chg_63d",
}

CURRENT_DAILY_OI_PRIOR_DAY_MAP = {
    "prior_day_total_call_oi": "total_call_oi",
    "prior_day_total_put_oi": "total_put_oi",
    "prior_day_pc_oi_ratio": "pc_oi_ratio",
    "prior_day_oi_concentration": "oi_concentration",
    "prior_day_max_pain_strike": "max_pain_strike",
}


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_master() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_FILE)
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values(["decision_datetime", "strategy"]).reset_index(drop=True)


def add_family_1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes_since_open = (
        out["decision_datetime"].dt.hour * 60
        + out["decision_datetime"].dt.minute
        - ENTRY_OPEN_MINUTE
    )
    out["entry_hour"] = out["decision_datetime"].dt.hour.astype(int)
    out["entry_minute"] = out["decision_datetime"].dt.minute.astype(int)
    out["minutes_since_open"] = minutes_since_open.astype(int)
    out["time_sin"] = np.sin(2 * np.pi * out["minutes_since_open"] / TOTAL_SESSION_MINUTES)
    out["time_cos"] = np.cos(2 * np.pi * out["minutes_since_open"] / TOTAL_SESSION_MINUTES)

    dummies = pd.get_dummies(out["strategy"], prefix="strat", dtype=int)
    for strat in [f"strat_{s}" for s in STRATEGIES]:
        if strat not in dummies.columns:
            dummies[strat] = 0
    dummies = dummies[[f"strat_{s}" for s in STRATEGIES]]
    out = pd.concat([out, dummies], axis=1)
    return out


def add_family_2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["decision_datetime"].dt.dayofweek.astype(int)
    out["is_monday"] = (out["day_of_week"] == 0).astype(int)
    out["is_friday"] = (out["day_of_week"] == 4).astype(int)
    out["is_month_end"] = out["decision_datetime"].dt.is_month_end.astype(int)
    out["is_quarter_end"] = out["decision_datetime"].dt.is_quarter_end.astype(int)
    out["cycle_quarter"] = out["decision_datetime"].dt.quarter.astype(int)
    return out


def add_family_3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for new_col, src_col in DAILY_PRIOR_BASE_MAP.items():
        if src_col in out.columns:
            out[new_col] = out[src_col]
    return out


def add_family_4(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for new_col, src_col in DAILY_LOOKBACK_MAP.items():
        if src_col in out.columns:
            out[new_col] = out[src_col]
    for new_col, src_col in CURRENT_DAILY_OI_PRIOR_DAY_MAP.items():
        if src_col in out.columns:
            out[new_col] = out[src_col]
    return out


def _daily_fixed_intraday_features(spx_1m: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, grp in spx_1m.groupby("date", sort=True):
        grp = grp.sort_values("datetime").reset_index(drop=True)
        first_bar = grp.iloc[0]
        first_15 = grp[grp["datetime"].dt.time <= pd.Timestamp("09:44:00").time()]
        first_30 = grp[grp["datetime"].dt.time <= pd.Timestamp("09:59:00").time()]

        daily_open = first_bar["open"]
        prev_close = np.nan
        if rows:
            prev_close = rows[-1]["session_close"]

        row = {
            "date": date,
            "spx_intraday_first_bar_open": daily_open,
            "session_close": grp.iloc[-1]["close"],
        }

        if pd.notna(prev_close) and prev_close > 0:
            row["daily_spx_open_vs_prev_close_gap_pct"] = (daily_open - prev_close) / prev_close * 100.0
        else:
            row["daily_spx_open_vs_prev_close_gap_pct"] = np.nan

        if not first_15.empty and daily_open > 0:
            row["first_15m_return_pct"] = (first_15.iloc[-1]["close"] - daily_open) / daily_open * 100.0
        else:
            row["first_15m_return_pct"] = np.nan

        if not first_30.empty and daily_open > 0:
            first_30_close = first_30.iloc[-1]["close"]
            first_30_high = first_30["high"].max()
            first_30_low = first_30["low"].min()
            first_30_range = first_30_high - first_30_low
            first_30_return = (first_30_close - daily_open) / daily_open * 100.0
            row["first_30m_return_pct"] = first_30_return
            row["first_30m_range_pct"] = first_30_range / daily_open * 100.0 if daily_open > 0 else np.nan
            row["first_30m_direction"] = (
                1 if first_30_return > 0 else -1 if first_30_return < 0 else 0
            )
            range_pct = row["first_30m_range_pct"]
            row["open_drive_strength"] = (
                first_30_return / range_pct if pd.notna(range_pct) and range_pct != 0 else np.nan
            )
            if pd.notna(prev_close) and prev_close > 0:
                gap_dir = np.sign(daily_open - prev_close)
                move_dir = np.sign(first_30_close - daily_open)
                row["prev_close_gap_reversal"] = float(gap_dir != 0 and move_dir == -gap_dir)
            else:
                row["prev_close_gap_reversal"] = np.nan
        else:
            row["first_30m_return_pct"] = np.nan
            row["first_30m_range_pct"] = np.nan
            row["first_30m_direction"] = np.nan
            row["open_drive_strength"] = np.nan
            row["prev_close_gap_reversal"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.drop(columns=["session_close"])


def _per_minute_intraday_features(spx_1m: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []

    for date, grp in spx_1m.groupby("date", sort=True):
        grp = grp.sort_values("datetime").reset_index(drop=True).copy()
        grp["log_ret"] = np.log(grp["close"] / grp["close"].shift(1))
        grp["typical"] = (grp["high"] + grp["low"] + grp["close"]) / 3.0
        grp["vwap_proxy"] = grp["typical"].expanding().mean()

        for w in (5, 15, 30):
            grp[f"ma{w}"] = grp["close"].rolling(w, min_periods=max(1, w // 3)).mean()
        grp["ma10"] = grp["close"].rolling(10, min_periods=3).mean()

        for w in (5, 10, 15, 30):
            grp[f"intraday_spx_ret_{w}m"] = grp["close"].pct_change(w) * 100.0
            grp[f"intraday_rvol_{w}m"] = (
                grp["log_ret"].rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(390 * 252)
            )

        log_hl = np.log(grp["high"] / grp["low"])
        log_co = np.log(grp["close"] / grp["open"])
        gk_component = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        grp["intraday_garman_klass_rv_30m"] = (
            gk_component.rolling(30, min_periods=10).mean().clip(lower=0).apply(
                lambda x: np.sqrt(x * 390 * 252) if pd.notna(x) else np.nan
            )
        )

        grp["running_high"] = grp["high"].cummax()
        grp["running_low"] = grp["low"].cummin()
        grp["running_range"] = grp["running_high"] - grp["running_low"]
        grp["intraday_pctB"] = np.where(
            grp["running_range"] > 0,
            (grp["close"] - grp["running_low"]) / grp["running_range"],
            0.5,
        )

        grp["intraday_spx_vs_vwap"] = np.where(
            grp["vwap_proxy"] > 0,
            (grp["close"] - grp["vwap_proxy"]) / grp["vwap_proxy"] * 100.0,
            np.nan,
        )
        for w in (5, 15, 30):
            grp[f"intraday_spx_vs_ma{w}"] = np.where(
                grp[f"ma{w}"] > 0,
                (grp["close"] - grp[f"ma{w}"]) / grp[f"ma{w}"] * 100.0,
                np.nan,
            )

        orb = grp[grp["datetime"].dt.time <= pd.Timestamp("09:59:00").time()]
        orb_high = orb["high"].max()
        orb_low = orb["low"].min()
        orb_range = orb_high - orb_low
        grp["intraday_orb_range_pct"] = np.where(grp["close"] > 0, orb_range / grp["close"] * 100.0, np.nan)
        grp["intraday_orb_containment"] = ((grp["close"] >= orb_low) & (grp["close"] <= orb_high)).astype(float)
        grp["intraday_orb_breakout_dir"] = np.where(
            grp["close"] > orb_high,
            1.0,
            np.where(grp["close"] < orb_low, -1.0, 0.0),
        )

        grp["bb_mid"] = grp["close"].rolling(20, min_periods=10).mean()
        bb_std = grp["close"].rolling(20, min_periods=10).std()
        grp["bb_upper"] = grp["bb_mid"] + 2 * bb_std
        grp["bb_lower"] = grp["bb_mid"] - 2 * bb_std
        atr_20 = (grp["high"] - grp["low"]).rolling(20, min_periods=10).mean()
        grp["kc_upper"] = grp["bb_mid"] + 1.5 * atr_20
        grp["kc_lower"] = grp["bb_mid"] - 1.5 * atr_20
        grp["intraday_ttm_squeeze"] = (
            (grp["bb_upper"] < grp["kc_upper"]) & (grp["bb_lower"] > grp["kc_lower"])
        ).astype(float)

        squeeze_duration = []
        count = 0
        for in_sq in grp["intraday_ttm_squeeze"].fillna(0).astype(int):
            count = count + 1 if in_sq == 1 else 0
            squeeze_duration.append(float(count))
        grp["intraday_bb_squeeze_duration"] = squeeze_duration

        keep_cols = [
            "datetime",
            "date",
            "intraday_spx_ret_5m",
            "intraday_spx_ret_10m",
            "intraday_spx_ret_15m",
            "intraday_spx_ret_30m",
            "intraday_spx_vs_vwap",
            "intraday_spx_vs_ma5",
            "intraday_spx_vs_ma15",
            "intraday_spx_vs_ma30",
            "intraday_rvol_5m",
            "intraday_rvol_10m",
            "intraday_rvol_15m",
            "intraday_rvol_30m",
            "intraday_garman_klass_rv_30m",
            "intraday_pctB",
            "intraday_orb_range_pct",
            "intraday_orb_containment",
            "intraday_orb_breakout_dir",
            "intraday_ttm_squeeze",
            "intraday_bb_squeeze_duration",
        ]
        part = grp[keep_cols].copy()
        part["decision_datetime"] = part["datetime"] + pd.Timedelta(minutes=1)
        all_rows.append(part.drop(columns=["datetime"]))

    out = pd.concat(all_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _vix_intraday_features() -> pd.DataFrame:
    cols = ["date", "decision_datetime", "vix_close", "vix_high", "vix_low"]
    ds = pd.read_parquet(DECISION_STATE_FILE, columns=cols)
    ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
    ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])
    ds = ds.sort_values(["decision_datetime"]).drop_duplicates(["decision_datetime"], keep="first")

    out_parts: list[pd.DataFrame] = []
    for _, grp in ds.groupby("date", sort=True):
        grp = grp.sort_values("decision_datetime").copy()
        grp["vix_intraday_prior_min_close"] = grp["vix_close"]
        grp["vix_intraday_running_high"] = grp["vix_high"].cummax()
        grp["vix_intraday_running_low"] = grp["vix_low"].cummin()
        out_parts.append(
            grp[
                [
                    "decision_datetime",
                    "vix_intraday_prior_min_close",
                    "vix_intraday_running_high",
                    "vix_intraday_running_low",
                ]
            ]
        )

    return pd.concat(out_parts, ignore_index=True)


def add_family_5(df: pd.DataFrame) -> pd.DataFrame:
    spx_1m = pd.read_parquet(SPX_1MIN_FILE)
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.normalize()
    spx_1m = spx_1m.sort_values("datetime").reset_index(drop=True)

    per_minute = _per_minute_intraday_features(spx_1m)
    fixed_daily = _daily_fixed_intraday_features(spx_1m)
    vix_state = _vix_intraday_features()

    intraday = per_minute.merge(fixed_daily, on="date", how="left").merge(
        vix_state, on="decision_datetime", how="left"
    )

    out = df.merge(intraday, on=["decision_datetime", "date"], how="left")
    return out


def _load_spx_daily() -> pd.DataFrame:
    daily = pd.read_parquet(SPX_DAILY_FILE)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily.sort_values("date").reset_index(drop=True)


def _daily_spx_context(spx_daily: pd.DataFrame) -> pd.DataFrame:
    daily = spx_daily.rename(
        columns={
            "spx_open": "daily_open",
            "spx_high": "daily_high",
            "spx_low": "daily_low",
            "spx_close": "daily_close",
        }
    )[["date", "daily_open", "daily_high", "daily_low", "daily_close"]].copy()

    close = daily["daily_close"]
    prev_close = close.shift(1)
    log_ret = np.log(close / prev_close)

    daily["daily_rv_5day"] = log_ret.rolling(5, min_periods=3).std() * np.sqrt(252)
    daily["daily_rv_20day"] = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    daily["atr_14day"] = (daily["daily_high"] - daily["daily_low"]).rolling(14, min_periods=7).mean()

    for w in (7, 20, 50, 180):
        daily[f"sma_{w}day"] = close.rolling(w, min_periods=max(3, w // 2)).mean()

    # Shift all daily context so decision date D only sees values ending at D-1.
    shift_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, shift_cols] = daily.loc[:, shift_cols].shift(1)
    return daily


def _intraday_range_points(spx_1m: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    daily_fixed = _daily_fixed_intraday_features(spx_1m)
    range_map = daily_fixed.set_index("date")["first_30m_range_pct"].to_dict()
    open_map = daily_fixed.set_index("date")["spx_intraday_first_bar_open"].to_dict()

    for date, grp in spx_1m.groupby("date", sort=True):
        grp = grp.sort_values("datetime").copy()
        grp["intraday_running_range_points"] = grp["high"].cummax() - grp["low"].cummin()
        first_open = open_map.get(date, np.nan)
        first_30m_range_pct = range_map.get(date, np.nan)
        if pd.notna(first_open) and pd.notna(first_30m_range_pct):
            grp["first_30m_range_points"] = first_open * first_30m_range_pct / 100.0
        else:
            grp["first_30m_range_points"] = np.nan
        grp["decision_datetime"] = grp["datetime"] + pd.Timedelta(minutes=1)
        rows.append(grp[["decision_datetime", "intraday_running_range_points", "first_30m_range_points"]])

    return pd.concat(rows, ignore_index=True)


def add_family_6(df: pd.DataFrame) -> pd.DataFrame:
    spx_1m = pd.read_parquet(SPX_1MIN_FILE)
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.normalize()
    spx_1m = spx_1m.sort_values("datetime").reset_index(drop=True)
    spx_daily = _load_spx_daily()

    daily_ctx = _daily_spx_context(spx_daily)
    intraday_ranges = _intraday_range_points(spx_1m)

    out = df.merge(daily_ctx, on="date", how="left").merge(intraday_ranges, on="decision_datetime", how="left")
    out["spx_intraday_atm_strike"] = np.round(out["spx_prev_min_close"] / 5.0) * 5.0

    for w in (7, 20, 50, 180):
        sma_col = f"sma_{w}day"
        out[f"intraday_spx_vs_sma_{w}day"] = np.where(
            out[sma_col] > 0,
            (out["spx_prev_min_close"] - out[sma_col]) / out[sma_col] * 100.0,
            np.nan,
        )

    ma15 = np.where(
        (1 + out["intraday_spx_vs_ma15"] / 100.0) != 0,
        out["spx_prev_min_close"] / (1 + out["intraday_spx_vs_ma15"] / 100.0),
        np.nan,
    )
    out["intraday_ma15_vs_sma_7day"] = np.where(
        out["sma_7day"] > 0,
        (ma15 - out["sma_7day"]) / out["sma_7day"] * 100.0,
        np.nan,
    )
    out["intraday_ma15_vs_sma_20day"] = np.where(
        out["sma_20day"] > 0,
        (ma15 - out["sma_20day"]) / out["sma_20day"] * 100.0,
        np.nan,
    )
    out["intraday_rvol_vs_daily_5day"] = np.where(
        out["daily_rv_5day"] > 0,
        out["intraday_rvol_30m"] / out["daily_rv_5day"],
        np.nan,
    )
    out["intraday_rvol_vs_daily_20day"] = np.where(
        out["daily_rv_20day"] > 0,
        out["intraday_rvol_30m"] / out["daily_rv_20day"],
        np.nan,
    )
    out["intraday_range_exhaustion"] = np.where(
        out["atr_14day"] > 0,
        out["intraday_running_range_points"] / out["atr_14day"],
        np.nan,
    )
    out["intraday_range_vs_atr"] = np.where(
        out["atr_14day"] > 0,
        out["first_30m_range_points"] / out["atr_14day"] * 100.0,
        np.nan,
    )

    return out.drop(
        columns=[
            "daily_open",
            "daily_high",
            "daily_low",
            "daily_close",
            "daily_rv_5day",
            "daily_rv_20day",
            "atr_14day",
            "sma_7day",
            "sma_20day",
            "sma_50day",
            "sma_180day",
            "intraday_running_range_points",
            "first_30m_range_points",
        ],
        errors="ignore",
    )


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd_hist(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd - signal


def _compute_efficiency_ratio(series: pd.Series, window: int = 10) -> pd.Series:
    direction = (series - series.shift(window)).abs()
    volatility = series.diff().abs().rolling(window, min_periods=window).sum()
    return direction / volatility.replace(0, np.nan)


def _compute_choppiness(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_sum = tr.rolling(window, min_periods=window).sum()
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    denom = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(tr_sum / denom) / np.log10(window)


def _compute_yz_hv(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    log_ho = np.log(high / open_.replace(0, np.nan))
    log_lo = np.log(low / open_.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    log_oc = np.log(open_ / close.shift(1).replace(0, np.nan))
    log_cc = np.log(close / close.shift(1).replace(0, np.nan))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_o = log_oc.rolling(window, min_periods=window).var()
    sigma_c = log_cc.rolling(window, min_periods=window).var()
    sigma_rs = rs.rolling(window, min_periods=window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = sigma_o + k * sigma_c + (1 - k) * sigma_rs
    return np.sqrt(yz_var.clip(lower=0) * 252)


def _build_daily_spx_technical_features(spx_daily: pd.DataFrame) -> pd.DataFrame:
    daily = spx_daily[["date", "spx_open", "spx_high", "spx_low", "spx_close"]].copy()
    daily = daily.sort_values("date").reset_index(drop=True)

    close = daily["spx_close"]
    open_ = daily["spx_open"]
    high = daily["spx_high"]
    low = daily["spx_low"]
    prev_close = close.shift(1)

    daily["daily_prior_open_to_close_pct"] = ((close - open_) / open_.replace(0, np.nan)) * 100.0
    daily["daily_prior_range_pct"] = ((high - low) / open_.replace(0, np.nan)) * 100.0

    for w in (5, 10, 20, 50, 100, 200):
        sma = close.rolling(w, min_periods=w).mean()
        daily[f"daily_prior_close_ratio_sma_{w}day"] = ((close - sma) / sma.replace(0, np.nan)) * 100.0

    for w in (5, 20):
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
        daily[f"daily_prior_close_ratio_ema_{w}day"] = ((close - ema) / ema.replace(0, np.nan)) * 100.0

    daily["daily_prior_rsi_14day"] = _compute_rsi(close, 14)
    daily["daily_prior_macd_hist"] = _compute_macd_hist(close)

    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    daily["daily_prior_bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    daily["daily_prior_bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    daily["daily_prior_bb_bandwidth_pctile"] = (
        daily["daily_prior_bb_bandwidth"]
        .rolling(60, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()
    daily["daily_prior_atr_pct"] = atr14 / close.replace(0, np.nan) * 100.0

    daily["daily_prior_efficiency_ratio"] = _compute_efficiency_ratio(close, 10)
    daily["daily_prior_choppiness_14day"] = _compute_choppiness(high, low, close, 14)

    for w in (5, 10, 20, 30):
        daily[f"daily_prior_yz_hv_{w}day"] = _compute_yz_hv(open_, high, low, close, w)

    # Shift so decision date D sees features ending at D-1.
    feature_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    return daily


def _build_daily_vix1d_features() -> pd.DataFrame:
    vix1d = pd.read_parquet(VIX1D_1MIN_FILE, columns=["datetime", "vix1d_close"])
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


def _build_daily_vix_features() -> pd.DataFrame:
    vix = pd.read_parquet(VIX_1MIN_FILE, columns=["datetime", "vix_close"])
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


def _build_daily_snapshot_iv_features() -> pd.DataFrame:
    cols = [
        "date",
        "decision_datetime",
        "strike",
        "call_delta",
        "put_delta",
        "call_implied_vol",
        "put_implied_vol",
    ]
    ds = pd.read_parquet(DECISION_STATE_FILE, columns=cols)
    ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
    ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])
    snap = ds[ds["decision_datetime"].dt.time == pd.Timestamp("10:00:00").time()].copy()

    rows: list[dict[str, object]] = []
    for date, grp in snap.groupby("date", sort=True):
        snapshot_rows = []
        for _, r in grp.iterrows():
            snapshot_rows.append(
                {
                    "strike": r["strike"],
                    "right": "call",
                    "delta": r["call_delta"],
                    "implied_vol": r["call_implied_vol"],
                    "iv_error": 0,
                }
            )
            snapshot_rows.append(
                {
                    "strike": r["strike"],
                    "right": "put",
                    "delta": r["put_delta"],
                    "implied_vol": r["put_implied_vol"],
                    "iv_error": 0,
                }
            )
        snapshot = pd.DataFrame(snapshot_rows)
        feats = extract_snapshot_iv_features(snapshot)
        feats["date"] = date
        rows.append(feats)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily

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
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0,
        raw=False,
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
    return daily


def add_family_7(df: pd.DataFrame) -> pd.DataFrame:
    spx_daily = _load_spx_daily()
    daily_spx = _build_daily_spx_technical_features(spx_daily)
    daily_vix = _build_daily_vix_features()
    daily_vix1d = _build_daily_vix1d_features()
    daily_iv = _build_daily_snapshot_iv_features()

    out = df.merge(daily_spx, on="date", how="left")
    out = out.merge(daily_vix, on="date", how="left")
    out = out.merge(daily_vix1d, on="date", how="left")
    out = out.merge(daily_iv, on="date", how="left")

    out["daily_prior_vix1d_vs_vix"] = np.where(
        out["daily_prior_vix_close"] > 0,
        out["daily_prior_vix1d_close"] / out["daily_prior_vix_close"],
        np.nan,
    )
    out["daily_prior_iv_rv_ratio"] = np.where(
        out["daily_prior_yz_hv_20day"] > 0,
        out["daily_prior_atm_iv"] / out["daily_prior_yz_hv_20day"],
        np.nan,
    )
    for w in (5, 10, 20, 30):
        hv_col = f"daily_prior_yz_hv_{w}day"
        out[f"daily_prior_hv_iv_ratio_{w}day"] = np.where(
            out[hv_col] > 0,
            out["daily_prior_atm_iv"] / out[hv_col],
            np.nan,
        )

    return out


def _round_strike(price: float, step: float = 5.0) -> float:
    return float(np.round(price / step) * step)


def _lookup_mid(grp: pd.DataFrame, strike: float, side: str) -> float:
    row = grp.loc[grp["strike"] == strike]
    if row.empty:
        return np.nan
    col = "call_mid_close" if side == "call" else "put_mid_close"
    val = row[col].iloc[0]
    return float(val) if pd.notna(val) and val > 0 else np.nan


def _decision_state_snapshot_features(master_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "decision_datetime",
        "strike",
        "call_delta",
        "put_delta",
        "call_mid_close",
        "put_mid_close",
        "call_implied_vol",
        "put_implied_vol",
        "spx_close",
    ]
    ds = pd.read_parquet(DECISION_STATE_FILE, columns=cols)
    ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])

    unique_master = master_df[["decision_datetime", "max_pain_strike", "ts_7dte_atm_mid", "ts_30dte_atm_mid"]].drop_duplicates("decision_datetime")
    want_times = set(unique_master["decision_datetime"])
    ds = ds[ds["decision_datetime"].isin(want_times)]

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
    rows: list[dict[str, object]] = []

    def nearest_iv(delta_arr: np.ndarray, iv_arr: np.ndarray, target: float, tolerance: float) -> float:
        mask = np.isfinite(delta_arr) & np.isfinite(iv_arr) & (iv_arr > 0.01) & (iv_arr < 5.0)
        if not mask.any():
            return np.nan
        d = np.abs(delta_arr[mask] - target)
        idx = np.argmin(d)
        return float(iv_arr[mask][idx]) if d[idx] <= tolerance else np.nan

    def strike_lookup(strikes: np.ndarray, values: np.ndarray, strike: float) -> float:
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
        atm = _round_strike(spx, 5.0)

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

        def ic_credit(short_offset: int, wing_offset: int) -> float:
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

        rows.append(
            {
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
            }
        )

    return pd.DataFrame(rows)


def add_family_8(df: pd.DataFrame) -> pd.DataFrame:
    snap = _decision_state_snapshot_features(df)
    return df.merge(snap, on="decision_datetime", how="left")


def _build_daily_term_structure_iv_features() -> pd.DataFrame:
    ts = pd.read_parquet(TERM_STRUCTURE_RAW_FILE, columns=["date", "dte", "strike", "right", "spx_close", "moneyness", "mid"])
    ts["date"] = pd.to_datetime(ts["date"]).dt.normalize()
    ts = ts.sort_values(["date", "dte", "right", "strike"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for date, grp in ts.groupby("date", sort=True):
        spx_close = grp["spx_close"].dropna().iloc[0] if grp["spx_close"].notna().any() else np.nan
        feats = extract_term_structure_iv(grp[["dte", "moneyness", "right", "mid", "strike"]], spx_close)
        feats["date"] = date
        rows.append(feats)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily

    daily["daily_prior_ts_iv_7dte"] = daily["ts_iv_7dte"]
    daily["daily_prior_ts_iv_14dte"] = daily["ts_iv_14dte"]
    daily["daily_prior_ts_iv_slope_7dte_14dte"] = np.where(
        daily["ts_iv_14dte"] > 0,
        daily["ts_iv_7dte"] / daily["ts_iv_14dte"],
        np.nan,
    )
    daily["daily_prior_ts_iv_inverted_7dte_14dte"] = np.where(
        daily["daily_prior_ts_iv_slope_7dte_14dte"].notna(),
        (daily["daily_prior_ts_iv_slope_7dte_14dte"] > 1.0).astype(float),
        np.nan,
    )
    feature_cols = [c for c in daily.columns if c != "date"]
    daily.loc[:, feature_cols] = daily.loc[:, feature_cols].shift(1)
    return daily[
        [
            "date",
            "daily_prior_ts_iv_7dte",
            "daily_prior_ts_iv_14dte",
            "daily_prior_ts_iv_slope_7dte_14dte",
            "daily_prior_ts_iv_inverted_7dte_14dte",
        ]
    ]


def _build_daily_options_volume_features() -> pd.DataFrame:
    oi = pd.read_parquet(OI_RAW_FILE, columns=["date", "expiration", "strike", "right", "open_interest"])
    oi["date"] = pd.to_datetime(oi["date"]).dt.normalize()
    oi["expiration"] = pd.to_datetime(oi["expiration"]).dt.normalize()
    oi = oi[oi["date"] == oi["expiration"]].copy()
    if oi.empty:
        return pd.DataFrame(columns=["date", "daily_prior_total_call_volume", "daily_prior_total_put_volume", "daily_prior_pc_volume_ratio"])

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
        daily["total_call_volume"] > 0,
        daily["total_put_volume"] / daily["total_call_volume"],
        np.nan,
    )
    daily["daily_prior_total_call_volume"] = daily["total_call_volume"].shift(1)
    daily["daily_prior_total_put_volume"] = daily["total_put_volume"].shift(1)
    daily["daily_prior_pc_volume_ratio"] = daily["pc_volume_ratio"].shift(1)
    return daily[
        [
            "date",
            "daily_prior_total_call_volume",
            "daily_prior_total_put_volume",
            "daily_prior_pc_volume_ratio",
        ]
    ]


def _build_daily_sector_spread_features() -> pd.DataFrame:
    raw = pd.read_parquet(BREADTH_RAW_FILE)
    raw = raw.reset_index().rename(columns={"Date": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.sort_values("date").reset_index(drop=True)
    for col in ("SPY", "XLF", "XLK", "XLE"):
        raw[f"{col.lower()}_ret_1day"] = raw[col].pct_change()
    raw["daily_prior_xlf_spy_ret_spread_1day"] = (raw["xlf_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    raw["daily_prior_xlk_spy_ret_spread_1day"] = (raw["xlk_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    raw["daily_prior_xle_spy_ret_spread_1day"] = (raw["xle_ret_1day"] - raw["spy_ret_1day"]).shift(1)
    return raw[
        [
            "date",
            "daily_prior_xlf_spy_ret_spread_1day",
            "daily_prior_xlk_spy_ret_spread_1day",
            "daily_prior_xle_spy_ret_spread_1day",
        ]
    ]


def _build_regime_features() -> pd.DataFrame:
    regime = pd.read_parquet(REGIME_FEATURES_FILE)
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    regime = regime.sort_values("date").reset_index(drop=True)
    regime["daily_prior_regime_state"] = regime["regime_state"].shift(1)
    regime["daily_prior_regime_prob_highvol"] = regime["regime_prob_highvol"].shift(1)
    regime["daily_prior_regime_duration"] = regime["regime_duration"].shift(1)
    regime["daily_prior_regime_switch"] = regime["regime_switch"].shift(1)
    return regime[
        [
            "date",
            "daily_prior_regime_state",
            "daily_prior_regime_prob_highvol",
            "daily_prior_regime_duration",
            "daily_prior_regime_switch",
        ]
    ]


def _build_event_calendar_features() -> pd.DataFrame:
    cal = pd.read_parquet(ECON_CALENDAR_FILE)
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    return cal[
        [
            "date",
            "is_cpi_day",
            "is_gdp_day",
            "is_nfp_day",
            "is_ppi_day",
            "days_to_next_cpi",
            "days_to_next_gdp",
            "days_to_next_nfp",
            "days_to_next_ppi",
            "days_to_next_fomc",
            "days_to_next_any_econ",
        ]
    ].copy()


def _build_mag7_earnings_features() -> pd.DataFrame:
    earn = pd.read_parquet(MAG7_EARNINGS_FILE)
    earn["date"] = pd.to_datetime(earn["date"]).dt.normalize()
    return earn[
        [
            "date",
            "is_earnings_aapl",
            "is_earnings_amzn",
            "is_earnings_googl",
            "is_earnings_meta",
            "is_earnings_msft",
            "is_earnings_nvda",
            "is_earnings_tsla",
            "mag7_earnings_count",
            "is_mag7_earnings_day",
        ]
    ].copy()


def add_family_9(df: pd.DataFrame) -> pd.DataFrame:
    term_iv = _build_daily_term_structure_iv_features()
    volume = _build_daily_options_volume_features()
    spreads = _build_daily_sector_spread_features()
    regime = _build_regime_features()
    events = _build_event_calendar_features()
    earnings = _build_mag7_earnings_features()

    out = df.merge(term_iv, on="date", how="left")
    out = out.merge(volume, on="date", how="left")
    out = out.merge(spreads, on="date", how="left")
    out = out.merge(regime, on="date", how="left")
    out = out.merge(events, on="date", how="left")
    out = out.merge(earnings, on="date", how="left")

    out["intraday_iv_ts_slope_0dte_7dte"] = np.where(
        out["daily_prior_ts_iv_7dte"] > 0,
        out["intraday_atm_iv"] / out["daily_prior_ts_iv_7dte"],
        np.nan,
    )
    out["intraday_iv_ts_slope_0dte_14dte"] = np.where(
        out["daily_prior_ts_iv_14dte"] > 0,
        out["intraday_atm_iv"] / out["daily_prior_ts_iv_14dte"],
        np.nan,
    )
    return out


def qa_frame(df: pd.DataFrame, label: str) -> None:
    key_dupes = int(df.duplicated(["decision_datetime", "strategy"]).sum())
    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy()).sum()) if len(numeric.columns) else 0
    print(f"\n[{label}] rows={len(df):,} cols={len(df.columns)} key_dupes={key_dupes} inf={inf_count}")


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def main() -> None:
    require_exists(INPUT_FILE)
    require_exists(SPX_1MIN_FILE)
    require_exists(DECISION_STATE_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = load_master()
    qa_frame(df, "base")

    df = add_family_1(df)
    qa_frame(df, "family_1")

    df = add_family_2(df)
    qa_frame(df, "family_2")

    df = add_family_3(df)
    qa_frame(df, "family_3")

    df = add_family_4(df)
    qa_frame(df, "family_4")

    df = add_family_5(df)
    df = sanitize_numeric(df)
    qa_frame(df, "family_5")

    df = add_family_6(df)
    df = sanitize_numeric(df)
    qa_frame(df, "family_6")

    df = add_family_7(df)
    df = sanitize_numeric(df)
    qa_frame(df, "family_7")

    df = add_family_8(df)
    df = sanitize_numeric(df)
    qa_frame(df, "family_8")

    df = add_family_9(df)
    df = sanitize_numeric(df)
    qa_frame(df, "family_9")

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUTPUT_FILE)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
