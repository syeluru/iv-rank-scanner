"""
Build master_data_v1_2 step 1: target aggregation spine.

This produces one row per decision_datetime x strategy from the intraday
decision-state table. It does not add daily joins yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DECISION_FILE = (
    PROJECT_DIR
    / "1_data_collection"
    / "data"
    / "v1_2"
    / "intraday_decision_state_v1_2.parquet"
)
OUT_FILE = PROJECT_DIR / "2_data_join" / "data" / "v1_2" / "master_data_v1_2.parquet"

ENTRY_START = "10:00"
ENTRY_END_EXCLUSIVE = "15:00"
CLOSE_TIME = "15:00"

STRATEGIES = [
    {"name": f"IC_{d:02d}d_25w", "short_delta": d / 100, "wing_width": 25}
    for d in range(5, 50, 5)
]


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_decision_state() -> pd.DataFrame:
    cols = [
        "decision_datetime",
        "date",
        "strike",
        "call_bid_close",
        "call_mid_close",
        "put_bid_close",
        "put_mid_close",
        "call_delta",
        "put_delta",
        "call_implied_vol",
        "put_implied_vol",
        "spx_close",
    ]
    df = pd.read_parquet(DECISION_FILE, columns=cols)
    df = df.rename(columns={"spx_close": "spx_prev_min_close"})
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"])

    t = df["decision_datetime"].dt.time
    df = df.loc[
        (t >= pd.Timestamp(f"{ENTRY_START}:00").time())
        & (t < pd.Timestamp(f"{ENTRY_END_EXCLUSIVE}:00").time())
    ].copy()

    df = df.sort_values(["decision_datetime", "strike"]).reset_index(drop=True)
    return df


def pick_long_idx(strikes: np.ndarray, valid_mask: np.ndarray, target_strike: float) -> int:
    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) == 0:
        return -1

    valid_strikes = strikes[valid_idx]
    pos = np.searchsorted(valid_strikes, target_strike)
    candidates = []
    if pos < len(valid_idx):
        candidates.append(valid_idx[pos])
    if pos > 0:
        candidates.append(valid_idx[pos - 1])
    if not candidates:
        return -1

    best = min(candidates, key=lambda idx: abs(strikes[idx] - target_strike))
    if abs(strikes[best] - target_strike) > 10:
        return -1
    return int(best)


def build_rows_for_slice(dt: pd.Timestamp, date: pd.Timestamp, g: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    spx_prev_min_close = g["spx_prev_min_close"].iloc[0]
    if pd.isna(spx_prev_min_close):
        return rows

    close_ts = pd.Timestamp(f"{date.date()} {CLOSE_TIME}:00")
    if int((close_ts - dt).total_seconds() / 60) <= 0:
        return rows

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
    call_long_valid = call_bid > 0
    put_long_valid = put_bid > 0

    if not call_short_valid.any() or not put_short_valid.any():
        return rows

    for strat in STRATEGIES:
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

        lc_idx = pick_long_idx(strikes, call_long_valid, sc_strike + wing)
        lp_idx = pick_long_idx(strikes, put_long_valid, sp_strike - wing)
        if lc_idx < 0 or lp_idx < 0:
            continue

        call_wing_width = float(strikes[lc_idx] - sc_strike)
        put_wing_width = float(sp_strike - strikes[lp_idx])
        if call_wing_width <= 0 or put_wing_width <= 0:
            continue

        credit = float(call_mid[sc_idx] + put_mid[sp_idx] - call_mid[lc_idx] - put_mid[lp_idx])
        if not np.isfinite(credit) or credit <= 0:
            continue

        rows.append(
            {
                "decision_datetime": dt,
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
            }
        )

    return rows


def main() -> None:
    require_exists(DECISION_FILE)
    df = load_decision_state()

    decision_dt = df["decision_datetime"].to_numpy()
    boundaries = np.flatnonzero(decision_dt[1:] != decision_dt[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(df)]))

    print("Building master_data_v1_2 target aggregation spine")
    print(f"  Input:  {DECISION_FILE}")
    print(f"  Output: {OUT_FILE}")
    print(f"  Decision timestamps: {len(starts):,}")
    print(f"  Strategies: {len(STRATEGIES)}")
    print()

    out_rows: list[dict] = []
    total = len(starts)
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        g = df.iloc[s:e]
        dt = g["decision_datetime"].iloc[0]
        date = g["date"].iloc[0]
        out_rows.extend(build_rows_for_slice(dt, date, g))

        if i % 5000 == 0 or i == total:
            print(f"[{i}/{total}] rows={len(out_rows):,}")

    out = pd.DataFrame(out_rows)
    out["target_delta"] = out["strategy"].str.extract(r"IC_(\d{2})d").astype(float) / 100.0

    pre_dedup_rows = len(out)
    combo_cols = ["decision_datetime", "short_call", "short_put", "long_call", "long_put"]
    out = out.sort_values(combo_cols + ["target_delta"], ascending=[True, True, True, True, True, False])
    out = out.drop_duplicates(subset=combo_cols, keep="first")

    out = out.drop(columns=["target_delta"])
    out = out.sort_values(["decision_datetime", "strategy"]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), OUT_FILE)

    print(f"\nRows before combo dedup: {pre_dedup_rows:,}")
    print(f"Rows after combo dedup: {len(out):,}")
    print(f"\nWrote {len(out):,} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
