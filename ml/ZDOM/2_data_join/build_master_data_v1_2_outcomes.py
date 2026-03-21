"""
Append TP outcome columns onto master_data_v1_2.

For each row in master_data_v1_2:
- use the locked strikes already selected in the master row
- walk forward through intraday_decision_state_v1_2 after entry
- compute debit path from the four locked legs
- resolve TP10..TP50 races against the 2x-credit stop

Writes the enriched table back to master_data_v1_2.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_DIR / "2_data_join" / "master_data_v1_2.parquet"
STATE_FILE = (
    PROJECT_DIR
    / "1_data_collection"
    / "data"
    / "v1_2"
    / "intraday_decision_state_v1_2.parquet"
)

SL_MULT = 2.0
TP_LEVELS = [i / 100 for i in range(10, 55, 5)]
CLOSE_TIME = "15:00:00"


OUTCOME_COLS = [
    f"tp{int(tp * 100)}_{suffix}"
    for tp in TP_LEVELS
    for suffix in ("target", "exit_reason", "exit_time", "exit_debit", "pnl")
]


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_master() -> pd.DataFrame:
    df = pd.read_parquet(MASTER_FILE)
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "decision_datetime", "strategy"]).reset_index(drop=True)


def load_state_for_dates(dates: list[pd.Timestamp]) -> pd.DataFrame:
    cols = ["date", "decision_datetime", "strike", "call_mid_close", "put_mid_close"]
    table = pq.read_table(STATE_FILE, columns=cols)
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["decision_datetime"] = pd.to_datetime(df["decision_datetime"])
    return df[df["date"].isin(dates)].copy()


def first_true_idx(arr: np.ndarray) -> int:
    idx = np.flatnonzero(arr)
    return int(idx[0]) if len(idx) else -1


def simulate_row(row, dt_to_idx, times, call_map, put_map, close_idx: int) -> dict:
    dt = row.decision_datetime
    start_idx = dt_to_idx.get(dt)
    if start_idx is None or start_idx >= close_idx:
        return {col: np.nan for col in OUTCOME_COLS}

    future_slice = slice(start_idx + 1, close_idx + 1)

    sc = call_map.get(row.short_call)
    sp = put_map.get(row.short_put)
    lc = call_map.get(row.long_call)
    lp = put_map.get(row.long_put)
    if any(x is None for x in (sc, sp, lc, lp)):
        return {col: np.nan for col in OUTCOME_COLS}

    debit = sc[future_slice] + sp[future_slice] - lc[future_slice] - lp[future_slice]
    future_times = times[future_slice]
    valid = ~np.isnan(debit)
    if not valid.any():
        return {col: np.nan for col in OUTCOME_COLS}

    debit = debit[valid]
    future_times = future_times[valid]

    credit = float(row.credit)
    sl_debit = credit * SL_MULT

    result: dict[str, object] = {}
    for pct in TP_LEVELS:
        key = f"tp{int(pct * 100)}"
        tp_debit = credit * (1 - pct)

        sl_idx = first_true_idx(debit >= sl_debit)
        tp_idx = first_true_idx(debit <= tp_debit)

        if sl_idx >= 0 and (tp_idx < 0 or sl_idx <= tp_idx):
            exit_idx = sl_idx
            exit_debit = float(debit[exit_idx])
            result[f"{key}_target"] = 0
            result[f"{key}_exit_reason"] = "sl"
            result[f"{key}_exit_time"] = pd.Timestamp(future_times[exit_idx]).isoformat()
            result[f"{key}_exit_debit"] = round(exit_debit, 4)
            result[f"{key}_pnl"] = round(credit - exit_debit, 4)
            continue

        if tp_idx >= 0:
            exit_idx = tp_idx
            exit_debit = float(debit[exit_idx])
            result[f"{key}_target"] = 1
            result[f"{key}_exit_reason"] = "tp"
            result[f"{key}_exit_time"] = pd.Timestamp(future_times[exit_idx]).isoformat()
            result[f"{key}_exit_debit"] = round(exit_debit, 4)
            result[f"{key}_pnl"] = round(credit - exit_debit, 4)
            continue

        exit_debit = float(debit[-1])
        exit_time = pd.Timestamp(future_times[-1]).isoformat()
        win = exit_debit < credit
        result[f"{key}_target"] = int(win)
        result[f"{key}_exit_reason"] = "close_win" if win else "close_loss"
        result[f"{key}_exit_time"] = exit_time
        result[f"{key}_exit_debit"] = round(exit_debit, 4)
        result[f"{key}_pnl"] = round(credit - exit_debit, 4)

    return result


def build_day_outcomes(day_master: pd.DataFrame, day_state: pd.DataFrame) -> pd.DataFrame:
    times = np.array(sorted(day_state["decision_datetime"].unique()), dtype="datetime64[ns]")
    if len(times) == 0:
        return pd.DataFrame(index=day_master.index, columns=OUTCOME_COLS)

    close_ts = np.datetime64(pd.Timestamp(f"{day_master['date'].iloc[0].date()} {CLOSE_TIME}"))
    close_matches = np.flatnonzero(times == close_ts)
    if len(close_matches) == 0:
        close_idx = len(times) - 1
    else:
        close_idx = int(close_matches[0])

    needed_strikes = np.unique(
        np.concatenate(
            [
                day_master["short_call"].to_numpy(dtype=float),
                day_master["short_put"].to_numpy(dtype=float),
                day_master["long_call"].to_numpy(dtype=float),
                day_master["long_put"].to_numpy(dtype=float),
            ]
        )
    )
    state_needed = day_state[day_state["strike"].isin(needed_strikes)].copy()

    call_piv = state_needed.pivot(index="decision_datetime", columns="strike", values="call_mid_close").reindex(pd.to_datetime(times))
    put_piv = state_needed.pivot(index="decision_datetime", columns="strike", values="put_mid_close").reindex(pd.to_datetime(times))

    call_map = {float(col): call_piv[col].to_numpy(dtype=float) for col in call_piv.columns}
    put_map = {float(col): put_piv[col].to_numpy(dtype=float) for col in put_piv.columns}
    dt_to_idx = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(times))}

    records = []
    for row in day_master.itertuples(index=False):
        records.append(simulate_row(row, dt_to_idx, times, call_map, put_map, close_idx))

    return pd.DataFrame(records, index=day_master.index)


def main() -> None:
    require_exists(MASTER_FILE)
    require_exists(STATE_FILE)

    master = load_master()
    dates = sorted(master["date"].drop_duplicates().tolist())
    state = load_state_for_dates(dates)

    print("Appending outcomes to master_data_v1_2")
    print(f"  Master rows: {len(master):,}")
    print(f"  Trading dates: {len(dates):,}")
    print()

    outcome_frames = []
    grouped_state = {d: g.copy() for d, g in state.groupby("date")}
    grouped_master = list(master.groupby("date", sort=True))

    for i, (date, day_master) in enumerate(grouped_master, start=1):
        day_state = grouped_state.get(date)
        if day_state is None or day_state.empty:
            outcome_frames.append(pd.DataFrame(index=day_master.index, columns=OUTCOME_COLS))
        else:
            outcome_frames.append(build_day_outcomes(day_master, day_state))

        if i % 25 == 0 or i == len(grouped_master):
            print(f"[{i}/{len(grouped_master)}] processed {date.date()} rows={sum(len(x) for x in outcome_frames):,}")

    outcomes = pd.concat(outcome_frames).sort_index()
    out = pd.concat([master, outcomes], axis=1)
    out = out.sort_values(["decision_datetime", "strategy"]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), MASTER_FILE)

    print(f"\nWrote {len(out):,} rows to {MASTER_FILE}")


if __name__ == "__main__":
    main()
