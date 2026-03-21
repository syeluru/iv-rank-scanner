"""
Join daily v1.2 feature tables onto master_data_v1_2.

Join rules:
- same-day: spxw_0dte_oi_daily_v1_2, presidential_cycles_v1_2
- T-1: spxw_term_structure_daily_v1_2, cross_asset_daily_features_v1_2,
       breadth_daily_features_v1_2, vol_context_daily_features_v1_2
- T-2: fred_daily_features_v1_2, macro_regime_v1_2

All joins preserve the master_data_v1_2 row grain of decision_datetime x strategy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_DIR / '2_data_join' / 'master_data_v1_2.parquet'
DATA_DIR = PROJECT_DIR / '1_data_collection' / 'data' / 'v1_2'
MASTER_MAX_DATE = pd.Timestamp('2026-03-13')

JOIN_SPECS = [
    ('spxw_0dte_oi_daily_v1_2.parquet', 'same_day'),
    ('spxw_term_structure_daily_v1_2.parquet', 't1'),
    ('fred_daily_features_v1_2.parquet', 't2'),
    ('cross_asset_daily_features_v1_2.parquet', 't1'),
    ('breadth_daily_features_v1_2.parquet', 't1'),
    ('vol_context_daily_features_v1_2.parquet', 't1'),
    ('macro_regime_v1_2.parquet', 't2'),
    ('presidential_cycles_v1_2.parquet', 'same_day'),
]


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing required input: {path}')


def build_effective_date_map(master_dates: pd.Series, lag: str) -> dict[pd.Timestamp, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(master_dates).drop_duplicates())
    idx = {d: i for i, d in enumerate(unique_dates)}
    out = {}
    for d in unique_dates:
        i = idx[d]
        if lag == 'same_day':
            out[d] = d
        elif lag == 't1':
            if i + 1 < len(unique_dates):
                out[d] = unique_dates[i + 1]
        elif lag == 't2':
            if i + 2 < len(unique_dates):
                out[d] = unique_dates[i + 2]
        else:
            raise ValueError(lag)
    return out


def load_master() -> pd.DataFrame:
    df = pd.read_parquet(MASTER_FILE)
    df['decision_datetime'] = pd.to_datetime(df['decision_datetime'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df[df['date'] <= MASTER_MAX_DATE].copy()
    return df.sort_values(['decision_datetime', 'strategy']).reset_index(drop=True)


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'date' not in df.columns:
        raise ValueError(f'{path.name} missing date column')
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df[df['date'] <= MASTER_MAX_DATE].copy()
    return df.sort_values('date').reset_index(drop=True)


def shift_daily_to_effective_date(df: pd.DataFrame, master_dates: pd.Series, lag: str) -> pd.DataFrame:
    mapping = build_effective_date_map(master_dates, lag)
    out = df.copy()
    out['effective_date'] = out['date'].map(mapping)
    out = out.dropna(subset=['effective_date']).copy()
    out['effective_date'] = pd.to_datetime(out['effective_date']).dt.normalize()
    out = out.drop(columns=['date'])
    out = out.rename(columns={'effective_date': 'date'})
    return out


def main() -> None:
    require_exists(MASTER_FILE)
    master = load_master()
    print(f'Loaded master rows: {len(master):,}')

    # avoid duplicate columns from overlapping macro/fred layers by taking only fields not already present
    current_cols = set(master.columns)

    for fname, lag in JOIN_SPECS:
        path = DATA_DIR / fname
        require_exists(path)
        daily = load_daily(path)
        daily = shift_daily_to_effective_date(daily, master['date'], lag)

        keep_cols = ['date'] + [c for c in daily.columns if c != 'date' and c not in current_cols]
        dropped = [c for c in daily.columns if c != 'date' and c in current_cols]
        daily = daily[keep_cols]

        before = len(master)
        master = master.merge(daily, on='date', how='left', validate='many_to_one')
        current_cols = set(master.columns)
        print(f'Joined {fname} lag={lag} cols_added={len(keep_cols)-1} overlap_skipped={len(dropped)} rows={before:,}')

    # normalize infs to NaN for cleaner QA
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].replace([np.inf, -np.inf], np.nan)

    master = master.sort_values(['decision_datetime', 'strategy']).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(master, preserve_index=False), MASTER_FILE)
    print(f'\nWrote joined master: {MASTER_FILE} rows={len(master):,} cols={len(master.columns)}')


if __name__ == '__main__':
    main()
