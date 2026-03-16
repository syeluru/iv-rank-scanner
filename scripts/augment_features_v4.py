#!/usr/bin/env python3
"""
Augment v3 training features with new v4 features (price action, S/R,
trendlines, statistical regime, expanded RV, expanded time/event).

Instead of re-running the full 8+ hour backfill, this script:
1. Reads existing training_features_v3.parquet to get (date, entry_time) pairs
2. Loads SPX/VIX 1-min data (already local)
3. Computes ONLY new features per slot
4. Outputs training_features_v4.parquet (v3 columns + new columns merged)

Uses checkpoint/resume system (every 25 days).

Usage:
    python -m scripts.augment_features_v4
    python -m scripts.augment_features_v4 --resume
"""

import argparse
import json
import logging
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features.v8.price_action import compute_price_action_features
from ml.features.v8.support_resistance import compute_support_resistance_features
from ml.features.v8.trendlines import compute_trendline_features
from ml.features.v8.statistical_regime import compute_statistical_regime_features
from ml.features.v8.realized_vol import compute_realized_vol_features
from ml.features.v8.time_event import compute_time_event_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("augment_v4")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("ml/features/v8/data")
V3_FILE = OUTPUT_DIR / "training_features_v3.parquet"
V4_FILE = OUTPUT_DIR / "training_features_v4.parquet"
PROGRESS_FILE = OUTPUT_DIR / "v4_augment_progress.json"
CHECKPOINT_EVERY = 25  # days

ENTRY_START = time(9, 35)
ENTRY_END = time(14, 0)
CLOSE_TIME = time(15, 0)


def load_spx_vix() -> dict:
    """Load SPX and VIX 1-min parquets."""
    logger.info("Loading SPX/VIX 1-min data...")

    spx = pd.read_parquet(DATA_DIR / "spx_1min.parquet")
    spx["date"] = pd.to_datetime(spx["datetime"]).dt.date
    spx = spx.rename(columns={"datetime": "timestamp"})
    spx["timestamp"] = pd.to_datetime(spx["timestamp"])
    logger.info(f"  SPX 1-min: {len(spx):,} rows")

    vix = pd.read_parquet(DATA_DIR / "vix_1min.parquet")
    vix["date"] = pd.to_datetime(vix["datetime"]).dt.date
    vix = vix.rename(columns={
        "datetime": "timestamp",
        "vix_open": "open", "vix_high": "high",
        "vix_low": "low", "vix_close": "close",
    })
    vix["timestamp"] = pd.to_datetime(vix["timestamp"])
    logger.info(f"  VIX 1-min: {len(vix):,} rows")

    return {"spx": spx, "vix": vix}


def get_day_data(local_data: dict, d: date) -> dict:
    """Get SPX/VIX 1-min bars for a single day."""
    spx = local_data["spx"]
    vix = local_data["vix"]

    spx_day = spx[spx["date"] == d][["timestamp", "open", "high", "low", "close"]].copy()
    vix_day = vix[vix["date"] == d][["timestamp", "open", "high", "low", "close"]].copy()

    return {
        "spx_1m": spx_day.reset_index(drop=True),
        "vix_1m": vix_day.reset_index(drop=True),
    }


def get_prior_days(local_data: dict, d: date, n_days: int = 5) -> list:
    """Get up to n_days prior trading day DataFrames."""
    spx = local_data["spx"]
    all_dates = sorted(spx["date"].unique())

    try:
        idx = list(all_dates).index(d)
    except ValueError:
        return []

    prior = []
    for i in range(1, min(n_days + 1, idx + 1)):
        prior_date = all_dates[idx - i]
        day_df = spx[spx["date"] == prior_date][
            ["timestamp", "open", "high", "low", "close"]
        ].copy().reset_index(drop=True)
        if not day_df.empty:
            prior.append(day_df)

    return prior


def compute_new_features_for_slot(
    spx_1m_up_to: pd.DataFrame,
    vix_1m_up_to: pd.DataFrame,
    prior_days_1m: list,
    spx_price: float,
    entry_dt: datetime,
) -> dict:
    """Compute all new v4 features for a single entry-time slot."""
    features = {}

    # Price Action (~32 features)
    pa = compute_price_action_features(spx_1m_up_to, vix_1m_up_to)
    features.update(pa)

    # Support & Resistance (~30 features)
    sr = compute_support_resistance_features(spx_1m_up_to, prior_days_1m, spx_price)
    features.update(sr)

    # Trendlines (~21 features)
    tl = compute_trendline_features(spx_1m_up_to, prior_days_1m, spx_price)
    features.update(tl)

    # Statistical & Regime (~14 features)
    stat = compute_statistical_regime_features(spx_1m_up_to, prior_days_1m)
    features.update(stat)

    # Expanded RV features (only the 5 new ones)
    rv = compute_realized_vol_features(spx_1m_up_to, prior_days_1m)
    for key in ['vrp_intraday_roc', 'yang_zhang_rv_30m', 'rv_5m_30m_ratio',
                'realized_skewness_30m', 'realized_kurtosis_30m']:
        features[key] = rv.get(key, np.nan)

    # Expanded time/event features (only the 10 new ones)
    te = compute_time_event_features(entry_dt)
    new_time_keys = [
        'theta_decay_proxy', 'session_fraction_squared',
        'fomc_proximity_kernel', 'cpi_proximity_kernel', 'nfp_proximity_kernel',
        'post_fomc_hours', 'post_cpi_hours', 'post_nfp_hours',
        'pre_event_compression', 'event_cluster_density',
    ]
    for key in new_time_keys:
        features[key] = te.get(key, np.nan)

    return features


def process_day(
    local_data: dict,
    d: date,
    v3_day_slots: pd.DataFrame,
) -> list:
    """Process all entry-time slots for a single day. Returns list of feature dicts."""
    day_data = get_day_data(local_data, d)
    spx_day = day_data["spx_1m"]
    vix_day = day_data["vix_1m"]

    if spx_day.empty:
        return []

    prior_days = get_prior_days(local_data, d, n_days=5)

    results = []
    for _, row in v3_day_slots.iterrows():
        entry_time_str = row["entry_time"]  # "HH:MM"
        h, m = int(entry_time_str[:2]), int(entry_time_str[3:5])
        entry_t = time(h, m)

        # Filter SPX/VIX up to entry time
        entry_dt = datetime(d.year, d.month, d.day, h, m)
        entry_ts = pd.Timestamp(entry_dt)

        spx_up_to = spx_day[spx_day["timestamp"] <= entry_ts].copy()
        vix_up_to = vix_day[vix_day["timestamp"] <= entry_ts].copy() if not vix_day.empty else pd.DataFrame()

        if spx_up_to.empty or len(spx_up_to) < 3:
            # Return NaN features
            dummy = compute_new_features_for_slot(
                pd.DataFrame(), pd.DataFrame(), [], 0.0, entry_dt
            )
            results.append(dummy)
            continue

        spx_price = float(spx_up_to["close"].iloc[-1])

        features = compute_new_features_for_slot(
            spx_up_to, vix_up_to, prior_days, spx_price, entry_dt
        )
        results.append(features)

    return results


def save_checkpoint(completed_dates: list, new_features_rows: list):
    """Save progress checkpoint."""
    progress = {
        "completed_dates": [str(d) for d in completed_dates],
        "n_rows": len(new_features_rows),
        "timestamp": datetime.now().isoformat(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

    # Also save partial results
    if new_features_rows:
        partial_df = pd.DataFrame(new_features_rows)
        partial_df.to_parquet(OUTPUT_DIR / "v4_augment_partial.parquet", index=False)
        logger.info(f"  Checkpoint: {len(completed_dates)} days, {len(new_features_rows):,} rows")


def load_checkpoint() -> tuple:
    """Load checkpoint if it exists."""
    if not PROGRESS_FILE.exists():
        return set(), []

    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    completed = {date.fromisoformat(d) for d in progress.get("completed_dates", [])}

    partial_file = OUTPUT_DIR / "v4_augment_partial.parquet"
    if partial_file.exists():
        partial_df = pd.read_parquet(partial_file)
        rows = partial_df.to_dict("records")
        logger.info(f"Resuming from checkpoint: {len(completed)} days, {len(rows):,} rows")
        return completed, rows

    return completed, []


def main():
    parser = argparse.ArgumentParser(description="Augment v3 features with new v4 features")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Process 2 days only")
    args = parser.parse_args()

    # Load v3 features
    if not V3_FILE.exists():
        logger.error(f"v3 features not found: {V3_FILE}")
        return

    logger.info("Loading v3 features...")
    v3_df = pd.read_parquet(V3_FILE)
    logger.info(f"  v3 features: {len(v3_df):,} rows × {len(v3_df.columns)} columns")

    # Get unique dates
    v3_dates = sorted(v3_df["date"].unique())
    logger.info(f"  {len(v3_dates)} unique dates")

    # Load local data
    local_data = load_spx_vix()

    # Resume from checkpoint
    if args.resume:
        completed_dates, new_features_rows = load_checkpoint()
    else:
        completed_dates, new_features_rows = set(), []

    # Process each date
    dates_to_process = [d for d in v3_dates if d not in completed_dates]
    if args.dry_run:
        dates_to_process = dates_to_process[:2]

    logger.info(f"Processing {len(dates_to_process)} dates...")
    start_time = _time.time()
    completed_list = list(completed_dates)

    for i, date_str in enumerate(dates_to_process):
        if isinstance(date_str, str):
            d = date.fromisoformat(date_str)
        else:
            d = date_str

        day_slots = v3_df[v3_df["date"] == date_str]

        t0 = _time.time()
        day_features = process_day(local_data, d, day_slots)

        if len(day_features) != len(day_slots):
            logger.warning(f"  {d}: expected {len(day_slots)} rows, got {len(day_features)}")
            # Pad with NaN
            while len(day_features) < len(day_slots):
                day_features.append({})

        new_features_rows.extend(day_features)
        completed_list.append(d)
        elapsed = _time.time() - t0

        if (i + 1) % 10 == 0 or (i + 1) == len(dates_to_process):
            total_elapsed = _time.time() - start_time
            rate = (i + 1) / total_elapsed * 60
            remaining = (len(dates_to_process) - i - 1) / max(rate, 0.01)
            logger.info(
                f"  [{i+1}/{len(dates_to_process)}] {d} "
                f"({len(day_slots)} slots, {elapsed:.1f}s) "
                f"| {rate:.0f} days/min | ETA {remaining:.0f} min"
            )

        # Checkpoint every N days
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(completed_list, new_features_rows)

    # Final save
    if not new_features_rows:
        logger.info("No new features computed.")
        return

    logger.info("Merging new features with v3...")
    new_df = pd.DataFrame(new_features_rows)
    logger.info(f"  New features: {len(new_df):,} rows × {len(new_df.columns)} columns")

    # Verify alignment
    if len(new_df) != len(v3_df):
        logger.error(f"Row count mismatch: v3={len(v3_df)}, new={len(new_df)}")
        # Save partial anyway
        new_df.to_parquet(OUTPUT_DIR / "v4_new_features_only.parquet", index=False)
        logger.info("Saved new features only (not merged) due to row mismatch.")
        return

    # Merge: v3 columns + new columns
    v4_df = pd.concat([v3_df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
    logger.info(f"  v4 features: {len(v4_df):,} rows × {len(v4_df.columns)} columns")

    # Check NaN rates
    new_cols = new_df.columns.tolist()
    nan_rates = v4_df[new_cols].isna().mean()
    high_nan = nan_rates[nan_rates > 0.20]
    if not high_nan.empty:
        logger.warning(f"  High NaN features (>20%): {len(high_nan)}")
        for col, rate in high_nan.items():
            logger.warning(f"    {col}: {rate:.1%}")

    # Save
    v4_df.to_parquet(V4_FILE, index=False)
    logger.info(f"Saved: {V4_FILE}")

    # Clean up checkpoint
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    partial = OUTPUT_DIR / "v4_augment_partial.parquet"
    if partial.exists():
        partial.unlink()

    total_time = _time.time() - start_time
    logger.info(f"Done in {total_time / 60:.1f} minutes")
    logger.info(f"  New columns added: {len(new_cols)}")
    logger.info(f"  Total columns: {len(v4_df.columns)}")


if __name__ == "__main__":
    main()
