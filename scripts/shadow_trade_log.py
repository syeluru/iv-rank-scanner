"""
Shadow Trading Log — Bill Benton's paper trading validator.

Simulates the full daily decision pipeline without executing real trades.
Run this every day to build a paper trading track record before going live.

Usage:
  # 10am ET — score today + record theoretical IC entry
  python3 scripts/shadow_trade_log.py

  # 3pm ET — record close and calculate P&L
  python3 scripts/shadow_trade_log.py --close

  # Score a specific date (backfill)
  python3 scripts/shadow_trade_log.py --date 2026-03-01

  # Close a specific date
  python3 scripts/shadow_trade_log.py --close --date 2026-03-01

  # View recent log
  python3 scripts/shadow_trade_log.py --review --n 10

Output: logs/shadow_trade_log.csv
Columns:
  date, signal, num_models_trade, models_voted_trade, model_scores,
  hard_blockers, soft_blockers, vix_level, spx_entry_price,
  suggested_structure, short_put, long_put, short_call, long_call,
  spread_width, entry_credit_est, entry_time_et,
  close_spx_price, close_time_et, close_inside_strikes,
  pnl_est, pnl_pct_of_credit, notes, logged_at
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"
LOGS_DIR    = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE   = LOGS_DIR / "shadow_trade_log.csv"
ET_TZ      = ZoneInfo("America/New_York")

# ── Log columns (ordered) ──────────────────────────────────────────────────────
LOG_COLUMNS = [
    "date",
    "signal",                  # TRADE / SKIP / SIZE_DOWN / BLOCKED
    "num_models_trade",        # how many of 5 models voted TRADE
    "models_voted_trade",      # comma-separated list
    "model_scores",            # JSON of {horizon: score}
    "hard_blockers",           # pipe-separated list or ""
    "soft_blockers",           # pipe-separated list or ""
    "vix_level",
    "spx_entry_price",
    "suggested_structure",     # IC / SINGLE_PUT / SINGLE_CALL / SKIP
    "short_put",
    "long_put",
    "short_call",
    "long_call",
    "spread_width",
    "entry_credit_est",
    "entry_time_et",
    "close_spx_price",         # filled at 3pm
    "close_time_et",
    "close_inside_strikes",    # True/False/NA
    "pnl_est",                 # estimated P&L in dollars per spread
    "pnl_pct_of_credit",       # pnl / entry_credit * 100
    "notes",
    "logged_at",
]

# ── Model horizon config ───────────────────────────────────────────────────────
HORIZONS = {
    "2yr":  "model_2yr.pkl",
    "1yr":  "model_1yr.pkl",
    "180d": "model_180d.pkl",
    "90d":  "model_90d.pkl",
    "7d":   "model_7d.pkl",
}

# Fallback to legacy single model if multi-horizon not trained yet
LEGACY_MODEL = "xgb_model.pkl"

# Minimum model score to call TRADE per horizon
TRADE_THRESHOLD = 0.55
# Soft-blocker requires this score to override
SOFT_THRESHOLD  = 0.70
# Need at least this many models to agree for a TRADE signal
MIN_MODELS_AGREE = 3


# ── Strike / credit estimation ─────────────────────────────────────────────────

def estimate_15delta_strikes(spx_price: float, vix_level: float,
                              spread_width_pts: int = None) -> dict:
    """
    Estimate 15-delta short strikes using a lognormal approximation.

    For 0DTE (T = 1/252 years):
      z = N^-1(0.85) ≈ 1.036  (15-delta put => d2 ≈ 1.036 in simplified model)
      short_put  ≈ S * exp(-z * sigma * sqrt(T))
      short_call ≈ S * exp(+z * sigma * sqrt(T))

    Long strikes are spread_width below/above the short strikes.
    """
    sigma   = vix_level / 100.0
    T       = 1.0 / 252.0
    z       = 1.036
    sqrt_T  = math.sqrt(T)

    raw_put  = spx_price * math.exp(-z * sigma * sqrt_T)
    raw_call = spx_price * math.exp( z * sigma * sqrt_T)

    # Round to nearest 5 (SPX strikes are in increments of 5)
    short_put  = round(raw_put  / 5) * 5
    short_call = round(raw_call / 5) * 5

    # Ensure minimum distance from ATM of at least 10 points
    short_put  = min(short_put,  int(spx_price) - 10)
    short_call = max(short_call, int(spx_price) + 10)

    # Spread width: default ~1% of SPX, rounded to nearest 25
    if spread_width_pts is None:
        raw_width = max(50, round(spx_price * 0.01 / 25) * 25)
        spread_width_pts = int(raw_width)

    long_put  = short_put  - spread_width_pts
    long_call = short_call + spread_width_pts

    return {
        "short_put":    short_put,
        "long_put":     long_put,
        "short_call":   short_call,
        "long_call":    long_call,
        "spread_width": spread_width_pts,
    }


def estimate_ic_credit(spread_width: int, vix_level: float,
                        spx_price: float, short_put: float, short_call: float) -> float:
    """
    Rough credit estimate for 15-delta IC.

    Empirical rule: 15-delta IC on SPX typically collects ~10-15% of spread width.
    Scales with vol: at VIX=15, ~10%; at VIX=25, ~15%.
    """
    # VIX-scaled credit ratio (capped between 0.08 and 0.20)
    base_ratio   = 0.10
    vol_adj      = (vix_level - 15.0) / 100.0  # adds ~1% per VIX point above 15
    credit_ratio = max(0.08, min(0.20, base_ratio + vol_adj))

    # Total IC = two spreads
    credit = spread_width * credit_ratio
    return round(credit, 2)


# ── Live data fetching ─────────────────────────────────────────────────────────

def fetch_live_spx_vix(target_date: str = None) -> tuple[float | None, float | None]:
    """Fetch SPX and VIX from yfinance. Returns (spx_price, vix_level)."""
    try:
        import yfinance as yf
        period = "5d"
        spx_df = yf.download("^GSPC", period=period, interval="1d",
                              progress=False, auto_adjust=True)
        vix_df = yf.download("^VIX",  period=period, interval="1d",
                              progress=False, auto_adjust=True)

        if spx_df.empty or vix_df.empty:
            return None, None

        if target_date:
            tdt = pd.Timestamp(target_date)
            spx_row = spx_df[spx_df.index.normalize() == tdt]
            vix_row = vix_df[vix_df.index.normalize() == tdt]
        else:
            spx_row = spx_df.tail(1)
            vix_row = vix_df.tail(1)

        if spx_row.empty:
            spx_row = spx_df.tail(1)
        if vix_row.empty:
            vix_row = vix_df.tail(1)

        # Handle MultiIndex columns from yfinance
        spx_close = float(spx_row["Close"].iloc[-1]) if "Close" in spx_row.columns else float(spx_row.iloc[-1, 3])
        vix_close = float(vix_row["Close"].iloc[-1]) if "Close" in vix_row.columns else float(vix_row.iloc[-1, 3])

        return spx_close, vix_close

    except Exception as e:
        print(f"  [warn] yfinance error: {e}")
        return None, None


def fetch_spx_close_at_3pm(target_date: str = None) -> float | None:
    """
    Fetch SPX closing price for the day (proxy for 3pm close).
    For intraday use, this returns the current or last available price.
    """
    try:
        import yfinance as yf
        period = "5d"

        # Try to get intraday data for today
        spx_1d = yf.download("^GSPC", period=period, interval="1h",
                              progress=False, auto_adjust=True)
        if spx_1d.empty:
            spx_price, _ = fetch_live_spx_vix(target_date)
            return spx_price

        if target_date:
            tdt = pd.Timestamp(target_date)
            day_data = spx_1d[spx_1d.index.normalize() == tdt.normalize()]
        else:
            today = pd.Timestamp.today().normalize()
            day_data = spx_1d[spx_1d.index.normalize() == today]

        if day_data.empty:
            day_data = spx_1d.tail(1)

        close_col = "Close" if "Close" in day_data.columns else day_data.columns[-2]
        return float(day_data[close_col].iloc[-1])

    except Exception as e:
        print(f"  [warn] close fetch error: {e}")
        return None


# ── Model loading + scoring ────────────────────────────────────────────────────

def load_models() -> dict:
    """
    Load all available horizon models. Falls back to legacy model if needed.
    Returns dict of {horizon_name: model_data_dict}.
    """
    loaded = {}

    for horizon, fname in HORIZONS.items():
        path = MODELS_DIR / fname
        if path.exists():
            try:
                with open(path, "rb") as f:
                    loaded[horizon] = pickle.load(f)
                print(f"  [model] loaded {horizon} ({fname})")
            except Exception as e:
                print(f"  [warn] failed to load {horizon}: {e}")

    # Fallback: use legacy model for any missing horizon
    if not loaded:
        legacy_path = MODELS_DIR / LEGACY_MODEL
        if legacy_path.exists():
            try:
                with open(legacy_path, "rb") as f:
                    legacy = pickle.load(f)
                for horizon in HORIZONS:
                    loaded[horizon] = legacy
                print(f"  [model] using legacy model for all horizons (multi-horizon not trained yet)")
            except Exception as e:
                print(f"  [warn] failed to load legacy model: {e}")

    return loaded


def build_features_for_date(target_date: str, feature_cols: list) -> dict:
    """
    Pull features for a specific date from existing parquet files.
    Same logic as score_live.py's build_live_features.
    """
    target_dt = pd.Timestamp(target_date)
    features  = {}

    for fname in ["spy_features.parquet", "options_features.parquet",
                  "model_table.parquet"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
            df["date"] = pd.to_datetime(df["date"])
            row = df[df["date"] == target_dt]
            if row.empty:
                row = df[df["date"] <= target_dt].tail(1)
            if not row.empty:
                for col in row.columns:
                    if col in feature_cols and col not in features:
                        val = row[col].iloc[0]
                        if not pd.isna(val):
                            features[col] = val
        except Exception as e:
            print(f"  [warn] could not read {fname}: {e}")

    return features


def score_all_models(target_date: str, models: dict) -> dict:
    """
    Score target_date across all loaded models.
    Returns {horizon: score} — score is None if model couldn't run.
    """
    scores = {}

    for horizon, model_data in models.items():
        try:
            feature_cols = model_data["feature_cols"]
            model        = model_data["model"]

            features = build_features_for_date(target_date, feature_cols)
            X_row = pd.DataFrame([{col: features.get(col, np.nan)
                                    for col in feature_cols}])
            X_row = X_row.fillna(0)

            score = float(model.predict_proba(X_row)[0, 1])
            scores[horizon] = round(score, 4)
        except Exception as e:
            print(f"  [warn] {horizon} scoring failed: {e}")
            scores[horizon] = None

    return scores


# ── Hard/soft blocker checks ───────────────────────────────────────────────────

def check_hard_blockers(target_date: str, vix_level: float | None = None) -> list[str]:
    """Return list of hard blocker reasons (empty = no blockers)."""
    target_dt = pd.Timestamp(target_date)
    blockers  = []

    # Event-based blockers
    for fname, label in [
        ("fomc_dates.parquet",    "FOMC day"),
        ("econ_calendar.parquet", "Economic release"),
        ("mag7_earnings.parquet", "MAG7 earnings"),
    ]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
            df["date"] = pd.to_datetime(df["date"])
            hits = df[df["date"] == target_dt]
            if not hits.empty:
                if "event" in df.columns:
                    for _, r in hits.iterrows():
                        evt = r.get("event", "event")
                        blockers.append(f"{label}: {evt}")
                else:
                    blockers.append(label)
        except Exception:
            pass

    # FOMC Eve check
    fomc_path = DATA_DIR / "fomc_dates.parquet"
    if fomc_path.exists():
        try:
            fomc = pd.read_parquet(fomc_path)
            fomc["date"] = pd.to_datetime(fomc["date"])
            fomc_dates = set(fomc["date"])
            tomorrow = target_dt + pd.Timedelta(days=1)
            if tomorrow in fomc_dates:
                blockers.append("FOMC Eve")
        except Exception:
            pass

    # VIX hard blocker
    if vix_level is not None and vix_level > 35:
        blockers.append(f"VIX={vix_level:.1f} > 35 (hard blocker)")

    return blockers


def check_soft_blockers(target_date: str, vix_level: float | None = None) -> list[str]:
    """Return list of soft blocker reasons."""
    target_dt = pd.Timestamp(target_date)
    soft      = []

    if vix_level is not None and 25 <= vix_level <= 35:
        soft.append(f"VIX={vix_level:.1f} in 25-35 (elevated vol)")

    # Month-end / quarter-end
    dt = target_dt.to_pydatetime()
    if dt.day >= 28:
        soft.append("Month-end (rebalancing flows)")
    if dt.month in (3, 6, 9, 12) and dt.day >= 28:
        soft.append("Quarter-end")

    # FOMC week (not day — already a hard blocker)
    fomc_path = DATA_DIR / "fomc_dates.parquet"
    if fomc_path.exists():
        try:
            fomc = pd.read_parquet(fomc_path)
            fomc["date"] = pd.to_datetime(fomc["date"])
            week_start = target_dt - pd.Timedelta(days=target_dt.dayofweek)
            week_end   = week_start + pd.Timedelta(days=6)
            in_fomc_week = ((fomc["date"] >= week_start) &
                            (fomc["date"] <= week_end)).any()
            if in_fomc_week:
                soft.append("FOMC week")
        except Exception:
            pass

    return soft


# ── Signal determination ───────────────────────────────────────────────────────

def determine_signal(scores: dict, soft_blockers: list,
                     spx_price: float | None, vix_level: float | None) -> tuple[str, list, str]:
    """
    Apply voting logic to determine final signal.

    Returns (signal, models_voted_trade, suggested_structure).
    signal: TRADE | SIZE_DOWN | SKIP | NO_MODEL
    """
    valid_scores = {h: s for h, s in scores.items() if s is not None}

    if not valid_scores:
        return "NO_MODEL", [], "SKIP"

    has_soft_blockers = len(soft_blockers) > 0
    threshold = SOFT_THRESHOLD if has_soft_blockers else TRADE_THRESHOLD

    models_trade = [h for h, s in valid_scores.items() if s >= threshold]
    n_trade      = len(models_trade)
    n_valid      = len(valid_scores)

    if n_trade >= MIN_MODELS_AGREE:
        # Enough models agree — determine structure
        avg_score = sum(valid_scores.values()) / n_valid
        structure = suggest_structure(spx_price, vix_level, valid_scores)

        if has_soft_blockers:
            return "SIZE_DOWN", models_trade, structure
        return "TRADE", models_trade, structure

    return "SKIP", models_trade, "SKIP"


def suggest_structure(spx_price: float | None, vix_level: float | None,
                      scores: dict) -> str:
    """Suggest IC structure based on current conditions."""
    if vix_level is None:
        return "IC"

    if vix_level > 30:
        return "SINGLE_PUT"  # avoid call side in high vol

    # Check directional bias from model score variance
    valid = [s for s in scores.values() if s is not None]
    if not valid:
        return "IC"

    avg = sum(valid) / len(valid)
    if avg > 0.70:
        return "IC"          # high confidence, full IC
    elif avg > 0.60:
        return "IC"          # normal confidence, full IC
    else:
        return "SINGLE_PUT"  # lower confidence, one-sided


# ── Log I/O ────────────────────────────────────────────────────────────────────

def init_log():
    """Create log file with header if it doesn't exist."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        print(f"  [log] created {LOG_FILE}")


def read_log() -> pd.DataFrame:
    """Read existing log into DataFrame."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(LOG_FILE, dtype=str)
        # Fill missing columns
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LOG_COLUMNS]
    except Exception as e:
        print(f"  [warn] could not read log: {e}")
        return pd.DataFrame(columns=LOG_COLUMNS)


def find_entry_row(df: pd.DataFrame, target_date: str) -> int | None:
    """Find row index for a given date's ENTRY record."""
    if df.empty:
        return None
    matches = df[df["date"] == target_date].index.tolist()
    return matches[0] if matches else None


def append_entry_row(row: dict):
    """Append a new entry row to the log CSV."""
    init_log()
    df = read_log()

    # Remove existing row for this date if it exists (re-score)
    if not df.empty and (df["date"] == row["date"]).any():
        df = df[df["date"] != row["date"]]
        print(f"  [log] overwriting existing entry for {row['date']}")

    new_row = {col: row.get(col, "") for col in LOG_COLUMNS}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    print(f"  [log] entry saved for {row['date']}")


def update_close_row(target_date: str, close_spx: float):
    """Update an existing entry row with 3pm close data."""
    df = read_log()
    if df.empty:
        print(f"  [error] no log entries found — run score first")
        return

    idx = find_entry_row(df, target_date)
    if idx is None:
        print(f"  [error] no entry for {target_date} — run score first")
        return

    row = df.loc[idx].copy()

    # Parse entry data
    short_put  = float(row["short_put"])  if row["short_put"]  else None
    long_put   = float(row["long_put"])   if row["long_put"]   else None
    short_call = float(row["short_call"]) if row["short_call"] else None
    long_call  = float(row["long_call"])  if row["long_call"]  else None
    spread_w   = float(row["spread_width"]) if row["spread_width"] else 100.0
    entry_cr   = float(row["entry_credit_est"]) if row["entry_credit_est"] else None
    signal     = row["signal"]

    now_et = datetime.now(ET_TZ).strftime("%H:%M ET")

    df.at[idx, "close_spx_price"] = round(close_spx, 2)
    df.at[idx, "close_time_et"]   = now_et

    # Calculate P&L only if we had an active trade
    if signal in ("TRADE", "SIZE_DOWN") and all(
            v is not None for v in [short_put, short_call, long_put, long_call, entry_cr]):

        inside = short_put <= close_spx <= short_call
        df.at[idx, "close_inside_strikes"] = inside

        if inside:
            # Full credit collected (max profit)
            pnl = entry_cr
            pnl_pct = 100.0
        else:
            # Check breach side and estimate loss
            if close_spx < short_put:
                # Put side breached
                breach = short_put - close_spx
                if close_spx <= long_put:
                    # Max loss (full spread minus credit)
                    pnl = -(spread_w - entry_cr)
                else:
                    # Partial loss (proportional)
                    pnl = -(breach - entry_cr) if breach > entry_cr else entry_cr - breach
            else:
                # Call side breached
                breach = close_spx - short_call
                if close_spx >= long_call:
                    pnl = -(spread_w - entry_cr)
                else:
                    pnl = -(breach - entry_cr) if breach > entry_cr else entry_cr - breach

            pnl_pct = (pnl / entry_cr * 100) if entry_cr else 0

        df.at[idx, "pnl_est"]           = round(pnl, 2)
        df.at[idx, "pnl_pct_of_credit"] = round(pnl_pct, 1)
        df.at[idx, "notes"] = (
            f"SPX closed {'INSIDE' if inside else 'OUTSIDE'} short strikes "
            f"[{short_put}P / {short_call}C]"
        )
    else:
        df.at[idx, "close_inside_strikes"] = "NA"
        df.at[idx, "notes"] = "SKIP day — no trade simulation"

    df.at[idx, "logged_at"] = datetime.now(ET_TZ).isoformat()
    df.to_csv(LOG_FILE, index=False)
    print(f"  [log] close recorded for {target_date}")


# ── Main scoring action ────────────────────────────────────────────────────────

def run_score(target_date: str, dry_run: bool = False):
    """Full 10am scoring pass — the core shadow trade log entry."""
    print(f"\n{'='*65}")
    print(f"  SHADOW TRADE LOG — Score: {target_date}")
    print(f"{'='*65}")

    # ── Fetch live data ──
    print(f"\n[1/5] Fetching SPX + VIX...")
    spx_price, vix_level = fetch_live_spx_vix(target_date)
    if spx_price:
        print(f"  SPX: {spx_price:.2f}  |  VIX: {vix_level:.2f}")
    else:
        print(f"  [warn] could not fetch live data — using parquet fallback")
        # Try to load from stored features
        spx_f = DATA_DIR / "spy_features.parquet"
        vix_f = DATA_DIR / "vix_daily.parquet"
        if spx_f.exists():
            df = pd.read_parquet(spx_f)
            df["date"] = pd.to_datetime(df["date"])
            row = df[df["date"] <= pd.Timestamp(target_date)].tail(1)
            if not row.empty:
                spx_price = float(row.get("d_close", row.get("spx_close", pd.Series([None]))).iloc[0] or 0) or None
        if vix_f.exists() and vix_level is None:
            df = pd.read_parquet(vix_f)
            df["date"] = pd.to_datetime(df["date"])
            row = df[df["date"] <= pd.Timestamp(target_date)].tail(1)
            if not row.empty and "vix_close" in df.columns:
                vix_level = float(row["vix_close"].iloc[0])

    # ── Hard blockers ──
    print(f"\n[2/5] Checking hard blockers...")
    hard_blockers = check_hard_blockers(target_date, vix_level)
    if hard_blockers:
        print(f"  HARD BLOCKERS FIRED:")
        for b in hard_blockers:
            print(f"    • {b}")
    else:
        print(f"  ✓ No hard blockers")

    # ── Soft blockers ──
    soft_blockers = check_soft_blockers(target_date, vix_level)
    if soft_blockers:
        print(f"  SOFT BLOCKERS:")
        for b in soft_blockers:
            print(f"    ~ {b}")

    # ── If hard blocker: record BLOCKED and exit ──
    if hard_blockers:
        row = {
            "date":              target_date,
            "signal":            "BLOCKED",
            "num_models_trade":  0,
            "models_voted_trade": "",
            "model_scores":      "{}",
            "hard_blockers":     " | ".join(hard_blockers),
            "soft_blockers":     " | ".join(soft_blockers),
            "vix_level":         round(vix_level, 2) if vix_level else "",
            "spx_entry_price":   round(spx_price, 2) if spx_price else "",
            "suggested_structure": "SKIP",
            "short_put": "", "long_put": "", "short_call": "", "long_call": "",
            "spread_width": "", "entry_credit_est": "",
            "entry_time_et":     datetime.now(ET_TZ).strftime("%H:%M ET"),
            "close_spx_price": "", "close_time_et": "",
            "close_inside_strikes": "NA",
            "pnl_est": 0.0, "pnl_pct_of_credit": 0.0,
            "notes":   f"Hard blocked: {'; '.join(hard_blockers)}",
            "logged_at": datetime.now(ET_TZ).isoformat(),
        }
        if not dry_run:
            append_entry_row(row)
        print(f"\n  SIGNAL: BLOCKED — DO NOT TRADE")
        return row

    # ── Load models ──
    print(f"\n[3/5] Loading models...")
    models = load_models()
    if not models:
        print(f"  [warn] no models loaded — signal will be NO_MODEL")

    # ── Score all models ──
    print(f"\n[4/5] Scoring {target_date} across {len(models)} model(s)...")
    scores = score_all_models(target_date, models)
    for horizon, score in scores.items():
        tag = "TRADE" if (score or 0) >= TRADE_THRESHOLD else "SKIP"
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {horizon:5s}: {score_str}  → {tag}")

    # ── Voting + signal ──
    signal, models_voted, structure = determine_signal(
        scores, soft_blockers, spx_price, vix_level
    )

    print(f"\n[5/5] Determining signal...")
    print(f"  Models voted TRADE: {len(models_voted)}/{len(scores)} ({', '.join(models_voted) or 'none'})")
    print(f"  Soft blockers: {len(soft_blockers)}")
    print(f"  => SIGNAL: {signal}")

    # ── Strikes + credit (if trading) ──
    strikes = {}
    entry_credit = ""
    if signal in ("TRADE", "SIZE_DOWN") and spx_price and vix_level:
        strikes = estimate_15delta_strikes(spx_price, vix_level)
        entry_credit = estimate_ic_credit(
            strikes["spread_width"], vix_level, spx_price,
            strikes["short_put"], strikes["short_call"]
        )
        print(f"\n  Theoretical IC:")
        print(f"    Long  put:  {strikes['long_put']}")
        print(f"    Short put:  {strikes['short_put']}")
        print(f"    ATM:        {int(round(spx_price/5)*5)}")
        print(f"    Short call: {strikes['short_call']}")
        print(f"    Long  call: {strikes['long_call']}")
        print(f"    Width:      {strikes['spread_width']} pts")
        print(f"    Est credit: ~${entry_credit:.2f}/pt")

    # ── Build log row ──
    row = {
        "date":               target_date,
        "signal":             signal,
        "num_models_trade":   len(models_voted),
        "models_voted_trade": ", ".join(models_voted),
        "model_scores":       json.dumps(scores),
        "hard_blockers":      " | ".join(hard_blockers),
        "soft_blockers":      " | ".join(soft_blockers),
        "vix_level":          round(vix_level, 2)   if vix_level   else "",
        "spx_entry_price":    round(spx_price, 2)   if spx_price   else "",
        "suggested_structure": structure,
        "short_put":          strikes.get("short_put",  ""),
        "long_put":           strikes.get("long_put",   ""),
        "short_call":         strikes.get("short_call", ""),
        "long_call":          strikes.get("long_call",  ""),
        "spread_width":       strikes.get("spread_width", ""),
        "entry_credit_est":   entry_credit,
        "entry_time_et":      datetime.now(ET_TZ).strftime("%H:%M ET"),
        "close_spx_price":    "",
        "close_time_et":      "",
        "close_inside_strikes": "",
        "pnl_est":            "",
        "pnl_pct_of_credit":  "",
        "notes":              "",
        "logged_at":          datetime.now(ET_TZ).isoformat(),
    }

    if not dry_run:
        append_entry_row(row)

    print(f"\n{'='*65}")
    print(f"  LOG ENTRY SAVED → {LOG_FILE}")
    print(f"  Run with --close at 3pm ET to record P&L")
    print(f"{'='*65}\n")

    return row


def run_close(target_date: str):
    """3pm close pass — record SPX price and calculate P&L."""
    print(f"\n{'='*65}")
    print(f"  SHADOW TRADE LOG — Close: {target_date}")
    print(f"{'='*65}")

    print(f"\nFetching SPX close price...")
    close_spx = fetch_spx_close_at_3pm(target_date)

    if close_spx is None:
        print(f"  [error] could not fetch SPX close — please provide manually:")
        print(f"  Edit {LOG_FILE} directly or re-run with --close after market close")
        return

    print(f"  SPX close: {close_spx:.2f}")
    update_close_row(target_date, close_spx)

    print(f"\n  Close recorded. Run --review to see P&L summary.")


def run_review(n: int = 20):
    """Print recent shadow trade log entries."""
    df = read_log()
    if df.empty:
        print("No shadow trade log entries yet.")
        return

    df_sorted = df.sort_values("date", ascending=False).head(n)
    print(f"\n{'='*80}")
    print(f"  SHADOW TRADE LOG — Last {min(n, len(df))} entries")
    print(f"{'='*80}")

    view_cols = ["date", "signal", "num_models_trade", "vix_level",
                 "spx_entry_price", "short_put", "short_call",
                 "entry_credit_est", "close_spx_price",
                 "close_inside_strikes", "pnl_est", "notes"]
    view_cols = [c for c in view_cols if c in df_sorted.columns]

    print(df_sorted[view_cols].to_string(index=False))

    # Summary stats
    trade_rows = df[df["signal"].isin(["TRADE", "SIZE_DOWN"])]
    if not trade_rows.empty and "pnl_est" in trade_rows.columns:
        closed = trade_rows[trade_rows["pnl_est"] != ""]
        if not closed.empty:
            pnls = pd.to_numeric(closed["pnl_est"], errors="coerce").dropna()
            print(f"\n  Closed trades: {len(pnls)}")
            print(f"  Win rate:  {(pnls > 0).mean():.0%}")
            print(f"  Total P&L: ${pnls.sum():.2f}/pt")
            print(f"  Avg P&L:   ${pnls.mean():.2f}/pt per trade")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shadow trading log — daily paper trade validator"
    )
    parser.add_argument("--date",   default=None,
                        help="Date to score/close (YYYY-MM-DD, default: today)")
    parser.add_argument("--close",  action="store_true",
                        help="Record 3pm close for a given date")
    parser.add_argument("--review", action="store_true",
                        help="Print recent log entries")
    parser.add_argument("--n",      type=int, default=20,
                        help="Number of rows for --review (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score but do not write to log")
    args = parser.parse_args()

    target_date = args.date or str(date.today())

    if args.review:
        run_review(args.n)
    elif args.close:
        run_close(target_date)
    else:
        run_score(target_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
