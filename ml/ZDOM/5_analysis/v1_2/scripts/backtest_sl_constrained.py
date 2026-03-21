"""
ZDOM V1.1 Stop-Loss Constrained Backtest.

Three strategies, each with a portfolio-level max stop-loss budget:

  S1 — Constrained Best EV (single entry):
       At each entry point find the (strategy, TP, qty) that maximises
       qty × EV subject to:
         qty × max_loss_per_ct  ≤  sl_budget_remaining
         qty × BP_per_ct        ≤  bp_available
       One real entry is placed and held.  On TP: re-enter with updated
       state.  5-delta: always shadow-placed (locks BP, no SL budget).

  S2 — Constrained Diversified (greedy fill):
       Sort passing combos by per-contract EV.  Walk down the list adding
       1 contract at a time until either SL budget or BP is exhausted.
       5-delta: shadow-placed, consumes BP, does NOT consume SL budget.
       On TP: re-fill remaining capacity.  On SL: actual loss is permanently
       deducted from daily SL budget.

  S3 — Optimised Portfolio (integer knapsack):
       At each entry point solve an integer LP that maximises Σ(qty_i × ev_i)
       subject to SL and BP constraints.  Multiple contracts of the same
       combo are allowed.  Re-optimise on every TP exit.
       5-delta: shadow-placed (excluded from SL constraint, included in BP).

SL budget mechanics:
  - sl_budget starts at portfolio × max_sl_pct each day.
  - When a position OPENS: the budget for new entries is reduced by
    qty × max_loss_per_ct  (i.e. we "reserve" the worst-case loss).
  - When a position closes for TP: the reservation is returned and actual
    profit is added  →  available_budget increases.
  - When a position closes for SL: the reservation is consumed (actual loss
    replaces the reservation).  No net change vs the reservation already in
    place.

Usage:
  python3 scripts/backtest_sl_constrained.py --max-sl-pct 0.20
  python3 scripts/backtest_sl_constrained.py --max-sl-pct 0.10 0.15 0.20 0.25 0.30
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent   # ml/zdom_v1/
V1_2_DIR    = PROJECT_DIR.parent / "zdom_v1_2"
DATA_DIR    = PROJECT_DIR / "3_data_join"
MODELS_DIR  = V1_2_DIR / "4_models" / "v1_2"
MASTER_DATA = V1_2_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final_scored.parquet"
OUTPUT_BASE = PROJECT_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

STRATEGIES_ALL  = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]   # 9 deltas
STRATEGIES_NO5D = [f"IC_{d:02d}d_25w" for d in range(10, 50, 5)]  # no 5-delta
SHADOW_STRATEGY = "IC_05d_25w"

def discover_tp_levels():
    """Discover the latest available v1_2 TP model targets from disk."""
    levels = set()
    for pkl_path in MODELS_DIR.glob("tp*_target_*_v1_2.pkl"):
        tp = pkl_path.name.split("_target_")[0]
        if tp.startswith("tp"):
            levels.add(tp)
    if not levels:
        # Conservative fallback if models have not been pulled yet.
        levels = {f"tp{p}" for p in range(10, 30, 5)}
    return sorted(levels, key=lambda tp: int(tp[2:]))


TP_LEVELS = discover_tp_levels()
TARGETS   = [f"{tp}_target" for tp in TP_LEVELS]

TRAIN_PCT = 0.70
GAP_DAYS  = 7
BUYING_POWER_PER_CONTRACT = 2500   # per iron-condor contract
FEES_PER_SHARE = 0.052             # $5.20 RT / 100 shares
START_PORTFOLIO = 10_000

# Fixed run parameters
SKIP_RATES  = [round(0.05 + i * 0.01, 2) for i in range(31)]   # 0.05 → 0.35
SLIP_PER_SIDE = 0.00
EFFECTIVE_SLIP = 0.00   # perfect mid fill — no slippage

# Meta columns (not features)
_TP_META = []
for _pct in range(10, 55, 5):
    _k = f"tp{_pct}"
    _TP_META += [f"{_k}_target", f"{_k}_exit_reason", f"{_k}_exit_time",
                 f"{_k}_exit_debit", f"{_k}_pnl"]

META_COLS = [
    "datetime", "decision_datetime", "date", "strategy",
    "spx_at_entry",
    "short_call", "short_put", "long_call", "long_put",
    "call_wing_width", "put_wing_width",
    "sc_delta", "sp_delta", "sc_iv", "sp_iv",
    "time_to_close_min",
    "blocked",
] + _TP_META


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    strategy:    str
    tp_level:    str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    exit_reason: str
    credit:      float
    exit_debit:  float
    qty:         int
    max_loss:    float   # qty × credit × 100  (total worst-case loss in $)
    shadow:      bool = False


@dataclass
class TradeRecord:
    day:          str
    strategy:     str
    tp_level:     str
    entry_time:   str
    exit_time:    str
    exit_reason:  str
    credit:       float
    exit_debit:   float
    fees:         float
    pnl_per_share: float
    qty:          int
    pnl_total:    float
    portfolio_after: float
    shadow:       bool = False
    sl_budget_after: float = 0.0


# ── Data loading / splitting / scoring (reuse from existing backtest) ─────────

def load_model_table():
    if MASTER_DATA.exists():
        print(f"Loading v1_2 master data: {MASTER_DATA.name} …")
        df = pd.read_parquet(MASTER_DATA)
        df["date"]     = pd.to_datetime(df["date"])
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "decision_datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["decision_datetime"])
        df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
        print(f"  {len(df):,} rows × {df.shape[1]} cols, {df['date'].nunique()} days")
        return df
    print("[error] model_table_v1.parquet not found"); sys.exit(1)


def apply_blockers(df):
    blocked = pd.Series(False, index=df.index)
    def _safe_col(name, default=0):
        return df[name] if name in df.columns else default
    blocked |= (_safe_col("days_to_next_fomc") == 0)
    blocked |= (_safe_col("days_to_next_fomc") == 1)
    for col in ["is_cpi_day", "is_ppi_day", "is_nfp_day", "is_gdp_day", "is_mag7_earnings_day"]:
        blocked |= (_safe_col(col) == 1)
    vix = df.get("vix_close", df.get("vix_intraday_prior_min_close",
                 df.get("daily_prior_vix_close", pd.Series(0, index=df.index))))
    blocked |= (vix.fillna(0) > 35)
    if "gap_pct" in df.columns:
        blocked |= (df["gap_pct"].abs() > 1.5)
    blocked = blocked.fillna(False)
    df["blocked"] = blocked
    print(f"  Hard blockers: {blocked.sum():,} rows blocked "
          f"({df[blocked]['date'].nunique()} days)")
    return df


def split_data(df):
    holdout_start = pd.Timestamp("2025-06-12")
    holdout_end   = pd.Timestamp("2026-02-27")
    train_end_dt  = pd.Timestamp("2024-09-04")

    # Prefer the split metadata embedded in Matt's latest model artifacts.
    for pkl_path in sorted(MODELS_DIR.glob("tp*_target_*_v1_2.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                art = pickle.load(f)
            split_info = art.get("split_info") or {}
            train_range = split_info.get("train_range")
            holdout_range = split_info.get("holdout_range")
            if train_range and holdout_range:
                train_end_dt = pd.Timestamp(train_range.split(" -> ")[1])
                holdout_start = pd.Timestamp(holdout_range.split(" -> ")[0])
                holdout_end = pd.Timestamp(holdout_range.split(" -> ")[1])
                print(f"  Split source: {pkl_path.name}")
                break
        except Exception as exc:
            print(f"  [warn] failed to read split_info from {pkl_path.name}: {exc}")

    train_df   = df[df["date"] <= train_end_dt].reset_index(drop=True)

    # Test set: gap between train end and holdout start
    test_start = train_end_dt + pd.Timedelta(days=1)
    test_end   = holdout_start - pd.Timedelta(days=1)
    test_df    = df[(df["date"] >= test_start) & (df["date"] <= test_end)].reset_index(drop=True)

    holdout_df = df[(df["date"] >= holdout_start) & (df["date"] <= holdout_end)].reset_index(drop=True)

    train_days   = train_df["date"].nunique()
    test_days    = test_df["date"].nunique()
    holdout_days = holdout_df["date"].nunique()
    print(f"  Train:   {len(train_df):>10,} rows ({train_days:>3d} days)")
    print(f"  Test:    {len(test_df):>10,} rows ({test_days:>3d} days)  "
          f"{test_df['date'].min().date()} → {test_df['date'].max().date()}")
    print(f"  Holdout: {len(holdout_df):>10,} rows ({holdout_days:>3d} days)  "
          f"{holdout_df['date'].min().date()} → {holdout_df['date'].max().date()}")
    return train_df, test_df, holdout_df


def get_feature_cols(df):
    all_feat = [c for c in df.columns if c not in META_COLS]
    return [c for c in all_feat if pd.api.types.is_numeric_dtype(df[c])]


def load_models():
    models = {}
    for tp in TP_LEVELS:
        target = f"{tp}_target"
        best_model, best_auc, best_algo = None, -1, None
        for algo in ["xgb", "lgbm"]:
            mf = MODELS_DIR / f"{target}_{algo}_v1_2.pkl"
            if not mf.exists():
                mf = MODELS_DIR / f"{target}_{algo}.pkl"
            if not mf.exists(): continue
            ho_auc = 0
            with open(mf, "rb") as f2:
                art_tmp = pickle.load(f2)
            ho_auc = art_tmp.get("holdout_auc", 0)
            if ho_auc > best_auc:
                best_model = art_tmp["model"]
                best_auc   = ho_auc
                best_algo  = algo
        if best_model is not None:
            models[tp] = (best_model, art_tmp.get("feature_cols", []))
            print(f"  {tp}: {best_algo.upper()} holdout AUC={best_auc:.4f}")
    return models


def fit_calibrators(train_df, models, feature_cols):
    """Fit isotonic regression calibrators on training data per TP level.

    Matt's v1_2 models output uncalibrated probabilities (~0.50) for targets
    with ~90% base rate, making raw EV deeply negative for all trades.
    Isotonic regression maps raw probs → calibrated probs that reflect the
    true win rate, restoring meaningful EV computation.
    """
    calibrators = {}
    for tp, (model, model_feat_cols) in models.items():
        target = f"{tp}_target"
        if target not in train_df.columns:
            continue
        fcols = model_feat_cols if model_feat_cols else feature_cols
        X = train_df[fcols]
        raw_probs = model.predict_proba(X)[:, 1]
        y = train_df[target].values

        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        ir.fit(raw_probs, y)
        cal_probs = ir.predict(raw_probs)
        print(f"  Calibrator {tp}: raw mean={raw_probs.mean():.3f} → "
              f"cal mean={cal_probs.mean():.3f} (base rate={y.mean():.3f})")
        calibrators[tp] = ir
    return calibrators


def score_holdout(holdout_df, models, feature_cols, calibrators=None):
    scored = holdout_df.copy()
    for tp, (model, model_feat_cols) in models.items():
        fcols = model_feat_cols if model_feat_cols else feature_cols
        X = scored[fcols]
        probs = model.predict_proba(X)[:, 1]
        if calibrators and tp in calibrators:
            probs = calibrators[tp].predict(probs)
        scored[f"prob_{tp}"] = probs
        print(f"  Scored {tp}: mean={probs.mean():.3f}")
    return scored


def precompute_ev_lookup(train_df):
    ev_lookup = {}
    for strat in STRATEGIES_ALL:
        strat_df = train_df[train_df["strategy"] == strat]
        if len(strat_df) == 0: continue
        for tp in TP_LEVELS:
            target  = f"{tp}_target"
            pnl_col = f"{tp}_pnl"
            if target not in strat_df.columns: continue
            wins   = strat_df[strat_df[target] == 1]
            losses = strat_df[strat_df[target] == 0]
            if len(wins) == 0 or len(losses) == 0: continue
            ev_lookup[(strat, tp)] = (
                wins[pnl_col].mean(),
                losses[pnl_col].mean(),
                len(wins) / len(strat_df),
            )
    print(f"  EV lookup: {len(ev_lookup)} combos")
    return ev_lookup


def compute_ev(ev_lookup, strategy, tp, prob):
    key = (strategy, tp)
    if key not in ev_lookup: return None
    avg_win, avg_loss, _ = ev_lookup[key]
    return prob * avg_win + (1 - prob) * avg_loss - FEES_PER_SHARE - EFFECTIVE_SLIP


def precompute_skip_cutoffs(sim_scored, skip_rates):
    cutoffs = {}
    for strat in STRATEGIES_ALL:
        sdf = sim_scored[sim_scored["strategy"] == strat]
        if len(sdf) == 0: continue
        for tp in TP_LEVELS:
            pcol = f"prob_{tp}"
            if pcol not in sdf.columns: continue
            probs = sdf[pcol].values
            for sr in skip_rates:
                cutoffs[(strat, tp, sr)] = np.percentile(probs, sr * 100)
    return cutoffs


# ── Candidate precomputation ───────────────────────────────────────────────────

def precompute_all_candidates(sim_scored, ev_lookup):
    """
    Precompute ALL candidates (regardless of skip-rate cutoff) once.
    Returns nested dict: date -> entry_time -> list[base_candidate_dicts]
    Base candidates have all fields + prob stored; EV precomputed.
    Filter by cutoff per simulation run using filter_candidates().
    """
    required_cols = [
        "date", "datetime", "strategy", "blocked", "credit",
    ]
    for tp in TP_LEVELS:
        required_cols.extend([
            f"prob_{tp}",
            f"{tp}_exit_time",
            f"{tp}_exit_debit",
            f"{tp}_exit_reason",
        ])
    slim = sim_scored[[c for c in required_cols if c in sim_scored.columns]].copy()

    pool = {}
    prob_cols = {tp: f"prob_{tp}" for tp in TP_LEVELS}
    for date, date_df in slim.groupby("date"):
        pool[date] = {}
        for entry_time, time_df in date_df.groupby("datetime"):
            candidates = []
            for row in time_df.itertuples(index=False):
                strat = row.strategy
                if getattr(row, "blocked", False): continue
                credit = row.credit
                if pd.isna(credit): continue
                credit_f = float(credit)
                is_shadow = (strat == SHADOW_STRATEGY)
                for tp in TP_LEVELS:
                    pcol = prob_cols[tp]
                    prob = getattr(row, pcol, None)
                    if prob is None or pd.isna(prob): continue
                    ev = compute_ev(ev_lookup, strat, tp, prob)
                    # Shadow (5D) candidates bypass EV filter — they lock BP
                    # but carry $0 PnL and $0 SL budget by design.
                    if not is_shadow and (ev is None or ev <= 0):
                        continue
                    if ev is None:
                        ev = 0.0
                    exit_ts     = getattr(row, f"{tp}_exit_time", None)
                    exit_debit  = getattr(row, f"{tp}_exit_debit", None)
                    exit_reason = getattr(row, f"{tp}_exit_reason", None)
                    if exit_ts is None or pd.isna(exit_ts) or pd.isna(exit_debit): continue
                    # exit_ts is already a Timestamp (pre-converted in main)
                    candidates.append({
                        "strategy":        strat,
                        "tp_level":        tp,
                        "prob":            float(prob),
                        "ev":              ev,
                        "credit":          credit_f,
                        "exit_debit":      float(exit_debit),
                        "exit_time":       exit_ts,
                        "exit_reason":     exit_reason,
                        "is_shadow":       is_shadow,
                        "max_loss_per_ct": credit_f * 100,
                    })
            # Pre-sort by EV desc (order within same EV may change after cutoff filter,
            # but for speed we sort once and rely on filter preserving relative order)
            candidates.sort(key=lambda x: x["ev"], reverse=True)
            pool[date][entry_time] = candidates
    return pool


def _get_all_candidates(day_pool, entry_time, cutoffs, skip_rate):
    """
    Filter precomputed candidate pool for a given entry_time and skip_rate.
    Returns list of candidate dicts sorted EV desc.
    """
    base = day_pool.get(entry_time, [])
    if not base:
        return []
    filtered = [c for c in base
                if c["prob"] >= cutoffs.get((c["strategy"], c["tp_level"], skip_rate), 1.0)]
    return filtered   # already sorted EV desc from precomputation


# ── Position close helper ─────────────────────────────────────────────────────

def close_position(pos: OpenPosition, portfolio: float, sl_reserved: float):
    """
    Close a position.  Returns (new_portfolio, new_sl_reserved, pnl_total,
                                 pnl_per_share, fees).
    sl_reserved = total SL reservation currently held for open positions.
    """
    fees          = FEES_PER_SHARE * pos.qty
    pnl_per_share = pos.credit - pos.exit_debit - EFFECTIVE_SLIP
    pnl_total     = pnl_per_share * pos.qty * 100 - fees

    new_portfolio   = portfolio + pnl_total
    # Release the reservation (max_loss was pre-reserved on open)
    new_sl_reserved = sl_reserved - pos.max_loss

    return new_portfolio, new_sl_reserved, pnl_total, pnl_per_share, fees


# ── Strategy 1: Constrained Best EV (single entry) ───────────────────────────

def simulate_s1_day(day_pool, cutoffs, skip_rate,
                    portfolio, sl_budget_start):
    """
    Run one day under S1.
    Returns (trades, final_portfolio, final_sl_budget_start).
    sl_budget_start: daily SL budget at the start of the day, adjusted
                     as positions close.
    """
    trades      = []
    open_pos    = None        # at most 1 real + 1 shadow at a time
    shadow_pos  = None
    sl_reserved = 0.0         # currently reserved for open real position
    sl_budget   = sl_budget_start

    entry_times = sorted(day_pool.keys())

    for et in entry_times:
        # ── Close positions that have exited at or before this entry_time ──
        for pos in [open_pos, shadow_pos]:
            if pos is None: continue
            if pos.exit_time > et: continue

            if pos.shadow:
                # Shadow: no PnL, no SL impact, just free BP
                t = TradeRecord(
                    day=str(et.date()), strategy=pos.strategy,
                    tp_level=pos.tp_level, entry_time=str(pos.entry_time),
                    exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
                    credit=pos.credit, exit_debit=pos.exit_debit,
                    fees=0, pnl_per_share=0, qty=pos.qty,
                    pnl_total=0, portfolio_after=portfolio,
                    shadow=True, sl_budget_after=sl_budget - sl_reserved,
                )
                trades.append(t)
                shadow_pos = None
            else:
                portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
                    close_position(pos, portfolio, sl_reserved)
                # Adjust daily SL budget: losses reduce it, profits restore it
                sl_budget += pnl_total
                t = TradeRecord(
                    day=str(et.date()), strategy=pos.strategy,
                    tp_level=pos.tp_level, entry_time=str(pos.entry_time),
                    exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
                    credit=pos.credit, exit_debit=pos.exit_debit,
                    fees=fees / pos.qty, pnl_per_share=pnl_per_share,
                    qty=pos.qty, pnl_total=pnl_total,
                    portfolio_after=portfolio,
                    shadow=False,
                    sl_budget_after=sl_budget - sl_reserved,
                )
                trades.append(t)
                open_pos = None

        # ── Entry logic ──
        # Already have both a shadow and a real position open? Skip.
        if open_pos is not None and shadow_pos is not None:
            continue

        candidates = _get_all_candidates(day_pool, et, cutoffs, skip_rate)
        if not candidates:
            continue

        # Available resources
        bp_used_shadow = (shadow_pos.qty * BUYING_POWER_PER_CONTRACT
                          if shadow_pos else 0)
        bp_used_real   = (open_pos.qty * BUYING_POWER_PER_CONTRACT
                          if open_pos else 0)
        bp_avail       = portfolio - bp_used_shadow - bp_used_real

        # SL available for NEW reservations
        sl_avail = (sl_budget - sl_reserved)

        # ── Handle shadow (5D) if not already open ──
        if shadow_pos is None:
            five_d_cands = [c for c in candidates if c["is_shadow"]]
            if five_d_cands:
                c5 = five_d_cands[0]
                shadow_qty = max(0, int(bp_avail // BUYING_POWER_PER_CONTRACT))
                if shadow_qty > 0:
                    shadow_pos = OpenPosition(
                        strategy=c5["strategy"], tp_level=c5["tp_level"],
                        entry_time=et, exit_time=c5["exit_time"],
                        exit_reason=c5["exit_reason"],
                        credit=c5["credit"], exit_debit=c5["exit_debit"],
                        qty=shadow_qty, max_loss=0.0, shadow=True,
                    )
                    bp_avail -= shadow_qty * BUYING_POWER_PER_CONTRACT

        # ── Handle real position if not already open ──
        if open_pos is None and sl_avail > 0 and bp_avail > 0:
            real_cands = [c for c in candidates if not c["is_shadow"]]
            best_ev, best_c, best_qty = -1e9, None, 0

            for c in real_cands:
                ml  = c["max_loss_per_ct"]
                sl_qty = max(0, int(sl_avail // ml)) if ml > 0 else 0
                bp_qty = max(0, int(bp_avail // BUYING_POWER_PER_CONTRACT))
                qty    = min(sl_qty, bp_qty)
                if qty == 0: continue
                total_ev = qty * c["ev"]
                if total_ev > best_ev:
                    best_ev, best_c, best_qty = total_ev, c, qty

            if best_c is not None and best_qty > 0:
                open_pos = OpenPosition(
                    strategy=best_c["strategy"], tp_level=best_c["tp_level"],
                    entry_time=et, exit_time=best_c["exit_time"],
                    exit_reason=best_c["exit_reason"],
                    credit=best_c["credit"], exit_debit=best_c["exit_debit"],
                    qty=best_qty,
                    max_loss=best_qty * best_c["max_loss_per_ct"],
                    shadow=False,
                )
                sl_reserved += open_pos.max_loss

    # ── End of day: close any still-open positions ──
    for pos in [open_pos, shadow_pos]:
        if pos is None: continue
        if pos.shadow:
            shadow_pos = None
            continue
        portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
            close_position(pos, portfolio, sl_reserved)
        sl_budget += pnl_total
        t = TradeRecord(
            day=str(pos.entry_time.date()), strategy=pos.strategy,
            tp_level=pos.tp_level, entry_time=str(pos.entry_time),
            exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
            credit=pos.credit, exit_debit=pos.exit_debit,
            fees=fees / pos.qty, pnl_per_share=pnl_per_share,
            qty=pos.qty, pnl_total=pnl_total,
            portfolio_after=portfolio,
            shadow=False, sl_budget_after=max(sl_budget - sl_reserved, 0),
        )
        trades.append(t)

    return trades, portfolio, sl_budget


# ── Strategy 2: Constrained Diversified (greedy fill) ────────────────────────

def simulate_s2_day(day_pool, cutoffs, skip_rate,
                    portfolio, sl_budget_start):
    trades      = []
    open_pos    = []   # list of OpenPosition (real)
    shadow_pos  = None
    sl_reserved = 0.0
    sl_budget   = sl_budget_start

    entry_times = sorted(day_pool.keys())

    def _close_expired(et):
        nonlocal portfolio, sl_reserved, sl_budget, open_pos, shadow_pos
        still_open = []
        for pos in open_pos:
            if pos.exit_time > et:
                still_open.append(pos); continue
            portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
                close_position(pos, portfolio, sl_reserved)
            sl_budget += pnl_total
            t = TradeRecord(
                day=str(et.date()), strategy=pos.strategy,
                tp_level=pos.tp_level, entry_time=str(pos.entry_time),
                exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
                credit=pos.credit, exit_debit=pos.exit_debit,
                fees=fees / pos.qty, pnl_per_share=pnl_per_share,
                qty=pos.qty, pnl_total=pnl_total,
                portfolio_after=portfolio, shadow=False,
                sl_budget_after=max(sl_budget - sl_reserved, 0),
            )
            trades.append(t)
        open_pos = still_open
        if shadow_pos and shadow_pos.exit_time <= et:
            trades.append(TradeRecord(
                day=str(et.date()), strategy=shadow_pos.strategy,
                tp_level=shadow_pos.tp_level,
                entry_time=str(shadow_pos.entry_time),
                exit_time=str(shadow_pos.exit_time),
                exit_reason=shadow_pos.exit_reason,
                credit=shadow_pos.credit, exit_debit=shadow_pos.exit_debit,
                fees=0, pnl_per_share=0, qty=shadow_pos.qty,
                pnl_total=0, portfolio_after=portfolio, shadow=True,
                sl_budget_after=max(sl_budget - sl_reserved, 0),
            ))
            shadow_pos = None

    for et in entry_times:
        _close_expired(et)

        candidates = _get_all_candidates(day_pool, et, cutoffs, skip_rate)
        if not candidates: continue

        # Already-open strategies (no double-entering)
        open_strat_tp = {(p.strategy, p.tp_level) for p in open_pos}

        bp_used = sum(p.qty * BUYING_POWER_PER_CONTRACT for p in open_pos)
        if shadow_pos:
            bp_used += shadow_pos.qty * BUYING_POWER_PER_CONTRACT
        bp_avail = portfolio - bp_used
        sl_avail = sl_budget - sl_reserved

        # Shadow fill (once per entry time, if no shadow open)
        if shadow_pos is None:
            five_d = [c for c in candidates if c["is_shadow"]]
            if five_d:
                c5  = five_d[0]
                qty = max(0, int(bp_avail // BUYING_POWER_PER_CONTRACT))
                if qty > 0:
                    shadow_pos = OpenPosition(
                        strategy=c5["strategy"], tp_level=c5["tp_level"],
                        entry_time=et, exit_time=c5["exit_time"],
                        exit_reason=c5["exit_reason"],
                        credit=c5["credit"], exit_debit=c5["exit_debit"],
                        qty=qty, max_loss=0.0, shadow=True,
                    )
                    bp_avail -= qty * BUYING_POWER_PER_CONTRACT

        # Greedy fill: 1 contract at a time, sorted by EV desc
        for c in candidates:
            if c["is_shadow"]: continue
            if (c["strategy"], c["tp_level"]) in open_strat_tp: continue
            ml = c["max_loss_per_ct"]
            if ml <= 0: continue
            if sl_avail < ml: continue
            if bp_avail < BUYING_POWER_PER_CONTRACT: continue
            # Place 1 contract
            pos = OpenPosition(
                strategy=c["strategy"], tp_level=c["tp_level"],
                entry_time=et, exit_time=c["exit_time"],
                exit_reason=c["exit_reason"],
                credit=c["credit"], exit_debit=c["exit_debit"],
                qty=1, max_loss=ml, shadow=False,
            )
            open_pos.append(pos)
            open_strat_tp.add((c["strategy"], c["tp_level"]))
            sl_reserved += ml
            sl_avail    -= ml
            bp_avail    -= BUYING_POWER_PER_CONTRACT

    # End-of-day close
    for pos in open_pos:
        portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
            close_position(pos, portfolio, sl_reserved)
        sl_budget += pnl_total
        trades.append(TradeRecord(
            day=str(pos.entry_time.date()), strategy=pos.strategy,
            tp_level=pos.tp_level, entry_time=str(pos.entry_time),
            exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
            credit=pos.credit, exit_debit=pos.exit_debit,
            fees=fees / pos.qty, pnl_per_share=pnl_per_share,
            qty=pos.qty, pnl_total=pnl_total,
            portfolio_after=portfolio, shadow=False,
            sl_budget_after=max(sl_budget - sl_reserved, 0),
        ))
    if shadow_pos:
        trades.append(TradeRecord(
            day=str(shadow_pos.entry_time.date()), strategy=shadow_pos.strategy,
            tp_level=shadow_pos.tp_level,
            entry_time=str(shadow_pos.entry_time),
            exit_time=str(shadow_pos.exit_time),
            exit_reason=shadow_pos.exit_reason,
            credit=shadow_pos.credit, exit_debit=shadow_pos.exit_debit,
            fees=0, pnl_per_share=0, qty=shadow_pos.qty,
            pnl_total=0, portfolio_after=portfolio, shadow=True,
            sl_budget_after=max(sl_budget - sl_reserved, 0),
        ))

    return trades, portfolio, sl_budget


# ── Strategy 3: Optimised Portfolio (integer knapsack) ───────────────────────

def _solve_knapsack(candidates, sl_avail, bp_avail, max_contracts=50):
    """
    Integer LP: maximise Σ(ev_i × qty_i)
    s.t. Σ(max_loss_i × qty_i) ≤ sl_avail   (real only)
         Σ(BP_i × qty_i)       ≤ bp_avail   (real + shadow handled outside)
         0 ≤ qty_i ≤ max_contracts  (integer)

    Returns dict: {(strategy, tp_level): qty}
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        n = len(candidates)
        if n == 0: return {}

        c_obj = np.array([-cand["ev"] for cand in candidates], dtype=float)
        A = np.array([
            [cand["max_loss_per_ct"] for cand in candidates],
            [BUYING_POWER_PER_CONTRACT for _ in candidates],
        ], dtype=float)
        b_ub  = np.array([sl_avail, bp_avail], dtype=float)
        bounds = Bounds(lb=0, ub=max_contracts)
        constraints = LinearConstraint(A, lb=-np.inf, ub=b_ub)
        integrality = np.ones(n)

        res = milp(c_obj, constraints=constraints,
                   integrality=integrality, bounds=bounds)
        if res.success:
            return {(candidates[i]["strategy"], candidates[i]["tp_level"]): int(round(res.x[i]))
                    for i in range(n) if round(res.x[i]) > 0}
    except Exception:
        pass

    # Fallback: greedy knapsack sorted by EV/max_loss ratio
    result = {}
    remaining_sl = sl_avail
    remaining_bp = bp_avail
    sorted_cands = sorted(candidates,
                          key=lambda c: c["ev"] / max(c["max_loss_per_ct"], 1),
                          reverse=True)
    for cand in sorted_cands:
        ml  = cand["max_loss_per_ct"]
        qty = min(int(remaining_sl // ml) if ml > 0 else 0,
                  int(remaining_bp // BUYING_POWER_PER_CONTRACT),
                  max_contracts)
        if qty > 0:
            result[(cand["strategy"], cand["tp_level"])] = qty
            remaining_sl -= qty * ml
            remaining_bp -= qty * BUYING_POWER_PER_CONTRACT
    return result


def simulate_s3_day(day_pool, cutoffs, skip_rate,
                    portfolio, sl_budget_start):
    trades      = []
    open_pos    = []
    shadow_pos  = None
    sl_reserved = 0.0
    sl_budget   = sl_budget_start

    entry_times = sorted(day_pool.keys())

    def _close_expired(et):
        nonlocal portfolio, sl_reserved, sl_budget, open_pos, shadow_pos
        still_open = []
        for pos in open_pos:
            if pos.exit_time > et:
                still_open.append(pos); continue
            portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
                close_position(pos, portfolio, sl_reserved)
            sl_budget += pnl_total
            trades.append(TradeRecord(
                day=str(et.date()), strategy=pos.strategy,
                tp_level=pos.tp_level, entry_time=str(pos.entry_time),
                exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
                credit=pos.credit, exit_debit=pos.exit_debit,
                fees=fees / pos.qty, pnl_per_share=pnl_per_share,
                qty=pos.qty, pnl_total=pnl_total,
                portfolio_after=portfolio, shadow=False,
                sl_budget_after=max(sl_budget - sl_reserved, 0),
            ))
        open_pos = still_open
        if shadow_pos and shadow_pos.exit_time <= et:
            trades.append(TradeRecord(
                day=str(et.date()), strategy=shadow_pos.strategy,
                tp_level=shadow_pos.tp_level,
                entry_time=str(shadow_pos.entry_time),
                exit_time=str(shadow_pos.exit_time),
                exit_reason=shadow_pos.exit_reason,
                credit=shadow_pos.credit, exit_debit=shadow_pos.exit_debit,
                fees=0, pnl_per_share=0, qty=shadow_pos.qty,
                pnl_total=0, portfolio_after=portfolio, shadow=True,
                sl_budget_after=max(sl_budget - sl_reserved, 0),
            ))
            shadow_pos = None

    for et in entry_times:
        _close_expired(et)

        candidates = _get_all_candidates(day_pool, et, cutoffs, skip_rate)
        if not candidates: continue

        open_strat_tp = {(p.strategy, p.tp_level) for p in open_pos}
        bp_used = sum(p.qty * BUYING_POWER_PER_CONTRACT for p in open_pos)
        if shadow_pos:
            bp_used += shadow_pos.qty * BUYING_POWER_PER_CONTRACT
        bp_avail = portfolio - bp_used
        sl_avail = sl_budget - sl_reserved

        # Shadow: place 5D if not already open
        if shadow_pos is None:
            five_d = [c for c in candidates if c["is_shadow"]]
            if five_d:
                c5  = five_d[0]
                qty = max(0, int(bp_avail // BUYING_POWER_PER_CONTRACT))
                if qty > 0:
                    shadow_pos = OpenPosition(
                        strategy=c5["strategy"], tp_level=c5["tp_level"],
                        entry_time=et, exit_time=c5["exit_time"],
                        exit_reason=c5["exit_reason"],
                        credit=c5["credit"], exit_debit=c5["exit_debit"],
                        qty=qty, max_loss=0.0, shadow=True,
                    )
                    bp_avail -= qty * BUYING_POWER_PER_CONTRACT

        # Knapsack on real candidates not already open
        real_cands = [c for c in candidates
                      if not c["is_shadow"]
                      and (c["strategy"], c["tp_level"]) not in open_strat_tp]

        if real_cands and sl_avail > 0 and bp_avail > 0:
            allocation = _solve_knapsack(real_cands, sl_avail, bp_avail)
            for cand in real_cands:
                key = (cand["strategy"], cand["tp_level"])
                qty = allocation.get(key, 0)
                if qty == 0: continue
                pos = OpenPosition(
                    strategy=cand["strategy"], tp_level=cand["tp_level"],
                    entry_time=et, exit_time=cand["exit_time"],
                    exit_reason=cand["exit_reason"],
                    credit=cand["credit"], exit_debit=cand["exit_debit"],
                    qty=qty, max_loss=qty * cand["max_loss_per_ct"],
                    shadow=False,
                )
                open_pos.append(pos)
                open_strat_tp.add(key)
                sl_reserved += pos.max_loss

    # End-of-day close
    for pos in open_pos:
        portfolio, sl_reserved, pnl_total, pnl_per_share, fees = \
            close_position(pos, portfolio, sl_reserved)
        sl_budget += pnl_total
        trades.append(TradeRecord(
            day=str(pos.entry_time.date()), strategy=pos.strategy,
            tp_level=pos.tp_level, entry_time=str(pos.entry_time),
            exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
            credit=pos.credit, exit_debit=pos.exit_debit,
            fees=fees / pos.qty, pnl_per_share=pnl_per_share,
            qty=pos.qty, pnl_total=pnl_total,
            portfolio_after=portfolio, shadow=False,
            sl_budget_after=max(sl_budget - sl_reserved, 0),
        ))
    if shadow_pos:
        trades.append(TradeRecord(
            day=str(shadow_pos.entry_time.date()), strategy=shadow_pos.strategy,
            tp_level=shadow_pos.tp_level,
            entry_time=str(shadow_pos.entry_time),
            exit_time=str(shadow_pos.exit_time),
            exit_reason=shadow_pos.exit_reason,
            credit=shadow_pos.credit, exit_debit=shadow_pos.exit_debit,
            fees=0, pnl_per_share=0, qty=shadow_pos.qty,
            pnl_total=0, portfolio_after=portfolio, shadow=True,
            sl_budget_after=max(sl_budget - sl_reserved, 0),
        ))

    return trades, portfolio, sl_budget


# ── Full simulation (all days) ────────────────────────────────────────────────

def simulate(candidate_pool, strategy_type, skip_rate, max_sl_pct,
             cutoffs, start_portfolio=START_PORTFOLIO):
    portfolio     = start_portfolio
    all_trades    = []
    equity_curve  = [{"date": "start", "portfolio": portfolio}]

    sim_fn = {1: simulate_s1_day, 2: simulate_s2_day, 3: simulate_s3_day}[strategy_type]

    for day in sorted(candidate_pool.keys()):
        day_pool   = candidate_pool[day]
        # Daily SL budget = fresh each morning based on CURRENT portfolio
        daily_sl   = portfolio * max_sl_pct
        trades, portfolio, _ = sim_fn(
            day_pool, cutoffs, skip_rate,
            portfolio, daily_sl,
        )
        all_trades.extend(trades)
        equity_curve.append({"date": str(day.date()), "portfolio": portfolio})

    real_trades = [t for t in all_trades if not t.shadow]
    shadow_trades = [t for t in all_trades if t.shadow]

    final  = equity_curve[-1]["portfolio"]
    total_ret = (final - start_portfolio) / start_portfolio * 100

    vals   = [e["portfolio"] for e in equity_curve]
    peak   = vals[0]; max_dd = 0.0
    for v in vals:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    daily_rets = []
    prev = start_portfolio
    for e in equity_curve[1:]:
        if prev > 0: daily_rets.append((e["portfolio"] - prev) / prev)
        prev = e["portfolio"]

    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252)

    wins = [t for t in real_trades if t.pnl_total > 0]
    win_rate = len(wins) / len(real_trades) * 100 if real_trades else 0
    avg_pnl  = np.mean([t.pnl_total for t in real_trades]) if real_trades else 0

    summary = {
        "strategy":         strategy_type,
        "skip_rate":        skip_rate,
        "max_sl_pct":       max_sl_pct,
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe":           round(sharpe, 2),
        "win_rate":         round(win_rate, 1),
        "n_trades":         len(real_trades),
        "n_shadow":         len(shadow_trades),
        "trades_per_day":   round(len(real_trades) / max(len(equity_curve) - 1, 1), 2),
        "final_portfolio":  round(final, 2),
        "avg_pnl":          round(avg_pnl, 2),
    }
    return equity_curve, all_trades, summary


# ── Output helpers ────────────────────────────────────────────────────────────

def save_outputs(output_dir, summaries, all_trade_logs, all_equity_curves):
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)

    all_trades_flat = []
    for key, trades in all_trade_logs.items():
        for t in trades:
            all_trades_flat.append({
                "sim_key": key,
                "day": t.day, "strategy": t.strategy, "tp_level": t.tp_level,
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "exit_reason": t.exit_reason, "credit": t.credit,
                "exit_debit": t.exit_debit, "fees": t.fees,
                "pnl_per_share": t.pnl_per_share, "qty": t.qty,
                "pnl_total": t.pnl_total, "portfolio_after": t.portfolio_after,
                "shadow": t.shadow, "sl_budget_after": t.sl_budget_after,
            })
    pd.DataFrame(all_trades_flat).to_csv(output_dir / "trade_log.csv", index=False)

    eq_flat = []
    for key, curve in all_equity_curves.items():
        for e in curve:
            eq_flat.append({"sim_key": key, **e})
    pd.DataFrame(eq_flat).to_csv(output_dir / "equity_curves.csv", index=False)

    print(f"  Saved {len(summaries)} summaries, "
          f"{len(all_trades_flat)} trade records → {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global SLIP_PER_SIDE, EFFECTIVE_SLIP

    parser = argparse.ArgumentParser(description="ZDOM V1.1 SL-Constrained Backtest")
    parser.add_argument("--max-sl-pct", nargs="+", type=float,
                        default=[0.10, 0.15, 0.20, 0.25, 0.30],
                        help="Max portfolio stop-loss %% (e.g. 0.10 0.20 0.30)")
    parser.add_argument("--strategy", nargs="+", type=int, choices=[1, 2, 3],
                        default=None,
                        help="Run strategy 1, 2, and/or 3 (e.g. --strategy 1 2)")
    parser.add_argument("--portfolio", type=float, default=START_PORTFOLIO)
    parser.add_argument("--slip-per-side", nargs="+", type=float,
                        default=[0.00],
                        help="Per-side slippage(s) in dollars (e.g. --slip-per-side 0.00 0.20)")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Optional suffix for output dir isolation")
    args = parser.parse_args()

    strategies      = args.strategy if args.strategy else [1, 2, 3]
    sl_pct_list     = args.max_sl_pct
    slip_per_sides  = args.slip_per_side

    print(f"\n{'='*70}")
    print(f"  ZDOM V1.2 — SL-Constrained Backtest")
    print(f"  Strategies:   {strategies}")
    print(f"  SL levels:    {sl_pct_list}")
    print(f"  Slip levels:  {['${:.2f}/side (${:.2f} RT)'.format(s, s*2) for s in slip_per_sides]}")
    print(f"  TP levels:    {', '.join(TP_LEVELS)}")
    print(f"  Skip rates:   {SKIP_RATES[0]:.2f} → {SKIP_RATES[-1]:.2f} "
          f"({len(SKIP_RATES)} levels)")
    print(f"  Portfolio:    ${args.portfolio:,.0f}")
    if args.output_tag:
        print(f"  Output tag:   {args.output_tag}")
    print(f"{'='*70}\n")

    # Phase 1: Data
    print("── Phase 1: Data Loading ──")
    df = load_model_table()
    # Blockers disabled — production does not use them.
    # df = apply_blockers(df)
    df["blocked"] = False
    train_df, test_df, holdout_df = split_data(df)
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    # Phase 2: Models
    print("\n── Phase 2: Loading Models ──")
    models = load_models()
    print(f"  Loaded {len(models)}/{len(TP_LEVELS)} models")

    # Phase 3: EV lookup + calibration
    print("\n── Phase 3: EV Lookup ──")
    ev_lookup = precompute_ev_lookup(train_df)

    print("\n── Phase 3b: Probability Calibration ──")
    calibrators = fit_calibrators(train_df, models, feature_cols)

    # Evaluation periods: test first, then holdout
    eval_periods = [
        ("test",    test_df),
        ("holdout", holdout_df),
    ]

    for period_name, period_df in eval_periods:
        print(f"\n{'='*70}")
        print(f"  EVALUATION PERIOD: {period_name.upper()}")
        print(f"  {period_df['date'].min().date()} → {period_df['date'].max().date()} "
              f"({period_df['date'].nunique()} days, {len(period_df):,} rows)")
        print(f"{'='*70}")

        print(f"\n── Phase 4: Scoring {period_name} ──")
        sim_scored = score_holdout(period_df, models, feature_cols, calibrators)

        # Pre-convert exit_time columns from str → Timestamp
        for _tp in TP_LEVELS:
            col = f"{_tp}_exit_time"
            if col in sim_scored.columns:
                sim_scored[col] = pd.to_datetime(sim_scored[col], errors="coerce")

        print(f"\n── Phase 5: Skip Cutoffs ({period_name}) ──")
        cutoffs = precompute_skip_cutoffs(sim_scored, SKIP_RATES)
        print(f"  Cutoffs: {len(cutoffs)}")

        # Phase 6: Loop over slippage levels
        total_sims = len(slip_per_sides) * len(strategies) * len(sl_pct_list) * len(SKIP_RATES)
        print(f"\n── Phase 6: Running {total_sims} Simulations [{period_name}] "
              f"({len(slip_per_sides)} slip × {len(strategies)} strat × "
              f"{len(sl_pct_list)} SL × {len(SKIP_RATES)} skip) ──")

        for slip_val in slip_per_sides:
            SLIP_PER_SIDE  = slip_val
            EFFECTIVE_SLIP = slip_val * 2
            slip_tag = f"slip{int(slip_val*100):02d}"

            print(f"\n{'─'*60}")
            print(f"  Slippage: ${slip_val:.2f}/side (${EFFECTIVE_SLIP:.2f} RT)  "
                  f"[{period_name}_{slip_tag}]")
            print(f"{'─'*60}")

            print("\n  Precomputing Candidate Pool …")
            t_pre = time.time()
            candidate_pool = precompute_all_candidates(sim_scored, ev_lookup)
            n_cands = sum(len(cs) for dp in candidate_pool.values()
                          for cs in dp.values())
            print(f"  Pool: {len(candidate_pool)} days, {n_cands:,} candidates  "
                  f"({time.time()-t_pre:.1f}s)")

            for sl_pct in sl_pct_list:
                for strat_type in strategies:
                    sl_label = f"sl{int(sl_pct*100):02d}pct"
                    dir_name = (f"backtest_sl_s{strat_type}_{sl_label}"
                                f"_{slip_tag}_{period_name}")
                    if args.output_tag:
                        dir_name = f"{dir_name}_{args.output_tag}"
                    out_dir  = OUTPUT_BASE / dir_name

                    print(f"\n  S{strat_type} @ {sl_pct*100:.0f}% SL "
                          f"[{slip_tag}/{period_name}]  →  {out_dir.name}")
                    summaries       = []
                    all_trade_logs  = {}
                    all_equity_curves = {}

                    t0  = time.time()
                    for i, sr in enumerate(SKIP_RATES):
                        key = f"S{strat_type}_sl{sl_pct:.2f}_skip{sr:.2f}"
                        eq, trades, summary = simulate(
                            candidate_pool, strat_type, sr, sl_pct,
                            cutoffs, start_portfolio=args.portfolio,
                        )
                        summaries.append(summary)
                        all_trade_logs[key]    = trades
                        all_equity_curves[key] = eq

                        if (i + 1) % 10 == 0 or (i + 1) == len(SKIP_RATES):
                            elapsed = time.time() - t0
                            best = max(summaries, key=lambda s: s["sharpe"])
                            print(f"    [{i+1}/{len(SKIP_RATES)}] {elapsed:.0f}s  "
                                  f"best so far: skip={best['skip_rate']:.2f} "
                                  f"sharpe={best['sharpe']:.2f} "
                                  f"ret={best['total_return_pct']:+.1f}%")

                    save_outputs(out_dir, summaries, all_trade_logs,
                                 all_equity_curves)

    print(f"\n{'='*70}")
    print(f"  All simulations complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
