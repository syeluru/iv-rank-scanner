"""
ZDOM V1.2 Walk-Forward Backtest — S1 + S2 (this machine).

Full grid: S1 (best EV single) + S2 (diversified greedy) across 5 folds.
Partner script backtest_walkforward_s3.py runs S3 (knapsack) on Mac Mini.

Slippage: $0.20/side ($0.40 RT)

Usage:
  source ~/venvs/zdom/bin/activate
  python3 ml/ZDOM/5_analysis/v1_2/scripts/backtest_walkforward_s1s2.py
"""

import gc
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── Paths (restructured repo) ────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent                       # .../v1_2/scripts/
ZDOM_DIR    = SCRIPT_DIR.parent.parent.parent                       # ml/ZDOM/
MODELS_DIR  = ZDOM_DIR / "4_models" / "v1_2"
MASTER_DATA = ZDOM_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final.parquet"
OUTPUT_BASE = ZDOM_DIR / "5_analysis" / "v1_2" / "results" / "backtest_walkforward"

# ── Constants ─────────────────────────────────────────────────────────────────

STRATEGIES_ALL  = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]   # 9 deltas
STRATEGIES_NO5D = [f"IC_{d:02d}d_25w" for d in range(10, 50, 5)]  # no 5-delta
SHADOW_STRATEGY = "IC_05d_25w"


def discover_tp_levels():
    levels = set()
    for pkl_path in MODELS_DIR.glob("tp*_target_*_v1_2.pkl"):
        tp = pkl_path.name.split("_target_")[0]
        if tp.startswith("tp"):
            levels.add(tp)
    if not levels:
        levels = {f"tp{p}" for p in range(10, 30, 5)}
    return sorted(levels, key=lambda tp: int(tp[2:]))


TP_LEVELS = discover_tp_levels()
TARGETS   = [f"{tp}_target" for tp in TP_LEVELS]

GAP_DAYS  = 7
BUYING_POWER_PER_CONTRACT = 2500
FEES_PER_SHARE = 0.052            # $5.20 RT / 100 shares
START_PORTFOLIO = 10_000

# Slippage: $0.20 per side
SLIP_PER_SIDE  = 0.20
EFFECTIVE_SLIP = SLIP_PER_SIDE * 2   # $0.40 RT

SKIP_RATES = [round(0.05 + i * 0.01, 2) for i in range(31)]  # 0.05 → 0.35

# Which strategies THIS script runs
STRATEGIES = [1, 2]
SL_LEVELS  = [0.10, 0.15, 0.20, 0.25, 0.30]

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
    max_loss:    float
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


# ── Data loading / splitting / scoring ────────────────────────────────────────

def load_model_table():
    if MASTER_DATA.exists():
        print(f"Loading master data: {MASTER_DATA.name} …")
        df = pd.read_parquet(MASTER_DATA)
        df["date"] = pd.to_datetime(df["date"])
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "decision_datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["decision_datetime"])
        df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
        print(f"  {len(df):,} rows × {df.shape[1]} cols, {df['date'].nunique()} days")
        return df
    print(f"[error] {MASTER_DATA} not found"); sys.exit(1)


def split_data(df):
    """Split post-train data into 5 equal folds."""
    train_end_dt = pd.Timestamp("2024-09-04")

    for pkl_path in sorted(MODELS_DIR.glob("tp*_target_*_v1_2.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                art = pickle.load(f)
            split_info = art.get("split_info") or {}
            train_range = split_info.get("train_range")
            if train_range:
                train_end_dt = pd.Timestamp(train_range.split(" -> ")[1])
                print(f"  Split source: {pkl_path.name}  train_end={train_end_dt.date()}")
                break
        except Exception as exc:
            print(f"  [warn] failed to read split_info from {pkl_path.name}: {exc}")

    train_df = df[df["date"] <= train_end_dt].reset_index(drop=True)
    post_train = df[df["date"] > train_end_dt].copy()
    post_dates = sorted(post_train["date"].unique())
    chunk_size = len(post_dates) // 5

    folds = {}
    fold_names = ["test", "holdout_1", "holdout_2", "holdout_3", "holdout_4"]
    for i, name in enumerate(fold_names):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 4 else len(post_dates)
        fold_dates = post_dates[start_idx:end_idx]
        fold_df = post_train[post_train["date"].isin(fold_dates)].reset_index(drop=True)
        folds[name] = fold_df

    train_days = train_df["date"].nunique()
    print(f"  Train:      {len(train_df):>10,} rows ({train_days:>3d} days)")
    for name, fold_df in folds.items():
        n_days = fold_df["date"].nunique()
        print(f"  {name:10s}: {len(fold_df):>10,} rows ({n_days:>3d} days)  "
              f"{fold_df['date'].min().date()} → {fold_df['date'].max().date()}")

    return train_df, folds


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
            if not mf.exists(): continue
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
    calibrators = {}
    for tp, (model, model_feat_cols) in models.items():
        target = f"{tp}_target"
        if target not in train_df.columns: continue
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
                wins[pnl_col].mean(), losses[pnl_col].mean(),
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


# ── Candidate precomputation ─────────────────────────────────────────────────

def precompute_all_candidates(sim_scored, ev_lookup):
    required_cols = ["date", "datetime", "strategy", "blocked", "credit"]
    for tp in TP_LEVELS:
        required_cols.extend([
            f"prob_{tp}", f"{tp}_exit_time", f"{tp}_exit_debit", f"{tp}_exit_reason",
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
                    if not is_shadow and (ev is None or ev <= 0): continue
                    if ev is None: ev = 0.0
                    exit_ts     = getattr(row, f"{tp}_exit_time", None)
                    exit_debit  = getattr(row, f"{tp}_exit_debit", None)
                    exit_reason = getattr(row, f"{tp}_exit_reason", None)
                    if exit_ts is None or pd.isna(exit_ts) or pd.isna(exit_debit): continue
                    candidates.append({
                        "strategy": strat, "tp_level": tp, "prob": float(prob),
                        "ev": ev, "credit": credit_f, "exit_debit": float(exit_debit),
                        "exit_time": exit_ts, "exit_reason": exit_reason,
                        "is_shadow": is_shadow, "max_loss_per_ct": credit_f * 100,
                    })
            candidates.sort(key=lambda x: x["ev"], reverse=True)
            pool[date][entry_time] = candidates
    return pool


def _get_all_candidates(day_pool, entry_time, cutoffs, skip_rate):
    base = day_pool.get(entry_time, [])
    if not base: return []
    return [c for c in base
            if c["prob"] >= cutoffs.get((c["strategy"], c["tp_level"], skip_rate), 1.0)]


# ── Position close helper ────────────────────────────────────────────────────

def close_position(pos: OpenPosition, portfolio: float, sl_reserved: float):
    fees          = FEES_PER_SHARE * pos.qty
    pnl_per_share = pos.credit - pos.exit_debit - EFFECTIVE_SLIP
    pnl_total     = pnl_per_share * pos.qty * 100 - fees
    new_portfolio   = portfolio + pnl_total
    new_sl_reserved = sl_reserved - pos.max_loss
    return new_portfolio, new_sl_reserved, pnl_total, pnl_per_share, fees


# ── Strategy 1: Constrained Best EV (single entry) ──────────────────────────

def simulate_s1_day(day_pool, cutoffs, skip_rate, portfolio, sl_budget_start):
    trades      = []
    open_pos    = None
    shadow_pos  = None
    sl_reserved = 0.0
    sl_budget   = sl_budget_start

    entry_times = sorted(day_pool.keys())

    for et in entry_times:
        for pos in [open_pos, shadow_pos]:
            if pos is None: continue
            if pos.exit_time > et: continue
            if pos.shadow:
                t = TradeRecord(
                    day=str(et.date()), strategy=pos.strategy,
                    tp_level=pos.tp_level, entry_time=str(pos.entry_time),
                    exit_time=str(pos.exit_time), exit_reason=pos.exit_reason,
                    credit=pos.credit, exit_debit=pos.exit_debit,
                    fees=0, pnl_per_share=0, qty=pos.qty,
                    pnl_total=0, portfolio_after=portfolio,
                    shadow=True, sl_budget_after=sl_budget - sl_reserved)
                trades.append(t)
                shadow_pos = None
            else:
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
                    sl_budget_after=sl_budget - sl_reserved)
                trades.append(t)
                open_pos = None

        if open_pos is not None and shadow_pos is not None:
            continue

        candidates = _get_all_candidates(day_pool, et, cutoffs, skip_rate)
        if not candidates: continue

        bp_used_shadow = shadow_pos.qty * BUYING_POWER_PER_CONTRACT if shadow_pos else 0
        bp_used_real   = open_pos.qty * BUYING_POWER_PER_CONTRACT if open_pos else 0
        bp_avail       = portfolio - bp_used_shadow - bp_used_real
        sl_avail       = sl_budget - sl_reserved

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
                        qty=shadow_qty, max_loss=0.0, shadow=True)
                    bp_avail -= shadow_qty * BUYING_POWER_PER_CONTRACT

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
                    max_loss=best_qty * best_c["max_loss_per_ct"], shadow=False)
                sl_reserved += open_pos.max_loss

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
            portfolio_after=portfolio, shadow=False,
            sl_budget_after=max(sl_budget - sl_reserved, 0))
        trades.append(t)

    return trades, portfolio, sl_budget


# ── Strategy 2: Constrained Diversified (greedy fill) ────────────────────────

def simulate_s2_day(day_pool, cutoffs, skip_rate, portfolio, sl_budget_start):
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
                sl_budget_after=max(sl_budget - sl_reserved, 0)))
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
                sl_budget_after=max(sl_budget - sl_reserved, 0)))
            shadow_pos = None

    for et in entry_times:
        _close_expired(et)
        candidates = _get_all_candidates(day_pool, et, cutoffs, skip_rate)
        if not candidates: continue

        open_strat_tp = {(p.strategy, p.tp_level) for p in open_pos}
        bp_used = sum(p.qty * BUYING_POWER_PER_CONTRACT for p in open_pos)
        if shadow_pos: bp_used += shadow_pos.qty * BUYING_POWER_PER_CONTRACT
        bp_avail = portfolio - bp_used
        sl_avail = sl_budget - sl_reserved

        if shadow_pos is None:
            five_d = [c for c in candidates if c["is_shadow"]]
            if five_d:
                c5 = five_d[0]
                qty = max(0, int(bp_avail // BUYING_POWER_PER_CONTRACT))
                if qty > 0:
                    shadow_pos = OpenPosition(
                        strategy=c5["strategy"], tp_level=c5["tp_level"],
                        entry_time=et, exit_time=c5["exit_time"],
                        exit_reason=c5["exit_reason"],
                        credit=c5["credit"], exit_debit=c5["exit_debit"],
                        qty=qty, max_loss=0.0, shadow=True)
                    bp_avail -= qty * BUYING_POWER_PER_CONTRACT

        for c in candidates:
            if c["is_shadow"]: continue
            if (c["strategy"], c["tp_level"]) in open_strat_tp: continue
            ml = c["max_loss_per_ct"]
            if ml <= 0: continue
            if sl_avail < ml: continue
            if bp_avail < BUYING_POWER_PER_CONTRACT: continue
            pos = OpenPosition(
                strategy=c["strategy"], tp_level=c["tp_level"],
                entry_time=et, exit_time=c["exit_time"],
                exit_reason=c["exit_reason"],
                credit=c["credit"], exit_debit=c["exit_debit"],
                qty=1, max_loss=ml, shadow=False)
            open_pos.append(pos)
            open_strat_tp.add((c["strategy"], c["tp_level"]))
            sl_reserved += ml
            sl_avail    -= ml
            bp_avail    -= BUYING_POWER_PER_CONTRACT

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
            sl_budget_after=max(sl_budget - sl_reserved, 0)))
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
            sl_budget_after=max(sl_budget - sl_reserved, 0)))

    return trades, portfolio, sl_budget


# ── Full simulation ──────────────────────────────────────────────────────────

def simulate(candidate_pool, strategy_type, skip_rate, max_sl_pct,
             cutoffs, start_portfolio=START_PORTFOLIO):
    portfolio     = start_portfolio
    all_trades    = []
    equity_curve  = [{"date": "start", "portfolio": portfolio}]

    sim_fn = {1: simulate_s1_day, 2: simulate_s2_day}[strategy_type]

    for day in sorted(candidate_pool.keys()):
        day_pool = candidate_pool[day]
        daily_sl = portfolio * max_sl_pct
        trades, portfolio, _ = sim_fn(day_pool, cutoffs, skip_rate, portfolio, daily_sl)
        all_trades.extend(trades)
        equity_curve.append({"date": str(day.date()), "portfolio": portfolio})

    real_trades = [t for t in all_trades if not t.shadow]
    shadow_trades = [t for t in all_trades if t.shadow]

    final     = equity_curve[-1]["portfolio"]
    total_ret = (final - start_portfolio) / start_portfolio * 100

    vals = [e["portfolio"] for e in equity_curve]
    peak = vals[0]; max_dd = 0.0
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

    return equity_curve, all_trades, {
        "strategy": strategy_type, "skip_rate": skip_rate,
        "max_sl_pct": max_sl_pct, "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd * 100, 2), "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 1), "n_trades": len(real_trades),
        "n_shadow": len(shadow_trades),
        "trades_per_day": round(len(real_trades) / max(len(equity_curve) - 1, 1), 2),
        "final_portfolio": round(final, 2), "avg_pnl": round(avg_pnl, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_configs = len(STRATEGIES) * len(SL_LEVELS) * len(SKIP_RATES)

    print(f"\n{'='*70}")
    print(f"  ZDOM V1.2 — Walk-Forward Grid (S1 + S2)")
    print(f"  Strategies:   {STRATEGIES}")
    print(f"  SL levels:    {[f'{s:.0%}' for s in SL_LEVELS]}")
    print(f"  Skip rates:   {SKIP_RATES[0]:.2f} → {SKIP_RATES[-1]:.2f} ({len(SKIP_RATES)} levels)")
    print(f"  Configs/fold: {total_configs}  ({len(STRATEGIES)}×{len(SL_LEVELS)}×{len(SKIP_RATES)})")
    print(f"  Total sims:   {total_configs * 5}")
    print(f"  Slippage:     ${EFFECTIVE_SLIP:.2f} RT (${SLIP_PER_SIDE:.2f}/side)")
    print(f"  Portfolio:    ${START_PORTFOLIO:,.0f}")
    print(f"{'='*70}\n")

    print("── Phase 1: Data Loading ──")
    df = load_model_table()
    df["blocked"] = False
    train_df, folds = split_data(df)
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    print("\n── Phase 2: Loading Models ──")
    models = load_models()
    print(f"  Loaded {len(models)}/{len(TP_LEVELS)} models")

    print("\n── Phase 3: EV Lookup ──")
    ev_lookup = precompute_ev_lookup(train_df)

    print("\n── Phase 3b: Probability Calibration ──")
    calibrators = fit_calibrators(train_df, models, feature_cols)

    del df, train_df
    gc.collect()

    all_results = []

    for fold_idx, (period_name, period_df) in enumerate(folds.items()):
        print(f"\n{'='*70}")
        print(f"  PERIOD: {period_name.upper()} ({fold_idx+1}/{len(folds)})")
        print(f"  {period_df['date'].min().date()} → {period_df['date'].max().date()} "
              f"({period_df['date'].nunique()} days, {len(period_df):,} rows)")
        print(f"{'='*70}")

        print(f"\n── Scoring {period_name} ──")
        sim_scored = score_holdout(period_df, models, feature_cols, calibrators)

        for _tp in TP_LEVELS:
            col = f"{_tp}_exit_time"
            if col in sim_scored.columns:
                sim_scored[col] = pd.to_datetime(sim_scored[col], errors="coerce")

        print(f"\n── Skip Cutoffs ──")
        cutoffs = precompute_skip_cutoffs(sim_scored, SKIP_RATES)
        print(f"  Cutoffs: {len(cutoffs)}")

        print("\n  Precomputing Candidate Pool …")
        t_pre = time.time()
        candidate_pool = precompute_all_candidates(sim_scored, ev_lookup)
        n_cands = sum(len(cs) for dp in candidate_pool.values() for cs in dp.values())
        print(f"  Pool: {len(candidate_pool)} days, {n_cands:,} candidates  "
              f"({time.time()-t_pre:.1f}s)")

        del sim_scored
        gc.collect()

        fold_t0 = time.time()
        sim_count = 0
        for strat_type in STRATEGIES:
            for sl_pct in SL_LEVELS:
                sl_label = f"sl{int(sl_pct*100):02d}pct"
                best_sharpe, best_skip, best_ret = -999, None, None
                block_t0 = time.time()

                for i, skip_rate in enumerate(SKIP_RATES):
                    eq, trades, summary = simulate(
                        candidate_pool, strat_type, skip_rate, sl_pct, cutoffs)
                    sim_count += 1
                    summary["period"] = period_name
                    summary["config"] = f"S{strat_type}_{sl_label}_skip{int(skip_rate*100):02d}"
                    all_results.append(summary)

                    if summary["sharpe"] > best_sharpe:
                        best_sharpe = summary["sharpe"]
                        best_skip   = skip_rate
                        best_ret    = summary["total_return_pct"]

                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - block_t0
                        print(f"    S{strat_type} @ {sl_pct*100:.0f}% SL [{i+1}/{len(SKIP_RATES)}] "
                              f"{elapsed:.0f}s  best: skip={best_skip:.2f} "
                              f"sharpe={best_sharpe:.2f} ret={best_ret:+.1f}%")

                block_elapsed = time.time() - block_t0
                print(f"  S{strat_type} @ {sl_pct*100:.0f}% SL [{period_name}]  "
                      f"best: skip={best_skip:.2f} sharpe={best_sharpe:.2f} "
                      f"ret={best_ret:+.1f}%  ({block_elapsed:.0f}s)")

        fold_elapsed = time.time() - fold_t0
        print(f"\n  Fold complete: {sim_count} sims in {fold_elapsed:.0f}s "
              f"({fold_elapsed/sim_count:.1f}s/sim)")

        del candidate_pool, cutoffs
        gc.collect()

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    out_file = OUTPUT_BASE / "walkforward_s1s2_grid.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"\n  Saved {len(summary_df)} results → {out_file}")

    print(f"\n{'='*70}")
    print(f"  BEST CONFIGS (avg across {len(folds)} folds)")
    print(f"{'='*70}")
    avg = summary_df.groupby(["strategy", "max_sl_pct", "skip_rate"]).agg(
        avg_sharpe=("sharpe", "mean"), avg_return=("total_return_pct", "mean"),
        avg_winrate=("win_rate", "mean"), min_return=("total_return_pct", "min"),
        max_dd=("max_drawdown_pct", "max"), avg_trades=("n_trades", "mean"),
    ).reset_index()
    best = avg.loc[avg.groupby(["strategy", "max_sl_pct"])["avg_sharpe"].idxmax()]
    best = best.sort_values(["strategy", "max_sl_pct"])
    print(best[["strategy", "max_sl_pct", "skip_rate", "avg_sharpe", "avg_return",
                "min_return", "max_dd", "avg_trades"]].to_string(index=False))
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
