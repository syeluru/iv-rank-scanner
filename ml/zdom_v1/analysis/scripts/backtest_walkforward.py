"""
ZDOM V1 Walk-Forward Backtest Simulator.

Simulates realistic trading on holdout data using trained XGBoost/LightGBM models.
Evaluates 2 portfolio strategies × 7 slippage scenarios × 21 skip rates (294 configs).
Excludes IC_05d_25w (5-delta) strategies which were empirically shown to destroy returns.

Strategies:
  1. "Best EV, Max Contracts" — single best combo, all contracts
  2. "Diversified, Max Buying Power" — spread across top-K combos

Usage:
  python3 scripts/backtest_walkforward.py                    # full sim
  python3 scripts/backtest_walkforward.py --walkthrough      # 1-day trace only
  python3 scripts/backtest_walkforward.py --no-tune          # skip Optuna (fast)
  python3 scripts/backtest_walkforward.py --strategy 1       # single strategy
  python3 scripts/backtest_walkforward.py --skip-rates 0.10 0.20  # subset
"""

import argparse
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models" / "v1"
OUTPUT_DIR = PROJECT_DIR / "output" / "backtest"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(10, 50, 5)]  # Exclude 5-delta
TP_LEVELS = [f"tp{p}" for p in range(10, 55, 5)]
TARGETS = [f"{tp}_target" for tp in TP_LEVELS]

TRAIN_PCT = 0.70
GAP_DAYS = 7
BUYING_POWER_PER_CONTRACT = 2500
FEES_PER_SHARE = 0.052  # $5.20 RT / 100 shares
START_PORTFOLIO = 10_000

EXIT_SLIPS = [round(x * 0.05, 2) for x in range(7)]  # 0.00, 0.05, ..., 0.30
SKIP_RATES = [round(0.20 + i * 0.01, 2) for i in range(21)]  # 0.20, 0.21, ..., 0.40

# Meta columns (not features) — from train_v1.py
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
    "time_to_close_min",
    "_blocked",
] + _TP_META


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    strategy: str
    tp_level: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    credit: float      # per share
    exit_debit: float   # per share
    qty: int            # contracts
    ev: float           # EV at entry


@dataclass
class TradeRecord:
    day: str
    strategy: str
    tp_level: str
    entry_time: str
    exit_time: str
    exit_reason: str
    credit: float
    exit_debit: float
    exit_slip: float
    fees: float
    pnl_per_share: float
    qty: int
    pnl_total: float
    portfolio_after: float


# ── Data Loading & Splitting ─────────────────────────────────────────────────

def load_model_table():
    """Load and concatenate all model_table parts."""
    parts = sorted(DATA_DIR.glob("model_table_v1_part*.parquet"))
    if not parts:
        # Try single file
        single = DATA_DIR / "model_table_v1.parquet"
        if single.exists():
            return pd.read_parquet(single)
        print("[error] No model_table files found.")
        sys.exit(1)

    print(f"Loading {len(parts)} model table parts...")
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime", "strategy"]).reset_index(drop=True)
    print(f"  {len(df):,} rows × {df.shape[1]} cols, {df['date'].nunique()} days")
    return df


def apply_blockers(df):
    """Apply hard blockers (rules-based)."""
    blocked = (
        (df["days_to_next_fomc"] == 0) |
        (df["days_to_next_fomc"] == 1) |
        (df["is_cpi_day"] == 1) |
        (df["is_ppi_day"] == 1) |
        (df["is_nfp_day"] == 1) |
        (df["is_gdp_day"] == 1) |
        (df["is_mag7_earnings_day"] == 1) |
        (df["vix_close"] > 35) |
        (df["gap_pct"].abs() > 1.5)
    ).fillna(False)
    df["_blocked"] = blocked
    n_blocked = blocked.sum()
    print(f"  Hard blockers: {n_blocked:,} rows blocked ({df[blocked]['date'].nunique()} days)")
    return df


def split_data(df):
    """Reproduce train/test/holdout split from train_v1.py."""
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * TRAIN_PCT)
    remaining = n_dates - train_end_idx - 2 * GAP_DAYS
    test_days = remaining // 2
    test_end_idx = train_end_idx + GAP_DAYS + test_days
    holdout_start_idx = test_end_idx + GAP_DAYS

    train_dates = set(dates[:train_end_idx])
    test_dates = set(dates[train_end_idx + GAP_DAYS:test_end_idx])
    holdout_dates = set(dates[holdout_start_idx:])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    holdout_df = df[df["date"].isin(holdout_dates)].reset_index(drop=True)

    print(f"  Train:   {len(train_df):>10,} rows ({len(train_dates):>3d} days)  "
          f"{min(train_dates).date()} → {max(train_dates).date()}")
    print(f"  Test:    {len(test_df):>10,} rows ({len(test_dates):>3d} days)")
    print(f"  Holdout: {len(holdout_df):>10,} rows ({len(holdout_dates):>3d} days)  "
          f"{min(holdout_dates).date()} → {max(holdout_dates).date()}")

    return train_df, test_df, holdout_df


def get_feature_cols(df):
    """Get numeric feature columns (exclude meta)."""
    all_feat = [c for c in df.columns if c not in META_COLS]
    return [c for c in all_feat if pd.api.types.is_numeric_dtype(df[c])]


# ── Model Training ───────────────────────────────────────────────────────────

def train_models(train_df, test_df, feature_cols, use_optuna=True, n_trials=30):
    """Train 9 XGBoost models (one per TP level). Returns dict of models."""
    import xgboost as xgb

    models = {}

    for tp in TP_LEVELS:
        target = f"{tp}_target"
        if target not in train_df.columns:
            continue

        y_train = train_df[target].astype(int)
        y_test = test_df[target].astype(int)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": pos_weight,
            "random_state": 42,
            "eval_metric": "auc",
            "verbosity": 0,
            "early_stopping_rounds": 30,
        }

        if use_optuna:
            try:
                best_params = _optuna_tune(
                    X_train, y_train, X_test, y_test,
                    pos_weight, n_trials=n_trials,
                )
                params.update(best_params)
            except Exception as e:
                print(f"    [warn] Optuna failed for {tp}: {e}, using defaults")

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        from sklearn.metrics import roc_auc_score
        test_probs = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)

        models[tp] = model
        print(f"  {tp}: test AUC={test_auc:.4f}, pos_rate={y_train.mean():.1%}")

        # Save model
        artifact = {
            "model": model,
            "feature_cols": feature_cols,
            "target": target,
            "test_auc": test_auc,
        }
        model_file = MODELS_DIR / f"{target}_xgb.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(artifact, f)

    return models


def _optuna_tune(X_train, y_train, X_test, y_test, pos_weight, n_trials=30):
    """Bayesian tuning with subsampling for speed."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    TUNE_SAMPLE = 300_000
    if len(X_train) > TUNE_SAMPLE:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=TUNE_SAMPLE, random_state=42)
        idx, _ = next(sss.split(X_train, y_train))
        X_tune, y_tune = X_train.iloc[idx], y_train.iloc[idx]
    else:
        X_tune, y_tune = X_train, y_train

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "scale_pos_weight": pos_weight,
            "random_state": 42,
            "eval_metric": "auc",
            "verbosity": 0,
            "early_stopping_rounds": 30,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_tune, y_tune, eval_set=[(X_test, y_test)], verbose=False)
        return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── EV & Cutoff Pre-computation ──────────────────────────────────────────────

def precompute_ev_lookup(train_df, feature_cols):
    """Compute avg win/loss PnL per (strategy, TP) from training data.

    Returns: dict[(strategy, tp_level)] -> (avg_win_pnl, avg_loss_pnl, base_win_rate)
    All PnL values are per-share (credit - exit_debit).
    """
    ev_lookup = {}

    for strat in STRATEGIES:
        strat_mask = train_df["strategy"] == strat
        strat_df = train_df[strat_mask]
        if len(strat_df) == 0:
            continue

        for tp in TP_LEVELS:
            target = f"{tp}_target"
            pnl_col = f"{tp}_pnl"

            if target not in strat_df.columns:
                continue

            wins = strat_df[strat_df[target] == 1]
            losses = strat_df[strat_df[target] == 0]

            if len(wins) == 0 or len(losses) == 0:
                continue

            # PnL per share = credit - exit_debit (already in data as tp_pnl)
            avg_win = wins[pnl_col].mean()
            avg_loss = losses[pnl_col].mean()
            win_rate = len(wins) / len(strat_df)

            ev_lookup[(strat, tp)] = (avg_win, avg_loss, win_rate)

    print(f"  EV lookup: {len(ev_lookup)} (strategy, TP) combos")
    return ev_lookup


def compute_ev(ev_lookup, strategy, tp_level, prob, exit_slip):
    """Compute EV for a specific (strategy, TP) at a given model probability.

    EV = P × avg_win - (1-P) × |avg_loss| - fees - exit_slip
    Where P = model's predicted probability of TP hit.
    """
    key = (strategy, tp_level)
    if key not in ev_lookup:
        return None

    avg_win, avg_loss, _ = ev_lookup[key]

    # EV per share
    ev = prob * avg_win + (1 - prob) * avg_loss - FEES_PER_SHARE - exit_slip
    return ev


def precompute_skip_cutoffs(holdout_scored, skip_rates):
    """Compute probability cutoffs per (strategy, tp, skip_rate).

    Cutoffs are computed per (strategy, TP) cell so each cell's bottom N%
    gets filtered independently. Without this, strategies with naturally
    higher probabilities (e.g. IC_05d) almost never get filtered.

    Returns: dict[(strategy, tp_level, skip_rate)] -> cutoff_value
    """
    cutoffs = {}
    for strat in STRATEGIES:
        strat_mask = holdout_scored["strategy"] == strat
        strat_df = holdout_scored[strat_mask]
        if len(strat_df) == 0:
            continue
        for tp in TP_LEVELS:
            prob_col = f"prob_{tp}"
            if prob_col not in strat_df.columns:
                continue
            probs = strat_df[prob_col].values
            for sr in skip_rates:
                cutoffs[(strat, tp, sr)] = np.percentile(probs, sr * 100)

    return cutoffs


# ── Holdout Scoring ──────────────────────────────────────────────────────────

def score_holdout(holdout_df, models, feature_cols):
    """Score all holdout rows with 9 models."""
    X = holdout_df[feature_cols]
    scored = holdout_df.copy()

    for tp, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        scored[f"prob_{tp}"] = probs
        print(f"  Scored {tp}: mean={probs.mean():.3f}, std={probs.std():.3f}")

    return scored


# ── Simulation Engine ────────────────────────────────────────────────────────

def _get_candidates(day_df, entry_time, ev_lookup, cutoffs, skip_rate,
                    exit_slip, blocked_strategies=None, optimization="joint"):
    """Get eligible (strategy, TP) candidates at a given entry time.

    Returns list of (strategy, tp, ev, row_data) sorted by EV descending.
    """
    # Get rows at this entry time
    time_rows = day_df[day_df["datetime"] == entry_time]
    if len(time_rows) == 0:
        return []

    candidates = []
    for _, row in time_rows.iterrows():
        strat = row["strategy"]

        # Skip blocked strategies (already in a position)
        if blocked_strategies and strat in blocked_strategies:
            continue

        # Skip hard-blocked rows
        if row.get("_blocked", False):
            continue

        for tp in TP_LEVELS:
            prob_col = f"prob_{tp}"
            target = f"{tp}_target"
            if prob_col not in row.index:
                continue

            prob = row[prob_col]

            # Apply skip cutoff (per strategy × TP cell)
            cutoff_key = (strat, tp, skip_rate)
            if cutoff_key in cutoffs and prob < cutoffs[cutoff_key]:
                continue

            ev = compute_ev(ev_lookup, strat, tp, prob, exit_slip)
            if ev is None:
                continue

            # Get exit info
            exit_time_str = row.get(f"{tp}_exit_time")
            exit_reason = row.get(f"{tp}_exit_reason")
            exit_debit = row.get(f"{tp}_exit_debit")
            credit = row.get("credit")

            if pd.isna(exit_time_str) or pd.isna(exit_debit):
                continue

            exit_time = pd.Timestamp(exit_time_str)

            candidates.append({
                "strategy": strat,
                "tp_level": tp,
                "prob": prob,
                "ev": ev,
                "credit": credit,
                "exit_debit": exit_debit,
                "exit_time": exit_time,
                "exit_reason": exit_reason,
                "entry_time": entry_time,
            })

    # Sort by EV descending
    candidates.sort(key=lambda x: x["ev"], reverse=True)
    return candidates


def simulate_strategy1(day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                       portfolio, optimization="joint"):
    """Strategy 1: Best EV, Max Contracts.

    Pick best combo, enter all contracts. Re-enter after TP, stop after SL.
    """
    trades = []
    buying_power = int(portfolio // BUYING_POWER_PER_CONTRACT)
    if buying_power <= 0:
        return trades, portfolio

    entry_times = sorted(day_df["datetime"].unique())
    current_time_idx = 0  # Start at first available time

    while current_time_idx < len(entry_times):
        entry_time = entry_times[current_time_idx]

        candidates = _get_candidates(
            day_df, entry_time, ev_lookup, cutoffs, skip_rate,
            exit_slip, optimization=optimization,
        )

        if not candidates:
            current_time_idx += 1
            continue

        # Pick the best EV candidate
        best = candidates[0]
        qty = buying_power

        # Compute PnL
        pnl_per_share = (best["credit"] - best["exit_debit"]
                         - exit_slip - FEES_PER_SHARE)
        pnl_total = pnl_per_share * 100 * qty

        portfolio += pnl_total
        buying_power = int(portfolio // BUYING_POWER_PER_CONTRACT)

        trades.append(TradeRecord(
            day=str(entry_time.date()),
            strategy=best["strategy"],
            tp_level=best["tp_level"],
            entry_time=str(entry_time),
            exit_time=str(best["exit_time"]),
            exit_reason=best["exit_reason"],
            credit=best["credit"],
            exit_debit=best["exit_debit"],
            exit_slip=exit_slip,
            fees=FEES_PER_SHARE,
            pnl_per_share=pnl_per_share,
            qty=qty,
            pnl_total=pnl_total,
            portfolio_after=portfolio,
        ))

        # SL → done for the day
        if best["exit_reason"] in ("sl", "close_loss"):
            break

        # TP or close_win → re-enter at exit time
        if best["exit_reason"] in ("tp", "close_win"):
            # Find next available entry time >= exit_time
            exit_ts = best["exit_time"]
            found = False
            for i in range(current_time_idx + 1, len(entry_times)):
                if entry_times[i] >= exit_ts:
                    current_time_idx = i
                    found = True
                    break
            if not found:
                break
            # Update buying power for re-entry
            buying_power = int(portfolio // BUYING_POWER_PER_CONTRACT)
            if buying_power <= 0:
                break
        else:
            break

    return trades, portfolio


def simulate_strategy2(day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                       portfolio, optimization="joint"):
    """Strategy 2: Diversified, Max Buying Power.

    Distribute contracts across top-K combos. Replace TP exits, not SL.
    """
    trades = []
    total_bp = int(portfolio // BUYING_POWER_PER_CONTRACT)
    if total_bp <= 0:
        return trades, portfolio

    entry_times = sorted(day_df["datetime"].unique())

    # Active positions: list of (Position, expected_exit_time)
    active_positions = []
    available_bp = total_bp
    active_strategies = set()

    # Enter initial positions at first available time
    current_time_idx = 0

    def _enter_positions(entry_time, bp_available):
        """Enter positions using available buying power."""
        nonlocal available_bp, active_strategies

        candidates = _get_candidates(
            day_df, entry_time, ev_lookup, cutoffs, skip_rate,
            exit_slip, blocked_strategies=active_strategies,
            optimization=optimization,
        )

        if not candidates or bp_available <= 0:
            return []

        # Deduplicate by strategy (take best TP per strategy)
        seen_strats = set()
        unique_candidates = []
        for c in candidates:
            if c["strategy"] not in seen_strats:
                seen_strats.add(c["strategy"])
                unique_candidates.append(c)

        # Filter to positive EV only
        eligible = [c for c in unique_candidates if c["ev"] > 0]
        if not eligible:
            eligible = unique_candidates[:1]  # Take best if none positive

        if not eligible:
            return []

        # Allocate contracts
        k = min(len(eligible), bp_available)
        selected = eligible[:k]
        base = bp_available // k
        extra = bp_available % k

        new_positions = []
        for i, cand in enumerate(selected):
            qty = base + (1 if i < extra else 0)
            if qty <= 0:
                continue

            pos = Position(
                strategy=cand["strategy"],
                tp_level=cand["tp_level"],
                entry_time=entry_time,
                exit_time=cand["exit_time"],
                exit_reason=cand["exit_reason"],
                credit=cand["credit"],
                exit_debit=cand["exit_debit"],
                qty=qty,
                ev=cand["ev"],
            )
            new_positions.append(pos)
            active_strategies.add(cand["strategy"])
            available_bp -= qty

        return new_positions

    # Initial entry
    while current_time_idx < len(entry_times):
        new_pos = _enter_positions(entry_times[current_time_idx], available_bp)
        if new_pos:
            active_positions.extend(new_pos)
            break
        current_time_idx += 1

    if not active_positions:
        return trades, portfolio

    # Process exits in chronological order
    while active_positions:
        # Find earliest exit
        active_positions.sort(key=lambda p: p.exit_time)
        pos = active_positions.pop(0)
        active_strategies.discard(pos.strategy)

        # Record trade
        pnl_per_share = pos.credit - pos.exit_debit - exit_slip - FEES_PER_SHARE
        pnl_total = pnl_per_share * 100 * pos.qty

        portfolio += pnl_total
        available_bp += pos.qty  # Return buying power

        trades.append(TradeRecord(
            day=str(pos.entry_time.date()),
            strategy=pos.strategy,
            tp_level=pos.tp_level,
            entry_time=str(pos.entry_time),
            exit_time=str(pos.exit_time),
            exit_reason=pos.exit_reason,
            credit=pos.credit,
            exit_debit=pos.exit_debit,
            exit_slip=exit_slip,
            fees=FEES_PER_SHARE,
            pnl_per_share=pnl_per_share,
            qty=pos.qty,
            pnl_total=pnl_total,
            portfolio_after=portfolio,
        ))

        # TP → replace with new position
        if pos.exit_reason in ("tp", "close_win"):
            # Recalculate total available BP
            total_bp = int(portfolio // BUYING_POWER_PER_CONTRACT)
            # available_bp is already updated above; but cap it
            used_bp = sum(p.qty for p in active_positions)
            available_bp = min(available_bp, total_bp - used_bp)
            available_bp = max(available_bp, 0)

            if available_bp > 0:
                # Find entry time >= exit_time
                exit_ts = pos.exit_time
                for i, et in enumerate(entry_times):
                    if et >= exit_ts:
                        new_pos = _enter_positions(et, available_bp)
                        active_positions.extend(new_pos)
                        break

        # SL → don't replace, just continue monitoring others

    return trades, portfolio


# ── Full Simulation Loop ─────────────────────────────────────────────────────

def simulate(holdout_scored, strategy_type, exit_slip, skip_rate,
             ev_lookup, cutoffs, optimization="joint",
             start_portfolio=START_PORTFOLIO):
    """Run full simulation across all holdout days.

    Returns: (equity_curve, trade_log, summary_stats)
    """
    portfolio = start_portfolio
    all_trades = []
    equity_curve = [{"date": "start", "portfolio": portfolio}]

    holdout_days = sorted(holdout_scored["date"].unique())

    for day in holdout_days:
        day_df = holdout_scored[holdout_scored["date"] == day]

        if strategy_type == 1:
            day_trades, portfolio = simulate_strategy1(
                day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                portfolio, optimization,
            )
        else:
            day_trades, portfolio = simulate_strategy2(
                day_df, ev_lookup, cutoffs, skip_rate, exit_slip,
                portfolio, optimization,
            )

        all_trades.extend(day_trades)
        equity_curve.append({"date": str(day.date()), "portfolio": portfolio})

    # Compute summary stats
    summary = _compute_summary(
        equity_curve, all_trades, strategy_type, exit_slip,
        skip_rate, optimization, start_portfolio,
    )

    return equity_curve, all_trades, summary


def _compute_summary(equity_curve, trades, strategy_type, exit_slip,
                     skip_rate, optimization, start_portfolio):
    """Compute summary statistics for a simulation run."""
    if not trades:
        return {
            "strategy": strategy_type,
            "exit_slip": exit_slip,
            "skip_rate": skip_rate,
            "optimization": optimization,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "trades_per_day": 0.0,
            "final_portfolio": start_portfolio,
        }

    final_portfolio = equity_curve[-1]["portfolio"]
    total_return = (final_portfolio - start_portfolio) / start_portfolio

    # Max drawdown
    values = [e["portfolio"] for e in equity_curve]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Daily returns for Sharpe
    daily_vals = [e["portfolio"] for e in equity_curve[1:]]  # skip "start"
    if len(daily_vals) > 1:
        daily_returns = []
        prev = start_portfolio
        for v in daily_vals:
            daily_returns.append((v - prev) / prev if prev > 0 else 0)
            prev = v
        daily_returns = np.array(daily_returns)
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                  if np.std(daily_returns) > 0 else 0.0)
    else:
        sharpe = 0.0

    # Win rate
    wins = sum(1 for t in trades if t.pnl_total > 0)
    win_rate = wins / len(trades) if trades else 0.0

    n_days = len(equity_curve) - 1  # exclude "start"

    return {
        "strategy": strategy_type,
        "exit_slip": exit_slip,
        "skip_rate": skip_rate,
        "optimization": optimization,
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
        "trades_per_day": round(len(trades) / max(n_days, 1), 2),
        "final_portfolio": round(final_portfolio, 2),
        "avg_pnl": round(np.mean([t.pnl_total for t in trades]), 2),
    }


# ── Walkthrough (1-day trace) ────────────────────────────────────────────────

def run_walkthrough(holdout_scored, ev_lookup, cutoffs, skip_rate=0.20,
                    exit_slip=0.20):
    """Produce hand-traced walkthrough on 1 holdout day."""
    day = sorted(holdout_scored["date"].unique())[0]
    day_df = holdout_scored[holdout_scored["date"] == day].copy()

    print(f"\n{'='*80}")
    print(f"  WALKTHROUGH — {day.date()}")
    print(f"  Skip rate: {skip_rate:.0%}  |  Exit slip: ${exit_slip:.2f}")
    print(f"  Portfolio: ${START_PORTFOLIO:,.2f}  |  "
          f"BP: {int(START_PORTFOLIO // BUYING_POWER_PER_CONTRACT)} contracts")
    print(f"{'='*80}")

    # Show available combos at first entry time
    entry_time = sorted(day_df["datetime"].unique())[0]
    print(f"\n── Available combos at {entry_time} ──\n")

    candidates = _get_candidates(
        day_df, entry_time, ev_lookup, cutoffs, skip_rate, exit_slip,
    )

    print(f"  {'Strategy':>14s}  {'TP':>4s}  {'Prob':>6s}  {'EV':>8s}  "
          f"{'Credit':>7s}  {'ExitDeb':>7s}  {'ExitRsn':>8s}  {'ExitTime':>8s}")
    print(f"  {'-'*80}")
    for c in candidates[:15]:
        print(f"  {c['strategy']:>14s}  {c['tp_level']:>4s}  {c['prob']:.4f}  "
              f"${c['ev']:>+7.4f}  ${c['credit']:>6.2f}  ${c['exit_debit']:>6.2f}  "
              f"{c['exit_reason']:>8s}  {str(c['exit_time'])[11:16]:>8s}")
    if len(candidates) > 15:
        print(f"  ... and {len(candidates) - 15} more")

    print(f"\n  Total eligible: {len(candidates)}")

    # Run Strategy 1
    print(f"\n── Strategy 1: Best EV, Max Contracts ──\n")
    portfolio = START_PORTFOLIO
    trades1, final1 = simulate_strategy1(
        day_df, ev_lookup, cutoffs, skip_rate, exit_slip, portfolio,
    )
    for t in trades1:
        print(f"  ENTER {t.strategy} × {t.tp_level} @ {t.entry_time[11:16]}  "
              f"qty={t.qty}")
        print(f"    credit=${t.credit:.2f}  exit_debit=${t.exit_debit:.2f}  "
              f"slip=${t.exit_slip:.2f}  fees=${t.fees:.3f}")
        print(f"    pnl/share=${t.pnl_per_share:+.3f}  "
              f"total=${t.pnl_total:+,.2f}  "
              f"exit={t.exit_reason} @ {t.exit_time[11:16]}")
        print(f"    portfolio: ${t.portfolio_after:,.2f}")
    if not trades1:
        print("  No trades taken.")
    print(f"  End of day portfolio: ${final1:,.2f}")

    # Run Strategy 2
    print(f"\n── Strategy 2: Diversified, Max Buying Power ──\n")
    portfolio = START_PORTFOLIO
    trades2, final2 = simulate_strategy2(
        day_df, ev_lookup, cutoffs, skip_rate, exit_slip, portfolio,
    )
    for t in trades2:
        print(f"  {'ENTER' if t.pnl_total == 0 else 'CLOSE'} "
              f"{t.strategy} × {t.tp_level}  "
              f"entry={t.entry_time[11:16]}  exit={t.exit_time[11:16]}  "
              f"qty={t.qty}  {t.exit_reason}")
        print(f"    credit=${t.credit:.2f}  exit=${t.exit_debit:.2f}  "
              f"pnl=${t.pnl_total:+,.2f}  portfolio=${t.portfolio_after:,.2f}")
    if not trades2:
        print("  No trades taken.")
    print(f"  End of day portfolio: ${final2:,.2f}")

    return trades1, trades2


# ── Output ───────────────────────────────────────────────────────────────────

def print_summary_table(summaries):
    """Print formatted summary table."""
    print(f"\n{'='*110}")
    print(f"  BACKTEST RESULTS SUMMARY")
    print(f"{'='*110}\n")

    print(f"  {'Strat':>5s}  {'Slip':>5s}  {'Skip':>5s}  {'Opt':>6s}  "
          f"{'Return%':>8s}  {'MaxDD%':>7s}  {'Sharpe':>7s}  "
          f"{'WinRate':>7s}  {'Trades':>7s}  {'T/Day':>6s}  "
          f"{'Final$':>10s}  {'AvgPnL':>8s}")
    print(f"  {'-'*105}")

    for s in sorted(summaries, key=lambda x: (x["strategy"], x["exit_slip"],
                                               x["skip_rate"])):
        print(f"  S{s['strategy']:>4d}  ${s['exit_slip']:.2f}  "
              f"{s['skip_rate']:>4.0%}  {s['optimization']:>6s}  "
              f"{s['total_return_pct']:>+7.1f}%  {s['max_drawdown_pct']:>6.1f}%  "
              f"{s['sharpe']:>7.2f}  {s['win_rate']:>6.1f}%  "
              f"{s['n_trades']:>7,}  {s['trades_per_day']:>5.1f}  "
              f"${s['final_portfolio']:>9,.0f}  ${s.get('avg_pnl', 0):>+7.1f}")


def save_results(summaries, all_trades, equity_curves):
    """Save results to CSV files."""
    # Summary table
    summary_df = pd.DataFrame(summaries)
    summary_file = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n  Saved summary:  {summary_file}")

    # Trade log
    if all_trades:
        trade_records = []
        for key, trades in all_trades.items():
            for t in trades:
                trade_records.append({
                    "sim_key": key,
                    "day": t.day,
                    "strategy": t.strategy,
                    "tp_level": t.tp_level,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_reason": t.exit_reason,
                    "credit": t.credit,
                    "exit_debit": t.exit_debit,
                    "exit_slip": t.exit_slip,
                    "fees": t.fees,
                    "pnl_per_share": t.pnl_per_share,
                    "qty": t.qty,
                    "pnl_total": t.pnl_total,
                    "portfolio_after": t.portfolio_after,
                })
        trade_df = pd.DataFrame(trade_records)
        trade_file = OUTPUT_DIR / "trade_log.csv"
        trade_df.to_csv(trade_file, index=False)
        print(f"  Saved trades:   {trade_file}")

    # Equity curves
    if equity_curves:
        eq_records = []
        for key, curve in equity_curves.items():
            for point in curve:
                eq_records.append({"sim_key": key, **point})
        eq_df = pd.DataFrame(eq_records)
        eq_file = OUTPUT_DIR / "equity_curves.csv"
        eq_df.to_csv(eq_file, index=False)
        print(f"  Saved equity:   {eq_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZDOM V1 Walk-Forward Backtest")
    parser.add_argument("--walkthrough", action="store_true",
                        help="Run 1-day walkthrough only (for verification)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna tuning (faster, default params)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per model (default: 30)")
    parser.add_argument("--strategy", type=int, choices=[1, 2], default=None,
                        help="Run only one strategy (default: both)")
    parser.add_argument("--skip-rates", nargs="*", type=float, default=None,
                        help="Skip rates to test (default: 0.20 to 0.40 in 1%% increments)")
    parser.add_argument("--slips", nargs="*", type=float, default=None,
                        help="Exit slippage values (default: 0.00 to 0.30 in $0.05 increments)")
    parser.add_argument("--portfolio", type=float, default=START_PORTFOLIO,
                        help=f"Starting portfolio (default: ${START_PORTFOLIO:,})")
    args = parser.parse_args()

    skip_rates = args.skip_rates or SKIP_RATES
    exit_slips = args.slips or EXIT_SLIPS
    strategies = [args.strategy] if args.strategy else [1, 2]

    print(f"\n{'='*80}")
    print(f"  ZDOM V1 Walk-Forward Backtest Simulator")
    print(f"{'='*80}")
    print(f"  Strategies:  {strategies}")
    print(f"  Slippage:    {exit_slips}")
    print(f"  Skip rates:  {skip_rates}")
    print(f"  Portfolio:   ${args.portfolio:,.2f}")
    print(f"  Tuning:      {'OFF' if args.no_tune else f'Optuna ({args.n_trials} trials)'}")

    # Phase 1: Load data
    print(f"\n── Phase 1: Data Loading ──")
    df = load_model_table()
    df = apply_blockers(df)
    train_df, test_df, holdout_df = split_data(df)
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    # Phase 2: Train models
    print(f"\n── Phase 2: Training Models ──")
    t0 = time.time()

    # Load best pre-trained model per TP (compare XGB vs LGBM holdout AUC)
    existing_models = {}
    for tp in TP_LEVELS:
        target = f"{tp}_target"
        best_model, best_auc, best_algo = None, -1, None

        for algo in ["xgb", "lgbm"]:
            model_file = MODELS_DIR / f"{target}_{algo}.pkl"
            summary_file = MODELS_DIR / f"{target}_{algo}_summary.json"
            if not model_file.exists():
                continue

            # Get holdout AUC from summary if available
            ho_auc = 0
            if summary_file.exists():
                import json
                with open(summary_file) as f:
                    ho_auc = json.load(f).get("holdout_auc", 0)

            if ho_auc > best_auc:
                with open(model_file, "rb") as f:
                    artifact = pickle.load(f)
                best_model = artifact["model"]
                best_auc = ho_auc
                best_algo = algo

        if best_model is not None:
            existing_models[tp] = best_model
            print(f"  {tp}: loaded {best_algo.upper()} (holdout AUC={best_auc:.4f})")

    if len(existing_models) == len(TP_LEVELS):
        models = existing_models
    else:
        print(f"  Only found {len(existing_models)}/{len(TP_LEVELS)} models, training missing...")
        models = train_models(
            train_df, test_df, feature_cols,
            use_optuna=not args.no_tune,
            n_trials=args.n_trials,
        )

    train_time = time.time() - t0
    print(f"  Training complete: {train_time:.0f}s")

    # Pre-compute EV lookup from training data
    print(f"\n── Phase 3: Pre-computing EV & Cutoffs ──")
    ev_lookup = precompute_ev_lookup(train_df, feature_cols)

    # Score holdout
    print(f"\n── Phase 4: Scoring Holdout ──")
    holdout_scored = score_holdout(holdout_df, models, feature_cols)

    # Pre-compute skip cutoffs
    cutoffs = precompute_skip_cutoffs(holdout_scored, skip_rates)
    print(f"  Cutoffs computed for {len(cutoffs)} (strategy, TP, skip_rate) combos")

    # Walkthrough mode
    if args.walkthrough:
        run_walkthrough(holdout_scored, ev_lookup, cutoffs)
        return

    # Phase 5: Run all simulations
    print(f"\n── Phase 5: Running Simulations ──")
    total_sims = len(strategies) * len(exit_slips) * len(skip_rates)
    print(f"  {total_sims} simulation configs")

    all_summaries = []
    all_trade_logs = {}
    all_equity_curves = {}

    sim_count = 0
    t0 = time.time()

    for strat in strategies:
        for slip in exit_slips:
            for sr in skip_rates:
                sim_count += 1
                key = f"S{strat}_slip{slip:.2f}_skip{sr:.2f}"

                eq, trades, summary = simulate(
                    holdout_scored, strat, slip, sr,
                    ev_lookup, cutoffs,
                    optimization="joint",
                    start_portfolio=args.portfolio,
                )

                all_summaries.append(summary)
                all_trade_logs[key] = trades
                all_equity_curves[key] = eq

                # Progress
                if sim_count % 20 == 0 or sim_count == total_sims:
                    elapsed = time.time() - t0
                    print(f"  [{sim_count}/{total_sims}] {key}: "
                          f"return={summary['total_return_pct']:+.1f}%, "
                          f"trades={summary['n_trades']}, "
                          f"sharpe={summary['sharpe']:.2f}  "
                          f"({elapsed:.0f}s)")

    sim_time = time.time() - t0
    print(f"\n  Simulations complete: {sim_time:.0f}s")

    # Output
    print_summary_table(all_summaries)
    save_results(all_summaries, all_trade_logs, all_equity_curves)

    print(f"\n  Done. Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
