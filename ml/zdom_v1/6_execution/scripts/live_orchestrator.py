"""
ZDOM V1 Live Orchestrator — Paper Trading on Tradier Sandbox.

Data collection starts at 9:30 ET (1-min bars via ThetaData for ORB + features).
Scoring + execution loop runs from 10:00-15:00 ET.
One trade at a time. 5-delta shadow filter. Tracks internal $10K portfolio.

Usage:
  python3 execution/scripts/live_orchestrator.py                  # run live
  python3 execution/scripts/live_orchestrator.py --dry-run        # score only, no orders
  python3 execution/scripts/live_orchestrator.py --skip-rate 0.30 # override skip rate
"""

import argparse
import json
import os
import pickle
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from live_features import LiveFeatureBuilder

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_DIR / "models" / "v1"
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "execution" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────

TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"
TRADIER_TOKEN = os.environ.get("TRADIER_PAPER_TOKEN", "YOUR_TRADIER_TOKEN_HERE")
TRADIER_ACCOUNT_ID = os.environ.get("TRADIER_ACCOUNT_ID", "YOUR_ACCOUNT_ID_HERE")

PORTFOLIO_START = 10_000
BUYING_POWER_PER_CONTRACT = 2500
FEES_PER_SHARE = 0.052  # $5.20 RT / 100 multiplier
EXIT_SLIP = 0.10
DEFAULT_SKIP_RATE = 0.30

STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(5, 50, 5)]
TRADEABLE_STRATEGIES = [f"IC_{d:02d}d_25w" for d in range(10, 50, 5)]
TP_LEVELS = [f"tp{p}" for p in range(10, 55, 5)]

SL_MULT = 2.0
DATA_COLLECTION_START_HOUR = 9
DATA_COLLECTION_START_MIN = 30
ENTRY_START_HOUR = 10
ENTRY_START_MIN = 0
CLOSE_HOUR = 15
CLOSE_MIN = 0

THETADATA_BASE_URL = "http://127.0.0.1:25503"

# ── ThetaData API ───────────────────────────────────────────────────────────

def _parse_thetadata_response(data):
    """Parse ThetaData JSON response into a flat list of bar dicts.

    ThetaData v3 can return either:
      - {"response": [{...bar...}, ...]}
      - {"response": [{"data": [{...bar...}, ...]}]}
    This handles both formats.
    """
    response = data.get("response", data if isinstance(data, list) else [])
    if response and isinstance(response[0], dict) and "data" in response[0]:
        bars = []
        for entry in response:
            for dp in entry.get("data", []):
                bars.append(dp)
        return bars
    return response


def fetch_thetadata_bars(date_str, start_time=None, end_time=None):
    """Fetch 1-min SPX bars from ThetaData for a given date.

    Args:
        date_str: Date in YYYY-MM-DD format.
        start_time: Optional start time as "HH:MM:SS".
        end_time: Optional end time as "HH:MM:SS".

    Returns:
        List of bar dicts with keys: datetime, open, high, low, close.
        Returns empty list on failure.
    """
    try:
        date_compact = date_str.replace("-", "")
        params = {
            "symbol": "SPX",
            "start_date": date_compact,
            "end_date": date_compact,
            "interval": "1m",
            "format": "json",
        }
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{THETADATA_BASE_URL}/v3/index/history/ohlc?{query}"

        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        response = _parse_thetadata_response(data)
        if not response:
            return []

        bars = []
        for bar_data in response:
            ts = bar_data.get("timestamp") or bar_data.get("datetime")
            bars.append({
                "datetime": pd.Timestamp(ts) if ts else datetime.now(),
                "open": float(bar_data.get("open", bar_data.get("Open", 0))),
                "high": float(bar_data.get("high", bar_data.get("High", 0))),
                "low": float(bar_data.get("low", bar_data.get("Low", 0))),
                "close": float(bar_data.get("close", bar_data.get("Close", 0))),
            })
        return bars
    except Exception as e:
        print(f"  [warn] ThetaData fetch failed: {e}")
        return []


def fetch_thetadata_latest_bar(now):
    """Fetch the T-1 completed 1-min SPX bar from ThetaData.

    Uses the Universal T-1 rule: fetches the bar that just completed,
    not the one currently in progress.

    Args:
        now: Current datetime.

    Returns:
        Bar dict or None on failure.
    """
    try:
        # T-1: fetch the bar that ended 1 minute ago
        bar_end = now.replace(second=0, microsecond=0)
        bar_start = bar_end - timedelta(minutes=1)

        date_str = now.strftime("%Y-%m-%d")
        start_time = bar_start.strftime("%H:%M:%S")
        end_time = bar_end.strftime("%H:%M:%S")

        bars = fetch_thetadata_bars(date_str, start_time=start_time, end_time=end_time)
        if bars:
            return bars[-1]  # last completed bar
        return None
    except Exception:
        return None


# ── Tradier API ──────────────────────────────────────────────────────────────

def tradier_headers():
    return {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}


def get_spx_quote():
    """Get current SPX price."""
    r = requests.get(f"{TRADIER_BASE_URL}/markets/quotes",
                     headers=tradier_headers(), params={"symbols": "SPX"})
    if r.status_code == 200:
        q = r.json().get("quotes", {}).get("quote", {})
        return {
            "last": q.get("last"),
            "bid": q.get("bid"),
            "ask": q.get("ask"),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
        }
    return None


def get_option_chain(expiration):
    """Get SPX 0DTE option chain with greeks."""
    r = requests.get(f"{TRADIER_BASE_URL}/markets/options/chains",
                     headers=tradier_headers(),
                     params={"symbol": "SPX", "expiration": expiration, "greeks": "true"})
    if r.status_code == 200:
        chain = r.json().get("options", {}).get("option", [])
        return chain
    return []


def build_ic_from_chain(chain, target_delta, wing_width=25):
    """Build an IC from the live option chain at target delta."""
    calls = {o["strike"]: o for o in chain if o.get("option_type") == "call" and o.get("bid", 0) > 0}
    puts = {o["strike"]: o for o in chain if o.get("option_type") == "put" and o.get("bid", 0) > 0}

    if not calls or not puts:
        return None

    # Find short call closest to target delta
    sc = min(calls.values(), key=lambda o: abs(o.get("greeks", {}).get("delta", 0) - target_delta)
             if o.get("greeks") else 999)
    # Find short put closest to -target delta
    sp = min(puts.values(), key=lambda o: abs(o.get("greeks", {}).get("delta", 0) + target_delta)
             if o.get("greeks") else 999)

    if not sc or not sp:
        return None

    sc_strike = sc["strike"]
    sp_strike = sp["strike"]
    lc_strike = sc_strike + wing_width
    lp_strike = sp_strike - wing_width

    lc = calls.get(lc_strike)
    lp = puts.get(lp_strike)

    if not lc or not lp:
        return None

    # Credit at mid
    credit = ((sc["bid"] + sc["ask"]) / 2 + (sp["bid"] + sp["ask"]) / 2 -
              (lc["bid"] + lc["ask"]) / 2 - (lp["bid"] + lp["ask"]) / 2)

    if credit <= 0:
        return None

    return {
        "sc": sc, "sp": sp, "lc": lc, "lp": lp,
        "sc_strike": sc_strike, "sp_strike": sp_strike,
        "lc_strike": lc_strike, "lp_strike": lp_strike,
        "credit": round(credit, 2),
        "sc_delta": sc.get("greeks", {}).get("delta", 0),
        "sp_delta": sp.get("greeks", {}).get("delta", 0),
    }


def place_ic_order(ic, qty=1):
    """Place a multileg IC order on Tradier."""
    order_data = {
        "class": "multileg",
        "symbol": "SPX",
        "type": "credit",
        "duration": "day",
        "price": ic["credit"],
        "option_symbol[0]": ic["sc"]["symbol"],
        "side[0]": "sell_to_open",
        "quantity[0]": qty,
        "option_symbol[1]": ic["lc"]["symbol"],
        "side[1]": "buy_to_open",
        "quantity[1]": qty,
        "option_symbol[2]": ic["sp"]["symbol"],
        "side[2]": "sell_to_open",
        "quantity[2]": qty,
        "option_symbol[3]": ic["lp"]["symbol"],
        "side[3]": "buy_to_open",
        "quantity[3]": qty,
    }
    r = requests.post(f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/orders",
                      headers=tradier_headers(), data=order_data)
    if r.status_code == 200:
        return r.json().get("order", {})
    return {"error": r.text}


def get_order_status(order_id):
    """Check order fill status."""
    r = requests.get(f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/orders/{order_id}",
                     headers=tradier_headers())
    if r.status_code == 200:
        return r.json().get("order", {})
    return {}


def get_positions():
    """Get current open positions."""
    r = requests.get(f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/positions",
                     headers=tradier_headers())
    if r.status_code == 200:
        pos = r.json().get("positions", {})
        if pos and pos != "null":
            p = pos.get("position", [])
            return p if isinstance(p, list) else [p]
    return []


def close_ic_order(ic_symbols, debit_price, qty=1):
    """Place a closing order for an IC at debit + $0.10 tolerance."""
    # We're willing to pay up to mid + $0.10 to get out
    close_price = round(debit_price + EXIT_SLIP, 2)
    order_data = {
        "class": "multileg",
        "symbol": "SPX",
        "type": "debit",
        "duration": "day",
        "price": close_price,
        "option_symbol[0]": ic_symbols[0],
        "side[0]": "buy_to_close",
        "quantity[0]": qty,
        "option_symbol[1]": ic_symbols[1],
        "side[1]": "sell_to_close",
        "quantity[1]": qty,
        "option_symbol[2]": ic_symbols[2],
        "side[2]": "buy_to_close",
        "quantity[2]": qty,
        "option_symbol[3]": ic_symbols[3],
        "side[3]": "sell_to_close",
        "quantity[3]": qty,
    }
    r = requests.post(f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/orders",
                      headers=tradier_headers(), data=order_data)
    if r.status_code == 200:
        return r.json().get("order", {})
    return {"error": r.text}


# ── Model Scoring ────────────────────────────────────────────────────────────

def load_models():
    """Load best model per TP level."""
    models = {}
    for tp in range(10, 55, 5):
        best_auc = 0
        best_model = None
        best_algo = None
        best_features = None

        for algo in ["xgb", "lgbm"]:
            summary_file = MODELS_DIR / f"tp{tp}_target_{algo}_summary.json"
            model_file = MODELS_DIR / f"tp{tp}_target_{algo}.pkl"
            if not summary_file.exists() or not model_file.exists():
                continue
            with open(summary_file) as f:
                auc = json.load(f).get("holdout_auc", 0)
            if auc > best_auc:
                with open(model_file, "rb") as f:
                    art = pickle.load(f)
                best_model = art["model"]
                best_features = art["feature_cols"]
                best_auc = auc
                best_algo = algo

        if best_model:
            models[tp] = {"model": best_model, "features": best_features,
                          "algo": best_algo, "auc": best_auc}

    return models


def score_entry(models, feature_vector):
    """Score a single feature vector with all 9 models. Returns dict[tp] -> probability."""
    probs = {}
    for tp, info in models.items():
        X = feature_vector[info["features"]]
        prob = info["model"].predict_proba(X)[:, 1][0]
        probs[tp] = prob
    return probs


# ── Logging ──────────────────────────────────────────────────────────────────

def init_trade_log(log_path):
    """Initialize trade log CSV."""
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("timestamp,action,strategy,tp_level,credit,exit_debit,prob,ev,"
                    "qty,pnl_per_share,pnl_total,portfolio,order_id,shadow,notes\n")


def log_trade(log_path, **kwargs):
    """Append a trade record."""
    with open(log_path, "a") as f:
        fields = ["timestamp", "action", "strategy", "tp_level", "credit", "exit_debit",
                   "prob", "ev", "qty", "pnl_per_share", "pnl_total", "portfolio",
                   "order_id", "shadow", "notes"]
        values = [str(kwargs.get(k, "")) for k in fields]
        f.write(",".join(values) + "\n")


# ── Main Loop ────────────────────────────────────────────────────────────────

def run(args):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"trades_{today}.csv"
    init_trade_log(log_path)

    print(f"\n{'='*70}")
    print(f"  ZDOM V1 Live Orchestrator")
    print(f"  Date: {today}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'PAPER TRADING'}")
    print(f"  Portfolio: ${args.portfolio:,.2f}")
    print(f"  Skip rate: {args.skip_rate:.0%}")
    print(f"  Broker: Tradier Sandbox")
    print(f"{'='*70}\n")

    # Load models
    print("Loading models...")
    models = load_models()
    for tp, info in sorted(models.items()):
        print(f"  TP{tp}: {info['algo'].upper()} (AUC={info['auc']:.4f})")

    if len(models) < 9:
        print(f"[WARN] Only loaded {len(models)}/9 models")

    # Get feature columns from first model
    first_model = list(models.values())[0]
    feature_cols = first_model["features"]
    print(f"  Feature vector: {len(feature_cols)} features")

    # Initialize live feature builder
    print("Loading daily features...")
    fb = LiveFeatureBuilder()
    fb.load_daily_features()

    portfolio = args.portfolio
    position_open = False
    open_position = None
    last_quote_time = None
    thetadata_available = True  # assume available, set False on repeated failures

    # ── Data Collection Phase (9:30 - 10:00) ────────────────────────────────
    # Collect 1-min bars from 9:30 so the ORB and intraday features are
    # populated before the scoring loop starts at 10:00.
    data_collection_start = datetime.now().replace(
        hour=DATA_COLLECTION_START_HOUR, minute=DATA_COLLECTION_START_MIN, second=0)
    scoring_start = datetime.now().replace(
        hour=ENTRY_START_HOUR, minute=ENTRY_START_MIN, second=0)
    market_close_time = datetime.now().replace(
        hour=CLOSE_HOUR, minute=CLOSE_MIN, second=0)

    print(f"\nData collection starts at {DATA_COLLECTION_START_HOUR}:{DATA_COLLECTION_START_MIN:02d}")
    print(f"Scoring starts at {ENTRY_START_HOUR}:{ENTRY_START_MIN:02d}\n")

    # Wait until data collection start
    now = datetime.now()
    if now < data_collection_start:
        wait = (data_collection_start - now).seconds
        print(f"  Waiting {wait//60}m {wait%60}s until data collection start ({DATA_COLLECTION_START_HOUR}:{DATA_COLLECTION_START_MIN:02d})...")
        while datetime.now() < data_collection_start:
            remaining = (data_collection_start - datetime.now()).seconds
            if remaining > 60:
                time.sleep(60)
            else:
                time.sleep(1)

    # If we're past data collection start, backfill any bars we missed
    now = datetime.now()
    if now >= data_collection_start and now <= market_close_time:
        backfill_end = min(now, scoring_start)
        today_str = now.strftime("%Y-%m-%d")
        print(f"  [{now.strftime('%H:%M:%S')}] Backfilling bars from ThetaData (9:30 to {backfill_end.strftime('%H:%M')})...")

        bars = fetch_thetadata_bars(
            today_str,
            start_time="09:30:00",
            end_time=backfill_end.strftime("%H:%M:%S"),
        )
        if bars:
            for bar in bars:
                fb.add_bar(bar)
            print(f"  [{now.strftime('%H:%M:%S')}] Backfilled {len(bars)} bars from ThetaData")
        else:
            print(f"  [{now.strftime('%H:%M:%S')}] No ThetaData bars available — will use Tradier quotes as fallback")
            thetadata_available = False

    # Live data collection loop: accumulate bars from 9:30 until scoring starts
    now = datetime.now()
    if now < scoring_start and now >= data_collection_start:
        print(f"  [{now.strftime('%H:%M:%S')}] Collecting live bars until scoring starts at {ENTRY_START_HOUR}:{ENTRY_START_MIN:02d}...")

        while datetime.now() < scoring_start:
            now = datetime.now()
            # Wait until the next minute boundary + a few seconds for the bar to complete
            next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            sleep_secs = (next_min - now).total_seconds() + 3  # +3s for bar completion
            if sleep_secs > 0:
                time.sleep(sleep_secs)

            now = datetime.now()
            if now >= scoring_start:
                break

            bar = fetch_thetadata_latest_bar(now)
            if bar:
                fb.add_bar(bar)
                print(f"  [{now.strftime('%H:%M:%S')}] Bar #{len(fb.bars):>3d} | "
                      f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")
            elif thetadata_available:
                # ThetaData returned nothing — try a Tradier quote as fallback
                spx_quote = get_spx_quote()
                if spx_quote and spx_quote.get("last"):
                    fb.add_bar_from_quote(spx_quote)
                    print(f"  [{now.strftime('%H:%M:%S')}] Bar #{len(fb.bars):>3d} (Tradier fallback) | "
                          f"last={spx_quote['last']:.2f}")

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Data collection complete — {len(fb.bars)} bars accumulated")
        if fb.orb_range:
            print(f"  ORB: high={fb.orb_high:.2f} low={fb.orb_low:.2f} range={fb.orb_range:.2f}")

    # ── Main Scoring Loop ────────────────────────────────────────────────────
    print(f"\nStarting scoring loop...\n")

    while True:
        now = datetime.now()

        scoring_start = now.replace(hour=ENTRY_START_HOUR, minute=ENTRY_START_MIN, second=0)
        market_close = now.replace(hour=CLOSE_HOUR, minute=CLOSE_MIN, second=0)

        if now < scoring_start:
            wait = (scoring_start - now).seconds
            if wait > 60:
                print(f"  [{now.strftime('%H:%M:%S')}] Scoring starts in {wait//60}m {wait%60}s")
                time.sleep(min(wait - 30, 60))
                continue
            time.sleep(1)
            continue

        if now >= market_close:
            print(f"\n  [{now.strftime('%H:%M:%S')}] Market closed. Done for today.")
            break

        # If position is open, monitor it
        if position_open and open_position:
            # Check current debit to close
            chain = get_option_chain(today)
            if chain:
                # Get current mid prices for our strikes
                chain_by_sym = {o["symbol"]: o for o in chain}
                sc = chain_by_sym.get(open_position["sc_symbol"], {})
                sp = chain_by_sym.get(open_position["sp_symbol"], {})
                lc = chain_by_sym.get(open_position["lc_symbol"], {})
                lp = chain_by_sym.get(open_position["lp_symbol"], {})

                if all(sc.get("bid") for _ in [sc, sp, lc, lp]):
                    sc_mid = (sc.get("bid", 0) + sc.get("ask", 0)) / 2
                    sp_mid = (sp.get("bid", 0) + sp.get("ask", 0)) / 2
                    lc_mid = (lc.get("bid", 0) + lc.get("ask", 0)) / 2
                    lp_mid = (lp.get("bid", 0) + lp.get("ask", 0)) / 2
                    debit = (sc_mid + sp_mid) - (lc_mid + lp_mid)

                    credit = open_position["credit"]
                    tp_pct = open_position["tp_pct"]
                    sl_debit = credit * SL_MULT
                    tp_debit = credit * (1 - tp_pct)

                    pnl_unrealized = credit - debit

                    # SL check
                    if debit >= sl_debit:
                        pnl = credit - debit - EXIT_SLIP - FEES_PER_SHARE
                        pnl_total = pnl * 100 * open_position["qty"]
                        portfolio += pnl_total

                        # Place close order on Tradier
                        if not args.dry_run:
                            syms = [open_position["sc_symbol"], open_position["lc_symbol"],
                                    open_position["sp_symbol"], open_position["lp_symbol"]]
                            close_order = close_ic_order(syms, debit, open_position["qty"])
                            close_id = close_order.get("id", "N/A")
                        else:
                            close_id = "DRY_RUN"

                        print(f"  [{now.strftime('%H:%M:%S')}] SL HIT | debit=${debit:.2f} >= SL=${sl_debit:.2f} | "
                              f"PnL=${pnl_total:+,.0f} | Portfolio=${portfolio:,.0f} | close_order={close_id}")
                        log_trade(log_path, timestamp=now, action="SL_EXIT",
                                  strategy=open_position["strategy"], tp_level=open_position["tp_level"],
                                  credit=credit, exit_debit=round(debit, 4), pnl_per_share=round(pnl, 4),
                                  qty=open_position["qty"], pnl_total=round(pnl_total, 2),
                                  portfolio=round(portfolio, 2), order_id=close_id)
                        position_open = False
                        open_position = None
                        continue

                    # TP check
                    if debit <= tp_debit:
                        pnl = credit - debit - EXIT_SLIP - FEES_PER_SHARE
                        pnl_total = pnl * 100 * open_position["qty"]
                        portfolio += pnl_total

                        # Place close order on Tradier
                        if not args.dry_run:
                            syms = [open_position["sc_symbol"], open_position["lc_symbol"],
                                    open_position["sp_symbol"], open_position["lp_symbol"]]
                            close_order = close_ic_order(syms, debit, open_position["qty"])
                            close_id = close_order.get("id", "N/A")
                        else:
                            close_id = "DRY_RUN"

                        print(f"  [{now.strftime('%H:%M:%S')}] TP HIT | debit=${debit:.2f} <= TP=${tp_debit:.2f} | "
                              f"PnL=${pnl_total:+,.0f} | Portfolio=${portfolio:,.0f} | close_order={close_id}")
                        log_trade(log_path, timestamp=now, action="TP_EXIT",
                                  strategy=open_position["strategy"], tp_level=open_position["tp_level"],
                                  credit=credit, exit_debit=round(debit, 4), pnl_per_share=round(pnl, 4),
                                  qty=open_position["qty"], pnl_total=round(pnl_total, 2),
                                  portfolio=round(portfolio, 2), order_id=close_id)
                        position_open = False
                        open_position = None
                        continue

                    print(f"  [{now.strftime('%H:%M:%S')}] HOLDING {open_position['strategy']} x {open_position['tp_level']} | "
                          f"debit=${debit:.2f} | unrealized=${pnl_unrealized:+.2f} | "
                          f"TP=${tp_debit:.2f} SL=${sl_debit:.2f}")

            time.sleep(60)
            continue

        # No position open — score and potentially enter
        print(f"  [{now.strftime('%H:%M:%S')}] Scoring...", end="")

        # Fetch T-1 completed bar from ThetaData (preferred) or Tradier quote (fallback)
        bar = fetch_thetadata_latest_bar(now) if thetadata_available else None
        if bar:
            fb.add_bar(bar)
        else:
            spx_quote = get_spx_quote()
            if not spx_quote or not spx_quote.get("last"):
                print(" no SPX quote")
                time.sleep(60)
                continue
            fb.add_bar_from_quote(spx_quote)

        chain = get_option_chain(today)
        if not chain:
            print(" no chain available")
            time.sleep(60)
            continue

        # Build ICs for all 9 strategies and score with all 9 models
        delta_map = {f"IC_{d:02d}d_25w": d / 100 for d in range(5, 50, 5)}
        candidates = []

        for strat, delta in delta_map.items():
            ic = build_ic_from_chain(chain, delta)
            if ic is None:
                continue

            # Build full 284-feature vector for this strategy
            fv = fb.build_feature_vector(strat, ic["credit"], feature_cols)

            # Score with all 9 TP models
            for tp_num, model_info in models.items():
                tp_key = f"tp{tp_num}"
                tp_pct = tp_num / 100

                try:
                    prob = model_info["model"].predict_proba(fv[model_info["features"]])[:, 1][0]
                except Exception:
                    prob = 0.5

                ev = prob * ic["credit"]

                candidates.append({
                    "strategy": strat,
                    "tp_level": tp_key,
                    "tp_pct": tp_pct,
                    "prob": prob,
                    "ev": ev,
                    "credit": ic["credit"],
                    "ic": ic,
                })

        if not candidates:
            print(" no candidates")
            time.sleep(60)
            continue

        # Sort by EV descending
        candidates.sort(key=lambda x: x["ev"], reverse=True)
        best = candidates[0]

        # 5-delta shadow filter
        if best["strategy"] == "IC_05d_25w":
            print(f" SHADOW | best={best['strategy']} x {best['tp_level']} EV=${best['ev']:.2f} — skipping (5d filter)")
            log_trade(log_path, timestamp=now, action="SHADOW",
                      strategy=best["strategy"], tp_level=best["tp_level"],
                      credit=best["credit"], prob=round(best["prob"], 4),
                      ev=round(best["ev"], 4), shadow="true",
                      notes="5d_filter")
            time.sleep(60)
            continue

        # TODO: apply skip rate threshold (needs proper cutoffs from training)

        # Determine position size
        qty = max(1, int(portfolio // BUYING_POWER_PER_CONTRACT))

        print(f" ENTER | {best['strategy']} x {best['tp_level']} | "
              f"credit=${best['credit']:.2f} | EV=${best['ev']:.2f} | qty={qty}")

        if not args.dry_run:
            order = place_ic_order(best["ic"], qty=qty)
            order_id = order.get("id", "N/A")
            print(f"    Order placed: {order_id}")
        else:
            order_id = "DRY_RUN"
            print(f"    [DRY RUN] Would place order")

        log_trade(log_path, timestamp=now, action="ENTRY",
                  strategy=best["strategy"], tp_level=best["tp_level"],
                  credit=best["credit"], prob=round(best["prob"], 4),
                  ev=round(best["ev"], 4), qty=qty,
                  portfolio=round(portfolio, 2), order_id=order_id)

        # Track open position
        position_open = True
        open_position = {
            "strategy": best["strategy"],
            "tp_level": best["tp_level"],
            "tp_pct": best["tp_pct"],
            "credit": best["credit"],
            "qty": qty,
            "entry_time": now,
            "sc_symbol": best["ic"]["sc"]["symbol"],
            "sp_symbol": best["ic"]["sp"]["symbol"],
            "lc_symbol": best["ic"]["lc"]["symbol"],
            "lp_symbol": best["ic"]["lp"]["symbol"],
            "order_id": order_id,
        }

        time.sleep(60)

    # End of day — close any open position
    if position_open and open_position:
        print(f"\n  EOD: Closing open position {open_position['strategy']} x {open_position['tp_level']}")
        chain = get_option_chain(today)
        if chain:
            chain_by_sym = {o["symbol"]: o for o in chain}
            sc = chain_by_sym.get(open_position["sc_symbol"], {})
            sp = chain_by_sym.get(open_position["sp_symbol"], {})
            lc = chain_by_sym.get(open_position["lc_symbol"], {})
            lp = chain_by_sym.get(open_position["lp_symbol"], {})

            debit = ((sc.get("bid", 0) + sc.get("ask", 0)) / 2 +
                     (sp.get("bid", 0) + sp.get("ask", 0)) / 2 -
                     (lc.get("bid", 0) + lc.get("ask", 0)) / 2 -
                     (lp.get("bid", 0) + lp.get("ask", 0)) / 2)

            pnl = open_position["credit"] - debit - EXIT_SLIP - FEES_PER_SHARE
            pnl_total = pnl * 100 * open_position["qty"]
            portfolio += pnl_total

            # Place close order on Tradier
            if not args.dry_run:
                syms = [open_position["sc_symbol"], open_position["lc_symbol"],
                        open_position["sp_symbol"], open_position["lp_symbol"]]
                close_order = close_ic_order(syms, debit, open_position["qty"])
                close_id = close_order.get("id", "N/A")
            else:
                close_id = "DRY_RUN"

            result = "CLOSE_WIN" if pnl > 0 else "CLOSE_LOSS"
            print(f"  EOD: {result} | PnL=${pnl_total:+,.0f} | Portfolio=${portfolio:,.0f} | close_order={close_id}")
            log_trade(log_path, timestamp=datetime.now(), action=result,
                      strategy=open_position["strategy"], tp_level=open_position["tp_level"],
                      credit=open_position["credit"], exit_debit=round(debit, 4),
                      pnl_per_share=round(pnl, 4), qty=open_position["qty"],
                      pnl_total=round(pnl_total, 2), portfolio=round(portfolio, 2),
                      order_id=close_id)

    print(f"\n{'='*70}")
    print(f"  END OF DAY SUMMARY")
    print(f"  Final portfolio: ${portfolio:,.2f}")
    print(f"  Day P&L: ${portfolio - args.portfolio:+,.2f}")
    print(f"  Trade log: {log_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="ZDOM V1 Live Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Score only, don't place orders")
    parser.add_argument("--skip-rate", type=float, default=DEFAULT_SKIP_RATE, help="Skip rate (default: 0.30)")
    parser.add_argument("--portfolio", type=float, default=PORTFOLIO_START, help="Starting portfolio (default: $10,000)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
