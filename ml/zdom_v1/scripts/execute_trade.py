"""
Execute a trade signal on Tradier paper trading (sandbox) or live.

Reads signal from score_live.py output and submits a multi-leg options order
to Tradier. Supports iron condors, single spreads, and custom structures.

Credentials are ALWAYS loaded from environment files — never hardcoded.
  Dev/UAT: .env.development → TRADIER_PAPER_TOKEN + sandbox.tradier.com
  Prod:    .env.production  → TRADIER_ACCESS_TOKEN + api.tradier.com

RULE 16: Never use the live Tradier token outside prod.
         Always verify base URL before sending any order.

Usage:
  # Dry run — show what would be submitted (default, safe)
  python3 scripts/execute_trade.py --dry-run

  # Execute paper trade (reads signal from score_live or stdin)
  ENV=development python3 scripts/execute_trade.py

  # Execute with explicit signal JSON
  python3 scripts/execute_trade.py --signal '{"structure":"IC","spx_price":5820,"vix":18.5}'

  # Run in prod (requires explicit --env prod flag)
  python3 scripts/execute_trade.py --env prod

  # Check account balance
  python3 scripts/execute_trade.py --balance

  # View recent trade log
  python3 scripts/execute_trade.py --history

Output: logs/trade_log.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

warnings.filterwarnings("ignore")

# Load .env
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR    = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TRADE_LOG   = LOGS_DIR / "trade_log.csv"
ET_TZ       = ZoneInfo("America/New_York")

# ── Log schema ─────────────────────────────────────────────────────────────────
TRADE_LOG_COLUMNS = [
    "logged_at",
    "date",
    "env",
    "action",           # SUBMIT / CANCEL / FILL_CHECK / BALANCE_CHECK
    "structure",        # IC / SINGLE_PUT / SINGLE_CALL / etc.
    "spx_price",
    "vix_level",
    "short_put",
    "long_put",
    "short_call",
    "long_call",
    "spread_width",
    "quantity",
    "credit_limit",     # limit price (credit) for the order
    "order_id",
    "order_status",     # pending / filled / cancelled / rejected
    "fill_price",       # actual fill price (credit received)
    "account_id",
    "buying_power",
    "notes",
]

# ── SPX option symbol format ───────────────────────────────────────────────────
# Format: SPXW{YY}{MM}{DD}{C/P}{strike * 1000, zero-padded to 8 digits}
# Example: SPXW260306P05820000 = SPXW put, 2026-03-06, strike 5820

def format_spx_option_symbol(expiry: str, side: str, strike: float) -> str:
    """
    Build Tradier option symbol for SPX weekly (SPXW).

    Args:
        expiry: "YYYY-MM-DD"
        side:   "C" or "P"
        strike: strike price (e.g. 5820.0)
    Returns:
        e.g. "SPXW260306P05820000"
    """
    dt          = datetime.strptime(expiry, "%Y-%m-%d")
    yy          = dt.strftime("%y")
    mm          = dt.strftime("%m")
    dd          = dt.strftime("%d")
    strike_int  = int(round(strike * 1000))
    strike_str  = f"{strike_int:08d}"
    return f"SPXW{yy}{mm}{dd}{side}{strike_str}"


# ── Credential loading ─────────────────────────────────────────────────────────

ENVIRONMENTS = {
    "development": {
        "env_file":   ".env.development",
        "token_key":  "TRADIER_PAPER_TOKEN",
        "base_url":   "https://sandbox.tradier.com/v1",
        "label":      "PAPER (sandbox)",
    },
    "uat": {
        "env_file":   ".env.uat",
        "token_key":  "TRADIER_PAPER_TOKEN",
        "base_url":   "https://sandbox.tradier.com/v1",
        "label":      "UAT (sandbox)",
    },
    "prod": {
        "env_file":   ".env.production",
        "token_key":  "TRADIER_ACCESS_TOKEN",
        "base_url":   "https://api.tradier.com/v1",
        "label":      "LIVE (real money)",
    },
}


def load_credentials(env: str) -> dict:
    """
    Load Tradier credentials from the appropriate .env file.
    Never falls back to hardcoded values.
    """
    cfg      = ENVIRONMENTS[env]
    env_path = PROJECT_DIR / cfg["env_file"]

    if load_dotenv is None:
        raise RuntimeError("python-dotenv is required. Run: pip install python-dotenv")

    if not env_path.exists():
        raise FileNotFoundError(
            f"Env file not found: {env_path}\n"
            f"Create it with TRADIER credentials before trading."
        )

    load_dotenv(env_path, override=True)

    token      = os.getenv(cfg["token_key"])
    account_id = os.getenv("TRADIER_ACCOUNT_ID")
    base_url   = os.getenv("TRADIER_BASE_URL", cfg["base_url"])

    if not token:
        raise ValueError(
            f"{cfg['token_key']} not set in {env_path}.\n"
            f"Add it: {cfg['token_key']}=your_token_here"
        )
    if not account_id:
        raise ValueError(
            f"TRADIER_ACCOUNT_ID not set in {env_path}."
        )

    # Safety: prod check
    if env == "prod":
        sandbox_url = "sandbox.tradier.com"
        if sandbox_url in base_url:
            raise ValueError(
                f"PROD environment cannot use sandbox URL ({base_url}). "
                f"Check TRADIER_BASE_URL in {env_path}."
            )

    return {
        "token":      token,
        "account_id": account_id,
        "base_url":   base_url.rstrip("/"),
        "label":      cfg["label"],
        "env":        env,
    }


# ── Tradier API wrapper ────────────────────────────────────────────────────────

class TradierClient:
    """Thin wrapper around Tradier REST API."""

    def __init__(self, token: str, base_url: str, account_id: str, env: str):
        self.token      = token
        self.base_url   = base_url.rstrip("/")
        self.account_id = account_id
        self.env        = env
        self.session    = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        url  = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict) -> dict:
        url  = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.post(url, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_account_balances(self) -> dict:
        """GET /accounts/{id}/balances — returns account info + buying power."""
        return self._get(f"accounts/{self.account_id}/balances")

    def get_positions(self) -> list:
        """GET /accounts/{id}/positions — current open positions."""
        data = self._get(f"accounts/{self.account_id}/positions")
        positions = data.get("positions", {})
        if positions == "null" or not positions:
            return []
        pos = positions.get("position", [])
        return pos if isinstance(pos, list) else [pos]

    def get_orders(self) -> list:
        """GET /accounts/{id}/orders — all orders."""
        data   = self._get(f"accounts/{self.account_id}/orders")
        orders = data.get("orders", {})
        if not orders or orders == "null":
            return []
        o = orders.get("order", [])
        return o if isinstance(o, list) else [o]

    def get_order(self, order_id: str) -> dict:
        """GET /accounts/{id}/orders/{order_id}"""
        return self._get(f"accounts/{self.account_id}/orders/{order_id}")

    def get_options_chain(self, symbol: str, expiration: str) -> list:
        """GET /markets/options/chains — options chain for symbol + expiry."""
        data = self._get("markets/options/chains", params={
            "symbol":     symbol,
            "expiration": expiration,
            "greeks":     "true",
        })
        opts = data.get("options", {})
        if not opts:
            return []
        chain = opts.get("option", [])
        return chain if isinstance(chain, list) else [chain]

    def get_option_expirations(self, symbol: str) -> list:
        """GET /markets/options/expirations"""
        data = self._get("markets/options/expirations", params={
            "symbol": symbol,
            "includeAllRoots": "true",
        })
        exp = data.get("expirations", {})
        if not exp:
            return []
        dates = exp.get("date", [])
        return dates if isinstance(dates, list) else [dates]

    def place_multileg_order(self, legs: list[dict], order_type: str = "market",
                              duration: str = "day", price: float = None,
                              preview: bool = False) -> dict:
        """
        POST /accounts/{id}/orders — submit a multi-leg options order.

        Args:
            legs: list of dicts with keys:
                  option_symbol, side, quantity
                  side: sell_to_open / buy_to_open / sell_to_close / buy_to_close
            order_type: "limit" or "market" (use "limit" for real orders)
            duration:   "day" (0DTE — always use day orders)
            price:      limit price if order_type="limit" (net credit, positive = credit)
            preview:    if True, use class=multileg&preview=true (Tradier preview mode)

        Returns: Tradier order response dict
        """
        if order_type == "limit" and price is None:
            raise ValueError("price is required for limit orders")

        data = {
            "class":    "multileg",
            "symbol":   "SPXW",
            "type":     order_type,
            "duration": duration,
        }
        if order_type == "limit":
            data["price"] = f"{price:.2f}"
        if preview:
            data["preview"] = "true"

        for i, leg in enumerate(legs):
            data[f"legs[{i}][option_symbol]"] = leg["option_symbol"]
            data[f"legs[{i}][side]"]          = leg["side"]
            data[f"legs[{i}][quantity]"]       = str(leg["quantity"])

        return self._post(f"accounts/{self.account_id}/orders", data)

    def cancel_order(self, order_id: str) -> dict:
        """DELETE /accounts/{id}/orders/{order_id}"""
        url  = f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}"
        resp = self.session.delete(url, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ── Strike / credit helpers ────────────────────────────────────────────────────

def estimate_15delta_strikes(spx_price: float, vix_level: float,
                              spread_width: int = None) -> dict:
    """Same as shadow_trade_log.py — estimate 15-delta strikes."""
    sigma  = vix_level / 100.0
    T      = 1.0 / 252.0
    z      = 1.036
    sqrt_T = math.sqrt(T)

    short_put  = round(spx_price * math.exp(-z * sigma * sqrt_T) / 5) * 5
    short_call = round(spx_price * math.exp( z * sigma * sqrt_T) / 5) * 5
    short_put  = min(short_put,  int(spx_price) - 10)
    short_call = max(short_call, int(spx_price) + 10)

    if spread_width is None:
        spread_width = int(max(50, round(spx_price * 0.01 / 25) * 25))

    return {
        "short_put":    short_put,
        "long_put":     short_put  - spread_width,
        "short_call":   short_call,
        "long_call":    short_call + spread_width,
        "spread_width": spread_width,
    }


def find_best_strike_from_chain(chain: list, target_delta: float,
                                 option_type: str) -> dict | None:
    """
    Find the closest-to-target-delta strike from a live options chain.

    Args:
        chain:        list of option dicts (from Tradier /options/chains)
        target_delta: 0.15 for 15-delta
        option_type:  "put" or "call"
    Returns:
        option dict or None
    """
    candidates = [
        o for o in chain
        if o.get("option_type") == option_type
        and o.get("greeks") is not None
        and o.get("greeks", {}).get("delta") is not None
    ]
    if not candidates:
        return None

    # For puts: delta is negative, target is -0.15
    if option_type == "put":
        target = -abs(target_delta)
        best   = min(candidates,
                     key=lambda o: abs(float(o["greeks"]["delta"]) - target))
    else:
        target = abs(target_delta)
        best   = min(candidates,
                     key=lambda o: abs(float(o["greeks"]["delta"]) - target))

    return best


# ── Build IC legs ──────────────────────────────────────────────────────────────

def build_ic_legs(expiry: str, strikes: dict, quantity: int = 1) -> list[dict]:
    """
    Build the 4-leg IC order structure for Tradier.

    Iron condor:
      - Sell short_put  (sell_to_open)
      - Buy  long_put   (buy_to_open)
      - Sell short_call (sell_to_open)
      - Buy  long_call  (buy_to_open)
    """
    legs = [
        {
            "option_symbol": format_spx_option_symbol(expiry, "P", strikes["short_put"]),
            "side":          "sell_to_open",
            "quantity":      quantity,
        },
        {
            "option_symbol": format_spx_option_symbol(expiry, "P", strikes["long_put"]),
            "side":          "buy_to_open",
            "quantity":      quantity,
        },
        {
            "option_symbol": format_spx_option_symbol(expiry, "C", strikes["short_call"]),
            "side":          "sell_to_open",
            "quantity":      quantity,
        },
        {
            "option_symbol": format_spx_option_symbol(expiry, "C", strikes["long_call"]),
            "side":          "buy_to_open",
            "quantity":      quantity,
        },
    ]
    return legs


def build_single_put_legs(expiry: str, strikes: dict, quantity: int = 1) -> list[dict]:
    """Put credit spread (sell short_put, buy long_put)."""
    return [
        {
            "option_symbol": format_spx_option_symbol(expiry, "P", strikes["short_put"]),
            "side":          "sell_to_open",
            "quantity":      quantity,
        },
        {
            "option_symbol": format_spx_option_symbol(expiry, "P", strikes["long_put"]),
            "side":          "buy_to_open",
            "quantity":      quantity,
        },
    ]


def build_single_call_legs(expiry: str, strikes: dict, quantity: int = 1) -> list[dict]:
    """Call credit spread (sell short_call, buy long_call)."""
    return [
        {
            "option_symbol": format_spx_option_symbol(expiry, "C", strikes["short_call"]),
            "side":          "sell_to_open",
            "quantity":      quantity,
        },
        {
            "option_symbol": format_spx_option_symbol(expiry, "C", strikes["long_call"]),
            "side":          "buy_to_open",
            "quantity":      quantity,
        },
    ]


# ── Trade log ─────────────────────────────────────────────────────────────────

def init_trade_log():
    if not TRADE_LOG.exists():
        with open(TRADE_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS).writeheader()


def log_trade(row: dict):
    """Append a row to trade_log.csv."""
    init_trade_log()
    clean = {col: row.get(col, "") for col in TRADE_LOG_COLUMNS}
    with open(TRADE_LOG, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS).writerow(clean)


# ── Pre-flight checks ──────────────────────────────────────────────────────────

def check_buying_power(client: TradierClient,
                       required_capital: float) -> tuple[float, bool]:
    """
    Check Tradier account buying power.
    Returns (available_buying_power, is_sufficient).
    """
    try:
        data    = client.get_account_balances()
        account = data.get("balances", {})
        # Tradier balances response structure
        bp = (account.get("margin", {}).get("option_buying_power")
              or account.get("cash", {}).get("cash_available")
              or 0.0)
        bp = float(bp)
        return bp, bp >= required_capital
    except Exception as e:
        print(f"  [warn] could not fetch buying power: {e}")
        return 0.0, False


def get_current_spx_price(client: TradierClient) -> float | None:
    """Fetch live SPX price via Tradier quotes endpoint."""
    try:
        data  = client._get("markets/quotes", params={"symbols": "SPX", "greeks": "false"})
        quote = data.get("quotes", {}).get("quote", {})
        if isinstance(quote, list):
            quote = quote[0]
        price = float(quote.get("last") or quote.get("close") or 0)
        return price if price > 0 else None
    except Exception as e:
        print(f"  [warn] could not fetch SPX price from Tradier: {e}")
        return None


# ── Main execution flow ────────────────────────────────────────────────────────

def execute_trade(signal: dict, env: str, dry_run: bool = True,
                  quantity: int = 1, preview_only: bool = False) -> dict:
    """
    Core trade execution function.

    Args:
        signal:       dict from score_live.py with keys:
                      structure, spx_price, vix_level, short_put, long_put,
                      short_call, long_call, spread_width, credit_limit
        env:          "development" / "uat" / "prod"
        dry_run:      if True, log intent but don't call API
        quantity:     number of spreads
        preview_only: use Tradier preview mode (validates but doesn't fill)

    Returns:
        dict with order_id, status, fill_price, notes
    """
    today  = str(date.today())
    now_et = datetime.now(ET_TZ).isoformat()

    # ── Validate signal ──
    structure = signal.get("structure", "IC")
    spx_price = float(signal.get("spx_price") or 0)
    vix_level = float(signal.get("vix_level") or 20)

    if spx_price <= 0:
        print("[warn] SPX price not in signal — estimating from Tradier")
        spx_price = None  # will fetch live below

    # ── Load credentials ──
    print(f"\nLoading credentials for env={env}...")
    creds = load_credentials(env)
    print(f"  Environment: {creds['label']}")
    print(f"  Account:     {creds['account_id']}")
    print(f"  Base URL:    {creds['base_url']}")

    if env == "prod" and not dry_run:
        confirm = input("\n⚠️  LIVE TRADE — type 'CONFIRM' to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            sys.exit(0)

    client = TradierClient(
        token=creds["token"],
        base_url=creds["base_url"],
        account_id=creds["account_id"],
        env=env,
    )

    # ── Fetch live SPX price if needed ──
    if spx_price is None or spx_price <= 0:
        spx_price = get_current_spx_price(client) or 5800.0
        print(f"  Live SPX: {spx_price:.2f}")

    # ── Determine strikes ──
    if all(signal.get(k) for k in ["short_put", "long_put", "short_call", "long_call"]):
        strikes = {
            "short_put":    float(signal["short_put"]),
            "long_put":     float(signal["long_put"]),
            "short_call":   float(signal["short_call"]),
            "long_call":    float(signal["long_call"]),
            "spread_width": float(signal.get("spread_width", 100)),
        }
        print(f"  Using strikes from signal")
    else:
        strikes = estimate_15delta_strikes(spx_price, vix_level)
        print(f"  Using estimated 15-delta strikes (live chain not loaded)")

    print(f"\n  Structure: {structure}")
    print(f"  Strikes:   {strikes['long_put']} / {strikes['short_put']} / "
          f"{strikes['short_call']} / {strikes['long_call']}")
    print(f"  Width:     {strikes['spread_width']} pts")

    # ── Credit limit ──
    # Use signal's credit_limit if provided, else 0 (market order for paper)
    credit_limit = float(signal.get("credit_limit") or
                         signal.get("entry_credit_est") or 0)
    order_type   = "limit" if credit_limit > 0 else "market"
    print(f"  Order type: {order_type}" +
          (f" @ ${credit_limit:.2f} net credit" if credit_limit > 0 else ""))

    # ── Capital requirement ──
    # Max loss per IC = (spread_width - credit) * 100 per contract
    max_loss_per = (strikes["spread_width"] - credit_limit) * 100
    required_bp  = max_loss_per * quantity * 1.1  # 10% buffer
    print(f"  Required buying power: ~${required_bp:,.0f} (max loss: ${max_loss_per:,.0f}/contract)")

    # ── Pre-flight buying power check (skip in dry-run) ──
    buying_power = None
    if not dry_run:
        bp, sufficient = check_buying_power(client, required_bp)
        buying_power = bp
        print(f"  Account buying power: ${bp:,.0f}")
        if not sufficient:
            msg = (f"Insufficient buying power: ${bp:,.0f} available, "
                   f"${required_bp:,.0f} required")
            log_trade({
                "logged_at":    now_et,
                "date":         today,
                "env":          env,
                "action":       "SUBMIT",
                "structure":    structure,
                "spx_price":    round(spx_price, 2),
                "vix_level":    vix_level,
                **strikes,
                "quantity":     quantity,
                "credit_limit": credit_limit,
                "order_status": "REJECTED",
                "buying_power": bp,
                "notes":        msg,
            })
            print(f"\n  [error] {msg}")
            return {"status": "REJECTED", "notes": msg}

    # ── Build legs ──
    expiry = today  # 0DTE — expiry = today
    if structure == "IC":
        legs = build_ic_legs(expiry, strikes, quantity)
    elif structure in ("SINGLE_PUT", "PUT_SPREAD"):
        legs = build_single_put_legs(expiry, strikes, quantity)
    elif structure in ("SINGLE_CALL", "CALL_SPREAD"):
        legs = build_single_call_legs(expiry, strikes, quantity)
    else:
        # Default to IC
        legs = build_ic_legs(expiry, strikes, quantity)

    print(f"\n  Legs ({len(legs)}):")
    for leg in legs:
        print(f"    {leg['side']:20s} {leg['option_symbol']}")

    # ── Dry run — log and exit ──
    if dry_run:
        print(f"\n  [DRY RUN] Would submit {order_type} order — no API call made")
        log_trade({
            "logged_at":    now_et,
            "date":         today,
            "env":          env,
            "action":       "DRY_RUN",
            "structure":    structure,
            "spx_price":    round(spx_price, 2),
            "vix_level":    vix_level,
            **{k: v for k, v in strikes.items()},
            "quantity":     quantity,
            "credit_limit": credit_limit,
            "order_id":     "",
            "order_status": "DRY_RUN",
            "fill_price":   "",
            "account_id":   creds["account_id"],
            "notes":        f"Dry run — legs: {[l['option_symbol'] for l in legs]}",
        })
        return {"status": "DRY_RUN", "legs": legs}

    # ── Pre-submit log ──
    log_trade({
        "logged_at":    now_et,
        "date":         today,
        "env":          env,
        "action":       "SUBMIT",
        "structure":    structure,
        "spx_price":    round(spx_price, 2),
        "vix_level":    vix_level,
        **{k: v for k, v in strikes.items()},
        "quantity":     quantity,
        "credit_limit": credit_limit,
        "order_id":     "PENDING",
        "order_status": "SUBMITTING",
        "fill_price":   "",
        "account_id":   creds["account_id"],
        "buying_power": buying_power or "",
        "notes":        f"Submitting {structure} to Tradier {creds['label']}",
    })

    # ── Submit to Tradier ──
    print(f"\n  Submitting to Tradier {creds['label']}...")
    try:
        resp = client.place_multileg_order(
            legs        = legs,
            order_type  = order_type,
            duration    = "day",
            price       = credit_limit if order_type == "limit" else None,
            preview     = preview_only,
        )
        print(f"  Response: {json.dumps(resp, indent=2)}")

        order_resp   = resp.get("order", resp)
        order_id     = str(order_resp.get("id", ""))
        order_status = order_resp.get("status", "unknown")
        commission   = order_resp.get("commission", "")
        cost         = order_resp.get("cost", "")

        result = {
            "order_id":     order_id,
            "order_status": order_status,
            "fill_price":   cost,
            "commission":   commission,
        }

        # ── Post-submit log ──
        log_trade({
            "logged_at":    datetime.now(ET_TZ).isoformat(),
            "date":         today,
            "env":          env,
            "action":       "FILL_CHECK",
            "structure":    structure,
            "spx_price":    round(spx_price, 2),
            "vix_level":    vix_level,
            **{k: v for k, v in strikes.items()},
            "quantity":     quantity,
            "credit_limit": credit_limit,
            "order_id":     order_id,
            "order_status": order_status,
            "fill_price":   cost,
            "account_id":   creds["account_id"],
            "buying_power": buying_power or "",
            "notes":        f"commission={commission}" if commission else "",
        })

        print(f"\n  Order ID: {order_id}")
        print(f"  Status:   {order_status}")
        if commission:
            print(f"  Commission: ${commission}")
        return result

    except requests.exceptions.HTTPError as e:
        err_body = ""
        try:
            err_body = e.response.json()
        except Exception:
            err_body = str(e)

        msg = f"HTTP {e.response.status_code}: {err_body}"
        print(f"\n  [error] Order failed: {msg}")

        log_trade({
            "logged_at":    datetime.now(ET_TZ).isoformat(),
            "date":         today,
            "env":          env,
            "action":       "SUBMIT",
            "structure":    structure,
            "spx_price":    round(spx_price, 2),
            "vix_level":    vix_level,
            **{k: v for k, v in strikes.items()},
            "quantity":     quantity,
            "credit_limit": credit_limit,
            "order_status": "ERROR",
            "account_id":   creds["account_id"],
            "notes":        msg,
        })
        raise


def show_balance(env: str):
    """Print account balance + positions."""
    creds  = load_credentials(env)
    client = TradierClient(**{k: creds[k] for k in ["token", "base_url", "account_id", "env"]})

    print(f"\nAccount: {creds['account_id']} ({creds['label']})")
    print(f"{'='*50}")

    try:
        data    = client.get_account_balances()
        account = data.get("balances", {})
        print(f"  Account type: {account.get('account_type', 'N/A')}")
        for section in ["margin", "cash", "pdt"]:
            s = account.get(section, {})
            if s:
                print(f"\n  [{section.upper()}]")
                for k, v in s.items():
                    if v is not None:
                        print(f"    {k:30s}: {v}")
    except Exception as e:
        print(f"  [error] {e}")

    try:
        positions = client.get_positions()
        print(f"\nOpen positions: {len(positions)}")
        for p in positions:
            print(f"  {p.get('symbol'):30s} qty={p.get('quantity')} "
                  f"cost_basis={p.get('cost_basis')}")
    except Exception as e:
        print(f"  [error] positions: {e}")


def show_history(n: int = 20):
    """Print recent trade log entries."""
    if not TRADE_LOG.exists():
        print("No trade log found.")
        return

    try:
        import pandas as pd
        df = pd.read_csv(TRADE_LOG)
        print(f"\n{'='*80}")
        print(f"  TRADE LOG — Last {min(n, len(df))} entries")
        print(f"{'='*80}")
        view_cols = ["logged_at", "env", "action", "structure", "spx_price",
                     "short_put", "short_call", "credit_limit", "order_id",
                     "order_status", "fill_price", "notes"]
        view_cols = [c for c in view_cols if c in df.columns]
        print(df.tail(n)[view_cols].to_string(index=False))
    except Exception as e:
        print(f"Error reading log: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Execute SPX 0DTE options trade via Tradier"
    )
    parser.add_argument("--env",      default=None,
                        choices=["development", "uat", "prod"],
                        help="Environment (default: APP_ENV from .env or 'development')")
    parser.add_argument("--signal",   default=None,
                        help="JSON signal dict (e.g. from score_live.py output)")
    parser.add_argument("--structure", default="IC",
                        choices=["IC", "SINGLE_PUT", "SINGLE_CALL"],
                        help="Trade structure override")
    parser.add_argument("--quantity", type=int, default=1,
                        help="Number of spreads (default: 1)")
    parser.add_argument("--dry-run",  action="store_true", default=True,
                        help="Log intent without calling Tradier API (default: True)")
    parser.add_argument("--execute",  action="store_true",
                        help="Actually submit to Tradier (disables dry-run)")
    parser.add_argument("--preview",  action="store_true",
                        help="Use Tradier preview mode (validates but doesn't fill)")
    parser.add_argument("--balance",  action="store_true",
                        help="Show account balance and exit")
    parser.add_argument("--history",  action="store_true",
                        help="Show recent trade log and exit")
    parser.add_argument("--n",        type=int, default=20,
                        help="Rows for --history")
    args = parser.parse_args()

    # Determine env
    env = args.env or os.getenv("APP_ENV", "development")
    if env not in ENVIRONMENTS:
        print(f"[error] Unknown env '{env}'. Use: development, uat, prod")
        sys.exit(1)

    # ── Subcommands ──
    if args.history:
        show_history(args.n)
        return

    if args.balance:
        show_balance(env)
        return

    # ── Parse / build signal ──
    if args.signal:
        try:
            signal = json.loads(args.signal)
        except json.JSONDecodeError as e:
            print(f"[error] Invalid JSON signal: {e}")
            sys.exit(1)
    else:
        # Build minimal signal from args + current market data
        signal = {"structure": args.structure}

    signal.setdefault("structure", args.structure)

    # ── Dry run logic ──
    dry_run = not args.execute  # dry_run is default; --execute disables it
    if dry_run and not args.preview:
        print("\n[DRY RUN MODE] Use --execute to actually submit orders.")

    # ── Execute ──
    result = execute_trade(
        signal       = signal,
        env          = env,
        dry_run      = dry_run,
        quantity     = args.quantity,
        preview_only = args.preview,
    )

    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    print(f"\nAll trades logged to: {TRADE_LOG}")


if __name__ == "__main__":
    main()
