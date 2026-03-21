"""
Tradier API client wrapper with async support and rate limiting.

Drop-in replacement for SchwabClient — returns data in Schwab-compatible format
so the bot, order builder, and monitoring code work without changes.
"""

import asyncio
import urllib.parse
from typing import Optional, Any
from datetime import datetime, date, timedelta
from loguru import logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed. Install with: pip install aiohttp")

from execution.broker_api.rate_limiter import RateLimiter
from config.settings import settings


class TradierClient:
    """
    Async wrapper around Tradier REST API with rate limiting.

    Translates Tradier responses into Schwab-compatible format so the
    bot, order builder, and monitoring code work without changes.
    """

    def __init__(self):
        """Initialize Tradier client."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        self.api_token = settings.TRADIER_API_TOKEN
        self.account_id = settings.TRADIER_ACCOUNT_ID

        if not self.api_token:
            raise ValueError(
                "Tradier API token not configured. "
                "Set TRADIER_API_TOKEN in .env"
            )

        # Select endpoint based on sandbox mode
        if settings.TRADIER_SANDBOX:
            self.base_url = "https://sandbox.tradier.com/v1"
        else:
            self.base_url = "https://api.tradier.com/v1"

        self.rate_limiter = RateLimiter(
            max_calls=settings.TRADIER_RATE_LIMIT_CALLS,
            period=settings.TRADIER_RATE_LIMIT_PERIOD
        )

        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"TradierClient initialized "
            f"({'sandbox' if settings.TRADIER_SANDBOX else 'production'}) "
            f"account={self.account_id}"
        )

    def _headers(self) -> dict:
        """Standard headers for Tradier API requests."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers())
        return self._session

    async def _get(self, path: str, params: dict = None) -> dict:
        """Rate-limited GET request."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"GET {path} failed: {resp.status} — {text[:500]}")
                raise ValueError(f"Tradier API error: {resp.status}")
            return await resp.json()

    async def _post(self, path: str, data: dict = None) -> tuple[dict, int]:
        """Rate-limited POST request. Returns (json_body, status_code)."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.post(url, data=data) as resp:
            text = await resp.text()
            if resp.status not in [200, 201]:
                logger.error(f"POST {path} failed: {resp.status} — {text[:500]}")
                raise ValueError(f"Tradier API error: {resp.status}")
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = {"raw": text}
            return body, resp.status

    async def _put(self, path: str, data: dict = None) -> tuple[dict, int]:
        """Rate-limited PUT request."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.put(url, data=data) as resp:
            text = await resp.text()
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = {"raw": text}
            return body, resp.status

    async def _delete(self, path: str) -> tuple[dict, int]:
        """Rate-limited DELETE request."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.delete(url) as resp:
            text = await resp.text()
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = {"raw": text}
            return body, resp.status

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def re_authenticate(self) -> bool:
        """No-op for Tradier — uses static API token, no OAuth refresh needed."""
        logger.info("Tradier uses static API tokens — no re-authentication needed")
        return True

    # ===== Market Data Methods =====

    async def get_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for a symbol.

        Returns data in Schwab-compatible format.
        """
        # Tradier uses plain symbol names (SPX not $SPX)
        clean_symbol = symbol.lstrip("$")
        data = await self._get("/markets/quotes", {"symbols": clean_symbol})
        quotes = data.get("quotes", {})
        quote = quotes.get("quote", {})

        # Handle single vs multiple quotes
        if isinstance(quote, list):
            quote = quote[0] if quote else {}

        # Transform to Schwab-compatible format
        return {
            clean_symbol: {
                "quote": {
                    "bidPrice": quote.get("bid", 0),
                    "askPrice": quote.get("ask", 0),
                    "lastPrice": quote.get("last", 0),
                    "totalVolume": quote.get("volume", 0),
                    "quoteTimeInLong": quote.get("trade_date", ""),
                }
            }
        }

    async def get_quotes(self, symbols: list[str]) -> dict:
        """Get real-time quotes for multiple symbols."""
        clean_symbols = [s.lstrip("$") for s in symbols]
        data = await self._get("/markets/quotes", {"symbols": ",".join(clean_symbols)})
        quotes = data.get("quotes", {})
        quote_list = quotes.get("quote", [])
        if isinstance(quote_list, dict):
            quote_list = [quote_list]

        result = {}
        for q in quote_list:
            sym = q.get("symbol", "")
            result[sym] = {
                "quote": {
                    "bidPrice": q.get("bid", 0),
                    "askPrice": q.get("ask", 0),
                    "lastPrice": q.get("last", 0),
                    "totalVolume": q.get("volume", 0),
                }
            }
        return result

    async def get_option_chain(
        self,
        symbol: str,
        contract_type: Optional[str] = None,
        strike_count: Optional[int] = None,
        include_underlying_quote: bool = True,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        strike_range: Optional[str] = None
    ) -> dict:
        """
        Get option chain for a symbol.

        Returns data in Schwab-compatible nested format:
        {
            "callExpDateMap": { "2026-03-14:0": { "5800.0": [...], ... } },
            "putExpDateMap": { ... },
            "underlyingPrice": 5850.0
        }
        """
        clean_symbol = symbol.lstrip("$")

        # First get available expirations (with all roots for SPX/SPXW)
        exp_params = {"symbol": clean_symbol, "includeAllRoots": "true"}
        exp_data = await self._get("/markets/options/expirations", exp_params)
        expirations = exp_data.get("expirations", {}).get("date", [])

        if isinstance(expirations, str):
            expirations = [expirations]

        # Filter expirations by date range
        if from_date:
            expirations = [e for e in expirations if e >= str(from_date)]
        if to_date:
            expirations = [e for e in expirations if e <= str(to_date)]

        if not expirations:
            logger.error(f"No expirations found for {clean_symbol}")
            return {"callExpDateMap": {}, "putExpDateMap": {}}

        # For 0DTE, we want today's expiration
        today_str = date.today().strftime("%Y-%m-%d")
        target_expirations = [e for e in expirations if e == today_str]
        if not target_expirations:
            # Fall back to nearest expiration
            target_expirations = expirations[:1]

        # Fetch chains for target expirations
        call_exp_map = {}
        put_exp_map = {}
        underlying_price = 0.0

        for exp in target_expirations:
            params = {
                "symbol": clean_symbol,
                "expiration": exp,
                "greeks": "true",
            }
            if contract_type:
                params["optionType"] = contract_type.lower()  # "call", "put"

            chain_data = await self._get("/markets/options/chains", params)
            options = chain_data.get("options", {}).get("option", [])

            if isinstance(options, dict):
                options = [options]

            if not options:
                continue

            # Calculate DTE
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - date.today()).days

            # Schwab-style expiration key: "2026-03-14:0"
            exp_key = f"{exp}:{dte}"

            calls_at_exp = {}
            puts_at_exp = {}

            for opt in options:
                strike = opt.get("strike", 0)
                strike_str = f"{strike:.1f}"
                greeks = opt.get("greeks", {}) or {}

                # Get underlying price from first option with it
                if opt.get("underlying_last", 0) > 0:
                    underlying_price = opt["underlying_last"]

                # Build Schwab-compatible option contract dict
                schwab_contract = {
                    "symbol": opt.get("symbol", ""),
                    "description": opt.get("description", ""),
                    "bid": opt.get("bid", 0) or 0,
                    "ask": opt.get("ask", 0) or 0,
                    "last": opt.get("last", 0) or 0,
                    "mark": (
                        ((opt.get("bid", 0) or 0) + (opt.get("ask", 0) or 0)) / 2
                    ),
                    "delta": greeks.get("delta", 0) or 0,
                    "gamma": greeks.get("gamma", 0) or 0,
                    "theta": greeks.get("theta", 0) or 0,
                    "vega": greeks.get("vega", 0) or 0,
                    "volatility": opt.get("greeks", {}).get("mid_iv", 0) if greeks else 0,
                    "openInterest": opt.get("open_interest", 0) or 0,
                    "totalVolume": opt.get("volume", 0) or 0,
                    "strikePrice": strike,
                    "expirationDate": exp,
                    "daysToExpiration": dte,
                    "putCall": opt.get("option_type", "").upper(),
                }

                # Schwab format: each strike maps to a list with one contract
                if opt.get("option_type", "").lower() == "call":
                    calls_at_exp[strike_str] = [schwab_contract]
                elif opt.get("option_type", "").lower() == "put":
                    puts_at_exp[strike_str] = [schwab_contract]

            if calls_at_exp:
                call_exp_map[exp_key] = calls_at_exp
            if puts_at_exp:
                put_exp_map[exp_key] = puts_at_exp

        result = {
            "callExpDateMap": call_exp_map,
            "putExpDateMap": put_exp_map,
            "underlyingPrice": underlying_price,
        }

        logger.debug(
            f"Got option chain for {clean_symbol}: "
            f"{sum(len(v) for v in call_exp_map.values())} calls, "
            f"{sum(len(v) for v in put_exp_map.values())} puts"
        )
        return result

    async def get_price_history(
        self,
        symbol: str,
        period_type: str = 'year',
        period: int = 1,
        frequency_type: str = 'daily',
        frequency: int = 1
    ) -> dict:
        """Get historical price data."""
        clean_symbol = symbol.lstrip("$")

        # Map Schwab period to Tradier interval
        interval_map = {
            "minute": "1min",
            "daily": "daily",
            "weekly": "weekly",
            "monthly": "monthly",
        }
        interval = interval_map.get(frequency_type, "daily")

        # Calculate date range from period
        end = date.today()
        if period_type == "day":
            start = end - timedelta(days=period)
        elif period_type == "month":
            start = end - timedelta(days=period * 30)
        elif period_type == "year":
            start = end - timedelta(days=period * 365)
        else:  # ytd
            start = date(end.year, 1, 1)

        params = {
            "symbol": clean_symbol,
            "interval": interval,
            "start": str(start),
            "end": str(end),
        }
        data = await self._get("/markets/history", params)
        history = data.get("history", {})
        days = history.get("day", []) if history else []

        if isinstance(days, dict):
            days = [days]

        # Transform to Schwab-compatible format
        candles = []
        for d in days:
            candles.append({
                "open": d.get("open", 0),
                "high": d.get("high", 0),
                "low": d.get("low", 0),
                "close": d.get("close", 0),
                "volume": d.get("volume", 0),
                "datetime": d.get("date", ""),
            })

        return {"candles": candles, "symbol": clean_symbol, "empty": len(candles) == 0}

    # ===== Account Methods =====

    async def _get_account_hash(self) -> str:
        """Tradier uses account_id directly — no hash needed."""
        return self.account_id

    async def get_account(self, account_id: Optional[str] = None, include_positions: bool = True) -> dict:
        """
        Get account information in Schwab-compatible format.
        """
        acct = account_id or self.account_id
        bal_data = await self._get(f"/accounts/{acct}/balances")
        balances = bal_data.get("balances", {})

        # Get positions if requested
        positions = []
        if include_positions:
            pos_data = await self._get(f"/accounts/{acct}/positions")
            pos_container = pos_data.get("positions", {})
            if pos_container == "null" or not pos_container:
                raw_positions = []
            else:
                raw_positions = pos_container.get("position", [])
                if isinstance(raw_positions, dict):
                    raw_positions = [raw_positions]

            for p in raw_positions:
                qty = p.get("quantity", 0)
                positions.append({
                    "instrument": {
                        "symbol": p.get("symbol", ""),
                        "assetType": "OPTION" if len(p.get("symbol", "")) > 10 else "EQUITY",
                    },
                    "longQuantity": max(qty, 0),
                    "shortQuantity": abs(min(qty, 0)),
                    "currentDayProfitLoss": p.get("unrealized_intraday_pl", 0) or 0,
                    "marketValue": p.get("market_value", 0) or 0,
                    "averagePrice": p.get("cost_basis", 0) or 0,
                })

        # Map Tradier balances to Schwab-compatible format
        # Tradier has different fields for margin vs cash accounts
        total_equity = balances.get("total_equity", 0) or 0
        option_bp = balances.get("option_buying_power", 0) or 0
        stock_bp = balances.get("stock_buying_power", 0) or 0
        total_cash = balances.get("total_cash", 0) or 0

        # For margin sub-object
        margin = balances.get("margin", {}) or {}
        if margin:
            option_bp = margin.get("option_buying_power", option_bp) or option_bp
            stock_bp = margin.get("stock_buying_power", stock_bp) or stock_bp

        # For cash sub-object
        cash = balances.get("cash", {}) or {}
        if cash:
            total_cash = cash.get("cash_available", total_cash) or total_cash

        result = {
            "securitiesAccount": {
                "accountNumber": acct,
                "currentBalances": {
                    "liquidationValue": total_equity,
                    "buyingPower": stock_bp,
                    "availableFundsNonMarginableTrade": option_bp,
                    "marginBalance": 0.0,
                },
                "initialBalances": {
                    # Tradier doesn't have SOD snapshot — use current as fallback
                    "liquidationValue": total_equity,
                },
                "positions": positions,
            }
        }

        logger.debug(f"Got account data: equity=${total_equity:,.2f}, option_bp=${option_bp:,.2f}")
        return result

    async def get_account_positions(self, account_id: Optional[str] = None) -> list[dict]:
        """Get current positions in Schwab-compatible format."""
        account_data = await self.get_account(account_id)
        positions = account_data.get("securitiesAccount", {}).get("positions", [])
        logger.debug(f"Found {len(positions)} positions")
        return positions

    async def get_buying_power(self, account_id: Optional[str] = None) -> float:
        """Get available buying power for options trading."""
        account_data = await self.get_account(account_id, include_positions=False)
        balances = (
            account_data
            .get("securitiesAccount", {})
            .get("currentBalances", {})
        )
        option_bp = balances.get("availableFundsNonMarginableTrade", 0.0)
        equity_bp = balances.get("buyingPower", 0.0)
        buying_power = option_bp if option_bp > 0 else equity_bp

        logger.info(
            f"Buying power: option_bp=${option_bp:,.2f}, "
            f"equity_bp=${equity_bp:,.2f}, using=${buying_power:,.2f}"
        )
        return float(buying_power)

    # ===== Order Methods =====

    async def place_order(
        self,
        order: dict,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Place an order. Accepts Schwab-format order dict and translates to Tradier.

        Args:
            order: Schwab-format order specification dictionary
            account_id: Account ID (defaults to configured account)

        Returns:
            Dict with orderId and status
        """
        if settings.PAPER_TRADING:
            logger.warning("Paper trading mode — order not actually placed")
            return {
                "status": "PAPER_TRADING",
                "order": order,
                "message": "Paper trading mode — order simulation only",
            }

        acct = account_id or self.account_id
        tradier_order = self._translate_order_to_tradier(order)

        body, status = await self._post(f"/accounts/{acct}/orders", data=tradier_order)

        order_resp = body.get("order", {})
        order_id = str(order_resp.get("id", ""))

        if not order_id:
            logger.warning(f"No order ID in response: {body}")

        result = {
            "orderId": order_id,
            "status": order_resp.get("status", "pending"),
        }

        logger.info(f"Order placed: id={order_id}, status={result['status']}")
        return result

    async def get_order(
        self,
        order_id: str,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Get order status in Schwab-compatible format.
        """
        acct = account_id or self.account_id
        data = await self._get(f"/accounts/{acct}/orders/{order_id}")
        order = data.get("order", {})

        return self._translate_order_from_tradier(order)

    async def get_orders_for_account(
        self,
        account_id: Optional[str] = None,
        status: Optional[str] = None,
        from_entered_datetime: Optional[datetime] = None,
        to_entered_datetime: Optional[datetime] = None,
        max_results: Optional[int] = None
    ) -> list[dict]:
        """Get orders in Schwab-compatible format."""
        acct = account_id or self.account_id
        params = {}

        if status:
            # Map Schwab status to Tradier
            status_map = {
                "FILLED": "filled",
                "WORKING": "open",
                "PENDING_ACTIVATION": "pending",
                "CANCELED": "canceled",
                "EXPIRED": "expired",
                "REJECTED": "rejected",
            }
            tradier_status = status_map.get(status, status.lower())
            # Tradier doesn't filter by status in the endpoint — we filter after
            params["includeTags"] = "true"

        try:
            data = await self._get(f"/accounts/{acct}/orders", params)
        except ValueError:
            logger.error("Failed to get orders")
            return []

        orders_container = data.get("orders", {})
        if not orders_container or orders_container == "null":
            return []

        raw_orders = orders_container.get("order", [])
        if isinstance(raw_orders, dict):
            raw_orders = [raw_orders]

        # Translate each order to Schwab format
        result = []
        for o in raw_orders:
            translated = self._translate_order_from_tradier(o)

            # Filter by status if requested
            if status and translated.get("status") != status:
                continue

            # Filter by date range
            if from_entered_datetime:
                order_time = translated.get("enteredTime", "")
                if order_time and order_time < from_entered_datetime.isoformat():
                    continue
            if to_entered_datetime:
                order_time = translated.get("enteredTime", "")
                if order_time and order_time > to_entered_datetime.isoformat():
                    continue

            result.append(translated)

        if max_results:
            result = result[:max_results]

        logger.debug(f"Got {len(result)} orders")
        return result

    async def get_transactions(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_types: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> list[dict]:
        """Get account history/transactions."""
        acct = account_id or self.account_id
        params = {}
        if start_date:
            params["start"] = str(start_date)
        if end_date:
            params["end"] = str(end_date)

        try:
            data = await self._get(f"/accounts/{acct}/history", params)
        except ValueError:
            return []

        history = data.get("history", {})
        if not history or history == "null":
            return []

        events = history.get("event", [])
        if isinstance(events, dict):
            events = [events]

        # Translate to Schwab-compatible format
        result = []
        for e in events:
            result.append({
                "transactionId": e.get("id", ""),
                "transactionDate": e.get("date", ""),
                "transactionType": e.get("type", "").upper(),
                "cash": e.get("amount", 0),
            })

        logger.info(f"Got {len(result)} transactions")
        return result

    async def cancel_order(
        self,
        order_id: str,
        account_id: Optional[str] = None
    ) -> bool:
        """Cancel an order."""
        if settings.PAPER_TRADING:
            logger.warning("Paper trading mode — cancel simulation")
            return True

        acct = account_id or self.account_id
        try:
            body, status = await self._delete(f"/accounts/{acct}/orders/{order_id}")
            success = status in [200, 204]
            if success:
                logger.info(f"Order {order_id} cancelled successfully")
            else:
                logger.error(f"Failed to cancel order {order_id}: {status}")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    # ===== Translation Helpers =====

    def _translate_order_to_tradier(self, schwab_order: dict) -> dict:
        """
        Translate Schwab-format order dict to Tradier form-encoded params.

        Schwab uses JSON body with orderLegCollection.
        Tradier uses form-encoded POST with indexed option_symbol[N], side[N], quantity[N].
        """
        legs = schwab_order.get("orderLegCollection", [])
        order_type = schwab_order.get("orderType", "MARKET").upper()
        price = schwab_order.get("price", "")

        # Map order type
        if order_type in ("NET_CREDIT", "NET_DEBIT"):
            tradier_type = "credit" if order_type == "NET_CREDIT" else "debit"
        elif order_type == "MARKET":
            tradier_type = "market"
        else:
            tradier_type = "limit"

        # Determine symbol from first leg
        first_symbol = ""
        if legs:
            raw = legs[0].get("instrument", {}).get("symbol", "")
            # Extract root symbol from OCC option symbol
            # Schwab: "SPXW  260314P05800000" → root is "SPXW" or "SPX"
            first_symbol = self._extract_root_from_occ(raw)

        # Build Tradier multileg order
        tradier_data = {
            "class": "multileg",
            "symbol": first_symbol,
            "type": tradier_type,
            "duration": schwab_order.get("duration", "DAY").lower(),
        }

        if price and tradier_type not in ("market",):
            tradier_data["price"] = str(price)

        # Map each leg
        instruction_map = {
            "BUY_TO_OPEN": "buy_to_open",
            "SELL_TO_OPEN": "sell_to_open",
            "BUY_TO_CLOSE": "buy_to_close",
            "SELL_TO_CLOSE": "sell_to_close",
        }

        for i, leg in enumerate(legs):
            occ_symbol = leg.get("instrument", {}).get("symbol", "")
            # Convert Schwab OCC format to Tradier OCC format
            tradier_symbol = self._schwab_symbol_to_tradier(occ_symbol)

            tradier_data[f"option_symbol[{i}]"] = tradier_symbol
            tradier_data[f"side[{i}]"] = instruction_map.get(
                leg.get("instruction", ""), "buy_to_open"
            )
            tradier_data[f"quantity[{i}]"] = str(leg.get("quantity", 1))

        return tradier_data

    def _translate_order_from_tradier(self, tradier_order: dict) -> dict:
        """
        Translate Tradier order response to Schwab-compatible format.
        """
        # Status mapping
        status_map = {
            "pending": "PENDING_ACTIVATION",
            "open": "WORKING",
            "partially_filled": "WORKING",
            "filled": "FILLED",
            "expired": "EXPIRED",
            "canceled": "CANCELED",
            "rejected": "REJECTED",
        }
        tradier_status = tradier_order.get("status", "pending")
        schwab_status = status_map.get(tradier_status, tradier_status.upper())

        # Build order legs from Tradier's leg array
        tradier_legs = tradier_order.get("leg", [])
        if isinstance(tradier_legs, dict):
            tradier_legs = [tradier_legs]

        side_map = {
            "buy_to_open": "BUY_TO_OPEN",
            "sell_to_open": "SELL_TO_OPEN",
            "buy_to_close": "BUY_TO_CLOSE",
            "sell_to_close": "SELL_TO_CLOSE",
        }

        order_legs = []
        execution_legs = []

        for idx, leg in enumerate(tradier_legs):
            order_legs.append({
                "instruction": side_map.get(leg.get("side", ""), "BUY_TO_OPEN"),
                "quantity": leg.get("quantity", 0),
                "instrument": {
                    "symbol": leg.get("option_symbol", ""),
                    "assetType": "OPTION",
                },
            })

            # If order is filled, create execution leg
            if schwab_status == "FILLED" and leg.get("avg_fill_price"):
                execution_legs.append({
                    "legId": idx + 1,  # 1-indexed like Schwab
                    "price": leg.get("avg_fill_price", 0),
                    "quantity": leg.get("quantity", 0),
                })

        result = {
            "orderId": str(tradier_order.get("id", "")),
            "status": schwab_status,
            "filledQuantity": tradier_order.get("exec_quantity", 0) or 0,
            "orderType": tradier_order.get("type", "").upper(),
            "price": tradier_order.get("price", 0),
            "enteredTime": tradier_order.get("create_date", ""),
            "orderLegCollection": order_legs,
        }

        # Add execution activity if filled
        if execution_legs:
            result["orderActivityCollection"] = [
                {
                    "executionLegs": execution_legs,
                    "executionTime": tradier_order.get("transaction_date", ""),
                }
            ]

        return result

    @staticmethod
    def _schwab_symbol_to_tradier(schwab_symbol: str) -> str:
        """
        Convert Schwab OCC symbol to Tradier format.

        Schwab: "SPXW  260314P05800000" (padded to 6 chars for root)
        Tradier: "SPXW260314P05800000" (no padding)
        """
        return schwab_symbol.replace(" ", "")

    @staticmethod
    def _extract_root_from_occ(occ_symbol: str) -> str:
        """Extract root symbol from OCC option symbol."""
        clean = occ_symbol.strip()
        if len(clean) < 15:
            return clean
        # Root is everything before the last 15 chars (YYMMDD + P/C + 8-digit strike)
        root = clean[:-15].strip()
        return root if root else clean


# Lazy global instance — only created when first accessed
_tradier_client = None


def get_tradier_client() -> TradierClient:
    """Get or create global TradierClient instance."""
    global _tradier_client
    if _tradier_client is None:
        _tradier_client = TradierClient()
    return _tradier_client
