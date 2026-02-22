"""ThetaData v3 REST API client wrapper.

ThetaData v3 uses a local Theta Terminal that exposes a REST API.
See: https://docs.thetadata.us/

Endpoints used (all work with STANDARD subscription):
- /v3/option/history/greeks/first_order - Greeks + bid/ask + underlying_price
- /v3/option/history/eod - End of day option data
- /v3/option/list/expirations - Available expirations

Setup:
1. Install Java 21+
2. Download ThetaTerminalv3.jar
3. Create creds.txt with email (line 1) and password (line 2)
4. Run: java -jar ThetaTerminalv3.jar
"""

from datetime import date, datetime, time, timedelta
from typing import Optional
import logging

import requests

from .models import OptionQuote, OptionsChain, Settlement

logger = logging.getLogger(__name__)


class ThetaDataClient:
    """Client for interacting with ThetaData v3 REST API via local Theta Terminal."""

    # Default local Theta Terminal endpoint (v3 uses port 25503)
    DEFAULT_HOST = "http://127.0.0.1:25503"

    def __init__(
        self,
        username: str = "",
        password: str = "",
        host: str = DEFAULT_HOST,
        timeout: int = 60,
    ):
        """
        Initialize ThetaData client.

        Args:
            username: ThetaData account email (for reference, auth via creds.txt)
            password: ThetaData account password (for reference, auth via creds.txt)
            host: Theta Terminal host URL (default: http://127.0.0.1:25503)
            timeout: Request timeout in seconds

        Note:
            ThetaData v3 requires their terminal application to be running.
            1. Download ThetaTerminalv3.jar
            2. Create creds.txt with email and password
            3. Run: java -jar ThetaTerminalv3.jar
        """
        self.username = username
        self.password = password
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._connected = False

    def connect(self) -> bool:
        """
        Test connection to Theta Terminal.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Test with option expirations endpoint (works with STANDARD subscription)
            response = self._session.get(
                f"{self.host}/v3/option/list/expirations",
                params={"symbol": "SPXW"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                self._connected = True
                logger.info("Successfully connected to Theta Terminal v3")
                return True
            else:
                logger.error(f"Connection failed: {response.status_code} - {response.text[:200]}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error(
                "Could not connect to Theta Terminal. "
                "Make sure it's running: java -jar ThetaTerminalv3.jar"
            )
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def ensure_connected(self):
        """Ensure we have an active connection."""
        if not self._connected:
            if not self.connect():
                raise ConnectionError(
                    "Could not connect to Theta Terminal. "
                    "Ensure it's running: java -jar ThetaTerminalv3.jar"
                )

    def _format_date(self, d: date) -> str:
        """Format date for API (YYYYMMDD)."""
        return d.strftime("%Y%m%d")

    def _format_date_dash(self, d: date) -> str:
        """Format date for API (YYYY-MM-DD)."""
        return d.strftime("%Y-%m-%d")

    def _format_time(self, t: time) -> str:
        """Format time for API (HH:mm:ss)."""
        return t.strftime("%H:%M:%S")

    def _parse_date(self, date_val) -> date:
        """Parse date from API format."""
        if isinstance(date_val, int):
            return datetime.strptime(str(date_val), "%Y%m%d").date()
        elif isinstance(date_val, str):
            # Try different formats
            for fmt in ["%Y-%m-%d", "%Y%m%d"]:
                try:
                    return datetime.strptime(date_val, fmt).date()
                except ValueError:
                    continue
        return date_val

    def _parse_datetime(self, dt_val) -> Optional[date]:
        """Parse date from various datetime formats."""
        if dt_val is None:
            return None

        if isinstance(dt_val, int):
            return datetime.strptime(str(dt_val), "%Y%m%d").date()

        if isinstance(dt_val, str):
            # Try ISO datetime format first (e.g., '2025-01-06T17:14:58.39')
            if "T" in dt_val:
                try:
                    return datetime.fromisoformat(dt_val.split(".")[0]).date()
                except ValueError:
                    pass

            # Try other formats
            for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(dt_val, fmt).date()
                except ValueError:
                    continue

        return None

    def get_options_chain(
        self,
        trade_date: date,
        trade_time: time,
        expiration: date,
    ) -> Optional[OptionsChain]:
        """
        Get options chain at a specific date/time using first_order Greeks endpoint.

        This endpoint returns bid/ask, all first-order Greeks, AND underlying_price
        in a single call, avoiding the need for separate index/stock endpoints.

        Args:
            trade_date: The trading date
            trade_time: Time of day (ET)
            expiration: Option expiration date

        Returns:
            OptionsChain object or None if data unavailable.
        """
        self.ensure_connected()

        try:
            timestamp = datetime.combine(trade_date, trade_time)
            quotes = []
            underlying_price = None

            # Fetch Greeks (includes bid/ask and underlying_price) for both calls and puts
            for right in ["call", "put"]:
                greeks_data = self._get_first_order_greeks(
                    symbol="SPXW",
                    expiration=expiration,
                    right=right,
                    trade_date=trade_date,
                    trade_time=trade_time,
                )

                for row in greeks_data:
                    strike = row.get("strike", 0)
                    if isinstance(strike, str):
                        strike = float(strike)

                    # Get underlying price from the response (same for all rows)
                    if underlying_price is None:
                        underlying_price = row.get("underlying_price")
                        if underlying_price:
                            underlying_price = float(underlying_price)

                    # Get bid/ask and Greeks directly from the response
                    bid = row.get("bid", 0) or 0
                    ask = row.get("ask", 0) or 0

                    quote = OptionQuote(
                        root="SPXW",
                        expiration=expiration,
                        strike=strike,
                        option_type="C" if right == "call" else "P",
                        bid=float(bid),
                        ask=float(ask),
                        delta=float(row.get("delta", 0) or 0),
                        gamma=float(row.get("gamma", 0) or 0),
                        theta=float(row.get("theta", 0) or 0),
                        vega=float(row.get("vega", 0) or 0),
                        iv=float(row.get("implied_vol", 0) or row.get("implied_volatility", 0) or 0),
                        timestamp=timestamp,
                        underlying_price=underlying_price or 0,
                        open_interest=0,  # Not in Greeks response
                        volume=0,  # Not in Greeks response
                    )
                    quotes.append(quote)

            if not quotes:
                logger.warning(f"No quotes found for {trade_date} {trade_time}")
                return None

            return OptionsChain(
                underlying="SPX",
                timestamp=timestamp,
                expiration=expiration,
                underlying_price=underlying_price or 0,
                quotes=quotes,
            )

        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_first_order_greeks(
        self,
        symbol: str,
        expiration: date,
        right: str,
        trade_date: date,
        trade_time: time,
        end_time: Optional[time] = None,
    ) -> list[dict]:
        """
        Get first-order Greeks at a specific time or time range using v3 API.

        This endpoint returns: bid, ask, delta, theta, vega, rho, epsilon, lambda,
        implied_vol, underlying_price - everything we need in one call!

        Response format (nested):
        {
          "response": [
            {
              "contract": {"right": "CALL", "strike": 4740.0, ...},
              "data": [{"delta": 0.17, "bid": 2.35, "ask": 2.45, "underlying_price": 4716.2, ...}]
            },
            ...
          ]
        }

        Endpoint: GET /v3/option/history/greeks/first_order
        Docs: https://docs.thetadata.us/operations/option_history_greeks_first_order.html

        Args:
            symbol: Option symbol (e.g., "SPXW")
            expiration: Option expiration date
            right: "call" or "put"
            trade_date: The trading date
            trade_time: Start time (ET)
            end_time: Optional end time (ET). If None, fetches 1 minute after trade_time.
        """
        try:
            # Calculate end time if not provided
            if end_time is None:
                end_time = time(trade_time.hour, trade_time.minute + 1 if trade_time.minute < 59 else 59)

            params = {
                "symbol": symbol,
                "expiration": self._format_date(expiration),
                "right": right,
                "date": self._format_date(trade_date),
                "interval": "1m",  # 1 minute intervals
                "start_time": self._format_time(trade_time),
                "end_time": self._format_time(end_time),
                "strike": "*",  # All strikes
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/option/history/greeks/first_order",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Greeks request failed: {response.status_code} - {response.text[:200]}"
                )
                return []

            data = response.json()

            # v3 API returns nested structure: {"response": [{"contract": {...}, "data": [...]}]}
            if isinstance(data, dict) and "response" in data:
                rows = data["response"]
            elif isinstance(data, list):
                rows = data
            else:
                return []

            # Flatten the nested structure: combine contract info with first data point
            flattened = []
            for row in rows:
                if not isinstance(row, dict):
                    continue

                contract = row.get("contract", {})
                data_points = row.get("data", [])

                if not data_points:
                    continue

                # Use the first data point (closest to requested time)
                data_point = data_points[0]

                # Merge contract info with data
                merged = {
                    "strike": contract.get("strike", 0),
                    "right": contract.get("right", ""),
                    "symbol": contract.get("symbol", ""),
                    "expiration": contract.get("expiration", ""),
                    # Data fields
                    "bid": data_point.get("bid", 0),
                    "ask": data_point.get("ask", 0),
                    "delta": data_point.get("delta", 0),
                    "gamma": data_point.get("gamma", 0),
                    "theta": data_point.get("theta", 0),
                    "vega": data_point.get("vega", 0),
                    "rho": data_point.get("rho", 0),
                    "implied_vol": data_point.get("implied_vol", 0),
                    "underlying_price": data_point.get("underlying_price", 0),
                    "timestamp": data_point.get("timestamp", ""),
                }
                flattened.append(merged)

            return flattened

        except Exception as e:
            logger.error(f"Error fetching first-order Greeks: {e}")
            return []

    def get_underlying_price(
        self, trade_date: date, trade_time: time
    ) -> Optional[float]:
        """
        Get SPX price at a specific time by fetching from Greeks endpoint.

        Note: The first_order Greeks endpoint includes underlying_price,
        so we fetch a single option to get it.

        Args:
            trade_date: The trading date
            trade_time: Time of day (ET)

        Returns:
            SPX price or None if unavailable.
        """
        self.ensure_connected()

        try:
            # Use the Greeks endpoint which includes underlying_price
            # We just need one option to get the underlying price
            greeks_data = self._get_first_order_greeks(
                symbol="SPXW",
                expiration=trade_date,  # 0DTE
                right="call",
                trade_date=trade_date,
                trade_time=trade_time,
            )

            if greeks_data:
                underlying = greeks_data[0].get("underlying_price")
                if underlying:
                    return float(underlying)

            return None

        except Exception as e:
            logger.error(f"Error getting underlying price: {e}")
            return None

    def get_index_price(
        self,
        symbol: str,
        trade_date: date,
        trade_time: time,
    ) -> Optional[float]:
        """
        Get index price (SPX, VIX, etc.) at a specific time.

        Uses the /v3/index/history/price endpoint.

        Args:
            symbol: Index symbol (e.g., 'SPX', 'VIX')
            trade_date: The trading date
            trade_time: Time of day (ET)

        Returns:
            Index price or None if unavailable.
        """
        self.ensure_connected()

        try:
            # Calculate end time (1 minute after start)
            end_minute = trade_time.minute + 1 if trade_time.minute < 59 else 59
            end_time = time(trade_time.hour, end_minute)

            params = {
                "symbol": symbol,
                "date": self._format_date(trade_date),
                "interval": "1m",
                "start_time": self._format_time(trade_time),
                "end_time": self._format_time(end_time),
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/index/history/price",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Index price request failed for {symbol}: {response.status_code}"
                )
                return None

            data = response.json()

            # Parse response - v3 format
            if isinstance(data, dict) and "response" in data:
                rows = data["response"]
            elif isinstance(data, list):
                rows = data
            else:
                return None

            if rows:
                # Get the price from the first row
                row = rows[0]
                # Try common field names
                for field in ["price", "close", "last", "open"]:
                    if field in row and row[field]:
                        return float(row[field])

            return None

        except Exception as e:
            logger.error(f"Error getting index price for {symbol}: {e}")
            return None

    def get_index_prices_for_day(
        self,
        symbol: str,
        trade_date: date,
        times: list[time],
    ) -> dict[str, Optional[float]]:
        """
        Get index prices at multiple times in a single day.

        More efficient than calling get_index_price multiple times.

        Args:
            symbol: Index symbol (e.g., 'SPX', 'VIX')
            trade_date: The trading date
            times: List of times to get prices for

        Returns:
            Dict mapping time string (HHMM) to price or None.
        """
        self.ensure_connected()

        try:
            # Get full day of data in one call
            params = {
                "symbol": symbol,
                "date": self._format_date(trade_date),
                "interval": "1m",
                "start_time": "09:30:00",
                "end_time": "16:00:00",
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/index/history/price",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Index prices request failed for {symbol}: {response.status_code}"
                )
                return {t.strftime("%H%M"): None for t in times}

            data = response.json()

            # Parse response
            if isinstance(data, dict) and "response" in data:
                rows = data["response"]
            elif isinstance(data, list):
                rows = data
            else:
                return {t.strftime("%H%M"): None for t in times}

            # Build a map of timestamp -> price
            price_by_time = {}
            for row in rows:
                ts = row.get("timestamp") or row.get("time")
                if ts:
                    # Parse timestamp to extract time
                    if isinstance(ts, str) and "T" in ts:
                        # ISO format: 2024-01-05T10:30:00
                        time_part = ts.split("T")[1][:5].replace(":", "")
                    elif isinstance(ts, str) and " " in ts:
                        time_part = ts.split(" ")[1][:5].replace(":", "")
                    else:
                        continue

                    # Get price
                    for field in ["price", "close", "last", "open"]:
                        if field in row and row[field]:
                            price_by_time[time_part] = float(row[field])
                            break

            # Map requested times to prices
            result = {}
            for t in times:
                time_key = t.strftime("%H%M")
                result[time_key] = price_by_time.get(time_key)

            return result

        except Exception as e:
            logger.error(f"Error getting index prices for {symbol}: {e}")
            return {t.strftime("%H%M"): None for t in times}

    def get_settlement_price(self, trade_date: date) -> Optional[float]:
        """
        Get SPX settlement price for 0DTE options.

        Uses the underlying_price from option EOD data or late-day Greeks.

        Args:
            trade_date: The trading date

        Returns:
            Settlement price or None if unavailable.
        """
        self.ensure_connected()

        try:
            # Try to get underlying price from option EOD endpoint
            params = {
                "symbol": "SPXW",
                "expiration": self._format_date(trade_date),
                "start_date": self._format_date(trade_date),
                "end_date": self._format_date(trade_date),
                "strike": "*",
                "right": "call",
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/option/history/eod",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "response" in data:
                    rows = data["response"]
                elif isinstance(data, list):
                    rows = data
                else:
                    rows = []

                # Option EOD might have underlying price or we can derive it
                # For settlement, we can also use the close price from Greeks at 4:00 PM
                if rows:
                    # Check if underlying_price is in the response
                    underlying = rows[0].get("underlying_price")
                    if underlying:
                        return float(underlying)

            # Fallback: get underlying price from Greeks at 4:00 PM (settlement time)
            return self.get_underlying_price(trade_date, time(16, 0))

        except Exception as e:
            logger.error(f"Error getting settlement price: {e}")
            return None

    def get_trading_days(
        self, start: date, end: date
    ) -> list[date]:
        """
        Get list of trading days in date range.

        Uses SPXW expirations to determine trading days (SPXW has 0DTE every day).

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of trading days.
        """
        self.ensure_connected()

        try:
            # Get all SPXW expirations - each expiration is a trading day
            params = {
                "symbol": "SPXW",
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/option/list/expirations",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(f"Expirations request failed: {response.status_code}")
                return self._generate_business_days(start, end)

            data = response.json()

            if isinstance(data, dict) and "response" in data:
                rows = data["response"]
            elif isinstance(data, list):
                rows = data
            else:
                return self._generate_business_days(start, end)

            # Parse expirations and filter to date range
            trading_days = []
            for row in rows:
                # Each row has 'expiration' field
                if isinstance(row, dict):
                    exp_str = row.get("expiration")
                else:
                    exp_str = row

                if exp_str:
                    parsed = self._parse_datetime(exp_str)
                    if parsed and start <= parsed <= end:
                        trading_days.append(parsed)

            if trading_days:
                logger.info(f"Found {len(trading_days)} trading days from SPXW expirations")
                return sorted(trading_days)

            # Fallback to business days if no expirations found
            return self._generate_business_days(start, end)

        except Exception as e:
            logger.error(f"Error getting trading days: {e}")
            return self._generate_business_days(start, end)

    def _generate_business_days(self, start: date, end: date) -> list[date]:
        """Generate approximate business days (excludes weekends)."""
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                days.append(current)
            current += timedelta(days=1)
        return days

    def get_expirations(
        self, trade_date: date, root: str = "SPXW"
    ) -> list[date]:
        """
        Get available expiration dates.

        Args:
            trade_date: Reference date
            root: Option root symbol

        Returns:
            List of expiration dates.
        """
        self.ensure_connected()

        try:
            params = {
                "symbol": root,
                "format": "json",
            }

            response = self._session.get(
                f"{self.host}/v3/option/list/expirations",
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return []

            data = response.json()

            if isinstance(data, dict) and "response" in data:
                exp_list = data["response"]
            elif isinstance(data, list):
                exp_list = data
            else:
                return []

            expirations = []
            for exp in exp_list:
                # Handle both dict format {"expiration": "2024-01-05"} and string format
                if isinstance(exp, dict):
                    exp_val = exp.get("expiration")
                else:
                    exp_val = exp

                exp_date = self._parse_datetime(exp_val)
                if exp_date and exp_date >= trade_date:
                    expirations.append(exp_date)

            return sorted(expirations)

        except Exception as e:
            logger.error(f"Error getting expirations: {e}")
            return []

    def disconnect(self):
        """Close the session."""
        self._session.close()
        self._connected = False
        logger.info("Disconnected from Theta Terminal")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
