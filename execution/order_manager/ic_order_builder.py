"""
Iron Condor order construction for 0DTE automated trading.

Handles strike selection from Schwab option chain data, order building
for entry/close, position sizing, and TP/SL calculations.
"""

import math
from typing import Dict, Optional, Tuple
from loguru import logger
from config.settings import settings


class IronCondorOrderBuilder:
    """Builds iron condor orders from Schwab option chain data."""

    def select_strikes(
        self,
        chain_data: dict,
        target_delta: float = 0.10,
        wing_width: float = 25.0,
    ) -> Optional[Dict]:
        """
        Select iron condor strikes from a 0DTE option chain.

        Finds short put/call at target delta, then adds wings at wing_width.

        Args:
            chain_data: Schwab option chain response
            target_delta: Target absolute delta for short strikes
            wing_width: Width of each wing in dollars

        Returns:
            Dict with all 4 legs' details, net credit, and expiration,
            or None if suitable strikes not found.
        """
        call_exp_map = chain_data.get('callExpDateMap', {})
        put_exp_map = chain_data.get('putExpDateMap', {})

        if not call_exp_map or not put_exp_map:
            logger.error("Option chain missing callExpDateMap or putExpDateMap")
            return None

        # Find the 0DTE expiration key (lowest DTE)
        exp_key = self._find_0dte_expiration(call_exp_map)
        if not exp_key:
            logger.error("No 0DTE expiration found in chain")
            return None

        puts_at_exp = put_exp_map.get(exp_key, {})
        calls_at_exp = call_exp_map.get(exp_key, {})

        if not puts_at_exp or not calls_at_exp:
            logger.error(f"No puts or calls at expiration {exp_key}")
            return None

        # Find short strikes by delta
        short_put = self._find_option_by_delta(puts_at_exp, target_delta)
        short_call = self._find_option_by_delta(calls_at_exp, target_delta)

        if not short_put or not short_call:
            logger.error("Could not find short options at target delta")
            return None

        # Find long wings
        long_put_target = short_put['strike'] - wing_width
        long_put = self._find_option_by_strike(
            puts_at_exp, long_put_target,
            exclude_strike=short_put['strike'], direction='below'
        )

        long_call_target = short_call['strike'] + wing_width
        long_call = self._find_option_by_strike(
            calls_at_exp, long_call_target,
            exclude_strike=short_call['strike'], direction='above'
        )

        if not long_put or not long_call:
            logger.error("Could not find wing options")
            return None

        # Calculate net credit at mark (mid) price
        net_credit = (
            short_put['mid'] + short_call['mid']
            - long_put['mid'] - long_call['mid']
        )

        # Parse expiration date from key (format: "2026-02-18:0")
        exp_date = exp_key.split(':')[0]

        result = {
            'long_put': long_put,
            'short_put': short_put,
            'short_call': short_call,
            'long_call': long_call,
            'net_credit': round(math.floor(net_credit * 20) / 20, 2),
            'expiration': exp_date,
            'exp_key': exp_key,
        }

        logger.info(
            f"IC strikes selected: "
            f"{long_put['strike']}/{short_put['strike']}p - "
            f"{short_call['strike']}/{long_call['strike']}c "
            f"Credit: ${net_credit:.2f}"
        )

        return result

    def build_entry_order(self, strikes: Dict, quantity: int) -> Dict:
        """
        Build Schwab API order dict for iron condor entry (SELL_TO_OPEN).

        Args:
            strikes: Output from select_strikes()
            quantity: Number of contracts

        Returns:
            Schwab order specification dict
        """
        return {
            "orderType": "NET_CREDIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "complexOrderStrategyType": "IRON_CONDOR",
            "price": f"{strikes['net_credit']:.2f}",
            "orderLegCollection": [
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "SELL_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "SELL_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
            ]
        }

    def build_close_order(self, strikes: Dict, quantity: int, debit_price: float) -> Dict:
        """
        Build Schwab API order dict for iron condor close (BUY_TO_CLOSE) at limit.

        Args:
            strikes: Output from select_strikes()
            quantity: Number of contracts
            debit_price: Net debit limit price

        Returns:
            Schwab order specification dict
        """
        return {
            "orderType": "NET_DEBIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "complexOrderStrategyType": "IRON_CONDOR",
            "price": f"{debit_price:.2f}",
            "orderLegCollection": [
                {
                    "instruction": "SELL_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "SELL_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
            ]
        }

    def build_market_close_order(self, strikes: Dict, quantity: int) -> Dict:
        """
        Build Schwab API order dict for market close (emergency exit).

        Args:
            strikes: Output from select_strikes()
            quantity: Number of contracts

        Returns:
            Schwab order specification dict
        """
        return {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "complexOrderStrategyType": "IRON_CONDOR",
            "orderLegCollection": [
                {
                    "instruction": "SELL_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_put']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['short_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "SELL_TO_CLOSE",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": strikes['long_call']['symbol'],
                        "assetType": "OPTION"
                    }
                },
            ]
        }

    def calculate_position_size(
        self,
        credit: float,
        portfolio_value: float,
        daily_target_pct: float,
        tp_pct: float,
        cost_per_leg: float,
        wing_width: float,
        buying_power: float,
        max_contracts: int,
    ) -> int:
        """
        Calculate position size in contracts.

        Sizes to hit daily_target_pct profit at take-profit, capped by
        buying power and max_contracts.

        Args:
            credit: Net credit per contract (e.g. 3.00)
            portfolio_value: Total portfolio value
            daily_target_pct: Target daily return (e.g. 0.02 = 2%)
            tp_pct: Take-profit as fraction of credit (e.g. 0.25)
            cost_per_leg: Transaction cost per leg per contract
            wing_width: Width of each wing in dollars
            buying_power: Available buying power
            max_contracts: Hard cap on contracts

        Returns:
            Number of contracts to trade
        """
        # Cost per contract: 4 legs open + 4 legs close = 8 legs total
        total_cost_per_contract = cost_per_leg * 8

        # Net profit per contract at TP: credit * tp_pct * 100 - costs
        net_profit_per_contract = (credit * tp_pct * 100) - total_cost_per_contract

        if net_profit_per_contract <= 0:
            logger.warning(
                f"Net profit per contract <= 0 "
                f"(credit={credit}, tp_pct={tp_pct}, costs={total_cost_per_contract})"
            )
            return 0

        # Target dollar profit
        target = portfolio_value * daily_target_pct

        # Contracts needed to hit target
        contracts = math.ceil(target / net_profit_per_contract)

        # Margin required per contract: full wing width (broker requires
        # collateral upfront; credit offset is applied after fill)
        margin_per = wing_width * 100
        if margin_per <= 0:
            logger.warning(f"Invalid margin calculation: wing_width={wing_width}")
            return 0

        # Cap by buying power (90% buffer for fees/rounding)
        usable_bp = buying_power * 0.90
        max_by_bp = int(usable_bp / margin_per)

        result = min(contracts, max_by_bp, max_contracts)
        result = max(result, 1)  # At least 1 contract

        logger.info(
            f"Position sizing: target=${target:.0f}, "
            f"profit/contract=${net_profit_per_contract:.2f}, "
            f"needed={contracts}, bp_max={max_by_bp}, "
            f"final={result}"
        )

        return result

    def calculate_tp_debit(
        self,
        credit: float,
        tp_pct: float,
        cost_per_leg: float,
    ) -> float:
        """
        Calculate take-profit debit price.

        TP debit = credit * (1 - tp_pct) - (cost adjustment for round-trip close legs).
        We subtract a small amount to ensure the limit captures costs.

        Args:
            credit: Net credit received
            tp_pct: Take-profit percentage of credit to keep (e.g. 0.25)
            cost_per_leg: Transaction cost per leg

        Returns:
            Debit price for the close order
        """
        # We want to buy back for 75% of credit (keeping 25%)
        # Subtract cost adjustment: 4 close legs / 100 shares per contract
        cost_adjustment = (cost_per_leg * 4) / 100
        debit = credit * (1.0 - tp_pct) - cost_adjustment

        # Ceil to $0.05 (fill-friendly: debit = buying back, higher = easier fill)
        debit = math.ceil(debit * 20) / 20

        return max(debit, 0.05)  # Floor at minimum tick

    def calculate_sl_threshold(self, credit: float, sl_multiplier: float) -> float:
        """
        Calculate stop-loss threshold (debit to close that triggers exit).

        If the cost to close exceeds credit * sl_multiplier, we exit.

        Args:
            credit: Net credit received
            sl_multiplier: Stop-loss multiplier (e.g. 4.0 = 4x credit)

        Returns:
            Debit threshold that triggers stop-loss
        """
        return credit * sl_multiplier

    def get_current_position_value(self, chain_data: dict, strikes: Dict) -> Optional[float]:
        """
        Calculate current mid-price cost to close all 4 legs from fresh chain data.

        Args:
            chain_data: Fresh Schwab option chain response
            strikes: Original strikes dict from select_strikes()

        Returns:
            Current debit to close (positive number), or None on error
        """
        call_exp_map = chain_data.get('callExpDateMap', {})
        put_exp_map = chain_data.get('putExpDateMap', {})
        exp_key = strikes['exp_key']

        puts = put_exp_map.get(exp_key, {})
        calls = call_exp_map.get(exp_key, {})

        if not puts or not calls:
            logger.warning(f"No chain data at {exp_key} for position valuation")
            return None

        # Find current quotes for each leg
        lp = self._find_option_by_strike(puts, strikes['long_put']['strike'])
        sp = self._find_option_by_strike(puts, strikes['short_put']['strike'])
        sc = self._find_option_by_strike(calls, strikes['short_call']['strike'])
        lc = self._find_option_by_strike(calls, strikes['long_call']['strike'])

        if not all([lp, sp, sc, lc]):
            logger.warning("Could not find all 4 legs in current chain for valuation")
            return None

        # Cost to close: buy back shorts at ask, sell longs at bid
        debit_to_close = (
            sp['ask'] + sc['ask']   # Buy back shorts
            - lp['bid'] - lc['bid']  # Sell longs
        )

        return round(max(debit_to_close, 0), 2)

    def select_strikes_manual(
        self,
        chain_data: dict,
        short_put_strike: float,
        short_call_strike: float,
        wing_width: float = 25.0,
    ) -> Optional[Dict]:
        """
        Select IC strikes using explicit short strikes (for testing or when greeks unavailable).

        Args:
            chain_data: Schwab option chain response
            short_put_strike: Explicit short put strike
            short_call_strike: Explicit short call strike
            wing_width: Width of each wing in dollars
        """
        call_exp_map = chain_data.get('callExpDateMap', {})
        put_exp_map = chain_data.get('putExpDateMap', {})

        if not call_exp_map or not put_exp_map:
            logger.error("Option chain missing callExpDateMap or putExpDateMap")
            return None

        exp_key = self._find_0dte_expiration(call_exp_map)
        if not exp_key:
            return None

        puts_at_exp = put_exp_map.get(exp_key, {})
        calls_at_exp = call_exp_map.get(exp_key, {})

        short_put = self._find_option_by_strike(puts_at_exp, short_put_strike)
        short_call = self._find_option_by_strike(calls_at_exp, short_call_strike)

        if not short_put or not short_call:
            logger.error("Could not find specified short strikes in chain")
            return None

        long_put = self._find_option_by_strike(
            puts_at_exp, short_put_strike - wing_width,
            exclude_strike=short_put['strike'], direction='below'
        )
        long_call = self._find_option_by_strike(
            calls_at_exp, short_call_strike + wing_width,
            exclude_strike=short_call['strike'], direction='above'
        )

        if not long_put or not long_call:
            logger.error("Could not find wing options at specified width")
            return None

        net_credit = (
            short_put['mid'] + short_call['mid']
            - long_put['mid'] - long_call['mid']
        )

        exp_date = exp_key.split(':')[0]

        result = {
            'long_put': long_put,
            'short_put': short_put,
            'short_call': short_call,
            'long_call': long_call,
            'net_credit': round(math.floor(net_credit * 20) / 20, 2),
            'expiration': exp_date,
            'exp_key': exp_key,
        }

        logger.info(
            f"IC strikes (manual): "
            f"{long_put['strike']}/{short_put['strike']}p - "
            f"{short_call['strike']}/{long_call['strike']}c "
            f"Credit: ${net_credit:.2f}"
        )

        return result

    @staticmethod
    def parse_option_symbol(symbol: str) -> Optional[Dict]:
        """
        Parse OCC option symbol into components.

        Format: "SPXW  260217P06780000" or "SPX   260217C06100000"
          - Root: variable length, left-padded to 6 chars
          - Date: YYMMDD (6 chars)
          - Type: P or C (1 char)
          - Strike: 8 digits = 5 integer + 3 decimal (e.g. 06780000 = 6780.000)

        Returns:
            Dict with strike, option_type, expiration, or None on parse failure.
        """
        # Strip whitespace from root portion
        clean = symbol.strip()
        if len(clean) < 15:
            logger.warning(f"Symbol too short to parse: '{symbol}'")
            return None

        # Last 15 chars: YYMMDD + P/C + 8-digit strike
        suffix = clean[-15:]
        date_part = suffix[:6]       # YYMMDD
        opt_type = suffix[6]          # P or C
        strike_raw = suffix[7:]       # 8 digits

        if opt_type not in ('P', 'C'):
            logger.warning(f"Invalid option type '{opt_type}' in symbol '{symbol}'")
            return None

        try:
            year = 2000 + int(date_part[:2])
            month = int(date_part[2:4])
            day = int(date_part[4:6])
            exp_date = f"{year}-{month:02d}-{day:02d}"

            strike = int(strike_raw) / 1000.0
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse date/strike from symbol '{symbol}'")
            return None

        return {
            'strike': strike,
            'option_type': opt_type,
            'expiration': exp_date,
        }

    # ===== Private helpers =====

    def _find_0dte_expiration(self, exp_map: dict) -> Optional[str]:
        """Find the 0DTE expiration key from the chain map."""
        best_key = None
        best_dte = float('inf')

        for key in exp_map:
            # Key format: "2026-02-18:0" where the number after : is DTE
            parts = key.split(':')
            if len(parts) == 2:
                try:
                    dte = int(parts[1])
                    if 0 <= dte < best_dte:
                        best_dte = dte
                        best_key = key
                except ValueError:
                    continue

        if best_key:
            logger.debug(f"Found 0DTE expiration: {best_key}")

        return best_key

    def _find_option_by_delta(self, strikes: dict, target_delta: float) -> Optional[Dict]:
        """Find option closest to target absolute delta."""
        best_option = None
        best_diff = float('inf')

        for strike_str, contracts in strikes.items():
            contract = contracts[0] if isinstance(contracts, list) else list(contracts.values())[0]

            delta = contract.get('delta', 0)
            # Skip invalid deltas (Schwab returns -999 when market is closed)
            if not delta or abs(delta) >= 999:
                continue

            diff = abs(abs(delta) - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_option = {
                    'symbol': contract.get('symbol', ''),
                    'strike': float(strike_str),
                    'delta': delta,
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'mid': (contract.get('bid', 0) + contract.get('ask', 0)) / 2,
                    'open_interest': contract.get('openInterest', 0),
                    'volume': contract.get('totalVolume', 0),
                }

        return best_option

    def _find_option_by_strike(
        self,
        strikes: dict,
        target_strike: float,
        exclude_strike: float = None,
        direction: str = None,
    ) -> Optional[Dict]:
        """
        Find option closest to target strike.

        Args:
            strikes: Strike map from chain
            target_strike: Target strike price
            exclude_strike: Strike to skip
            direction: 'below' or 'above' to constrain search
        """
        best_option = None
        best_diff = float('inf')

        for strike_str, contracts in strikes.items():
            strike = float(strike_str)

            if exclude_strike is not None and abs(strike - exclude_strike) < 0.01:
                continue

            if direction == 'below' and strike >= exclude_strike:
                continue
            if direction == 'above' and strike <= exclude_strike:
                continue

            diff = abs(strike - target_strike)
            if diff < best_diff:
                best_diff = diff
                contract = contracts[0] if isinstance(contracts, list) else list(contracts.values())[0]
                best_option = {
                    'symbol': contract.get('symbol', ''),
                    'strike': strike,
                    'delta': contract.get('delta', 0),
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'mid': (contract.get('bid', 0) + contract.get('ask', 0)) / 2,
                    'open_interest': contract.get('openInterest', 0),
                    'volume': contract.get('totalVolume', 0),
                }

        return best_option
