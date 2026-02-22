#!/usr/bin/env python3
"""
Schwab Order Placement Script

This script can operate in two modes:
1. Interactive mode: Accepts natural language order descriptions
2. Programmatic mode: Accepts structured order data from other scripts

Examples:
    # Interactive natural language mode
    python scripts/place_schwab_order.py
    > place an iron condor on SPX for the mid-feb weekly at 0.1 delta

    # Programmatic mode (from algo trading script)
    python scripts/place_schwab_order.py --order-json '{"symbol": "AAPL", ...}'

    # Direct order placement
    python scripts/place_schwab_order.py --symbol AAPL --action BUY --type CALL --strike 180 --expiration 2026-03-20 --quantity 1
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from loguru import logger

from execution.broker_api.schwab_client import schwab_client
from schwab.orders import options as schwab_options

console = Console()


class NaturalLanguageParser:
    """
    Enhanced natural language parser for option orders.

    Supports patterns like:
    - "buy 1 AAPL 180 call expiring March 20"
    - "sell iron condor on SPX at 0.1 delta for Feb expiration"
    - "iron condor on SPX: 6550/6600 put spread, 7100/7150 call spread"
    - "SPX iron condor with strikes 6550, 6600, 7100, 7150"
    - "short the 6600/7100 strangle, long the 6550/7150 wings"
    """

    def __init__(self):
        self.keywords = {
            'actions': ['buy', 'sell', 'place', 'open', 'enter'],
            'strategies': ['iron condor', 'iron_condor', 'ic', 'vertical spread', 'call spread',
                          'put spread', 'butterfly', 'straddle', 'strangle', 'credit spread', 'debit spread'],
            'option_types': ['call', 'calls', 'put', 'puts'],
            'symbols': ['SPX', 'SPXW', 'SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'GOOGL', 'AMZN', 'MSFT',
                       'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'GLD', 'SLV', 'TLT', 'VIX'],
        }
        # Index symbols that require $ prefix for Schwab API
        # SPXW (weekly) maps to $SPX - Schwab uses same underlying for weekly/monthly
        self.index_symbols = {'SPX', 'SPXW', 'NDX', 'NDXP', 'RUT', 'RUTW', 'VIX', 'DJX', 'OEX', 'XSP'}
        # Map weekly symbols to their base index symbol
        self.symbol_aliases = {'SPXW': 'SPX', 'NDXP': 'NDX', 'RUTW': 'RUT'}

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Schwab API. Index symbols like SPX need $ prefix."""
        if not symbol:
            return symbol
        symbol_upper = symbol.upper()
        # Map aliases (e.g., SPXW -> SPX)
        base_symbol = self.symbol_aliases.get(symbol_upper, symbol_upper)
        # Add $ prefix for index symbols
        if base_symbol in self.index_symbols or symbol_upper in self.index_symbols:
            return f"${base_symbol}"
        return symbol_upper

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language into structured order.

        Args:
            text: Natural language order description

        Returns:
            Dictionary with order details or None if parsing fails
        """
        import re
        original_text = text
        text = text.lower().strip()

        # Detect strategy type
        strategy = self._detect_strategy(text)

        # Extract symbol
        symbol = self._extract_symbol(text)

        # Extract expiration
        expiration = self._extract_expiration(text)

        # Extract quantity
        quantity = self._extract_quantity(text)

        # Determine action
        action = self._detect_action(text)

        # === ENHANCED: Try to extract explicit strikes for iron condor ===
        explicit_strikes = None
        if strategy == 'iron_condor':
            explicit_strikes = self._extract_iron_condor_strikes(original_text)

        # Extract delta if mentioned (fallback if no explicit strikes)
        delta = self._extract_delta(text)

        # Extract single strike if single-leg
        strike = self._extract_strike(text) if strategy in ['call', 'put'] else None

        # Normalize symbol for Schwab API (e.g., SPX -> $SPX)
        api_symbol = self.normalize_symbol(symbol)

        result = {
            'strategy': strategy,
            'symbol': api_symbol,
            'display_symbol': symbol,
            'action': action,
            'quantity': quantity,
            'delta': delta,
            'expiration': expiration,
            'strike': strike,
            'explicit_strikes': explicit_strikes,  # New: explicit strike specifications
        }

        # If we have explicit strikes, show what was parsed
        if explicit_strikes:
            console.print(f"[green]✓ Parsed explicit strikes:[/green]")
            console.print(f"  Long Put:   ${explicit_strikes.get('long_put', 'auto')}")
            console.print(f"  Short Put:  ${explicit_strikes.get('short_put', 'auto')}")
            console.print(f"  Short Call: ${explicit_strikes.get('short_call', 'auto')}")
            console.print(f"  Long Call:  ${explicit_strikes.get('long_call', 'auto')}")

        # Validate we have minimum required fields
        if not symbol:
            console.print("[red]❌ Could not identify symbol[/red]")
            return None

        return result

    def _detect_strategy(self, text: str) -> Optional[str]:
        """Detect options strategy type."""
        # Check for IC abbreviation (case insensitive)
        text_lower = text.lower()
        if ' ic ' in f' {text_lower} ' or text_lower.startswith('ic ') or text_lower.endswith(' ic') or ' ic:' in text_lower:
            return 'iron_condor'

        for strategy in self.keywords['strategies']:
            if strategy in text_lower:
                return strategy.replace(' ', '_')

        # Check for simple option types
        if 'call' in text_lower and 'spread' not in text_lower:
            return 'call'
        if 'put' in text_lower and 'spread' not in text_lower:
            return 'put'

        return None

    def _extract_iron_condor_strikes(self, text: str) -> Optional[Dict[str, float]]:
        """
        Extract explicit strike prices for iron condor from natural language.

        Handles patterns like:
        - "6550/6600 put spread, 7100/7150 call spread"
        - "strikes 6550, 6600, 7100, 7150"
        - "long put 6550, short put 6600, short call 7100, long call 7150"
        - "6550-6600-7100-7150"
        - "put side 6550/6600, call side 7100/7150"
        - "short 6600 put and 7100 call, wings at 6550 and 7150"

        Returns:
            Dictionary with long_put, short_put, short_call, long_call strikes
        """
        import re

        strikes = {}
        text_lower = text.lower()

        # Pattern 1: "X/Y put spread" and "X/Y call spread"
        put_spread = re.search(r'(\d{3,5})\s*/\s*(\d{3,5})\s*(?:put|p)\s*(?:spread)?', text_lower)
        call_spread = re.search(r'(\d{3,5})\s*/\s*(\d{3,5})\s*(?:call|c)\s*(?:spread)?', text_lower)

        if put_spread:
            s1, s2 = float(put_spread.group(1)), float(put_spread.group(2))
            strikes['long_put'] = min(s1, s2)
            strikes['short_put'] = max(s1, s2)
        if call_spread:
            s1, s2 = float(call_spread.group(1)), float(call_spread.group(2))
            strikes['short_call'] = min(s1, s2)
            strikes['long_call'] = max(s1, s2)

        if len(strikes) == 4:
            return strikes

        # Pattern 2: "put side X/Y" and "call side X/Y"
        put_side = re.search(r'put\s*(?:side|wing)?\s*:?\s*(\d{3,5})\s*/\s*(\d{3,5})', text_lower)
        call_side = re.search(r'call\s*(?:side|wing)?\s*:?\s*(\d{3,5})\s*/\s*(\d{3,5})', text_lower)

        if put_side:
            s1, s2 = float(put_side.group(1)), float(put_side.group(2))
            strikes['long_put'] = min(s1, s2)
            strikes['short_put'] = max(s1, s2)
        if call_side:
            s1, s2 = float(call_side.group(1)), float(call_side.group(2))
            strikes['short_call'] = min(s1, s2)
            strikes['long_call'] = max(s1, s2)

        if len(strikes) == 4:
            return strikes

        # Pattern 3: Four numbers in sequence "strikes 6550, 6600, 7100, 7150" or "6550-6600-7100-7150"
        four_strikes = re.findall(r'\b(\d{3,5})\b', text)
        # Filter to reasonable strike values (> 100 for most underlyings)
        four_strikes = [float(s) for s in four_strikes if 100 < float(s) < 50000]

        if len(four_strikes) >= 4:
            # Take the first 4 unique strikes, sorted
            unique_strikes = sorted(set(four_strikes))[:4]
            if len(unique_strikes) == 4:
                strikes['long_put'] = unique_strikes[0]
                strikes['short_put'] = unique_strikes[1]
                strikes['short_call'] = unique_strikes[2]
                strikes['long_call'] = unique_strikes[3]
                return strikes

        # Pattern 4: Explicit leg specifications
        # "long put 6550" or "long put at 6550" or "buy 6550 put"
        long_put = re.search(r'(?:long|buy)\s*(?:the\s*)?(\d{3,5})?\s*put|put\s*(?:at\s*)?(\d{3,5})\s*(?:long|buy)', text_lower)
        if not long_put:
            long_put = re.search(r'(?:long|buy)\s*put\s*(?:at|@|:)?\s*(\d{3,5})', text_lower)
        if long_put:
            strike_val = next((g for g in long_put.groups() if g), None)
            if strike_val:
                strikes['long_put'] = float(strike_val)

        short_put = re.search(r'(?:short|sell)\s*(?:the\s*)?(\d{3,5})?\s*put|put\s*(?:at\s*)?(\d{3,5})\s*(?:short|sell)', text_lower)
        if not short_put:
            short_put = re.search(r'(?:short|sell)\s*put\s*(?:at|@|:)?\s*(\d{3,5})', text_lower)
        if short_put:
            strike_val = next((g for g in short_put.groups() if g), None)
            if strike_val:
                strikes['short_put'] = float(strike_val)

        short_call = re.search(r'(?:short|sell)\s*(?:the\s*)?(\d{3,5})?\s*call|call\s*(?:at\s*)?(\d{3,5})\s*(?:short|sell)', text_lower)
        if not short_call:
            short_call = re.search(r'(?:short|sell)\s*call\s*(?:at|@|:)?\s*(\d{3,5})', text_lower)
        if short_call:
            strike_val = next((g for g in short_call.groups() if g), None)
            if strike_val:
                strikes['short_call'] = float(strike_val)

        long_call = re.search(r'(?:long|buy)\s*(?:the\s*)?(\d{3,5})?\s*call|call\s*(?:at\s*)?(\d{3,5})\s*(?:long|buy)', text_lower)
        if not long_call:
            long_call = re.search(r'(?:long|buy)\s*call\s*(?:at|@|:)?\s*(\d{3,5})', text_lower)
        if long_call:
            strike_val = next((g for g in long_call.groups() if g), None)
            if strike_val:
                strikes['long_call'] = float(strike_val)

        # Pattern 5: "short X and Y" for the short strikes, "wings at A and B"
        short_strikes = re.search(r'short\s*(?:the\s*)?(\d{3,5})\s*(?:and|&|,)\s*(\d{3,5})', text_lower)
        if short_strikes:
            s1, s2 = float(short_strikes.group(1)), float(short_strikes.group(2))
            strikes['short_put'] = min(s1, s2)
            strikes['short_call'] = max(s1, s2)

        wings = re.search(r'(?:wings?|long)\s*(?:at|@|:)?\s*(\d{3,5})\s*(?:and|&|,)\s*(\d{3,5})', text_lower)
        if wings:
            s1, s2 = float(wings.group(1)), float(wings.group(2))
            strikes['long_put'] = min(s1, s2)
            strikes['long_call'] = max(s1, s2)

        # Return strikes only if we have all 4
        if len(strikes) == 4:
            return strikes

        # Return partial strikes so the builder can fill in the rest with delta
        if strikes:
            console.print(f"[yellow]Partial strikes parsed: {strikes}[/yellow]")
            console.print("[dim]Missing strikes will be auto-selected based on delta[/dim]")
            return strikes

        return None

    def _extract_symbol(self, text: str) -> Optional[str]:
        """Extract ticker symbol."""
        # Skip action/strategy words that might look like tickers
        skip_words = ['PLACE', 'BUY', 'SELL', 'CALL', 'PUT', 'FOR', 'THE', 'AND', 'AT',
                      'IRON', 'CONDOR', 'SPREAD', 'WITH', 'STRIKES', 'DELTA', 'SIDE',
                      'LONG', 'SHORT', 'WING', 'WINGS', 'MID', 'FEB', 'MAR', 'APR', 'JAN',
                      'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'WEEKLY']

        words = text.upper().split()

        # First pass: check for known symbols (highest priority)
        for word in words:
            # Clean punctuation
            clean_word = word.strip(':,.-/')
            if clean_word in self.keywords['symbols']:
                return clean_word

        # Second pass: check for valid ticker patterns
        for word in words:
            clean_word = word.strip(':,.-/')
            if clean_word not in skip_words and clean_word.isalpha() and 2 <= len(clean_word) <= 5:
                # Additional validation: not a common English word
                common_words = {'ON', 'IN', 'TO', 'OF', 'OR', 'AN', 'IT', 'IS', 'BE', 'AS', 'BY', 'SO', 'WE', 'HE', 'UP', 'DO', 'IF', 'MY', 'NO', 'GO'}
                if clean_word not in common_words:
                    return clean_word

        return None

    def _extract_delta(self, text: str) -> Optional[float]:
        """Extract delta value."""
        import re
        # Look for patterns like "0.1 delta", ".1 delta", "10 delta"
        match = re.search(r'(\d*\.?\d+)\s*delta', text)
        if match:
            delta = float(match.group(1))
            # Normalize: if > 1, assume it's in percentage (e.g., "10 delta" = 0.10)
            if delta > 1:
                delta = delta / 100
            return delta
        return None

    def _extract_expiration(self, text: str) -> Optional[str]:
        """Extract expiration description."""
        import re
        from datetime import datetime, timedelta

        # Pattern 0: "today" or "0dte" or "0 dte"
        if 'today' in text or '0dte' in text.replace(' ', '') or 'same day' in text:
            today = datetime.now()
            return f"{today.year}-{today.month:02d}-{today.day:02d}"

        # Pattern 0b: "tomorrow"
        if 'tomorrow' in text:
            tomorrow = datetime.now() + timedelta(days=1)
            return f"{tomorrow.year}-{tomorrow.month:02d}-{tomorrow.day:02d}"

        # Pattern 1: YYYY-MM-DD format
        match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
        if match:
            return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

        # Pattern 2: YYYY MM/DD or YYYY/MM/DD format (e.g., "2026 03/06" or "2026/03/06")
        match = re.search(r'(\d{4})\s*[/\s]\s*(\d{1,2})/(\d{1,2})', text)
        if match:
            return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

        # Pattern 3: MM/DD/YYYY or MM-DD-YYYY format
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
        if match:
            return f"{match.group(3)}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"

        # Pattern 4: MM/DD format (assume current/next year)
        match = re.search(r'(?:expir\w*|exp|on)\s*(\d{1,2})/(\d{1,2})(?!\d)', text)
        if match:
            month, day = int(match.group(1)), int(match.group(2))
            year = datetime.now().year
            target_date = datetime(year, month, day)
            if target_date < datetime.now():
                year += 1
            return f"{year}-{month:02d}-{day:02d}"

        # Look for month names with day numbers (e.g., "feb 4", "february 4th", "4 feb")
        months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        for month_name, month_num in months.items():
            # Pattern: "feb 4" or "feb 4th" or "february 4"
            match = re.search(rf'{month_name}\w*\s+(\d{{1,2}})(?:st|nd|rd|th)?', text)
            if match:
                day = int(match.group(1))
                year = datetime.now().year
                target_date = datetime(year, month_num, day)
                if target_date.date() < datetime.now().date():
                    year += 1
                return f"{year}-{month_num:02d}-{day:02d}"

            # Pattern: "4 feb" or "4th feb"
            match = re.search(rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}', text)
            if match:
                day = int(match.group(1))
                year = datetime.now().year
                target_date = datetime(year, month_num, day)
                if target_date.date() < datetime.now().date():
                    year += 1
                return f"{year}-{month_num:02d}-{day:02d}"

        # Look for month names without specific day (weekly/monthly)
        for month_name in months.keys():
            if month_name in text:
                # Check for timing modifiers (mid, early, late)
                timing = ''
                if 'mid' in text:
                    timing = 'mid_'
                elif 'early' in text:
                    timing = 'early_'
                elif 'late' in text:
                    timing = 'late_'

                # Check if it's a weekly or monthly
                if 'weekly' in text or 'week' in text:
                    return f"{timing}{month_name}_weekly"
                else:
                    return f"{timing}{month_name}_monthly"

        # Look for "X days" or "X weeks"
        match = re.search(r'(\d+)\s*(day|week|month)s?', text)
        if match:
            count = int(match.group(1))
            unit = match.group(2)
            return f"{count}_{unit}s"

        return None

    def _extract_strike(self, text: str) -> Optional[float]:
        """Extract strike price for single option orders."""
        import re
        text_lower = text.lower()

        # Look for explicit strike patterns: "strike 270", "270 strike", "$270"
        # But avoid matching dates like "03/06" or "2026"

        # Pattern 1: "strike X" or "X strike" or "@ X"
        match = re.search(r'(?:strike|@)\s*\$?(\d+(?:\.\d+)?)', text_lower)
        if match:
            return float(match.group(1))

        match = re.search(r'\$?(\d+(?:\.\d+)?)\s*strike', text_lower)
        if match:
            return float(match.group(1))

        # Pattern 2: "SYMBOL STRIKE call/put" like "AAPL 180 put" or "AMD 270 call"
        # Also handles "SYMBOL STRIKE calls/puts" (plural)
        match = re.search(r'\b[A-Za-z]{1,5}\s+(\d+(?:\.\d+)?)\s*(?:call|put|calls|puts|c|p)\b', text_lower)
        if match:
            val = float(match.group(1))
            if val > 1 and not (2020 <= val <= 2035):
                return val

        # Pattern 3: "call/put STRIKE" like "call 270" or "put 150"
        match = re.search(r'(?:call|put)\s+(\d+(?:\.\d+)?)\b', text_lower)
        if match:
            val = float(match.group(1))
            if val > 1 and not (2020 <= val <= 2035):
                return val

        # Pattern 4: Number before "calls/puts" that's likely a strike (3+ digits)
        # e.g., "250 calls" where 250 is a strike
        match = re.search(r'\b(\d{3,})(?:\.\d+)?\s*(?:calls?|puts?)\b', text_lower)
        if match:
            val = float(match.group(1))
            if not (2020 <= val <= 2035):  # Not a year
                return val

        # Pattern 4: Standalone dollar amount like "$270"
        match = re.search(r'\$(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        return None

    def _detect_action(self, text: str) -> str:
        """Detect buy/sell action."""
        if 'sell' in text.lower():
            return 'SELL'
        return 'BUY'  # Default to buy

    def _extract_quantity(self, text: str) -> int:
        """Extract quantity."""
        import re
        text_lower = text.lower()

        # Pattern 1: "X contract(s)" - explicit contract count
        match = re.search(r'(\d+)\s*contracts?', text_lower)
        if match:
            return int(match.group(1))

        # Pattern 2: "buy/sell X SYMBOL" at the start - e.g., "buy 2 AAPL"
        match = re.search(r'(?:buy|sell)\s+(\d+)\s+[A-Za-z]{1,5}\b', text_lower)
        if match:
            qty = int(match.group(1))
            # Sanity check: quantity is usually small (1-100), not a strike price
            if qty <= 100:
                return qty

        # Pattern 3: "X calls" or "X puts" but NOT "270 call" (which is strike)
        # Look for small numbers (1-20) followed by calls/puts
        match = re.search(r'\b(\d{1,2})\s*(?:calls?|puts?)\b', text_lower)
        if match:
            qty = int(match.group(1))
            if qty <= 20:  # Reasonable quantity
                return qty

        return 1  # Default to 1


class OrderBuilder:
    """
    Builds Schwab orders from structured data.
    """

    async def build_from_parsed(self, parsed: Dict[str, Any]) -> Optional[Dict]:
        """
        Build order from parsed natural language.

        Args:
            parsed: Parsed order dictionary

        Returns:
            Schwab order spec or None
        """
        strategy = parsed.get('strategy')
        symbol = parsed['symbol']

        console.print(f"\n[cyan]Building order for {symbol}...[/cyan]")

        # Get option chain - for multi-leg strategies we need the full strike range
        # Using date filtering instead of strike_count to get all strikes for target expiration
        console.print("  Fetching option chain from Schwab...")
        try:
            from datetime import datetime, timedelta
            today = datetime.now().date()

            # Determine date range based on parsed expiration
            expiration_desc = parsed.get('expiration')
            from_date = None
            to_date = None

            if expiration_desc:
                # Check if it's a specific date (YYYY-MM-DD format)
                import re
                date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', expiration_desc)
                if date_match:
                    target_date = datetime.strptime(expiration_desc, '%Y-%m-%d').date()
                    # For specific dates, use a 1-day window
                    from_date = target_date
                    to_date = target_date
                    console.print(f"  Target expiration: {target_date}")

            # Default range if no specific date
            if from_date is None:
                from_date = today
                to_date = today + timedelta(days=45)  # Get wider range, will filter later

            # For iron condors, fetch puts and calls separately to avoid overflow
            if strategy == 'iron_condor':
                console.print("    Fetching put chain...")
                put_chain = await schwab_client.get_option_chain(
                    symbol=symbol,
                    contract_type='PUT',
                    include_underlying_quote=True,
                    from_date=from_date,
                    to_date=to_date
                )

                console.print("    Fetching call chain...")
                call_chain = await schwab_client.get_option_chain(
                    symbol=symbol,
                    contract_type='CALL',
                    include_underlying_quote=True,
                    from_date=from_date,
                    to_date=to_date
                )

                # Merge the chains
                chain_data = put_chain.copy()
                chain_data['callExpDateMap'] = call_chain.get('callExpDateMap', {})
            else:
                chain_data = await schwab_client.get_option_chain(
                    symbol=symbol,
                    strike_count=50,
                    include_underlying_quote=True,
                    from_date=from_date,
                    to_date=to_date
                )
        except Exception as e:
            console.print(f"[red]❌ Failed to fetch option chain: {e}[/red]")
            return None

        underlying_price = chain_data.get('underlyingPrice', 0)
        console.print(f"  ✓ Underlying price: ${underlying_price:.2f}")

        # Build order based on strategy
        if strategy == 'iron_condor':
            return await self._build_iron_condor(parsed, chain_data, underlying_price)
        elif strategy == 'call' or strategy == 'put':
            return await self._build_single_option(parsed, chain_data, underlying_price)
        else:
            console.print(f"[yellow]⚠ Strategy '{strategy}' not yet supported[/yellow]")
            return None

    async def _build_iron_condor(self, parsed: Dict, chain_data: Dict, underlying_price: float) -> Dict:
        """Build an iron condor order."""
        target_delta = parsed.get('delta', 0.10)
        expiration_desc = parsed.get('expiration')
        explicit_strikes = parsed.get('explicit_strikes')

        # Find appropriate expiration
        exp_date, dte = self._find_expiration(chain_data, expiration_desc)
        if not exp_date:
            console.print("[red]❌ Could not find suitable expiration[/red]")
            return None

        console.print(f"  ✓ Selected expiration: {exp_date} ({dte} DTE)")

        # Find options at target delta
        call_exp_map = chain_data.get('callExpDateMap', {})
        put_exp_map = chain_data.get('putExpDateMap', {})

        exp_key = f"{exp_date}:{dte}"

        puts_at_exp = put_exp_map.get(exp_key, {})
        calls_at_exp = call_exp_map.get(exp_key, {})

        # === USE EXPLICIT STRIKES IF PROVIDED ===
        if explicit_strikes:
            console.print("  [cyan]Using explicit strikes from your input...[/cyan]")

            long_put = None
            short_put = None
            short_call = None
            long_call = None

            if 'long_put' in explicit_strikes:
                long_put = self._find_option_by_strike(
                    puts_at_exp, explicit_strikes['long_put'], 'PUT'
                )
                if long_put:
                    console.print(f"    ✓ Long put @ ${long_put['strike']:.0f}")
                else:
                    console.print(f"    [yellow]⚠ Could not find put at ${explicit_strikes['long_put']:.0f}[/yellow]")

            if 'short_put' in explicit_strikes:
                short_put = self._find_option_by_strike(
                    puts_at_exp, explicit_strikes['short_put'], 'PUT'
                )
                if short_put:
                    console.print(f"    ✓ Short put @ ${short_put['strike']:.0f}")
                else:
                    console.print(f"    [yellow]⚠ Could not find put at ${explicit_strikes['short_put']:.0f}[/yellow]")

            if 'short_call' in explicit_strikes:
                short_call = self._find_option_by_strike(
                    calls_at_exp, explicit_strikes['short_call'], 'CALL'
                )
                if short_call:
                    console.print(f"    ✓ Short call @ ${short_call['strike']:.0f}")
                else:
                    console.print(f"    [yellow]⚠ Could not find call at ${explicit_strikes['short_call']:.0f}[/yellow]")

            if 'long_call' in explicit_strikes:
                long_call = self._find_option_by_strike(
                    calls_at_exp, explicit_strikes['long_call'], 'CALL'
                )
                if long_call:
                    console.print(f"    ✓ Long call @ ${long_call['strike']:.0f}")
                else:
                    console.print(f"    [yellow]⚠ Could not find call at ${explicit_strikes['long_call']:.0f}[/yellow]")

            # Fill in any missing legs with delta-based selection
            if not short_put:
                console.print("    [dim]Auto-selecting short put by delta...[/dim]")
                short_put = self._find_option_by_delta(puts_at_exp, target_delta, 'PUT')
            if not short_call:
                console.print("    [dim]Auto-selecting short call by delta...[/dim]")
                short_call = self._find_option_by_delta(calls_at_exp, target_delta, 'CALL')

            if not short_put or not short_call:
                console.print("[red]❌ Could not find short options[/red]")
                return None

            # Calculate wing width if needed
            if underlying_price > 1000:
                wing_width = 50
            elif underlying_price > 100:
                wing_width = 10
            else:
                wing_width = 5

            if not long_put:
                console.print("    [dim]Auto-selecting long put...[/dim]")
                long_put_target = short_put['strike'] - wing_width
                long_put = self._find_option_by_strike(
                    puts_at_exp, long_put_target, 'PUT',
                    exclude_strike=short_put['strike'], direction='below'
                )

            if not long_call:
                console.print("    [dim]Auto-selecting long call...[/dim]")
                long_call_target = short_call['strike'] + wing_width
                long_call = self._find_option_by_strike(
                    calls_at_exp, long_call_target, 'CALL',
                    exclude_strike=short_call['strike'], direction='above'
                )

        else:
            # === DELTA-BASED SELECTION (original logic) ===
            # Find short put (sell put at -delta)
            short_put = self._find_option_by_delta(puts_at_exp, target_delta, 'PUT')

            # Find short call (sell call at +delta)
            short_call = self._find_option_by_delta(calls_at_exp, target_delta, 'CALL')

            if not short_put or not short_call:
                console.print("[red]❌ Could not find options at target delta[/red]")
                return None

            # Calculate appropriate wing width based on underlying price
            if underlying_price > 1000:
                wing_width = 50  # SPX, NDX - $50 wide wings
            elif underlying_price > 100:
                wing_width = 10  # Higher-priced stocks
            else:
                wing_width = 5   # Regular stocks

            # Find long put (buy put further OTM than short put - must be BELOW short strike)
            long_put_target = short_put['strike'] - wing_width
            long_put = self._find_option_by_strike(
                puts_at_exp, long_put_target, 'PUT',
                exclude_strike=short_put['strike'], direction='below'
            )

            # Find long call (buy call further OTM than short call - must be ABOVE short strike)
            long_call_target = short_call['strike'] + wing_width
            long_call = self._find_option_by_strike(
                calls_at_exp, long_call_target, 'CALL',
                exclude_strike=short_call['strike'], direction='above'
            )

        if not long_put or not long_call:
            console.print("[red]❌ Could not find wing options[/red]")
            return None

        return {
            'type': 'IRON_CONDOR',
            'symbol': parsed['symbol'],
            'legs': [
                {'action': 'BUY', 'option': long_put},
                {'action': 'SELL', 'option': short_put},
                {'action': 'SELL', 'option': short_call},
                {'action': 'BUY', 'option': long_call},
            ],
            'quantity': parsed.get('quantity', 1),
            'expiration': exp_date,
            'dte': dte,
            'max_credit': (short_put['bid'] + short_call['bid'] - long_put['ask'] - long_call['ask'])
        }

    async def _build_single_option(self, parsed: Dict, chain_data: Dict, underlying_price: float) -> Dict:
        """Build a single option order."""
        option_type = parsed['strategy'].upper()
        target_delta = parsed.get('delta')
        target_strike = parsed.get('strike')
        expiration_desc = parsed.get('expiration')

        # Find expiration
        exp_date, dte = self._find_expiration(chain_data, expiration_desc)
        if not exp_date:
            console.print("[red]❌ Could not find suitable expiration[/red]")
            return None

        console.print(f"  ✓ Selected expiration: {exp_date} ({dte} DTE)")

        # Get appropriate map
        exp_map = chain_data.get('callExpDateMap' if option_type == 'CALL' else 'putExpDateMap', {})
        exp_key = f"{exp_date}:{dte}"

        strikes_at_exp = exp_map.get(exp_key, {})

        # Find option by strike if provided, otherwise by delta
        if target_strike:
            console.print(f"  Looking for {option_type} at strike ${target_strike}...")
            option = self._find_option_by_strike(strikes_at_exp, target_strike, option_type)
            if not option:
                console.print(f"[red]❌ Could not find {option_type} at strike ${target_strike}[/red]")
                # Show available strikes
                available = sorted([float(k) for k in strikes_at_exp.keys()])
                if available:
                    console.print(f"[dim]Available strikes: {available[:10]}... to {available[-10:]}[/dim]")
                return None
            console.print(f"  ✓ Found {option_type} at ${option['strike']:.2f}")
        else:
            # Use delta-based selection
            if target_delta is None:
                target_delta = 0.30  # Default delta
            console.print(f"  Looking for {option_type} at {target_delta:.0%} delta...")
            option = self._find_option_by_delta(strikes_at_exp, target_delta, option_type)
            if not option:
                console.print(f"[red]❌ Could not find {option_type} at delta {target_delta}[/red]")
                return None
            console.print(f"  ✓ Found {option_type} at ${option['strike']:.2f} (delta: {option['delta']:.2f})")

        return {
            'type': 'SINGLE_OPTION',
            'symbol': parsed['symbol'],
            'action': parsed['action'],
            'option': option,
            'quantity': parsed.get('quantity', 1),
            'expiration': exp_date,
            'dte': dte,
            'option_type': option_type
        }

    def _find_expiration(self, chain_data: Dict, exp_desc: Optional[str]) -> tuple[Optional[str], Optional[int]]:
        """Find expiration date from description."""
        put_exp_map = chain_data.get('putExpDateMap', {})
        import re
        from datetime import datetime

        # If specific date provided
        if exp_desc and '-' in exp_desc:
            for exp_key in put_exp_map.keys():
                if exp_key.startswith(exp_desc):
                    parts = exp_key.split(':')
                    return parts[0], int(parts[1])

        # Find based on description
        target_dte = 30  # Default to ~30 DTE
        target_month = None

        if exp_desc:
            # Check for month name in description
            months = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            for month_name, month_num in months.items():
                if month_name in exp_desc.lower():
                    target_month = month_num
                    break

            # Check for timing hints
            is_mid_month = 'mid' in exp_desc.lower()
            is_weekly = 'weekly' in exp_desc or 'week' in exp_desc

            if target_month:
                # Calculate target date based on month
                today = datetime.now()
                target_year = today.year
                if target_month < today.month:
                    target_year += 1  # Next year

                if is_mid_month:
                    # Mid-month is around the 14th-15th
                    target_day = 14
                else:
                    # Default to 3rd Friday (monthly expiration) or end of month
                    target_day = 20

                try:
                    target_date = datetime(target_year, target_month, target_day)
                    target_dte = (target_date - today).days
                except ValueError:
                    pass  # Fall back to default
            elif 'weekly' in exp_desc or 'week' in exp_desc:
                target_dte = 7
            elif 'monthly' in exp_desc or 'month' in exp_desc:
                target_dte = 30
            elif 'day' in exp_desc:
                # Extract number of days
                match = re.search(r'(\d+)', exp_desc)
                if match:
                    target_dte = int(match.group(1))

        # Find closest expiration to target DTE
        best_exp = None
        best_dte = None
        best_diff = float('inf')

        for exp_key in put_exp_map.keys():
            parts = exp_key.split(':')
            if len(parts) == 2:
                exp_date, dte = parts[0], int(parts[1])
                diff = abs(dte - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_date
                    best_dte = dte

        return best_exp, best_dte

    def _find_option_by_delta(self, strikes: Dict, target_delta: float, option_type: str) -> Optional[Dict]:
        """Find option closest to target delta."""
        best_option = None
        best_diff = float('inf')

        for strike_str, contracts in strikes.items():
            contract = contracts[0] if isinstance(contracts, list) else list(contracts.values())[0]

            delta = contract.get('delta', 0)
            if not delta:
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

    def _find_option_by_strike(self, strikes: Dict, target_strike: float, option_type: str,
                                exclude_strike: float = None, direction: str = None) -> Optional[Dict]:
        """Find option at specific strike.

        Args:
            strikes: Dictionary of strikes to contracts
            target_strike: Target strike price
            option_type: 'PUT' or 'CALL'
            exclude_strike: Strike to exclude (to avoid selecting same strike as short)
            direction: 'below' to only find strikes below target, 'above' for above target
        """
        # Find closest strike with optional direction constraint
        best_option = None
        best_diff = float('inf')

        for strike_str, contracts in strikes.items():
            strike = float(strike_str)

            # Skip excluded strike
            if exclude_strike is not None and abs(strike - exclude_strike) < 0.01:
                continue

            # Apply direction constraint
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

    async def build_from_signal(self, signal: Dict[str, Any]) -> Optional[Dict]:
        """
        Build order from algo trading signal.

        Args:
            signal: Signal dictionary from algo strategy

        Returns:
            Schwab order spec
        """
        # Extract option details from signal
        return {
            'type': 'SINGLE_OPTION',
            'symbol': signal['symbol'],
            'action': 'BUY' if signal['direction'] == 'LONG' else 'SELL',
            'option': {
                'symbol': signal.get('option_symbol', ''),
                'strike': signal['strike'],
                'delta': signal.get('delta'),
                'bid': signal.get('option_bid'),
                'ask': signal.get('option_ask'),
                'mid': signal['entry_price'],
            },
            'quantity': signal.get('quantity', 1),
            'expiration': signal.get('expiration'),
            'option_type': signal.get('option_type')
        }


def display_order_preview(order_spec: Dict):
    """Display comprehensive order details for review."""
    console.print("\n" + "="*80)
    console.print("[bold cyan]📋 DETAILED ORDER PREVIEW[/bold cyan]")
    console.print("="*80 + "\n")

    # Basic Order Info
    console.print(Panel.fit(
        f"[bold]Strategy:[/bold] {order_spec['type'].replace('_', ' ').title()}\n"
        f"[bold]Underlying:[/bold] {order_spec['symbol']}\n"
        f"[bold]Expiration:[/bold] {order_spec['expiration']} ([cyan]{order_spec['dte']} DTE[/cyan])\n"
        f"[bold]Quantity:[/bold] {order_spec['quantity']} contract(s)",
        title="[bold cyan]Order Details[/bold cyan]",
        border_style="cyan"
    ))

    if order_spec['type'] == 'IRON_CONDOR':
        # Detailed legs table
        console.print("\n[bold cyan]Order Legs:[/bold cyan]")
        table = Table(show_header=True, header_style="bold cyan", border_style="cyan")
        table.add_column("Leg", style="bold", width=6)
        table.add_column("Action", width=8, justify="center")
        table.add_column("Option Symbol", width=25)
        table.add_column("Strike", width=10, justify="right")
        table.add_column("Delta", width=8, justify="right")
        table.add_column("Bid", width=8, justify="right")
        table.add_column("Ask", width=8, justify="right")
        table.add_column("Mid", width=8, justify="right")

        leg_names = ["Long Put", "Short Put", "Short Call", "Long Call"]
        for i, (leg, name) in enumerate(zip(order_spec['legs'], leg_names), 1):
            opt = leg['option']
            action_style = "green" if leg['action'] == 'BUY' else "red"
            table.add_row(
                f"{i}",
                f"[{action_style}]{leg['action']}[/{action_style}]",
                f"[dim]{opt.get('symbol', 'N/A')[:25]}[/dim]",
                f"[bold]${opt['strike']:.2f}[/bold]",
                f"{opt['delta']:+.3f}",
                f"${opt['bid']:.2f}",
                f"${opt['ask']:.2f}",
                f"[bold yellow]${opt['mid']:.2f}[/bold yellow]"
            )

        console.print(table)

        # Liquidity info
        console.print("\n[bold cyan]Liquidity Check:[/bold cyan]")
        liq_table = Table(show_header=True, header_style="bold cyan", border_style="cyan")
        liq_table.add_column("Leg", width=12)
        liq_table.add_column("Open Interest", justify="right")
        liq_table.add_column("Volume", justify="right")
        liq_table.add_column("Status", justify="center")

        for i, (leg, name) in enumerate(zip(order_spec['legs'], leg_names), 1):
            opt = leg['option']
            oi = opt.get('open_interest', 0)
            vol = opt.get('volume', 0)
            # Simple liquidity check
            if oi >= 100 and vol >= 10:
                status = "[green]✓ Good[/green]"
            elif oi >= 50 and vol >= 5:
                status = "[yellow]⚠ Fair[/yellow]"
            else:
                status = "[red]✗ Low[/red]"

            liq_table.add_row(
                name,
                f"{oi:,}",
                f"{vol:,}",
                status
            )

        console.print(liq_table)

        # Pricing and Risk
        console.print("\n[bold cyan]Pricing & Risk Analysis:[/bold cyan]")
        max_credit = order_spec['max_credit']
        width = 10  # Strike width (should calculate from actual strikes)
        max_loss = width - max_credit

        risk_table = Table(show_header=False, border_style="cyan", show_edge=False)
        risk_table.add_column("Label", style="bold", width=25)
        risk_table.add_column("Value", width=20)
        risk_table.add_column("Per Contract", width=20)

        risk_table.add_row(
            "Credit Received (Limit):",
            f"[green bold]${max_credit:.2f}[/green bold]",
            f"[green]${max_credit * 100:.2f}[/green]"
        )
        risk_table.add_row(
            "Maximum Loss:",
            f"[red bold]${max_loss:.2f}[/red bold]",
            f"[red]${max_loss * 100:.2f}[/red]"
        )
        risk_table.add_row(
            "Risk/Reward Ratio:",
            f"[yellow]{max_loss / max_credit:.2f}:1[/yellow]",
            ""
        )
        risk_table.add_row(
            "Total Credit (all contracts):",
            "",
            f"[green bold]${max_credit * order_spec['quantity'] * 100:.2f}[/green bold]"
        )
        risk_table.add_row(
            "Total Risk (all contracts):",
            "",
            f"[red bold]${max_loss * order_spec['quantity'] * 100:.2f}[/red bold]"
        )

        console.print(risk_table)

        # Order Type
        console.print("\n[bold cyan]Order Execution:[/bold cyan]")
        console.print(f"  [bold]Order Type:[/bold] Limit Order (Net Credit)")
        console.print(f"  [bold]Limit Price:[/bold] ${max_credit:.2f} credit per contract")
        console.print(f"  [bold]Time in Force:[/bold] Day Order (GTC optional)")

    elif order_spec['type'] == 'SINGLE_OPTION':
        opt = order_spec['option']
        action = order_spec['action']

        # Single option details
        console.print("\n[bold cyan]Option Details:[/bold cyan]")
        detail_table = Table(show_header=False, border_style="cyan", show_edge=False)
        detail_table.add_column("Field", style="bold", width=25)
        detail_table.add_column("Value", width=40)

        action_color = "green" if action == 'BUY' else "red"
        detail_table.add_row("Action:", f"[{action_color} bold]{action}[/{action_color} bold]")
        detail_table.add_row("Option Symbol:", f"[dim]{opt.get('symbol', 'N/A')}[/dim]")
        detail_table.add_row("Strike Price:", f"[bold]${opt['strike']:.2f}[/bold]")
        detail_table.add_row("Option Type:", f"[bold]{order_spec.get('option_type', 'OPTION')}[/bold]")
        detail_table.add_row("Delta:", f"{opt.get('delta', 0):+.3f}")
        detail_table.add_row("", "")
        detail_table.add_row("Bid Price:", f"${opt['bid']:.2f}")
        detail_table.add_row("Ask Price:", f"${opt['ask']:.2f}")
        detail_table.add_row("Mid Price:", f"[bold yellow]${opt.get('mid', 0):.2f}[/bold yellow]")
        detail_table.add_row("", "")
        detail_table.add_row("Open Interest:", f"{opt.get('open_interest', 0):,}")
        detail_table.add_row("Volume Today:", f"{opt.get('volume', 0):,}")

        console.print(detail_table)

        # Pricing
        console.print("\n[bold cyan]Pricing:[/bold cyan]")
        price_table = Table(show_header=False, border_style="cyan", show_edge=False)
        price_table.add_column("Label", style="bold", width=30)
        price_table.add_column("Value", width=30)

        cost_per_contract = opt.get('mid', 0) * 100
        total_cost = cost_per_contract * order_spec['quantity']

        if action == 'BUY':
            price_table.add_row("Cost per Contract:", f"[red]${cost_per_contract:.2f}[/red]")
            price_table.add_row("Total Cost:", f"[red bold]${total_cost:.2f}[/red bold]")
        else:  # SELL
            price_table.add_row("Credit per Contract:", f"[green]${cost_per_contract:.2f}[/green]")
            price_table.add_row("Total Credit:", f"[green bold]${total_cost:.2f}[/green bold]")

        console.print(price_table)

        # Order Type
        console.print("\n[bold cyan]Order Execution:[/bold cyan]")
        console.print(f"  [bold]Order Type:[/bold] Limit Order")
        console.print(f"  [bold]Limit Price:[/bold] ${opt.get('mid', 0):.2f} per share (${opt.get('mid', 0) * 100:.2f} per contract)")
        console.print(f"  [bold]Quantity:[/bold] {order_spec['quantity']} contract(s)")
        console.print(f"  [bold]Time in Force:[/bold] Day Order")

    # Final confirmation message
    console.print("\n" + "="*80)
    console.print("[bold yellow]⚠️  REVIEW CAREFULLY BEFORE PROCEEDING[/bold yellow]")
    console.print("="*80)


def display_confirmation_summary(order_spec: Dict):
    """Display a concise summary of critical order details for final review."""
    console.print("\n")

    if order_spec['type'] == 'IRON_CONDOR':
        strikes = [leg['option']['strike'] for leg in order_spec['legs']]
        summary_text = f"""
[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]
                            [bold yellow]FINAL ORDER CONFIRMATION[/bold yellow]
[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]

  [bold]Strategy:[/bold]      Iron Condor on {order_spec['symbol']}
  [bold]Expiration:[/bold]    {order_spec['expiration']} ({order_spec['dte']} days)
  [bold]Strikes:[/bold]       ${strikes[0]:.0f} / ${strikes[1]:.0f} / ${strikes[2]:.0f} / ${strikes[3]:.0f}
  [bold]Quantity:[/bold]      {order_spec['quantity']} contract(s)

  [bold green]Credit:[/bold green]         ${order_spec['max_credit']:.2f} per contract
  [bold green]Total Credit:[/bold green]   ${order_spec['max_credit'] * order_spec['quantity'] * 100:.2f}

  [bold]Order Type:[/bold]    Limit Order (Net Credit)
  [bold]Time in Force:[/bold] Day Order

[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]
"""
    else:  # SINGLE_OPTION
        opt = order_spec['option']
        action = order_spec['action']
        cost = opt.get('mid', 0) * 100 * order_spec['quantity']

        summary_text = f"""
[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]
                            [bold yellow]FINAL ORDER CONFIRMATION[/bold yellow]
[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]

  [bold]Action:[/bold]        {action}
  [bold]Underlying:[/bold]    {order_spec['symbol']}
  [bold]Strike:[/bold]        ${opt['strike']:.2f} {order_spec.get('option_type', '')}
  [bold]Expiration:[/bold]    {order_spec['expiration']} ({order_spec['dte']} days)
  [bold]Quantity:[/bold]      {order_spec['quantity']} contract(s)

  [bold]Limit Price:[/bold]   ${opt.get('mid', 0):.2f} per share
  [bold]Total {'Cost' if action == 'BUY' else 'Credit'}:[/bold]     [{'red' if action == 'BUY' else 'green'}]${abs(cost):.2f}[/{'red' if action == 'BUY' else 'green'}]

  [bold]Order Type:[/bold]    Limit Order
  [bold]Time in Force:[/bold] Day Order

[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]
"""

    console.print(Panel.fit(summary_text, border_style="yellow", padding=(0, 2)))


def build_schwab_order(order_spec: Dict) -> Dict:
    """Build the Schwab API order structure from order spec."""
    if order_spec['type'] == 'IRON_CONDOR':
        # Iron condor is a 4-leg spread
        schwab_order = {
            "orderType": "NET_CREDIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "price": f"{order_spec['max_credit']:.2f}",
            "complexOrderStrategyType": "IRON_CONDOR",
            "quantity": order_spec['quantity'],
            "orderLegCollection": []
        }

        instructions = ["BUY_TO_OPEN", "SELL_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_OPEN"]
        for leg, instruction in zip(order_spec['legs'], instructions):
            opt = leg['option']
            schwab_order["orderLegCollection"].append({
                "instruction": instruction,
                "quantity": order_spec['quantity'],
                "instrument": {
                    "symbol": opt['symbol'],
                    "assetType": "OPTION"
                }
            })

    elif order_spec['type'] == 'SINGLE_OPTION':
        opt = order_spec['option']
        action = order_spec['action']
        instruction = f"{action}_TO_OPEN"

        schwab_order = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "price": f"{opt.get('mid', 0):.2f}",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": order_spec['quantity'],
                    "instrument": {
                        "symbol": opt['symbol'],
                        "assetType": "OPTION"
                    }
                }
            ]
        }
    else:
        raise ValueError(f"Unknown order type: {order_spec['type']}")

    return schwab_order


def display_schwab_order_structure(order_spec: Dict):
    """Display the exact Schwab API order structure."""
    console.print("\n" + "="*80)
    console.print("[bold cyan]📡 SCHWAB API ORDER STRUCTURE[/bold cyan]")
    console.print("="*80)
    console.print("\n[dim]This is the exact order that will be sent to Schwab:[/dim]\n")

    schwab_order = build_schwab_order(order_spec)

    # Display as formatted JSON
    order_json = json.dumps(schwab_order, indent=2)
    syntax = Syntax(order_json, "json", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n" + "="*80)


async def modify_order_interactive(order_spec: Dict, chain_data: Dict = None) -> Dict:
    """
    Allow user to interactively modify order parameters using natural language.

    Args:
        order_spec: Current order specification
        chain_data: Option chain data for looking up new strikes

    Returns:
        Modified order specification
    """
    console.print("\n" + "="*80)
    console.print("[bold cyan]📝 MODIFY ORDER[/bold cyan]")
    console.print("="*80)

    if order_spec['type'] == 'IRON_CONDOR':
        legs = order_spec['legs']
        console.print("\n[bold]Current legs:[/bold]")
        console.print(f"  1. Long Put:   ${legs[0]['option']['strike']:.0f}")
        console.print(f"  2. Short Put:  ${legs[1]['option']['strike']:.0f}")
        console.print(f"  3. Short Call: ${legs[2]['option']['strike']:.0f}")
        console.print(f"  4. Long Call:  ${legs[3]['option']['strike']:.0f}")
        console.print(f"  Quantity: {order_spec['quantity']} | Credit: ${order_spec['max_credit']:.2f}")

        console.print("\n[bold]Describe your modification in natural language:[/bold]")
        console.print("[dim]Examples:[/dim]")
        console.print("[dim]  • 'strikes 6830, 6835, 6970, 6975' (change all 4 strikes)[/dim]")
        console.print("[dim]  • 'change quantity to 2'[/dim]")
        console.print("[dim]  • 'set credit to 5.50'[/dim]")
        console.print("[dim]  • 'move short put to 6600'[/dim]")
        console.print("[dim]  • 'widen put spread to 75 wide'[/dim]")
        console.print("[dim]  • 'move call side up 20 points'[/dim]")
        console.print("[dim]  • 'cancel' to go back[/dim]")

        mod_text = Prompt.ask("\n[bold cyan]Modification[/bold cyan]")

        if mod_text.lower() in ['cancel', 'back', 'none', 'n']:
            console.print("[yellow]Modification cancelled[/yellow]")
            return order_spec

        # Parse the modification
        order_spec = parse_and_apply_modification(order_spec, mod_text, chain_data)

    elif order_spec['type'] == 'SINGLE_OPTION':
        opt = order_spec['option']
        console.print(f"\n[bold]Current option:[/bold]")
        console.print(f"  Strike: ${opt['strike']:.0f}")
        console.print(f"  Quantity: {order_spec['quantity']}")
        console.print(f"  Limit: ${opt.get('mid', 0):.2f}")

        console.print("\n[bold]Describe your modification in natural language:[/bold]")
        console.print("[dim]Examples: 'change quantity to 5', 'set limit to 3.50', 'cancel'[/dim]")

        mod_text = Prompt.ask("\n[bold cyan]Modification[/bold cyan]")

        if mod_text.lower() in ['cancel', 'back', 'none', 'n']:
            console.print("[yellow]Modification cancelled[/yellow]")
            return order_spec

        order_spec = parse_and_apply_modification(order_spec, mod_text, chain_data)

    return order_spec


def parse_and_apply_modification(order_spec: Dict, mod_text: str, chain_data: Dict = None) -> Dict:
    """
    Parse natural language modification and apply it to the order.

    Args:
        order_spec: Current order specification
        mod_text: Natural language modification text
        chain_data: Option chain data for looking up new strikes

    Returns:
        Modified order specification
    """
    import re
    mod_text = mod_text.lower().strip()

    # === QUANTITY MODIFICATIONS ===
    qty_match = re.search(r'(?:quantity|qty|contracts?)\s*(?:to|=|:)?\s*(\d+)', mod_text)
    if not qty_match:
        qty_match = re.search(r'(\d+)\s*(?:contracts?|qty)', mod_text)
    if qty_match:
        new_qty = int(qty_match.group(1))
        order_spec['quantity'] = new_qty
        console.print(f"[green]✓ Quantity updated to {new_qty}[/green]")
        return order_spec

    # === CREDIT/PRICE MODIFICATIONS ===
    credit_match = re.search(r'(?:credit|price|limit)\s*(?:to|=|:)?\s*\$?(\d+\.?\d*)', mod_text)
    if not credit_match:
        credit_match = re.search(r'\$(\d+\.?\d*)\s*(?:credit|limit)?', mod_text)
    if credit_match:
        new_credit = float(credit_match.group(1))
        if order_spec['type'] == 'IRON_CONDOR':
            order_spec['max_credit'] = new_credit
            console.print(f"[green]✓ Credit updated to ${new_credit:.2f}[/green]")
        else:
            order_spec['option']['mid'] = new_credit
            console.print(f"[green]✓ Limit price updated to ${new_credit:.2f}[/green]")
        return order_spec

    # === STRIKE MODIFICATIONS (Iron Condor) ===
    if order_spec['type'] == 'IRON_CONDOR':
        legs = order_spec['legs']

        # === ALL FOUR STRIKES AT ONCE ===
        # Pattern: "change strikes to 6830, 6835, 6970, 6975" or "strikes 6830 6835 6970 6975"
        # or "6830/6835/6970/6975" or just four numbers
        four_strikes = re.findall(r'\b(\d{3,5})\b', mod_text)
        # Filter to reasonable strike values
        four_strikes = [float(s) for s in four_strikes if 100 < float(s) < 50000]

        if len(four_strikes) >= 4:
            # Take first 4 strikes and sort them
            strikes = sorted(four_strikes[:4])
            old_strikes = [legs[i]['option']['strike'] for i in range(4)]

            # Assign: long put (lowest), short put, short call, long call (highest)
            legs[0]['option']['strike'] = strikes[0]  # long put
            legs[1]['option']['strike'] = strikes[1]  # short put
            legs[2]['option']['strike'] = strikes[2]  # short call
            legs[3]['option']['strike'] = strikes[3]  # long call

            # Update symbols
            for i in range(4):
                legs[i]['option']['symbol'] = update_symbol_strike(
                    legs[i]['option']['symbol'], strikes[i]
                )

            console.print(f"[green]✓ All strikes updated:[/green]")
            console.print(f"    Long Put:   ${old_strikes[0]:.0f} → ${strikes[0]:.0f}")
            console.print(f"    Short Put:  ${old_strikes[1]:.0f} → ${strikes[1]:.0f}")
            console.print(f"    Short Call: ${old_strikes[2]:.0f} → ${strikes[2]:.0f}")
            console.print(f"    Long Call:  ${old_strikes[3]:.0f} → ${strikes[3]:.0f}")
            console.print("[yellow]⚠ Note: Prices are estimated. Consider refreshing chain data.[/yellow]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move short put to 6600" or "short put 6600"
        short_put_match = re.search(r'short\s*put\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if short_put_match:
            new_strike = float(short_put_match.group(1))
            old_strike = legs[1]['option']['strike']
            legs[1]['option']['strike'] = new_strike
            # Update symbol (approximate - would need chain lookup for exact)
            legs[1]['option']['symbol'] = update_symbol_strike(legs[1]['option']['symbol'], new_strike)
            console.print(f"[green]✓ Short put moved from ${old_strike:.0f} to ${new_strike:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move long put to 6550"
        long_put_match = re.search(r'long\s*put\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if long_put_match:
            new_strike = float(long_put_match.group(1))
            old_strike = legs[0]['option']['strike']
            legs[0]['option']['strike'] = new_strike
            legs[0]['option']['symbol'] = update_symbol_strike(legs[0]['option']['symbol'], new_strike)
            console.print(f"[green]✓ Long put moved from ${old_strike:.0f} to ${new_strike:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move short call to 7150"
        short_call_match = re.search(r'short\s*call\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if short_call_match:
            new_strike = float(short_call_match.group(1))
            old_strike = legs[2]['option']['strike']
            legs[2]['option']['strike'] = new_strike
            legs[2]['option']['symbol'] = update_symbol_strike(legs[2]['option']['symbol'], new_strike)
            console.print(f"[green]✓ Short call moved from ${old_strike:.0f} to ${new_strike:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move long call to 7200"
        long_call_match = re.search(r'long\s*call\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if long_call_match:
            new_strike = float(long_call_match.group(1))
            old_strike = legs[3]['option']['strike']
            legs[3]['option']['strike'] = new_strike
            legs[3]['option']['symbol'] = update_symbol_strike(legs[3]['option']['symbol'], new_strike)
            console.print(f"[green]✓ Long call moved from ${old_strike:.0f} to ${new_strike:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "widen put spread to 75" or "put width 75"
        put_width_match = re.search(r'(?:put\s*(?:spread|width|side)|widen\s*put)\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if put_width_match:
            new_width = float(put_width_match.group(1))
            short_put_strike = legs[1]['option']['strike']
            new_long_put_strike = short_put_strike - new_width
            old_long_put_strike = legs[0]['option']['strike']
            legs[0]['option']['strike'] = new_long_put_strike
            legs[0]['option']['symbol'] = update_symbol_strike(legs[0]['option']['symbol'], new_long_put_strike)
            console.print(f"[green]✓ Put spread widened: long put ${old_long_put_strike:.0f} → ${new_long_put_strike:.0f} (${new_width:.0f} wide)[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "widen call spread to 75" or "call width 75"
        call_width_match = re.search(r'(?:call\s*(?:spread|width|side)|widen\s*call)\s*(?:to|=|:)?\s*\$?(\d+)', mod_text)
        if call_width_match:
            new_width = float(call_width_match.group(1))
            short_call_strike = legs[2]['option']['strike']
            new_long_call_strike = short_call_strike + new_width
            old_long_call_strike = legs[3]['option']['strike']
            legs[3]['option']['strike'] = new_long_call_strike
            legs[3]['option']['symbol'] = update_symbol_strike(legs[3]['option']['symbol'], new_long_call_strike)
            console.print(f"[green]✓ Call spread widened: long call ${old_long_call_strike:.0f} → ${new_long_call_strike:.0f} (${new_width:.0f} wide)[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move call side up 20" or "shift calls up 20"
        call_shift_match = re.search(r'(?:move|shift)\s*call(?:s|side)?\s*(up|down)\s*\$?(\d+)', mod_text)
        if call_shift_match:
            direction = 1 if call_shift_match.group(1) == 'up' else -1
            shift = float(call_shift_match.group(2)) * direction
            old_short = legs[2]['option']['strike']
            old_long = legs[3]['option']['strike']
            legs[2]['option']['strike'] += shift
            legs[3]['option']['strike'] += shift
            legs[2]['option']['symbol'] = update_symbol_strike(legs[2]['option']['symbol'], legs[2]['option']['strike'])
            legs[3]['option']['symbol'] = update_symbol_strike(legs[3]['option']['symbol'], legs[3]['option']['strike'])
            console.print(f"[green]✓ Call side shifted {call_shift_match.group(1)} ${abs(shift):.0f}: ${old_short:.0f}/${old_long:.0f} → ${legs[2]['option']['strike']:.0f}/${legs[3]['option']['strike']:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

        # "move put side down 20" or "shift puts down 20"
        put_shift_match = re.search(r'(?:move|shift)\s*put(?:s|side)?\s*(up|down)\s*\$?(\d+)', mod_text)
        if put_shift_match:
            direction = 1 if put_shift_match.group(1) == 'up' else -1
            shift = float(put_shift_match.group(2)) * direction
            old_long = legs[0]['option']['strike']
            old_short = legs[1]['option']['strike']
            legs[0]['option']['strike'] += shift
            legs[1]['option']['strike'] += shift
            legs[0]['option']['symbol'] = update_symbol_strike(legs[0]['option']['symbol'], legs[0]['option']['strike'])
            legs[1]['option']['symbol'] = update_symbol_strike(legs[1]['option']['symbol'], legs[1]['option']['strike'])
            console.print(f"[green]✓ Put side shifted {put_shift_match.group(1)} ${abs(shift):.0f}: ${old_long:.0f}/${old_short:.0f} → ${legs[0]['option']['strike']:.0f}/${legs[1]['option']['strike']:.0f}[/green]")
            recalculate_iron_condor_credit(order_spec)
            return order_spec

    console.print(f"[yellow]Could not understand modification: '{mod_text}'[/yellow]")
    console.print("[dim]Try: 'strikes 6830, 6835, 6970, 6975', 'quantity to 2', 'credit to 5.50', 'short put to 6600'[/dim]")
    return order_spec


def update_symbol_strike(symbol: str, new_strike: float) -> str:
    """
    Update the strike price in an option symbol.

    Option symbols are formatted like: SPXW  260213P06540000
    The last 8 digits are the strike * 1000
    """
    import re
    # Match the pattern: base + date + type + strike (8 digits)
    match = re.match(r'(.+[PC])(\d{8})$', symbol.replace(' ', ''))
    if match:
        base = symbol[:-8]  # Keep everything except last 8 chars
        new_strike_str = f"{int(new_strike * 1000):08d}"
        return base + new_strike_str
    return symbol  # Return unchanged if pattern doesn't match


def recalculate_iron_condor_credit(order_spec: Dict):
    """
    Recalculate the max credit for an iron condor based on leg prices.
    Note: This uses stored bid/ask which may be stale after strike changes.
    """
    legs = order_spec['legs']
    # Credit = short put bid + short call bid - long put ask - long call ask
    short_put_bid = legs[1]['option'].get('bid', 0)
    short_call_bid = legs[2]['option'].get('bid', 0)
    long_put_ask = legs[0]['option'].get('ask', 0)
    long_call_ask = legs[3]['option'].get('ask', 0)

    new_credit = short_put_bid + short_call_bid - long_put_ask - long_call_ask

    # If we don't have accurate prices, estimate based on strike changes
    if new_credit <= 0:
        console.print("[yellow]⚠ Prices may be stale after strike changes. Consider refreshing.[/yellow]")
        # Keep existing credit as estimate
    else:
        order_spec['max_credit'] = new_credit
        console.print(f"[dim]Estimated credit updated to ${new_credit:.2f}[/dim]")


async def place_order(order_spec: Dict, dry_run: bool = False, show_api_structure: bool = True) -> bool:
    """
    Place order via Schwab.

    Args:
        order_spec: Order specification
        dry_run: If True, don't actually place the order
        show_api_structure: Show the exact API payload

    Returns:
        True if order placed successfully
    """
    # Show exact API structure before placing
    if show_api_structure:
        display_schwab_order_structure(order_spec)

        if not Confirm.ask("\n[bold]Does this order look correct?[/bold]", default=False):
            console.print("[yellow]Order cancelled[/yellow]")
            return False

    if dry_run:
        console.print("\n[yellow]✓ DRY RUN - Order validated but not placed[/yellow]")
        return True

    try:
        console.print("\n[cyan]Placing order via Schwab API...[/cyan]")

        # Build the Schwab order structure
        schwab_order = build_schwab_order(order_spec)

        # Place the order via Schwab client
        response = await schwab_client.place_order(order=schwab_order)

        console.print(f"[green]✓ Order submitted successfully![/green]")
        console.print(f"[dim]Response: {response}[/dim]")

        return True

    except Exception as e:
        console.print(f"[red]❌ Failed to place order: {e}[/red]")
        logger.exception("Order placement error")
        return False


async def interactive_mode():
    """Run in interactive natural language mode."""
    console.print(Panel.fit(
        "[bold cyan]Schwab Order Placement - Interactive Mode[/bold cyan]\n\n"
        "Describe your order in natural language.\n"
        "Examples:\n"
        "  • place an iron condor on SPX for the mid-feb weekly at 0.1 delta\n"
        "  • buy 1 AAPL 180 call expiring March 20\n"
        "  • sell 5 SPY puts at 0.30 delta for next week\n\n"
        "Type 'quit' to exit",
        border_style="cyan"
    ))

    parser = NaturalLanguageParser()
    builder = OrderBuilder()

    while True:
        console.print()
        order_text = Prompt.ask("[bold cyan]Order[/bold cyan]")

        if order_text.lower() in ['quit', 'exit', 'q']:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # Parse natural language
        console.print("\n[cyan]Parsing order...[/cyan]")
        parsed = parser.parse(order_text)

        if not parsed:
            console.print("[red]❌ Could not parse order. Please try again.[/red]")
            continue

        console.print("[green]✓ Order parsed successfully[/green]")
        console.print(f"  Strategy: {parsed.get('strategy')}")
        console.print(f"  Symbol: {parsed.get('symbol')}")
        console.print(f"  Delta: {parsed.get('delta')}")

        # Build order
        order_spec = await builder.build_from_parsed(parsed)

        if not order_spec:
            continue

        # Display comprehensive preview
        display_order_preview(order_spec)

        # Show confirmation summary
        display_confirmation_summary(order_spec)

        # Interactive modification loop
        while True:
            console.print("\n[bold cyan]Options:[/bold cyan]")
            console.print("  [green]y[/green] - Place this order")
            console.print("  [yellow]m[/yellow] - Modify order")
            console.print("  [red]n[/red] - Cancel order")

            choice = Prompt.ask("\n[bold]Your choice[/bold]", choices=["y", "m", "n"], default="n")

            if choice == "n":
                console.print("[yellow]Order cancelled[/yellow]")
                break

            elif choice == "m":
                # Modify order (pass chain_data for potential strike lookups)
                order_spec = await modify_order_interactive(order_spec, chain_data=None)
                # Re-display the updated order
                display_order_preview(order_spec)
                display_confirmation_summary(order_spec)
                continue

            elif choice == "y":
                dry_run = not Confirm.ask("  [yellow]⚠️  FINAL CONFIRMATION: Place REAL order (not dry run)?[/yellow]", default=False)
                success = await place_order(order_spec, dry_run=dry_run, show_api_structure=True)

                if success:
                    console.print("\n[bold green]✓ Order placed successfully![/bold green]")
                else:
                    console.print("\n[bold red]✗ Order placement failed[/bold red]")
                break


async def programmatic_mode(args):
    """Run in programmatic mode with structured data."""
    builder = OrderBuilder()

    # Parse signal data
    if args.order_json:
        signal = json.loads(args.order_json)
    elif args.signal_file:
        with open(args.signal_file) as f:
            signal = json.load(f)
    else:
        # Build from individual args
        signal = {
            'symbol': args.symbol,
            'action': args.action,
            'strike': args.strike,
            'expiration': args.expiration,
            'quantity': args.quantity,
            'option_type': args.type,
            'direction': 'LONG' if args.action == 'BUY' else 'SHORT',
            'entry_price': args.price or 0,
        }

    # Build order
    console.print("[cyan]Building order from signal...[/cyan]")
    order_spec = await builder.build_from_signal(signal)

    if not order_spec:
        console.print("[red]❌ Failed to build order[/red]")
        return False

    # Display comprehensive preview
    display_order_preview(order_spec)

    # Show confirmation summary
    display_confirmation_summary(order_spec)

    # Ask for confirmation unless --auto-approve
    if args.auto_approve:
        console.print("\n[yellow]⚠️  AUTO-APPROVE enabled - proceeding without manual confirmation[/yellow]")
        success = await place_order(order_spec, dry_run=args.dry_run, show_api_structure=True)
        return success
    elif Confirm.ask("\n[bold]Proceed with this order?[/bold]", default=False):
        success = await place_order(order_spec, dry_run=args.dry_run, show_api_structure=True)
        return success
    else:
        console.print("[yellow]Order cancelled[/yellow]")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Place Schwab orders via natural language or structured data"
    )

    # Modes
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive natural language mode')

    # Programmatic mode options
    parser.add_argument('--order-json', type=str,
                       help='Order data as JSON string')
    parser.add_argument('--signal-file', type=str,
                       help='Path to JSON file with signal data')

    # Direct order specification
    parser.add_argument('--symbol', type=str,
                       help='Stock symbol')
    parser.add_argument('--action', type=str, choices=['BUY', 'SELL'],
                       help='Buy or sell')
    parser.add_argument('--type', type=str, choices=['CALL', 'PUT'],
                       help='Option type')
    parser.add_argument('--strike', type=float,
                       help='Strike price')
    parser.add_argument('--expiration', type=str,
                       help='Expiration date (YYYY-MM-DD)')
    parser.add_argument('--quantity', type=int, default=1,
                       help='Number of contracts')
    parser.add_argument('--price', type=float,
                       help='Limit price')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview order without placing it')
    parser.add_argument('--auto-approve', action='store_true',
                       help='Skip approval prompt (programmatic mode only)')

    args = parser.parse_args()

    # Determine mode
    if args.interactive or (not args.order_json and not args.signal_file and not args.symbol):
        # Interactive mode (default)
        asyncio.run(interactive_mode())
    else:
        # Programmatic mode
        success = asyncio.run(programmatic_mode(args))
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
