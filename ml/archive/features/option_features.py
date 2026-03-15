"""
Option-based features (8 features).

Extracts features from option chain data:
- IV spreads and skew (call/put smile)
- Option deltas (market-estimated probability)
- Put/call ratios (sentiment)
- ATM vs OTM IV comparison
"""

from typing import Dict, List, Optional
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class OptionFeaturesExtractor(BaseFeatureExtractor):
    """Extract option-based features from option chain."""

    async def extract(self, option_chain: dict, underlying_price: Optional[float] = None) -> Dict[str, float]:
        """
        Extract option features.

        Args:
            option_chain: Option chain data from Schwab API
            underlying_price: Current underlying price (optional, extracted from chain if not provided)

        Returns:
            Dict with 8 option features
        """
        features = {}

        try:
            if not option_chain or 'callExpDateMap' not in option_chain or 'putExpDateMap' not in option_chain:
                logger.warning("Invalid or empty option chain")
                return self._get_default_features()

            # Get underlying price
            if underlying_price is None:
                underlying_price = option_chain.get('underlyingPrice', 0)

            if underlying_price == 0:
                logger.warning("Could not determine underlying price")
                return self._get_default_features()

            # Extract nearest expiration (0DTE or closest)
            calls = self._get_nearest_expiration(option_chain['callExpDateMap'])
            puts = self._get_nearest_expiration(option_chain['putExpDateMap'])

            if not calls or not puts:
                logger.warning("No options found in chain")
                return self._get_default_features()

            # Find ATM and OTM options
            atm_call = self._find_strike_by_delta(calls, target_delta=0.50, is_call=True)
            atm_put = self._find_strike_by_delta(puts, target_delta=-0.50, is_call=False)
            otm_call_10 = self._find_strike_by_delta(calls, target_delta=0.10, is_call=True)
            otm_put_10 = self._find_strike_by_delta(puts, target_delta=-0.10, is_call=False)

            # Extract IVs (Schwab returns volatility already as percentage, e.g. 24.78 = 24.78%)
            features['iv_call_atm'] = atm_call.get('volatility', 20.0) if atm_call else 20.0
            features['iv_put_atm'] = atm_put.get('volatility', 20.0) if atm_put else 20.0
            features['iv_call_otm_10'] = otm_call_10.get('volatility', 22.0) if otm_call_10 else 22.0
            features['iv_put_otm_10'] = otm_put_10.get('volatility', 24.0) if otm_put_10 else 24.0

            # IV skew (put skew typically higher)
            # Positive skew = puts more expensive than calls (normal)
            features['iv_skew'] = features['iv_put_otm_10'] - features['iv_call_otm_10']

            # Deltas
            features['delta_call_atm'] = abs(atm_call.get('delta', 0.50)) if atm_call else 0.50
            features['delta_put_atm'] = abs(atm_put.get('delta', -0.50)) if atm_put else 0.50

            # Put/call ratio (volume or open interest)
            put_volume = sum(opt[0].get('totalVolume', 0) for opt in puts.values() if opt)
            call_volume = sum(opt[0].get('totalVolume', 0) for opt in calls.values() if opt)

            if call_volume > 0:
                features['put_call_ratio'] = put_volume / call_volume
            else:
                features['put_call_ratio'] = 1.0

            logger.debug(f"Extracted {len(features)} option features")

        except Exception as e:
            logger.error(f"Option feature extraction failed: {e}", exc_info=True)
            features = self._get_default_features()

        return features

    def _get_nearest_expiration(self, exp_date_map: dict) -> dict:
        """Get options for nearest expiration date."""
        if not exp_date_map:
            return {}

        # Sort by expiration date (keys are formatted as 'YYYY-MM-DD:DTE')
        sorted_expirations = sorted(exp_date_map.keys())

        if not sorted_expirations:
            return {}

        # Get first expiration's strikes
        first_exp = sorted_expirations[0]
        return exp_date_map[first_exp]

    def _find_strike_by_delta(
        self,
        options: dict,
        target_delta: float,
        is_call: bool
    ) -> Optional[dict]:
        """
        Find option closest to target delta.

        Args:
            options: Dict of strike -> option data
            target_delta: Target delta (e.g., 0.50 for ATM call, -0.10 for 10-delta put)
            is_call: True for calls, False for puts

        Returns:
            Option dict or None
        """
        if not options:
            return None

        best_option = None
        best_diff = float('inf')

        for strike_key, strike_options in options.items():
            # strike_options is a list of contracts at this strike
            if not strike_options:
                continue

            option = strike_options[0]  # Take first contract
            delta = option.get('delta', 0)

            if delta is None:
                continue

            # Calculate difference from target
            diff = abs(delta - target_delta)

            if diff < best_diff:
                best_diff = diff
                best_option = option

        return best_option

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values (neutral conditions)."""
        return {
            'iv_call_atm': 20.0,  # 20% IV
            'iv_put_atm': 20.0,
            'iv_call_otm_10': 22.0,  # Slight elevation
            'iv_put_otm_10': 24.0,  # Put skew
            'iv_skew': 2.0,  # 2% put skew
            'delta_call_atm': 0.50,
            'delta_put_atm': 0.50,
            'put_call_ratio': 1.0,
        }
