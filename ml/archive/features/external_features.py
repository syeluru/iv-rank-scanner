"""
External/macro data features (3 features).

Extracts features from external data sources:
- HY OAS (High Yield Option-Adjusted Spread) - Credit stress indicator
- Yield curve (10Y-2Y spread) - Economic outlook indicator
- Market stress index - Composite stress measure

Data sources:
- FRED API (Federal Reserve Economic Data)
- Stored in macro_data table (requires daily sync)
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from ml.features.base import BaseFeatureExtractor
from loguru import logger


class ExternalFeaturesExtractor(BaseFeatureExtractor):
    """Extract external macro data features."""

    # Normal baseline values
    HY_OAS_NORMAL = 350.0  # basis points
    YIELD_CURVE_NORMAL = 50.0  # basis points

    def __init__(self, db_manager=None):
        """
        Initialize external features extractor.

        Args:
            db_manager: Optional DatabaseManager for querying macro_data table
        """
        self.db_manager = db_manager

    async def extract(self, timestamp: datetime) -> Dict[str, float]:
        """
        Extract external features.

        Args:
            timestamp: Current timestamp

        Returns:
            Dict with 3 external features
        """
        features = {}

        try:
            # Try to get data from database
            if self.db_manager:
                features = await self._extract_from_db(timestamp)
            else:
                features = self._get_default_features()

            logger.debug(f"Extracted {len(features)} external features")

        except Exception as e:
            logger.error(f"External feature extraction failed: {e}", exc_info=True)
            features = self._get_default_features()

        return features

    async def _extract_from_db(self, timestamp: datetime) -> Dict[str, float]:
        """
        Extract features from macro_data database table.

        Args:
            timestamp: Current timestamp

        Returns:
            Dict with external features
        """
        # Get most recent data (macro data is typically daily)
        query_date = timestamp.date()

        # Look back up to 5 days to handle weekends/holidays
        for days_back in range(5):
            check_date = query_date - timedelta(days=days_back)

            try:
                # Query HY OAS
                hy_oas_query = """
                    SELECT value FROM macro_data
                    WHERE date = ? AND metric_name = 'HY_OAS'
                    LIMIT 1
                """
                hy_oas_result = self.db_manager.fetch_one(hy_oas_query, (check_date,))
                hy_oas = hy_oas_result[0] if hy_oas_result else self.HY_OAS_NORMAL

                # Query yield curve
                yield_curve_query = """
                    SELECT value FROM macro_data
                    WHERE date = ? AND metric_name = 'YIELD_CURVE_10Y_2Y'
                    LIMIT 1
                """
                yield_curve_result = self.db_manager.fetch_one(yield_curve_query, (check_date,))
                yield_curve = yield_curve_result[0] if yield_curve_result else self.YIELD_CURVE_NORMAL

                # If we got data, break
                if hy_oas_result or yield_curve_result:
                    break

            except Exception as e:
                logger.warning(f"DB query failed for {check_date}: {e}")
                continue

        # Calculate market stress index
        # Composite of HY OAS elevation and yield curve inversion
        stress_index = self._calculate_stress_index(hy_oas, yield_curve)

        return {
            'hy_oas': float(hy_oas),
            'yield_curve_10y_2y': float(yield_curve),
            'market_stress_index': float(stress_index),
        }

    def _calculate_stress_index(self, hy_oas: float, yield_curve: float) -> float:
        """
        Calculate composite market stress index (0-100 scale).

        Components:
        - HY OAS elevation (high = stress)
        - Yield curve inversion (negative = stress)

        Returns:
            Stress score: 0-30 = low, 30-70 = medium, 70-100 = high
        """
        # HY OAS stress (normalized)
        # Normal: 300-400 bps, Stress: >600 bps, Crisis: >800 bps
        oas_stress = 0.0
        if hy_oas > 400:
            oas_stress = min((hy_oas - 400) / 4, 50)  # Max 50 points

        # Yield curve stress (inversion = stress)
        # Normal: +50 bps, Warning: 0 to +20, Stress: negative
        curve_stress = 0.0
        if yield_curve < 20:
            curve_stress = max(20 - yield_curve, 0) * 2.5  # Max 50 points

        # Combine
        total_stress = oas_stress + curve_stress

        return min(total_stress, 100.0)

    def _get_default_features(self) -> Dict[str, float]:
        """
        Return default feature values (neutral conditions).

        Default values represent normal market conditions:
        - HY OAS: 350 bps (normal credit spreads)
        - Yield Curve: +50 bps (normal positive slope)
        - Stress Index: 25 (low-medium stress)
        """
        return {
            'hy_oas': self.HY_OAS_NORMAL,
            'yield_curve_10y_2y': self.YIELD_CURVE_NORMAL,
            'market_stress_index': 25.0,  # Low-medium stress
        }
