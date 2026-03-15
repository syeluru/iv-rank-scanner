"""Tests for src/gex.py — GEX calculator module."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gex import compute_gex_features, find_zero_gamma, compute_gex_profile, GEX_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_greeks(strikes, rights, gammas):
    """Helper to create a greeks DataFrame."""
    return pd.DataFrame({"strike": strikes, "right": rights, "gamma": gammas})


def make_oi(strikes, rights, ois):
    """Helper to create an OI DataFrame."""
    return pd.DataFrame({"strike": strikes, "right": rights, "open_interest": ois})


# ---------------------------------------------------------------------------
# compute_gex_features
# ---------------------------------------------------------------------------

class TestComputeGexFeatures:
    def test_empty_greeks_returns_nan(self):
        greeks = pd.DataFrame(columns=["strike", "right", "gamma"])
        oi = make_oi([5000], ["call"], [100])
        result = compute_gex_features(greeks, oi, 5000.0)
        assert all(np.isnan(result[k]) for k in GEX_FEATURE_NAMES)

    def test_empty_oi_returns_nan(self):
        greeks = make_greeks([5000], ["call"], [0.05])
        oi = pd.DataFrame(columns=["strike", "right", "open_interest"])
        result = compute_gex_features(greeks, oi, 5000.0)
        assert all(np.isnan(result[k]) for k in GEX_FEATURE_NAMES)

    def test_nan_spot_returns_nan(self):
        greeks = make_greeks([5000], ["call"], [0.05])
        oi = make_oi([5000], ["call"], [100])
        result = compute_gex_features(greeks, oi, np.nan)
        assert all(np.isnan(result[k]) for k in GEX_FEATURE_NAMES)

    def test_zero_spot_returns_nan(self):
        greeks = make_greeks([5000], ["call"], [0.05])
        oi = make_oi([5000], ["call"], [100])
        result = compute_gex_features(greeks, oi, 0.0)
        assert all(np.isnan(result[k]) for k in GEX_FEATURE_NAMES)

    def test_single_call_positive_gex(self):
        """A single call should produce positive net GEX."""
        spot = 5000.0
        gamma = 0.001
        oi_val = 1000
        greeks = make_greeks([5000], ["call"], [gamma])
        oi = make_oi([5000], ["call"], [oi_val])
        result = compute_gex_features(greeks, oi, spot)

        expected_gex = 1.0 * gamma * oi_val * 100 * spot**2 * 0.01
        assert abs(result["net_gex"] - expected_gex) < 1.0
        assert result["net_gex"] > 0
        assert result["gex_imbalance"] == 1.0  # all call, no put
        assert result["call_wall"] == 5000
        assert np.isnan(result["put_wall"])

    def test_single_put_negative_gex(self):
        """A single put should produce negative net GEX."""
        spot = 5000.0
        gamma = 0.001
        oi_val = 1000
        greeks = make_greeks([4900], ["put"], [gamma])
        oi = make_oi([4900], ["put"], [oi_val])
        result = compute_gex_features(greeks, oi, spot)

        assert result["net_gex"] < 0
        assert result["gex_imbalance"] == 0.0  # all put, no call
        assert np.isnan(result["call_wall"])
        assert result["put_wall"] == 4900

    def test_balanced_call_put_gex(self):
        """Equal call and put GEX should give ~0.5 imbalance."""
        spot = 5000.0
        greeks = make_greeks([5000, 5000], ["call", "put"], [0.001, 0.001])
        oi = make_oi([5000, 5000], ["call", "put"], [1000, 1000])
        result = compute_gex_features(greeks, oi, spot)

        # Net GEX should be ~0 (call + positive, put - negative, same magnitude)
        assert abs(result["net_gex"]) < 1.0
        assert abs(result["gex_imbalance"] - 0.5) < 0.01

    def test_call_wall_is_highest_gex_strike(self):
        """Call wall should be at the strike with highest call GEX."""
        spot = 5000.0
        greeks = make_greeks(
            [5000, 5050, 5100], ["call", "call", "call"], [0.001, 0.003, 0.001]
        )
        oi = make_oi(
            [5000, 5050, 5100], ["call", "call", "call"], [500, 500, 500]
        )
        result = compute_gex_features(greeks, oi, spot)
        assert result["call_wall"] == 5050

    def test_put_wall_is_highest_abs_gex_strike(self):
        """Put wall should be at the strike with highest absolute put GEX."""
        spot = 5000.0
        greeks = make_greeks(
            [4900, 4950, 4800], ["put", "put", "put"], [0.001, 0.005, 0.002]
        )
        oi = make_oi(
            [4900, 4950, 4800], ["put", "put", "put"], [500, 500, 500]
        )
        result = compute_gex_features(greeks, oi, spot)
        assert result["put_wall"] == 4950

    def test_wall_distance_calculation(self):
        """Wall distances should be (wall - spot) / spot * 100."""
        spot = 5000.0
        greeks = make_greeks([5100, 4900], ["call", "put"], [0.001, 0.001])
        oi = make_oi([5100, 4900], ["call", "put"], [1000, 1000])
        result = compute_gex_features(greeks, oi, spot)

        assert abs(result["call_wall_distance"] - 2.0) < 0.01  # (5100-5000)/5000*100
        assert abs(result["put_wall_distance"] - 2.0) < 0.01   # (5000-4900)/5000*100

    def test_normalized_gex(self):
        """net_gex_normalized should be net_gex / spot."""
        spot = 5000.0
        greeks = make_greeks([5000], ["call"], [0.001])
        oi = make_oi([5000], ["call"], [1000])
        result = compute_gex_features(greeks, oi, spot)

        assert abs(result["net_gex_normalized"] - result["net_gex"] / spot) < 0.01

    def test_zero_oi_rows_excluded(self):
        """Rows with OI=0 should be excluded."""
        spot = 5000.0
        greeks = make_greeks([5000, 5050], ["call", "call"], [0.001, 0.001])
        oi = make_oi([5000, 5050], ["call", "call"], [1000, 0])
        result = compute_gex_features(greeks, oi, spot)

        # Only the 5000 strike should contribute
        expected = 1.0 * 0.001 * 1000 * 100 * spot**2 * 0.01
        assert abs(result["net_gex"] - expected) < 1.0
        assert result["call_wall"] == 5000

    def test_returns_all_expected_keys(self):
        """Result dict should contain all GEX feature names."""
        greeks = make_greeks([5000], ["call"], [0.001])
        oi = make_oi([5000], ["call"], [1000])
        result = compute_gex_features(greeks, oi, 5000.0)
        assert set(result.keys()) == set(GEX_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# find_zero_gamma
# ---------------------------------------------------------------------------

class TestFindZeroGamma:
    def test_no_crossing_returns_nan(self):
        """If cumulative GEX never crosses zero, return NaN."""
        series = pd.Series([1.0, 2.0, 3.0], index=[4900, 4950, 5000])
        assert np.isnan(find_zero_gamma(series))

    def test_crossing_found(self):
        """Should interpolate the zero crossing correctly."""
        # Cumulative: [10, 10+(-30)=-20] → crosses zero between 4950 and 5000
        series = pd.Series([10.0, -30.0], index=[4950, 5000])
        zero = find_zero_gamma(series)
        # Linear interp: cum = [10, -20], cross at 4950 + 50 * (10/30) = 4966.67
        assert abs(zero - 4966.67) < 1.0

    def test_single_element_returns_nan(self):
        series = pd.Series([5.0], index=[5000])
        assert np.isnan(find_zero_gamma(series))

    def test_first_crossing_used(self):
        """Should return the first zero crossing, not later ones."""
        series = pd.Series([10.0, -30.0, 40.0, -20.0], index=[4900, 4950, 5000, 5050])
        zero = find_zero_gamma(series)
        # First crossing is between 4900 and 4950
        assert 4900 <= zero <= 4950


# ---------------------------------------------------------------------------
# compute_gex_profile
# ---------------------------------------------------------------------------

class TestComputeGexProfile:
    def test_basic_profile(self):
        spot = 5000.0
        greeks = make_greeks(
            [4900, 5000, 5000, 5100],
            ["put", "call", "put", "call"],
            [0.001, 0.002, 0.002, 0.001],
        )
        oi = make_oi(
            [4900, 5000, 5000, 5100],
            ["put", "call", "put", "call"],
            [500, 1000, 800, 600],
        )
        profile = compute_gex_profile(greeks, oi, spot)

        assert "strike" in profile.columns
        assert "call_gex" in profile.columns
        assert "put_gex" in profile.columns
        assert "net_gex" in profile.columns
        assert len(profile) == 3  # 3 distinct strikes

    def test_empty_returns_empty_df(self):
        greeks = pd.DataFrame(columns=["strike", "right", "gamma"])
        oi = pd.DataFrame(columns=["strike", "right", "open_interest"])
        profile = compute_gex_profile(greeks, oi, 5000.0)
        assert len(profile) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
