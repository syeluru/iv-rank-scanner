"""Tests for src/iv_surface.py — IV Surface feature extractor module."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.iv_surface import (
    bs_call_price, bs_put_price, implied_vol_from_price,
    find_delta_iv, extract_snapshot_iv_features,
    extract_term_structure_iv, add_rolling_features,
    compute_term_structure_slopes,
)


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_call_price_atm(self):
        """ATM call with reasonable params should give a positive price."""
        price = bs_call_price(S=5000, K=5000, T=1/365, r=0.05, sigma=0.20)
        assert price > 0
        assert price < 100  # 0DTE ATM call shouldn't be > 100

    def test_put_price_atm(self):
        price = bs_put_price(S=5000, K=5000, T=1/365, r=0.05, sigma=0.20)
        assert price > 0

    def test_zero_time_call_returns_intrinsic(self):
        price = bs_call_price(S=5050, K=5000, T=0, r=0.05, sigma=0.20)
        assert abs(price - 50.0) < 0.01

    def test_zero_time_put_returns_intrinsic(self):
        price = bs_put_price(S=4950, K=5000, T=0, r=0.05, sigma=0.20)
        assert abs(price - 50.0) < 0.01

    def test_deep_otm_call_is_small(self):
        price = bs_call_price(S=5000, K=5500, T=1/365, r=0.05, sigma=0.20)
        assert price < 1.0  # deep OTM 0DTE call is essentially 0

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) for European options."""
        S, K, T, r, sigma = 5000, 5050, 30/365, 0.05, 0.20
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.01


# ---------------------------------------------------------------------------
# Implied vol inversion
# ---------------------------------------------------------------------------

class TestImpliedVol:
    def test_roundtrip_call(self):
        """Price a call then recover the vol."""
        S, K, T, r = 5000, 5000, 30/365, 0.05
        target_vol = 0.25
        price = bs_call_price(S, K, T, r, target_vol)
        recovered = implied_vol_from_price(price, S, K, T, r, "call")
        assert abs(recovered - target_vol) < 0.001

    def test_roundtrip_put(self):
        S, K, T, r = 5000, 4900, 30/365, 0.05
        target_vol = 0.30
        price = bs_put_price(S, K, T, r, target_vol)
        recovered = implied_vol_from_price(price, S, K, T, r, "put")
        assert abs(recovered - target_vol) < 0.001

    def test_zero_price_returns_nan(self):
        assert np.isnan(implied_vol_from_price(0, 5000, 5000, 30/365, 0.05))

    def test_negative_time_returns_nan(self):
        assert np.isnan(implied_vol_from_price(10, 5000, 5000, -1, 0.05))

    def test_price_below_intrinsic_returns_nan(self):
        # Call with S=5050, K=5000 → intrinsic=50. Price=40 is below.
        result = implied_vol_from_price(40, 5050, 5000, 30/365, 0.05, "call")
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# find_delta_iv
# ---------------------------------------------------------------------------

class TestFindDeltaIV:
    def make_snapshot(self):
        return pd.DataFrame({
            "strike": [4800, 4900, 5000, 5000, 5100, 5200],
            "right": ["put", "put", "call", "put", "call", "call"],
            "delta": [-0.10, -0.25, 0.50, -0.50, 0.25, 0.10],
            "implied_vol": [0.30, 0.25, 0.20, 0.21, 0.22, 0.28],
            "iv_error": [0, 0, 0, 0, 0, 0],
        })

    def test_find_50d_call(self):
        snap = self.make_snapshot()
        iv, delta, strike = find_delta_iv(snap, 0.50, "call")
        assert abs(iv - 0.20) < 0.001
        assert strike == 5000

    def test_find_25d_put(self):
        snap = self.make_snapshot()
        iv, delta, strike = find_delta_iv(snap, -0.25, "put")
        assert abs(iv - 0.25) < 0.001
        assert strike == 4900

    def test_find_10d_call(self):
        snap = self.make_snapshot()
        iv, delta, strike = find_delta_iv(snap, 0.10, "call")
        assert abs(iv - 0.28) < 0.001

    def test_no_match_returns_nan(self):
        snap = self.make_snapshot()
        # Ask for 0.90 delta call — nothing close
        iv, delta, strike = find_delta_iv(snap, 0.90, "call", tolerance=0.05)
        assert np.isnan(iv)

    def test_iv_error_excluded(self):
        snap = self.make_snapshot()
        snap.loc[snap["strike"] == 5000, "iv_error"] = 1  # corrupt ATM
        iv, _, _ = find_delta_iv(snap, 0.50, "call")
        assert np.isnan(iv)  # should not find ATM call

    def test_empty_snapshot_returns_nan(self):
        snap = pd.DataFrame(columns=["strike", "right", "delta", "implied_vol", "iv_error"])
        iv, _, _ = find_delta_iv(snap, 0.50, "call")
        assert np.isnan(iv)


# ---------------------------------------------------------------------------
# extract_snapshot_iv_features
# ---------------------------------------------------------------------------

class TestExtractSnapshotIVFeatures:
    def make_full_snapshot(self):
        return pd.DataFrame({
            "strike": [4800, 4900, 5000, 5000, 5100, 5200],
            "right": ["put", "put", "call", "put", "call", "call"],
            "delta": [-0.10, -0.25, 0.50, -0.50, 0.25, 0.10],
            "implied_vol": [0.30, 0.26, 0.20, 0.21, 0.22, 0.28],
            "iv_error": [0, 0, 0, 0, 0, 0],
            "underlying_price": [5000] * 6,
            "mid": [5, 15, 25, 26, 14, 4],
        })

    def test_extracts_atm_iv(self):
        snap = self.make_full_snapshot()
        feats = extract_snapshot_iv_features(snap)
        # ATM = avg of 50d call (0.20) and 50d put (0.21)
        assert abs(feats["atm_iv"] - 0.205) < 0.001

    def test_extracts_25d_skew(self):
        snap = self.make_full_snapshot()
        feats = extract_snapshot_iv_features(snap)
        # 25d put IV (0.26) - 25d call IV (0.22) = 0.04
        assert abs(feats["skew_25d"] - 0.04) < 0.001

    def test_extracts_10d_skew(self):
        snap = self.make_full_snapshot()
        feats = extract_snapshot_iv_features(snap)
        # 10d put IV (0.30) - 10d call IV (0.28) = 0.02
        assert abs(feats["skew_10d"] - 0.02) < 0.001

    def test_smile_curvature(self):
        snap = self.make_full_snapshot()
        feats = extract_snapshot_iv_features(snap)
        # (25d_put + 25d_call)/2 - atm = (0.26+0.22)/2 - 0.205 = 0.035
        assert abs(feats["iv_smile_curve"] - 0.035) < 0.001

    def test_missing_puts_partial_result(self):
        """If puts are missing, should still extract call-side features."""
        snap = self.make_full_snapshot()
        snap = snap[snap["right"] == "call"]
        feats = extract_snapshot_iv_features(snap)
        assert not np.isnan(feats["atm_iv"])  # call ATM should work
        assert np.isnan(feats["skew_25d"])    # no put → no skew

    def test_all_features_returned(self):
        snap = self.make_full_snapshot()
        feats = extract_snapshot_iv_features(snap)
        expected_keys = {"atm_iv", "iv_25d_call", "iv_25d_put", "skew_25d",
                        "iv_10d_call", "iv_10d_put", "skew_10d", "iv_smile_curve"}
        assert expected_keys.issubset(set(feats.keys()))


# ---------------------------------------------------------------------------
# extract_term_structure_iv
# ---------------------------------------------------------------------------

class TestExtractTermStructureIV:
    def test_empty_returns_nan(self):
        feats = extract_term_structure_iv(pd.DataFrame(), 5000.0)
        assert np.isnan(feats["ts_iv_7dte"])
        assert np.isnan(feats["ts_iv_14dte"])

    def test_extracts_7dte_iv(self):
        ts_day = pd.DataFrame({
            "dte": [7, 7],
            "moneyness": [0.0, 0.0],
            "right": ["call", "put"],
            "mid": [50.0, 48.0],
            "strike": [5000, 5000],
        })
        feats = extract_term_structure_iv(ts_day, 5000.0)
        assert not np.isnan(feats["ts_iv_7dte"])
        assert 0.01 < feats["ts_iv_7dte"] < 3.0


# ---------------------------------------------------------------------------
# add_rolling_features / compute_term_structure_slopes
# ---------------------------------------------------------------------------

class TestRollingAndSlopes:
    def test_add_rolling_features(self):
        df = pd.DataFrame({"atm_iv": np.random.uniform(0.15, 0.25, 50)})
        add_rolling_features(df, "atm_iv", windows=(5, 21))
        assert "atm_iv_5d_mean" in df.columns
        assert "atm_iv_21d_std" in df.columns
        assert "atm_iv_zscore_21d" in df.columns
        # First few rows should be NaN due to min_periods
        assert df["atm_iv_5d_mean"].iloc[0] is np.nan or np.isnan(df["atm_iv_5d_mean"].iloc[0])

    def test_compute_term_structure_slopes(self):
        df = pd.DataFrame({
            "atm_iv": [0.25, 0.30],
            "ts_iv_7dte": [0.20, 0.25],
            "ts_iv_14dte": [0.18, 0.22],
        })
        df = compute_term_structure_slopes(df)
        assert "iv_ts_slope_0_7" in df.columns
        assert "ts_inverted_7_14" in df.columns
        # 0.25/0.20 = 1.25 > 1 → inverted
        assert df["iv_ts_slope_0_7"].iloc[0] == pytest.approx(1.25)
        assert df["ts_inverted_7_14"].iloc[0] == 1.0  # 7DTE > 14DTE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
