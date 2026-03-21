"""Tests for src/vanna_charm.py — Vanna and Charm exposure calculator."""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vanna_charm import (
    bs_vanna, bs_charm, compute_vanna_charm_features,
    VANNA_CHARM_FEATURES,
)


class TestBSVanna(unittest.TestCase):
    """Test Black-Scholes Vanna computation."""

    def test_atm_vanna_finite(self):
        """ATM option should have finite vanna."""
        v = bs_vanna(S=5000, K=5000, T=1/365, r=0.05, sigma=0.20)
        self.assertFalse(np.isnan(v))
        self.assertTrue(np.isfinite(v))

    def test_vanna_zero_time(self):
        """Vanna should be NaN when T=0."""
        v = bs_vanna(S=5000, K=5000, T=0, r=0.05, sigma=0.20)
        self.assertTrue(np.isnan(v))

    def test_vanna_zero_vol(self):
        """Vanna should be NaN when sigma=0."""
        v = bs_vanna(S=5000, K=5000, T=1/365, r=0.05, sigma=0)
        self.assertTrue(np.isnan(v))

    def test_vanna_negative_inputs(self):
        """Vanna should be NaN for invalid inputs."""
        self.assertTrue(np.isnan(bs_vanna(S=-1, K=5000, T=1/365, r=0.05, sigma=0.20)))
        self.assertTrue(np.isnan(bs_vanna(S=5000, K=-1, T=1/365, r=0.05, sigma=0.20)))

    def test_deep_otm_vanna_near_zero(self):
        """Deep OTM option should have vanna close to zero."""
        v = bs_vanna(S=5000, K=3000, T=1/365, r=0.05, sigma=0.20)
        self.assertAlmostEqual(v, 0.0, places=4)

    def test_vanna_sign_otm_call(self):
        """OTM call: d2 < 0 when K > S, so vanna = -N'(d1)*d2/sigma should be positive."""
        v = bs_vanna(S=5000, K=5100, T=30/365, r=0.05, sigma=0.20)
        self.assertGreater(v, 0)

    def test_vanna_longer_expiry(self):
        """Vanna with longer time to expiry should be finite and differ from short DTE."""
        v_short = bs_vanna(S=5000, K=5000, T=1/365, r=0.05, sigma=0.20)
        v_long = bs_vanna(S=5000, K=5000, T=30/365, r=0.05, sigma=0.20)
        self.assertFalse(np.isnan(v_long))
        self.assertNotAlmostEqual(v_short, v_long, places=4)


class TestBSCharm(unittest.TestCase):
    """Test Black-Scholes Charm computation."""

    def test_atm_charm_finite(self):
        """ATM option should have finite charm."""
        c = bs_charm(S=5000, K=5000, T=1/365, r=0.05, sigma=0.20, right="call")
        self.assertFalse(np.isnan(c))
        self.assertTrue(np.isfinite(c))

    def test_charm_zero_time(self):
        """Charm should be NaN when T=0."""
        c = bs_charm(S=5000, K=5000, T=0, r=0.05, sigma=0.20)
        self.assertTrue(np.isnan(c))

    def test_call_put_charm_differ(self):
        """Call and put charm at same strike should generally differ."""
        c_call = bs_charm(S=5000, K=5000, T=30/365, r=0.05, sigma=0.20, right="call")
        c_put = bs_charm(S=5000, K=5000, T=30/365, r=0.05, sigma=0.20, right="put")
        # With q=0, they're the same. Test with q != 0
        c_call_q = bs_charm(S=5000, K=5000, T=30/365, r=0.05, sigma=0.20, right="call", q=0.02)
        c_put_q = bs_charm(S=5000, K=5000, T=30/365, r=0.05, sigma=0.20, right="put", q=0.02)
        self.assertNotAlmostEqual(c_call_q, c_put_q, places=6)

    def test_charm_short_dte_large(self):
        """Charm magnitude should be larger for shorter DTE (delta decays faster)."""
        c_short = bs_charm(S=5000, K=5050, T=1/365, r=0.05, sigma=0.20, right="call")
        c_long = bs_charm(S=5000, K=5050, T=60/365, r=0.05, sigma=0.20, right="call")
        self.assertGreater(abs(c_short), abs(c_long))

    def test_deep_otm_charm_small(self):
        """Deep OTM option should have charm close to zero."""
        c = bs_charm(S=5000, K=3000, T=1/365, r=0.05, sigma=0.20, right="put")
        self.assertAlmostEqual(c, 0.0, places=4)


class TestComputeVannaCharmFeatures(unittest.TestCase):
    """Test the aggregate Vanna/Charm feature computation."""

    def _make_chain(self, spot=5000):
        """Create synthetic options chain for testing."""
        strikes = np.arange(spot - 200, spot + 250, 50)
        rows = []
        for k in strikes:
            for right in ["call", "put"]:
                delta = 0.5 if right == "call" else -0.5
                rows.append({
                    "strike": float(k),
                    "right": right,
                    "iv": 0.20,
                    "delta": delta,
                    "gamma": 0.0005,
                })
        greeks = pd.DataFrame(rows)

        oi_rows = []
        for k in strikes:
            for right in ["call", "put"]:
                oi_rows.append({
                    "strike": float(k),
                    "right": right,
                    "open_interest": 1000,
                })
        oi = pd.DataFrame(oi_rows)
        return greeks, oi

    def test_returns_all_features(self):
        """Should return all expected feature keys."""
        greeks, oi = self._make_chain()
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        for key in VANNA_CHARM_FEATURES:
            self.assertIn(key, feats)

    def test_non_nan_with_valid_data(self):
        """Should produce non-NaN values with valid input."""
        greeks, oi = self._make_chain()
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        for key in ["net_vanna_exposure", "net_vanna_normalized",
                     "net_charm_exposure", "net_charm_normalized"]:
            self.assertFalse(np.isnan(feats[key]), f"{key} should not be NaN")

    def test_empty_greeks_returns_nan(self):
        """Empty greeks should return all NaN."""
        _, oi = self._make_chain()
        feats = compute_vanna_charm_features(pd.DataFrame(), oi, 5000)
        for key in VANNA_CHARM_FEATURES:
            self.assertTrue(np.isnan(feats[key]))

    def test_zero_spot_returns_nan(self):
        """Zero spot price should return all NaN."""
        greeks, oi = self._make_chain()
        feats = compute_vanna_charm_features(greeks, oi, 0)
        for key in VANNA_CHARM_FEATURES:
            self.assertTrue(np.isnan(feats[key]))

    def test_imbalance_between_0_and_1(self):
        """Imbalance metrics should be between 0 and 1."""
        greeks, oi = self._make_chain()
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        if not np.isnan(feats["vanna_imbalance"]):
            self.assertGreaterEqual(feats["vanna_imbalance"], 0)
            self.assertLessEqual(feats["vanna_imbalance"], 1)
        if not np.isnan(feats["charm_imbalance"]):
            self.assertGreaterEqual(feats["charm_imbalance"], 0)
            self.assertLessEqual(feats["charm_imbalance"], 1)

    def test_vanna_charm_ratio_between_0_and_1(self):
        """Ratio should be between 0 and 1."""
        greeks, oi = self._make_chain()
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        if not np.isnan(feats["vanna_charm_ratio"]):
            self.assertGreaterEqual(feats["vanna_charm_ratio"], 0)
            self.assertLessEqual(feats["vanna_charm_ratio"], 1)

    def test_normalized_scales_by_spot(self):
        """Normalized features should scale with spot price."""
        greeks, oi = self._make_chain(5000)
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        if not np.isnan(feats["net_vanna_normalized"]):
            expected = feats["net_vanna_exposure"] / 5000
            self.assertAlmostEqual(feats["net_vanna_normalized"], expected, places=4)

    def test_single_option_works(self):
        """Should handle chain with just 2 options (1 call, 1 put)."""
        greeks = pd.DataFrame([
            {"strike": 5000.0, "right": "call", "iv": 0.20, "delta": 0.50, "gamma": 0.001},
            {"strike": 5000.0, "right": "put", "iv": 0.20, "delta": -0.50, "gamma": 0.001},
        ])
        oi = pd.DataFrame([
            {"strike": 5000.0, "right": "call", "open_interest": 500},
            {"strike": 5000.0, "right": "put", "open_interest": 500},
        ])
        feats = compute_vanna_charm_features(greeks, oi, 5000)
        self.assertFalse(np.isnan(feats["net_vanna_exposure"]))


class TestBSVannaCharmConsistency(unittest.TestCase):
    """Test consistency between Vanna and Charm computations."""

    def test_vanna_symmetric_for_same_moneyness(self):
        """Vanna should be the same regardless of call/put (it's symmetric)."""
        v1 = bs_vanna(S=5000, K=5100, T=30/365, r=0.05, sigma=0.25)
        # Vanna formula is the same for calls and puts
        self.assertFalse(np.isnan(v1))

    def test_various_vol_levels(self):
        """Test across different vol levels."""
        for sigma in [0.05, 0.15, 0.30, 0.60, 1.00]:
            v = bs_vanna(S=5000, K=5000, T=7/365, r=0.05, sigma=sigma)
            c = bs_charm(S=5000, K=5000, T=7/365, r=0.05, sigma=sigma)
            self.assertFalse(np.isnan(v), f"Vanna NaN at sigma={sigma}")
            self.assertFalse(np.isnan(c), f"Charm NaN at sigma={sigma}")


if __name__ == "__main__":
    unittest.main()
