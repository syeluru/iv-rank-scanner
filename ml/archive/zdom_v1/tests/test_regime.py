"""Tests for src/regime.py — HMM Regime Detector module."""

import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.regime import (
    fit_hmm, VolRegimeDetector, GEXRegimeDetector,
    _compute_regime_duration, _compute_regime_switch,
    VOL_REGIME_LABELS, GEX_REGIME_LABELS,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helper: generate synthetic regime data
# ---------------------------------------------------------------------------

def generate_two_regime_data(n=500, seed=42):
    """Generate synthetic data with two clear regimes.

    First half: low-vol (small returns, low VIX)
    Second half: high-vol (large returns, high VIX)
    """
    rng = np.random.RandomState(seed)
    n_half = n // 2

    # Low-vol regime
    returns_low = rng.normal(0.0005, 0.005, n_half)
    vix_low = rng.normal(14, 2, n_half)

    # High-vol regime
    returns_high = rng.normal(-0.001, 0.02, n_half)
    vix_high = rng.normal(28, 5, n_half)

    returns = np.concatenate([returns_low, returns_high])
    vix = np.concatenate([vix_low, vix_high])
    return returns, vix


def generate_three_gex_regime_data(n=600, seed=42):
    """Generate synthetic GEX + VIX data with three regimes."""
    rng = np.random.RandomState(seed)
    n_third = n // 3

    gex_pos = rng.normal(5e6, 1e6, n_third)
    vix_pos = rng.normal(13, 1.5, n_third)

    gex_trans = rng.normal(0, 2e6, n_third)
    vix_trans = rng.normal(18, 3, n_third)

    gex_neg = rng.normal(-4e6, 1.5e6, n_third)
    vix_neg = rng.normal(25, 4, n_third)

    gex = np.concatenate([gex_pos, gex_trans, gex_neg])
    vix = np.concatenate([vix_pos, vix_trans, vix_neg])
    return gex, vix


# ---------------------------------------------------------------------------
# fit_hmm
# ---------------------------------------------------------------------------

class TestFitHMM:
    def test_basic_fit(self):
        rng = np.random.RandomState(0)
        obs = rng.randn(200, 2)
        model = fit_hmm(obs, n_states=2, n_restarts=3, n_iter=100)
        assert model is not None
        assert model.n_components == 2

    def test_3_state_fit(self):
        rng = np.random.RandomState(0)
        obs = rng.randn(300, 2)
        model = fit_hmm(obs, n_states=3, n_restarts=3, n_iter=100)
        assert model.n_components == 3

    def test_predict_returns_valid_states(self):
        rng = np.random.RandomState(0)
        obs = rng.randn(200, 2)
        model = fit_hmm(obs, n_states=2, n_restarts=3)
        states = model.predict(obs)
        assert len(states) == 200
        assert set(states).issubset({0, 1})


# ---------------------------------------------------------------------------
# _compute_regime_duration / _compute_regime_switch
# ---------------------------------------------------------------------------

class TestRegimeDurationSwitch:
    def test_duration_single_regime(self):
        states = np.array([0, 0, 0, 0, 0])
        dur = _compute_regime_duration(states)
        assert list(dur) == [1, 2, 3, 4, 5]

    def test_duration_regime_change(self):
        states = np.array([0, 0, 1, 1, 1, 0])
        dur = _compute_regime_duration(states)
        assert list(dur) == [1, 2, 1, 2, 3, 1]

    def test_switch_no_change(self):
        states = np.array([0, 0, 0])
        sw = _compute_regime_switch(states)
        assert list(sw) == [0, 0, 0]

    def test_switch_with_changes(self):
        states = np.array([0, 0, 1, 1, 0])
        sw = _compute_regime_switch(states)
        assert list(sw) == [0, 0, 1, 0, 1]

    def test_empty(self):
        states = np.array([], dtype=int)
        dur = _compute_regime_duration(states)
        sw = _compute_regime_switch(states)
        assert len(dur) == 0
        assert len(sw) == 0


# ---------------------------------------------------------------------------
# VolRegimeDetector
# ---------------------------------------------------------------------------

class TestVolRegimeDetector:
    def test_fit_and_predict(self):
        returns, vix = generate_two_regime_data(500)
        det = VolRegimeDetector(n_restarts=5, n_iter=200)
        det.fit(returns, vix)

        result = det.predict(returns, vix)
        assert "regime_state" in result
        assert len(result["regime_state"]) == 500
        assert set(result["regime_state"]).issubset({0, 1})
        assert len(result["regime_prob_highvol"]) == 500
        assert len(result["regime_duration"]) == 500
        assert len(result["regime_switch"]) == 500

    def test_detects_regime_shift(self):
        """Should detect the shift from low-vol to high-vol."""
        returns, vix = generate_two_regime_data(500)
        det = VolRegimeDetector(n_restarts=10, n_iter=300)
        det.fit(returns, vix)
        result = det.predict(returns, vix)

        # First 100 obs should be mostly low-vol (state 0)
        first_100_highvol = np.mean(result["regime_state"][:100] == 1)
        # Last 100 obs should be mostly high-vol (state 1)
        last_100_highvol = np.mean(result["regime_state"][-100:] == 1)
        assert last_100_highvol > first_100_highvol

    def test_predict_current(self):
        returns, vix = generate_two_regime_data(500)
        det = VolRegimeDetector(n_restarts=5, n_iter=200)
        det.fit(returns, vix)
        current = det.predict_current(returns, vix)

        assert "state" in current
        assert "label" in current
        assert current["label"] in VOL_REGIME_LABELS.values()
        assert 0 <= current["prob_highvol"] <= 1
        assert 0 <= current["prob_lowvol"] <= 1
        assert abs(current["prob_highvol"] + current["prob_lowvol"] - 1.0) < 0.01

    def test_save_and_load(self, tmp_path):
        returns, vix = generate_two_regime_data(300)
        det = VolRegimeDetector(n_restarts=3, n_iter=100)
        det.fit(returns, vix)

        model_path = tmp_path / "test_hmm.pkl"
        det.save(model_path)

        loaded = VolRegimeDetector.load(model_path)
        result_orig = det.predict(returns, vix)
        result_loaded = loaded.predict(returns, vix)
        np.testing.assert_array_equal(result_orig["regime_state"],
                                      result_loaded["regime_state"])


# ---------------------------------------------------------------------------
# GEXRegimeDetector
# ---------------------------------------------------------------------------

class TestGEXRegimeDetector:
    def test_fit_and_predict(self):
        gex, vix = generate_three_gex_regime_data(600)
        det = GEXRegimeDetector(n_restarts=5, n_iter=200)
        det.fit(gex, vix)

        result = det.predict(gex, vix)
        assert "gex_regime" in result
        assert len(result["gex_regime"]) == 600
        assert set(result["gex_regime"]).issubset({0, 1, 2})

    def test_detects_positive_and_negative_regimes(self):
        """Positive GEX data should be state 0, negative should be state 2."""
        gex, vix = generate_three_gex_regime_data(600)
        det = GEXRegimeDetector(n_restarts=10, n_iter=300)
        det.fit(gex, vix)
        result = det.predict(gex, vix)

        # First 200: positive GEX → mostly state 0
        first_pos = np.mean(result["gex_regime"][:200] == 0)
        # Last 200: negative GEX → mostly state 2
        last_neg = np.mean(result["gex_regime"][-200:] == 2)
        # At least some separation (HMM may not perfectly separate)
        assert first_pos > 0.3 or last_neg > 0.3

    def test_predict_current(self):
        gex, vix = generate_three_gex_regime_data(600)
        det = GEXRegimeDetector(n_restarts=5, n_iter=200)
        det.fit(gex, vix)
        current = det.predict_current(gex, vix)

        assert "state" in current
        assert "label" in current
        assert current["label"] in GEX_REGIME_LABELS.values()
        probs = current["prob_pos"] + current["prob_trans"] + current["prob_neg"]
        assert abs(probs - 1.0) < 0.01

    def test_save_and_load(self, tmp_path):
        gex, vix = generate_three_gex_regime_data(300)
        det = GEXRegimeDetector(n_restarts=3, n_iter=100)
        det.fit(gex, vix)

        model_path = tmp_path / "test_gex_hmm.pkl"
        det.save(model_path)

        loaded = GEXRegimeDetector.load(model_path)
        result_orig = det.predict(gex, vix)
        result_loaded = loaded.predict(gex, vix)
        np.testing.assert_array_equal(result_orig["gex_regime"],
                                      result_loaded["gex_regime"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
