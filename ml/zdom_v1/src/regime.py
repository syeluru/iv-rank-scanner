"""HMM Regime Detector module.

Implements Hidden Markov Model regime detection using hmmlearn's GaussianHMM.
Supports two modes:
  1. Vol-based regime (2-state): low-vol vs high-vol using SPX returns + VIX
  2. GEX-based regime (3-state): positive/transition/negative GEX using GEX + VIX

Features produced (vol-based, 2-state):
    regime_state         — 0 = low-vol, 1 = high-vol
    regime_prob_highvol  — P(high-vol) from posterior
    regime_prob_lowvol   — P(low-vol) from posterior
    regime_duration      — consecutive days in current regime
    regime_switch        — 1 if regime changed from prior day

Features produced (GEX-based, 3-state):
    gex_regime           — 0 = positive_gex, 1 = transition, 2 = negative_gex
    gex_regime_prob_pos  — P(positive GEX)
    gex_regime_prob_trans — P(transition)
    gex_regime_prob_neg  — P(negative GEX)
    gex_regime_duration  — consecutive days in current regime
    gex_regime_switch    — 1 if regime changed from prior day
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

GEX_REGIME_LABELS = {0: "positive_gex", 1: "transition", 2: "negative_gex"}
VOL_REGIME_LABELS = {0: "low_vol", 1: "high_vol"}


def fit_hmm(observations, n_states=2, n_restarts=25, n_iter=1000, random_seed=42):
    """Fit GaussianHMM with multiple random restarts, return best model.

    Args:
        observations: 2D array of scaled observation features
        n_states: number of hidden states
        n_restarts: number of random initializations to try
        n_iter: max EM iterations per restart
        random_seed: base random seed

    Returns:
        Best-fit GaussianHMM model (highest log-likelihood)

    Raises:
        RuntimeError: if all fitting attempts fail
    """
    best_model = None
    best_score = -np.inf

    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_seed + i,
            tol=1e-4,
        )
        try:
            model.fit(observations)
            score = model.score(observations)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All HMM fitting attempts failed")

    return best_model


def _compute_regime_duration(states):
    """Compute consecutive days in current regime state."""
    duration = np.ones(len(states), dtype=int)
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            duration[i] = duration[i - 1] + 1
    return duration


def _compute_regime_switch(states):
    """Compute regime switch indicator (1 if changed from prior day)."""
    switch = np.zeros(len(states), dtype=int)
    if len(states) > 1:
        switch[1:] = (states[1:] != states[:-1]).astype(int)
    return switch


# ---------------------------------------------------------------------------
# Vol-based 2-state regime detector
# ---------------------------------------------------------------------------

class VolRegimeDetector:
    """2-state HMM regime detector: low-vol vs high-vol.

    Uses SPX daily log-returns and VIX close as observation features.
    State 0 = low-vol, State 1 = high-vol (relabeled by return variance).
    """

    def __init__(self, n_restarts=25, n_iter=1000, random_seed=42):
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.model = None
        self.scaler = None
        self.swap_labels = False

    def fit(self, returns, vix):
        """Fit the HMM on historical data.

        Args:
            returns: array of SPX daily log-returns
            vix: array of VIX close values (same length)
        """
        features = np.column_stack([returns, vix])
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]

        self.scaler = StandardScaler()
        obs_scaled = self.scaler.fit_transform(features)
        self.model = fit_hmm(obs_scaled, n_states=2,
                             n_restarts=self.n_restarts,
                             n_iter=self.n_iter,
                             random_seed=self.random_seed)

        # Relabel so state 1 = higher variance in returns
        covs = np.array([self.model.covars_[i][0, 0] for i in range(2)])
        self.swap_labels = (np.argmax(covs) == 0)

        return self

    def predict(self, returns, vix):
        """Predict regime states and probabilities.

        Args:
            returns: array of SPX daily log-returns
            vix: array of VIX close values

        Returns:
            dict with keys: states, prob_highvol, prob_lowvol, duration, switch
        """
        features = np.column_stack([returns, vix])
        obs_scaled = self.scaler.transform(features)

        states = self.model.predict(obs_scaled)
        posteriors = self.model.predict_proba(obs_scaled)

        if self.swap_labels:
            states = 1 - states
            posteriors = posteriors[:, ::-1]

        return {
            "regime_state": states,
            "regime_prob_highvol": posteriors[:, 1],
            "regime_prob_lowvol": posteriors[:, 0],
            "regime_duration": _compute_regime_duration(states),
            "regime_switch": _compute_regime_switch(states),
        }

    def predict_current(self, returns, vix):
        """Predict the current (latest) regime state.

        Returns:
            dict with current state, probabilities, and transition matrix
        """
        result = self.predict(returns, vix)
        idx = len(result["regime_state"]) - 1
        transmat = self.model.transmat_
        if self.swap_labels:
            transmat = transmat[::-1, ::-1]

        return {
            "state": int(result["regime_state"][idx]),
            "label": VOL_REGIME_LABELS[int(result["regime_state"][idx])],
            "prob_highvol": float(result["regime_prob_highvol"][idx]),
            "prob_lowvol": float(result["regime_prob_lowvol"][idx]),
            "duration": int(result["regime_duration"][idx]),
            "transition_matrix": transmat.tolist(),
        }

    def save(self, path):
        """Save fitted model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "swap_labels": self.swap_labels,
                "n_states": 2,
            }, f)

    @classmethod
    def load(cls, path):
        """Load a fitted model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls()
        det.model = data["model"]
        det.scaler = data["scaler"]
        det.swap_labels = data["swap_labels"]
        return det


# ---------------------------------------------------------------------------
# GEX-based 3-state regime detector
# ---------------------------------------------------------------------------

class GEXRegimeDetector:
    """3-state HMM regime detector based on GEX + VIX.

    State 0 = positive_gex (dealers long gamma, mean-reverting)
    State 1 = transition (mixed/uncertain)
    State 2 = negative_gex (dealers short gamma, trend-amplifying)
    """

    def __init__(self, n_states=3, n_restarts=25, n_iter=1000, random_seed=42):
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.model = None
        self.scaler = None
        self.perm = None

    def fit(self, gex_normalized, vix):
        """Fit the HMM on GEX + VIX data.

        Args:
            gex_normalized: array of net_gex_normalized values
            vix: array of VIX close values
        """
        features = np.column_stack([gex_normalized, vix])
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]

        self.scaler = StandardScaler()
        obs_scaled = self.scaler.fit_transform(features)
        self.model = fit_hmm(obs_scaled, n_states=self.n_states,
                             n_restarts=self.n_restarts,
                             n_iter=self.n_iter,
                             random_seed=self.random_seed)

        # Relabel: highest mean GEX → state 0 (positive), lowest → state 2 (negative)
        means_orig = self.scaler.inverse_transform(self.model.means_)
        gex_means = means_orig[:, 0]
        sorted_indices = np.argsort(-gex_means)  # descending
        self.perm = np.zeros(len(sorted_indices), dtype=int)
        for new_label, old_label in enumerate(sorted_indices):
            self.perm[old_label] = new_label

        return self

    def predict(self, gex_normalized, vix):
        """Predict GEX regime states and probabilities.

        Args:
            gex_normalized: array of net_gex_normalized values
            vix: array of VIX close values

        Returns:
            dict with keys: gex_regime, prob_pos, prob_trans, prob_neg, duration, switch
        """
        features = np.column_stack([gex_normalized, vix])
        obs_scaled = self.scaler.transform(features)

        raw_states = self.model.predict(obs_scaled)
        posteriors = self.model.predict_proba(obs_scaled)

        states = self.perm[raw_states]
        inv_perm = np.argsort(self.perm)
        posteriors_relabeled = posteriors[:, inv_perm]

        return {
            "gex_regime": states,
            "gex_regime_prob_pos": posteriors_relabeled[:, 0],
            "gex_regime_prob_trans": posteriors_relabeled[:, 1],
            "gex_regime_prob_neg": posteriors_relabeled[:, 2],
            "gex_regime_duration": _compute_regime_duration(states),
            "gex_regime_switch": _compute_regime_switch(states),
        }

    def predict_current(self, gex_normalized, vix):
        """Predict the current (latest) GEX regime state."""
        result = self.predict(gex_normalized, vix)
        idx = len(result["gex_regime"]) - 1

        inv_perm = np.argsort(self.perm)
        transmat = self.model.transmat_[np.ix_(inv_perm, inv_perm)]

        return {
            "state": int(result["gex_regime"][idx]),
            "label": GEX_REGIME_LABELS[int(result["gex_regime"][idx])],
            "prob_pos": float(result["gex_regime_prob_pos"][idx]),
            "prob_trans": float(result["gex_regime_prob_trans"][idx]),
            "prob_neg": float(result["gex_regime_prob_neg"][idx]),
            "duration": int(result["gex_regime_duration"][idx]),
            "transition_matrix": transmat.tolist(),
        }

    def save(self, path):
        """Save fitted model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "perm": self.perm,
                "n_states": self.n_states,
                "regime_labels": GEX_REGIME_LABELS,
            }, f)

    @classmethod
    def load(cls, path):
        """Load a fitted model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(n_states=data["n_states"])
        det.model = data["model"]
        det.scaler = data["scaler"]
        det.perm = data["perm"]
        return det
