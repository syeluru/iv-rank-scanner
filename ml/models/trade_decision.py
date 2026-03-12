"""
Unified trade decision engine — EV-based strategy selection across all 6 strategies.

Combines v3 (risk regime), v5 (per-delta short IC TP probability), and v7 (per-delta
long IC profitability) to recommend the best trade(s) for any given day.

Output: direction (SHORT_ONLY / LONG_ONLY / BOTH / NEITHER) + specific delta for each.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from loguru import logger

from ml.models.strategy_scorer import StrategyScorer, StrategyCandidate


# ── Enums & Data Classes ─────────────────────────────────────────────────────


class TradeAction(str, Enum):
    """What trade to execute."""
    SHORT_ONLY = "short_only"      # Sell IC (traditional, theta decay)
    LONG_ONLY = "long_only"        # Buy IC (gamma play, big move expected)
    BOTH = "both"                  # Both long and short IC (different deltas)
    NEITHER = "neither"            # Sit out — no positive EV strategies


@dataclass
class TradeDecision:
    """Complete trade recommendation with EV-based reasoning."""
    action: TradeAction
    confidence: float              # 0-1 overall confidence in recommendation

    # Short IC details (if action is SHORT_ONLY or BOTH)
    short_delta: Optional[float] = None
    short_ev: float = 0.0
    short_v3_confidence: float = 0.0
    short_v5_tp25: float = 0.0

    # Long IC details (if action is LONG_ONLY or BOTH)
    long_delta: Optional[float] = None
    long_ev: float = 0.0
    long_v7_score: float = 0.0
    long_recommended_entry_start: Optional[str] = None
    long_recommended_entry_end: Optional[str] = None

    # All scored strategies for display
    all_candidates: List[StrategyCandidate] = field(default_factory=list)

    # Reasoning
    reasons: List[str] = field(default_factory=list)

    @property
    def has_short(self) -> bool:
        return self.action in (TradeAction.SHORT_ONLY, TradeAction.BOTH)

    @property
    def has_long(self) -> bool:
        return self.action in (TradeAction.LONG_ONLY, TradeAction.BOTH)

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"Action={self.action.value}", f"Confidence={self.confidence:.2f}"]
        if self.has_short:
            d = int(self.short_delta * 100) if self.short_delta else "?"
            parts.append(f"Short@{d}δ(EV=${self.short_ev:+.2f})")
        if self.has_long:
            d = int(self.long_delta * 100) if self.long_delta else "?"
            parts.append(f"Long@{d}δ(EV=${self.long_ev:+.2f})")
        return " | ".join(parts)


# ── Decision Engine ──────────────────────────────────────────────────────────


class TradeDecisionEngine:
    """
    EV-based decision engine that scores all 6 strategies (short/long × 3 deltas)
    and selects the best trade(s).

    Parameters
    ----------
    min_ev : float
        Minimum EV per contract to consider a strategy viable.
    min_short_v3 : float
        Minimum v3 confidence to allow any short IC.
    long_entry_start / long_entry_end : str
        Recommended entry window for long ICs.
    """

    def __init__(
        self,
        min_ev: float = 0.10,
        min_short_v3: float = 0.45,
        long_entry_start: str = "11:00",
        long_entry_end: str = "12:30",
    ):
        self.min_ev = min_ev
        self.min_short_v3 = min_short_v3
        self.long_entry_start = long_entry_start
        self.long_entry_end = long_entry_end
        self.scorer = StrategyScorer()

        logger.info(
            "TradeDecisionEngine initialised | "
            f"min_ev=${min_ev:.2f} min_short_v3={min_short_v3}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def decide(
        self,
        v3_confidence: float,
        v5_scores: Optional[Dict[float, Dict]] = None,
        v7_scores: Optional[Dict[float, float]] = None,
    ) -> TradeDecision:
        """
        Make EV-based trade decision.

        Parameters
        ----------
        v3_confidence : float
            P(safe day) from v3 model. Higher = safer for short ICs.
        v5_scores : dict, optional
            Per-delta v5 scores: {0.10: {"tp25": 0.85, "tp50": 0.65}, ...}
        v7_scores : dict, optional
            Per-delta v7 scores: {0.10: 0.35, 0.15: 0.40, 0.20: 0.55}

        Returns
        -------
        TradeDecision
        """
        # Score all 6 strategies
        all_candidates = self.scorer.score_all(v3_confidence, v5_scores, v7_scores)
        reasons: List[str] = []

        # Filter viable candidates
        viable = [c for c in all_candidates if c.ev_per_contract >= self.min_ev]

        # Apply v3 safety gate to short strategies
        if v3_confidence < self.min_short_v3:
            blocked = [c for c in viable if c.direction == "short"]
            if blocked:
                reasons.append(
                    f"v3={v3_confidence:.2f} < {self.min_short_v3} — "
                    f"blocked {len(blocked)} short strategies"
                )
            viable = [c for c in viable if c.direction != "short"]

        # Select best of each direction
        best_short = next((c for c in viable if c.direction == "short"), None)
        best_long = next((c for c in viable if c.direction == "long"), None)

        # Determine action
        if best_short and best_long:
            action = TradeAction.BOTH
            reasons.append(
                f"Best short: {best_short.strategy_name} EV=${best_short.ev_per_contract:+.2f}"
            )
            reasons.append(
                f"Best long: {best_long.strategy_name} EV=${best_long.ev_per_contract:+.2f}"
            )
        elif best_short:
            action = TradeAction.SHORT_ONLY
            reasons.append(
                f"Best short: {best_short.strategy_name} EV=${best_short.ev_per_contract:+.2f}"
            )
            if not best_long and v7_scores:
                best_long_candidate = next(
                    (c for c in all_candidates if c.direction == "long"), None
                )
                if best_long_candidate:
                    reasons.append(
                        f"No viable long (best: {best_long_candidate.strategy_name} "
                        f"EV=${best_long_candidate.ev_per_contract:+.2f} < min ${self.min_ev:.2f})"
                    )
        elif best_long:
            action = TradeAction.LONG_ONLY
            reasons.append(
                f"Best long: {best_long.strategy_name} EV=${best_long.ev_per_contract:+.2f}"
            )
            if v3_confidence < self.min_short_v3:
                reasons.append(f"Shorts blocked by v3={v3_confidence:.2f}")
            else:
                best_short_candidate = next(
                    (c for c in all_candidates if c.direction == "short"), None
                )
                if best_short_candidate:
                    reasons.append(
                        f"No viable short (best: {best_short_candidate.strategy_name} "
                        f"EV=${best_short_candidate.ev_per_contract:+.2f})"
                    )
        else:
            action = TradeAction.NEITHER
            if all_candidates:
                best_overall = all_candidates[0]
                reasons.append(
                    f"No viable strategies. Best: {best_overall.strategy_name} "
                    f"EV=${best_overall.ev_per_contract:+.2f} (below ${self.min_ev:.2f})"
                )
            else:
                reasons.append("No strategies scored.")

        # Build decision
        decision = TradeDecision(
            action=action,
            confidence=self._compute_confidence(best_short, best_long, all_candidates),
            all_candidates=all_candidates,
            reasons=reasons,
        )

        if best_short:
            decision.short_delta = best_short.delta
            decision.short_ev = best_short.ev_per_contract
            decision.short_v3_confidence = v3_confidence
            decision.short_v5_tp25 = best_short.win_probability

        if best_long:
            decision.long_delta = best_long.delta
            decision.long_ev = best_long.ev_per_contract
            decision.long_v7_score = best_long.confidence
            decision.long_recommended_entry_start = self.long_entry_start
            decision.long_recommended_entry_end = self.long_entry_end

        logger.info(f"Decision: {decision.summary()}")
        for r in reasons:
            logger.debug(f"  → {r}")

        return decision

    # ── For backward compat with existing bot code ────────────────────────

    def decide_legacy(
        self,
        v3_confidence: float,
        v7_scores: Dict[float, float],
        v6_score: Optional[float] = None,
    ) -> TradeDecision:
        """
        Legacy interface that converts v3+v7 to the new EV-based system.
        Synthesizes v5 scores from v3 confidence (assumes v5 ≈ base TP rate
        when no v5 models are available).
        """
        base_tp25 = {0.10: 0.843, 0.15: 0.827, 0.20: 0.796}
        v5_scores = {}
        for delta in [0.10, 0.15, 0.20]:
            adj = base_tp25[delta] * (0.8 + 0.4 * v3_confidence)
            v5_scores[delta] = {"tp25": min(adj, 0.99), "tp50": 0.0}

        return self.decide(v3_confidence, v5_scores, v7_scores)

    # ── Private helpers ──────────────────────────────────────────────────

    def _compute_confidence(
        self,
        best_short: Optional[StrategyCandidate],
        best_long: Optional[StrategyCandidate],
        all_candidates: List[StrategyCandidate],
    ) -> float:
        """Confidence based on EV magnitude relative to thresholds."""
        evs = []
        if best_short:
            evs.append(best_short.ev_per_contract)
        if best_long:
            evs.append(best_long.ev_per_contract)

        if not evs:
            # NEITHER — confidence in sitting out
            if all_candidates:
                worst_ev = all_candidates[0].ev_per_contract
                return round(min(max(1.0 - worst_ev / self.min_ev, 0.0), 1.0), 4)
            return 0.5

        avg_ev = sum(evs) / len(evs)
        raw = 0.5 + 0.5 * (avg_ev - self.min_ev) / max(1.0 - self.min_ev, 0.01)
        return round(min(max(raw, 0.0), 1.0), 4)


# ── Demo / Self-Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level:<7} | {message}")

    engine = TradeDecisionEngine()

    scenarios = [
        (
            "Safe day, strong short signal at 15δ",
            0.82,
            {0.10: {"tp25": 0.90, "tp50": 0.72}, 0.15: {"tp25": 0.92, "tp50": 0.75}, 0.20: {"tp25": 0.85, "tp50": 0.60}},
            {0.10: 0.30, 0.15: 0.35, 0.20: 0.40},
        ),
        (
            "Safe day, both directions viable",
            0.78,
            {0.10: {"tp25": 0.88, "tp50": 0.70}, 0.15: {"tp25": 0.90, "tp50": 0.72}, 0.20: {"tp25": 0.84, "tp50": 0.62}},
            {0.10: 0.50, 0.15: 0.58, 0.20: 0.62},
        ),
        (
            "Dangerous day, long IC opportunity at 20δ",
            0.35,
            {0.10: {"tp25": 0.75, "tp50": 0.55}, 0.15: {"tp25": 0.70, "tp50": 0.50}, 0.20: {"tp25": 0.65, "tp50": 0.45}},
            {0.10: 0.48, 0.15: 0.55, 0.20: 0.60},
        ),
        (
            "Moderate risk, no strong signals",
            0.55,
            {0.10: {"tp25": 0.80, "tp50": 0.60}, 0.15: {"tp25": 0.78, "tp50": 0.58}, 0.20: {"tp25": 0.74, "tp50": 0.52}},
            {0.10: 0.30, 0.15: 0.35, 0.20: 0.38},
        ),
        (
            "Everything looks bad — sit out",
            0.30,
            {0.10: {"tp25": 0.70, "tp50": 0.50}, 0.15: {"tp25": 0.65, "tp50": 0.45}, 0.20: {"tp25": 0.60, "tp50": 0.40}},
            {0.10: 0.25, 0.15: 0.28, 0.20: 0.32},
        ),
    ]

    print("\n" + "=" * 80)
    print("UNIFIED TRADE DECISION ENGINE — EV-BASED STRATEGY SELECTION")
    print("=" * 80)

    for desc, v3, v5, v7 in scenarios:
        print(f"\n--- {desc} ---")
        result = engine.decide(v3_confidence=v3, v5_scores=v5, v7_scores=v7)
        print(f"  {result.summary()}")
        for reason in result.reasons:
            print(f"    → {reason}")

        table = engine.scorer.display_scorecard(result.all_candidates, engine.min_ev)
        for line in table.split("\n"):
            print(f"  {line}")

    print("\n" + "=" * 80)
    print("All scenarios complete.")
