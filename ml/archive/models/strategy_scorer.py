"""
Strategy Scorer — converts raw ML probabilities into Expected Value (EV) per
contract for all 6 possible 0DTE iron condor strategies, enabling apples-to-apples
comparison.

Strategies:
    Short IC @ 10δ, 15δ, 20δ
    Long  IC @ 10δ, 15δ, 20δ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyCandidate:
    direction: str        # "short" or "long"
    delta: float          # 0.10, 0.15, 0.20
    ev_per_contract: float  # Expected $ value per contract
    confidence: float     # Primary model confidence (0-1)
    win_probability: float  # Estimated win probability
    components: dict      # Breakdown of EV computation

    @property
    def strategy_name(self) -> str:
        d = int(self.delta * 100)
        return f"{'Short' if self.direction == 'short' else 'Long'} IC @ {d}δ"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class StrategyScorer:
    """Scores all 6 strategies and returns ranked candidates."""

    # Historical calibration constants from training data (test period 2025-06-01+).
    # Three P&L buckets: TP hit, moderate loss (missed TP, pnl >= -$3), big loss (pnl < -$3)
    SHORT_PARAMS: dict[float, dict] = {
        0.10: {
            "avg_credit": 1.59,
            "avg_tp_pnl": 0.43,         # avg P&L when hit_25pct=1 (84.3% base rate)
            "avg_moderate_pnl": -0.77,   # avg P&L when missed TP but pnl >= -$3 (10.0%)
            "avg_big_loss": -8.70,       # avg P&L on pnl < -$3 days (5.6%)
            "big_loss_rate": 0.056,
        },
        0.15: {
            "avg_credit": 2.70,
            "avg_tp_pnl": 0.75,          # (82.7% base rate)
            "avg_moderate_pnl": -0.84,   # (8.5%)
            "avg_big_loss": -9.12,       # (8.8%)
            "big_loss_rate": 0.088,
        },
        0.20: {
            "avg_credit": 3.94,
            "avg_tp_pnl": 1.12,          # (79.6% base rate)
            "avg_moderate_pnl": -0.87,   # (8.7%)
            "avg_big_loss": -9.66,       # (11.7%)
            "big_loss_rate": 0.117,
        },
    }

    LONG_PARAMS: dict[float, dict] = {
        0.10: {
            "avg_win": 3.72,       # avg long P&L when winning (short lost)
            "avg_loss": -1.01,     # avg long P&L when losing (short won)
            "slippage": 0.20,      # estimated slippage + spread cost
        },
        0.15: {
            "avg_win": 4.81,
            "avg_loss": -1.69,
            "slippage": 0.30,
        },
        0.20: {
            "avg_win": 5.60,
            "avg_loss": -2.31,
            "slippage": 0.40,
        },
    }

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def score_short(
        self,
        delta: float,
        v3_confidence: float,
        v5_tp25: float,
        v5_tp50: float = 0.0,
    ) -> StrategyCandidate:
        """
        Short IC EV = P(hit_TP) * avg_tp_profit
                     + P(big_loss) * avg_big_loss
                     + P(other) * avg_other_loss

        v3_confidence is NOT a calibrated probability. We scale the base big-loss
        rate by v3 confidence:
            v3=1.0 → big_loss ≈ 0 (perfect safety)
            v3=0.5 → big_loss = base rate
            v3=0.0 → big_loss = 2× base rate (maximum danger)
        """
        params = self.SHORT_PARAMS[delta]

        p_tp = v5_tp25
        # Scale big loss rate relative to baseline using v3 confidence
        # v3=1.0 → 0%, v3=0.5 → base rate, v3=0 → 2× base rate
        base_bl = params["big_loss_rate"]
        p_big_loss = min(base_bl * (2.0 - 2.0 * v3_confidence), 1.0 - p_tp)
        p_big_loss = max(0.0, p_big_loss)
        p_moderate = max(0.0, 1.0 - p_tp - p_big_loss)

        ev = (
            p_tp * params["avg_tp_pnl"]
            + p_big_loss * params["avg_big_loss"]
            + p_moderate * params["avg_moderate_pnl"]
        )

        logger.debug(
            "score_short delta={} | p_tp={:.3f} p_big={:.3f} p_mod={:.3f} | EV={:.3f}",
            delta, p_tp, p_big_loss, p_moderate, ev,
        )

        return StrategyCandidate(
            direction="short",
            delta=delta,
            ev_per_contract=round(ev, 4),
            confidence=v3_confidence,
            win_probability=p_tp,
            components={
                "p_tp": round(p_tp, 4),
                "p_big_loss": round(p_big_loss, 4),
                "p_moderate": round(p_moderate, 4),
                "avg_tp_pnl": params["avg_tp_pnl"],
                "avg_moderate_pnl": params["avg_moderate_pnl"],
                "avg_big_loss": params["avg_big_loss"],
                "v5_tp50": round(v5_tp50, 4),
            },
        )

    def score_long(
        self,
        delta: float,
        v7_score: float,
        v3_confidence: float,
    ) -> StrategyCandidate:
        """
        Long IC EV = P(profit) * avg_win * danger_boost
                   - (1 - P(profit)) * |avg_loss|
                   - slippage

        Where:
            P(profit)    = v7_score
            danger_boost = amplifies expected long win when v3 flags danger
                           (1.0 at conf=0.5, up to 1.5 at conf=0)
        """
        params = self.LONG_PARAMS[delta]

        p_profit = v7_score
        # Dangerous-day boost: when v3 flags danger, long IC wins are larger
        danger_boost = max(1.0, 1.5 - v3_confidence)

        ev = (
            p_profit * params["avg_win"] * danger_boost
            - (1 - p_profit) * abs(params["avg_loss"])
            - params["slippage"]
        )

        logger.debug(
            "score_long delta={} | p_profit={:.3f} danger_boost={:.2f} | EV={:.3f}",
            delta, p_profit, danger_boost, ev,
        )

        return StrategyCandidate(
            direction="long",
            delta=delta,
            ev_per_contract=round(ev, 4),
            confidence=v7_score,
            win_probability=p_profit,
            components={
                "p_profit": round(p_profit, 4),
                "danger_boost": round(danger_boost, 4),
                "avg_win": params["avg_win"],
                "avg_loss": params["avg_loss"],
                "slippage": params["slippage"],
            },
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def score_all(
        self,
        v3_confidence: float,
        v5_scores: dict | None,   # {delta: {"tp25": float, "tp50": float}}
        v7_scores: dict | None,   # {delta: float}
    ) -> list[StrategyCandidate]:
        """Score all available strategies, return sorted by EV descending."""
        candidates: list[StrategyCandidate] = []

        for delta in [0.10, 0.15, 0.20]:
            if v5_scores and delta in v5_scores:
                entry = v5_scores[delta]
                candidates.append(
                    self.score_short(
                        delta,
                        v3_confidence,
                        entry["tp25"],
                        entry.get("tp50", 0.0),
                    )
                )
            if v7_scores and delta in v7_scores:
                candidates.append(
                    self.score_long(delta, v7_scores[delta], v3_confidence)
                )

        candidates.sort(key=lambda c: c.ev_per_contract, reverse=True)
        return candidates

    def recommend(
        self,
        v3_confidence: float,
        v5_scores: dict | None,
        v7_scores: dict | None,
        min_ev: float = 0.10,
    ) -> dict:
        """
        Returns recommendation dict:
        {
            "action": "SHORT_ONLY" | "LONG_ONLY" | "BOTH" | "NEITHER",
            "best_short": StrategyCandidate | None,
            "best_long": StrategyCandidate | None,
            "all_candidates": [StrategyCandidate, ...],
            "reasoning": str,
        }
        """
        all_candidates = self.score_all(v3_confidence, v5_scores, v7_scores)
        viable = [c for c in all_candidates if c.ev_per_contract >= min_ev]

        best_short = next((c for c in viable if c.direction == "short"), None)
        best_long = next((c for c in viable if c.direction == "long"), None)

        if best_short and best_long:
            action = "BOTH"
        elif best_short:
            action = "SHORT_ONLY"
        elif best_long:
            action = "LONG_ONLY"
        else:
            action = "NEITHER"

        reasoning = self._build_reasoning(
            action, best_short, best_long, all_candidates, min_ev,
        )

        logger.info("Recommendation: {} | {}", action, reasoning)

        return {
            "action": action,
            "best_short": best_short,
            "best_long": best_long,
            "all_candidates": all_candidates,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_scorecard(
        self,
        candidates: list[StrategyCandidate],
        min_ev: float = 0.10,
    ) -> str:
        """Return a formatted table of all strategies with their EVs."""
        header = (
            f"{'Strategy':<18} {'EV/contract':>12} {'Win Prob':>9} "
            f"{'Confidence':>11} {'Viable':>7}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for c in candidates:
            viable_flag = " YES" if c.ev_per_contract >= min_ev else "  no"
            lines.append(
                f"{c.strategy_name:<18} "
                f"{'$' + f'{c.ev_per_contract:+.2f}':>12} "
                f"{c.win_probability:>8.1%} "
                f"{c.confidence:>10.1%} "
                f"{viable_flag:>7}"
            )

        lines.append(sep)
        table = "\n".join(lines)
        logger.info("Strategy scorecard:\n{}", table)
        return table

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        action: str,
        best_short: StrategyCandidate | None,
        best_long: StrategyCandidate | None,
        all_candidates: list[StrategyCandidate],
        min_ev: float,
    ) -> str:
        """Create a human-readable explanation of the decision."""
        parts: list[str] = []

        n_viable = sum(1 for c in all_candidates if c.ev_per_contract >= min_ev)
        n_total = len(all_candidates)
        parts.append(f"{n_viable}/{n_total} strategies above min EV ${min_ev:.2f}.")

        if action == "NEITHER":
            if all_candidates:
                best = all_candidates[0]
                parts.append(
                    f"Best was {best.strategy_name} at ${best.ev_per_contract:+.2f} "
                    f"(below threshold). Standing aside."
                )
            else:
                parts.append("No strategies scored. Standing aside.")
            return " ".join(parts)

        if best_short:
            parts.append(
                f"Best short: {best_short.strategy_name} "
                f"EV=${best_short.ev_per_contract:+.2f} "
                f"(TP prob {best_short.win_probability:.0%})."
            )

        if best_long:
            parts.append(
                f"Best long: {best_long.strategy_name} "
                f"EV=${best_long.ev_per_contract:+.2f} "
                f"(win prob {best_long.win_probability:.0%})."
            )

        if action == "BOTH":
            parts.append("Both directions offer positive EV.")

        return " ".join(parts)
