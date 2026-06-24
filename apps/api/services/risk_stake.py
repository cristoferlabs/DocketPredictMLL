"""Risk / Stake Layer — asignación de bankroll para SHARP y PARLAY."""

from __future__ import annotations

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.shared.config import get_settings


def allocate_sharp_stake(
    decision: BetDecisionResult,
    *,
    ev_final: float,
    mds: int,
    confidence_norm: float,
) -> float:
    """Stake % bankroll para single SHARP (conservador)."""
    settings = get_settings()
    base = decision.stake_pct
    if base <= 0 and decision.pick:
        kelly = (decision.pick.kelly_stake or 0) * 100
        base = kelly
    if base <= 0:
        base = 0.5 if ev_final >= 0.05 else 0.25

    mds_factor = min(1.0, mds / 85.0)
    conf_factor = min(1.0, confidence_norm / 0.85)
    stake = base * mds_factor * conf_factor * settings.kelly_fraction * 4
    cap = settings.sharp_max_stake_pct
    return round(max(0.1, min(cap, stake)), 2)


def allocate_parlay_stake(
    *,
    combined_prob: float,
    ev_parlay: float,
    combo_score: float,
    n_legs: int,
    correlation_penalty: float,
) -> float:
    """Stake % bankroll para combinada (menor que singles)."""
    settings = get_settings()
    if ev_parlay <= 0 or combined_prob <= 0:
        return 0.0
    base = settings.parlay_base_stake_pct
    leg_factor = 0.85 ** max(0, n_legs - 3)
    score_boost = min(1.3, 1.0 + combo_score * 2)
    stake = base * leg_factor * correlation_penalty * score_boost
    if ev_parlay >= 0.15:
        stake *= 1.1
    return round(max(0.05, min(settings.parlay_max_stake_pct, stake)), 2)
