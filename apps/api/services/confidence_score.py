"""
Confianza unificada — sin acumular penalizaciones por capa.

  confidence = 0.4 × MDS + 0.3 × model_reliability + 0.3 × trust
"""

from __future__ import annotations

from apps.api.services.market_dominance import MarketDominanceResult, market_agreement_score
from apps.api.services.trust_arbitration import TrustArbitration


def trust_component(trust: TrustArbitration) -> float:
    if trust.trust_side == "model":
        return trust.model_confidence
    if trust.trust_side == "market":
        return trust.market_confidence
    return (trust.model_confidence + trust.market_confidence) / 2.0


def compute_mds(
    dominance: MarketDominanceResult,
    trust: TrustArbitration | None = None,
) -> int:
    """Match Decision Score (0–100)."""
    agreement = market_agreement_score(dominance.max_raw_divergence)
    align_bonus = max(0.0, 1.0 - dominance.max_raw_divergence / 0.35)
    raw = (
        0.40 * dominance.model_reliability
        + 0.30 * agreement
        + 0.20 * align_bonus
        + 0.10 * dominance.market_reliability
    )
    if trust:
        if trust.trust_side == "model":
            raw += 0.14
            if trust.model_confidence >= 0.85:
                raw += 0.03
        elif trust.trust_side == "market":
            raw -= 0.10
        elif trust.trust_side == "ambiguous":
            raw -= 0.04
    return int(max(0, min(100, round(raw * 100))))


def compute_unified_confidence(
    *,
    mds: int,
    model_reliability: float,
    trust: TrustArbitration,
    cold_start: bool = False,
) -> int:
    mds_n = max(0.0, min(1.0, mds / 100.0))
    rel = max(0.0, min(1.0, model_reliability))
    tr = max(0.0, min(1.0, trust_component(trust)))
    score = 0.4 * mds_n + 0.3 * rel + 0.3 * tr
    out = int(max(0, min(100, round(score * 100))))
    if cold_start:
        out = min(out, 58)
    return out


def sharp_composite_passes(confidence_score: int, *, settings=None) -> bool:
    """Gate SHARP único — evita correlación MDS + confianza por separado."""
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    return confidence_score >= settings.sharp_min_composite
