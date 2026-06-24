"""
Fachada de compatibilidad — lógica migrada a market_dominance.py.

Nuevos imports deben usar apps.api.services.market_dominance directamente.
"""

from __future__ import annotations

from typing import Any

from apps.api.services.market_dominance import (
    DISCREPANCY_LABELS,
    LAYER_LABELS,
    DecisionLayer,
    DiscrepancyDiagnosis,
    MarketAdjustment,
    MarketDominanceResult,
    blend_1x2_probs,
    classify_decision_layer,
    detect_market_dominance,
    diagnose_discrepancy,
    market_agreement_score,
    market_confidence_weights,
    model_confidence_tier,
    normalized_market_probs,
    recalculate_outcome_edges,
)
from apps.api.services.odds_context import MarketContext1X2
from apps.api.services.worldcup_engine import ModelMarkets

__all__ = [
    "DISCREPANCY_LABELS",
    "LAYER_LABELS",
    "DecisionLayer",
    "DiscrepancyDiagnosis",
    "MarketAdjustment",
    "MarketDominanceResult",
    "apply_market_calibration",
    "blend_1x2_probs",
    "classify_decision_layer",
    "compute_recalibrated_confidence",
    "detect_market_dominance",
    "diagnose_discrepancy",
    "market_agreement_score",
    "market_confidence_weights",
    "model_confidence_tier",
    "normalized_market_probs",
    "recalculate_outcome_edges",
]


def compute_recalibrated_confidence(
    *,
    data_quality_pct: float,
    market_agreement: float,
    historical_accuracy: float | None = None,
    injury_penalty: float = 0.0,
    model_tier: str = "medium",
    layer: DecisionLayer = "normal",
    mds: int | None = None,
    model_reliability: float | None = None,
    trust: Any | None = None,
) -> int:
    """Compat — delega a fórmula unificada si hay trust; si no, fallback legacy."""
    if trust is not None and mds is not None and model_reliability is not None:
        from apps.api.services.confidence_score import compute_unified_confidence

        return compute_unified_confidence(
            mds=mds,
            model_reliability=model_reliability,
            trust=trust,
        )
    hist = historical_accuracy if historical_accuracy is not None else 0.33
    agreement = max(0.0, min(1.0, market_agreement))
    dq = max(0.0, min(100.0, data_quality_pct))
    tier_bonus = {"high": 8, "medium": 0, "low": -6}.get(model_tier, 0)
    layer_penalty = {"normal": 0, "doubt": -5, "extreme": -5}.get(layer, 0)
    score = (
        0.40 * hist * 100
        + 0.25 * agreement * 100
        + 0.35 * dq
        + tier_bonus
        + layer_penalty
        - injury_penalty
    )
    return int(max(0, min(100, round(score))))


def apply_market_calibration(
    model: ModelMarkets,
    market_ctx: MarketContext1X2 | None,
    team1: str,
    team2: str,
    *,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    extreme_threshold: float = 0.20,
    doubt_threshold: float = 0.12,
) -> tuple[MarketAdjustment | None, MarketContext1X2 | None]:
    """Compat: delega a detect_market_dominance y devuelve adjustment + adjusted_market."""
    from apps.api.services.worldcup_engine import MatchAnalysis

    if not market_ctx or not model:
        return None, market_ctx

    analysis = MatchAnalysis(
        team1=team1,
        team2=team2,
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    dom = detect_market_dominance(
        analysis,
        market_ctx,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        extreme_threshold=extreme_threshold,
        doubt_threshold=doubt_threshold,
    )
    return dom.adjustment, dom.adjusted_market
