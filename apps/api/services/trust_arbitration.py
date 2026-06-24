"""
Trust Arbitration — ¿quién es más confiable cuando modelo y mercado divergen?

NO altera probabilidades del MODEL layer.
Solo informa DECISION / SHARP sobre a quién seguir en desacoples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apps.api.services.market_dominance import MarketDominanceResult, model_confidence_tier
from apps.api.services.worldcup_engine import ModelMarkets

TrustSide = Literal["model", "market", "ambiguous"]


@dataclass(frozen=True)
class TrustArbitration:
    model_confidence: float
    market_confidence: float
    trust_side: TrustSide
    trust_ratio: float
    w_model: float
    w_market: float
    rationale: str


def compute_pick_model_confidence(
    *,
    pick_model_prob: float,
    pick_market_implied: float | None,
    model: ModelMarkets,
    data_quality_pct: float,
    max_divergence: float,
) -> float:
    tier = model_confidence_tier(model)
    tier_score = {"high": 0.88, "medium": 0.72, "low": 0.52}.get(tier, 0.72)
    dq = max(0.0, min(1.0, data_quality_pct / 100.0))
    mkt = pick_market_implied or 0.0
    edge_dir = pick_model_prob - mkt

    conf = tier_score * (0.55 + 0.45 * dq)
    if edge_dir >= 0.10:
        conf += 0.14
    elif edge_dir >= 0.05:
        conf += 0.08
    if 0.42 <= pick_model_prob <= 0.68:
        conf += 0.06
    if mkt < 0.12 and pick_model_prob > 0.28:
        conf -= 0.28
    if mkt > 0 and pick_model_prob > mkt + 0.12 and max_divergence < 0.22:
        conf += 0.05
    if max_divergence >= 0.28 and pick_model_prob < 0.38 and mkt < 0.35:
        conf -= 0.12
    return max(0.0, min(1.0, conf))


def compute_pick_market_confidence(
    *,
    pick_model_prob: float,
    pick_market_implied: float | None,
    dominance: MarketDominanceResult,
    pick_divergence: float,
) -> float:
    u = dominance.uncertainty
    base = u.confidence_market if u else dominance.market_reliability
    mkt = pick_market_implied or 0.0

    conf = base
    if mkt >= 0.68 and mkt > pick_model_prob + 0.10:
        conf += 0.12
    if mkt >= 0.75:
        conf += 0.08
    if mkt < 0.12 and pick_model_prob > 0.25:
        conf += 0.22
    if pick_divergence >= 0.22 and mkt > pick_model_prob:
        conf += 0.06
    if u and u.book_consistency >= 0.6:
        conf += 0.04
    if dominance.max_raw_divergence >= 0.30 and mkt < 0.40:
        conf -= 0.10
    return max(0.0, min(1.0, conf))


def arbitrate_pick_trust(
    *,
    pick_model_prob: float,
    pick_market_implied: float | None,
    pick_divergence: float,
    model: ModelMarkets,
    dominance: MarketDominanceResult,
    data_quality_pct: float = 100.0,
    market_favorite_implied: float | None = None,
) -> TrustArbitration:
    """
    Si discrepancia alta: investigar quién es más confiable (no bloqueo simétrico).

    model >> market → confiar modelo (edge mid-range válido)
    market >> model → seguir mercado (bloquear pick contrario al mercado)
    similar → ambiguo (NO BET o WATCH)
    """
    max_div = dominance.max_raw_divergence
    model_conf = compute_pick_model_confidence(
        pick_model_prob=pick_model_prob,
        pick_market_implied=pick_market_implied,
        model=model,
        data_quality_pct=data_quality_pct,
        max_divergence=max_div,
    )
    market_conf = compute_pick_market_confidence(
        pick_model_prob=pick_model_prob,
        pick_market_implied=pick_market_implied,
        dominance=dominance,
        pick_divergence=pick_divergence,
    )
    if market_favorite_implied and market_favorite_implied >= 0.68:
        mkt = pick_market_implied or 0.0
        if mkt < 0.22 and pick_model_prob < 0.45:
            market_conf = min(1.0, market_conf + 0.18)
            model_conf = max(0.0, model_conf - 0.18)
    ratio = model_conf / max(0.05, market_conf)

    if ratio >= 1.22:
        side: TrustSide = "model"
        rationale = f"modelo más confiable (ratio {ratio:.2f})"
    elif ratio <= 0.82:
        side = "market"
        rationale = f"mercado más confiable (ratio {ratio:.2f})"
    else:
        side = "ambiguous"
        rationale = f"confianzas similares — ambiguo (ratio {ratio:.2f})"

    total = model_conf + market_conf
    w_model = round(model_conf / total, 3) if total > 0 else 0.5
    w_market = round(1.0 - w_model, 3)

    return TrustArbitration(
        model_confidence=round(model_conf, 3),
        market_confidence=round(market_conf, 3),
        trust_side=side,
        trust_ratio=round(ratio, 3),
        w_model=w_model,
        w_market=w_market,
        rationale=rationale,
    )
