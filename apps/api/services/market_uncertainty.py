"""
Market Uncertainty Score (MUS) — cuánto confiar en el mercado en DECISION.

MUS = 1 - confidence_market

El mercado es otra estimación probabilística, no verdad absoluta.
Esta capa solo alimenta el árbol de decisión; nunca altera probabilidades del modelo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from apps.api.services.odds_context import MarketContext1X2, OutcomeEdge
from apps.worker.ml.odds_math import expected_value_fair, expected_value_raw

if TYPE_CHECKING:
    from apps.api.services.trust_arbitration import TrustArbitration

SoftBetAction = Literal["NO_BET", "WATCH", "WEAK_BET", "STRONG_BET"]


@dataclass(frozen=True)
class MarketUncertaintyResult:
    confidence_market: float
    mus: float
    book_consistency: float
    pricing_conviction: float
    news_volatility: float
    line_stability: float
    rationale: str


@dataclass(frozen=True)
class EvBand:
    selection: str
    optimistic: float
    base: float
    pessimistic: float


def _implied_entropy(implieds: list[float]) -> float:
    if not implieds:
        return 1.0
    total = sum(implieds)
    if total <= 0:
        return 1.0
    probs = [p / total for p in implieds if p > 0]
    if len(probs) < 2:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    max_ent = math.log(len(probs))
    return ent / max_ent if max_ent > 0 else 0.0


def _estimate_line_stability(market_ctx: MarketContext1X2) -> float:
    """Proxy de estabilidad: libros + convicción + baja divergencia modelo-mercado."""
    n_books = market_ctx.n_books or 1
    base = min(1.0, 0.28 + n_books * 0.12)
    implieds = [o.market_implied for o in market_ctx.outcomes if o.market_implied]
    if implieds:
        peak = max(implieds)
        base *= 0.65 + 0.35 * min(1.0, peak / 0.55)
    divs = [o.divergence for o in market_ctx.outcomes if o.divergence]
    if divs:
        base *= max(0.40, 1.0 - max(divs) / 0.38)
    return round(max(0.20, min(1.0, base)), 3)


def compute_market_uncertainty(
    market_ctx: MarketContext1X2 | None,
    *,
    has_injury_news: bool = False,
    has_suspensions: bool = False,
    line_stability: float | None = None,
) -> MarketUncertaintyResult:
    """Estima confianza del mercado (otra estimación, no verdad)."""
    if not market_ctx or not market_ctx.has_market:
        return MarketUncertaintyResult(
            confidence_market=0.0,
            mus=1.0,
            book_consistency=0.0,
            pricing_conviction=0.0,
            news_volatility=0.0,
            line_stability=0.5,
            rationale="sin mercado — máxima incertidumbre",
        )

    n_books = market_ctx.n_books or 1
    book_consistency = min(1.0, n_books / 5.0)

    implieds = [o.market_implied for o in market_ctx.outcomes if o.market_implied]
    pricing_conviction = max(implieds) if implieds else 0.5
    entropy = _implied_entropy([i for i in implieds if i])
    pricing_sharpness = pricing_conviction * (1.0 - 0.35 * entropy)

    if has_injury_news and has_suspensions:
        news_volatility = 0.75
    elif has_injury_news or has_suspensions:
        news_volatility = 0.55
    else:
        news_volatility = 0.15

    stability = line_stability if line_stability is not None else _estimate_line_stability(market_ctx)

    confidence = (
        0.30 * book_consistency
        + 0.25 * stability
        + 0.20 * (1.0 - news_volatility)
        + 0.25 * pricing_sharpness
    )
    confidence = max(0.0, min(1.0, confidence))
    mus = round(1.0 - confidence, 3)

    parts = [
        f"libros {book_consistency:.0%}",
        f"convicción {pricing_conviction:.0%}",
        f"news vol {news_volatility:.0%}",
    ]
    return MarketUncertaintyResult(
        confidence_market=round(confidence, 3),
        mus=mus,
        book_consistency=round(book_consistency, 3),
        pricing_conviction=round(pricing_conviction, 3),
        news_volatility=round(news_volatility, 3),
        line_stability=round(stability, 3),
        rationale="; ".join(parts),
    )


def compute_ev_band(
    outcome: OutcomeEdge,
    *,
    mus: float,
    market_confidence: float,
) -> EvBand:
    """Banda EV fair para decisión; pesimista usa fair odds."""
    base = outcome.ev_fair_pct / 100.0
    optimistic = base
    pick_div = outcome.divergence or 0.0
    if outcome.fair_implied and outcome.fair_odds and outcome.fair_odds > 1:
        pes_prob = min(outcome.model_prob, outcome.fair_implied)
        pessimistic = expected_value_fair(pes_prob, outcome.fair_odds)
    elif outcome.market_implied and outcome.market_odds and outcome.market_odds > 1:
        pes_prob = min(outcome.model_prob, outcome.market_implied)
        pessimistic = expected_value_raw(pes_prob, outcome.market_odds)
    else:
        haircut = market_confidence * pick_div * 0.75
        pessimistic = base * max(0.0, 1.0 - haircut)
    pessimistic = min(base, round(pessimistic, 4))
    return EvBand(
        selection=outcome.selection,
        optimistic=round(optimistic, 4),
        base=round(base, 4),
        pessimistic=pessimistic,
    )


def compute_ev_band_from_pick(
    *,
    selection: str,
    model_prob: float,
    market_odds: float,
    divergence: float | None,
    mus: float,
    market_confidence: float,
    ev_base: float,
) -> EvBand:
    outcome = OutcomeEdge(
        selection=selection,
        model_prob=model_prob,
        model_fair_odds=round(1 / model_prob, 2) if model_prob > 0 else 0,
        market_odds=market_odds,
        edge_pct=ev_base * 100,
        divergence=divergence,
    )
    return compute_ev_band(
        outcome, mus=mus, market_confidence=market_confidence
    )


def resolve_soft_decision(
    *,
    ev_band: EvBand,
    mus: float,
    max_divergence: float,
    pick_divergence: float,
    confidence_score: int,
    diagnosis_primary: str | None = None,
    trust: TrustArbitration | None = None,
) -> tuple[SoftBetAction, str]:
    """
    Árbol suave con arbitraje de confianza en desacoples.

    Ya no aplica regla simétrica «Δ alto = peligro».
    """
    mc = 1.0 - mus

    if ev_band.base <= 0 and ev_band.optimistic <= 0:
        return "NO_BET", "sin valor"

    high_discrepancy = max_divergence >= 0.18 or pick_divergence >= 0.10

    if high_discrepancy and trust is not None:
        if trust.trust_side == "model" and ev_band.base >= 0.02:
            if ev_band.base >= 0.04 and confidence_score >= 40:
                return "STRONG_BET", f"confiar modelo — {trust.rationale}"
            return "WEAK_BET", f"edge modelo mid-range — {trust.rationale}"
        if trust.trust_side == "market":
            if ev_band.optimistic >= 0.12:
                return "WATCH", f"mercado domina desacople — {trust.rationale}"
            return "NO_BET", f"seguir mercado — {trust.rationale}"
        if trust.trust_side == "ambiguous":
            if ev_band.optimistic >= 0.10:
                return "WATCH", f"desacople ambiguo — {trust.rationale}"
            return "NO_BET", f"sin ventaja clara — {trust.rationale}"

    if high_discrepancy and pick_divergence >= 0.18 and trust is None:
        if mus >= 0.42 and ev_band.pessimistic >= 0.025:
            return "WEAK_BET", "mercado incierto + EV pesimista positivo"
        return "WATCH", "mismatch — observar"

    if mus >= 0.40 and ev_band.pessimistic >= 0.025:
        if ev_band.base >= 0.05 and confidence_score >= 42:
            return "STRONG_BET", "MUS alto + EV pesimista robusto"
        return "WEAK_BET", "MUS alto + EV positivo post-haircut"

    if (
        ev_band.optimistic >= 0.15
        and mus >= 0.38
        and (high_discrepancy or pick_divergence >= 0.12)
        and (trust is None or trust.trust_side != "model")
    ):
        return "WATCH", "edge extremo — observar sin stake pleno"

    if ev_band.base <= 0:
        return "NO_BET", "EV base negativo"

    if not high_discrepancy and ev_band.base >= 0.04 and confidence_score >= 45:
        return "STRONG_BET", "modelo y mercado alineados"
    if ev_band.base >= 0.02 and confidence_score >= 35:
        return "WEAK_BET", "EV moderado"
    if ev_band.base > 0:
        return "WATCH", "edge marginal — vigilar"

    return "NO_BET", "sin valor tras contexto"
