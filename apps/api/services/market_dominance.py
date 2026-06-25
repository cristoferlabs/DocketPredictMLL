"""
Market Dominance Detection — capa formal independiente.

Responde: ¿quién es más confiable en este partido? (sin decidir stake ni pick).
NUNCA mezcla probabilidades modelo+mercado; el modelo base permanece intacto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from apps.api.services.market_uncertainty import MarketUncertaintyResult, compute_market_uncertainty
from apps.api.services.market_alignment import ALIGNMENT_TIERS, alignment_status
from apps.api.services.odds_context import MarketContext1X2, OutcomeEdge, max_market_divergence
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.worker.ml.odds_math import expected_value_raw

DecisionLayer = Literal["normal", "doubt", "extreme"]
DominanceLevel = Literal["aligned", "moderate", "high"]

DISCREPANCY_LABELS: dict[str, tuple[str, str]] = {
    "market_dominant": (
        "Market dominant signal",
        "Mercado domina el pricing; modelo no alineado en este partido",
    ),
    "information_asymmetry": (
        "Information asymmetry",
        "Mismatch estructural — el mercado probablemente incorpora info (news/lineup) que el modelo no tiene",
    ),
    "model_underconfidence": (
        "Model underconfidence",
        "El modelo no refleja la convicción implícita del mercado",
    ),
    "elo_drift": ("ELO drift", "Rating ELO desalineado con el precio de mercado"),
    "poisson_bias": ("Poisson bias", "λ ofensivos mal estimados vs implied del mercado"),
    "alignment_aligned": ALIGNMENT_TIERS["aligned"],
    "alignment_mild": ALIGNMENT_TIERS["mild"],
    "alignment_divergence": ALIGNMENT_TIERS["divergence"],
    "alignment_alert": ALIGNMENT_TIERS["alert"],
}

LAYER_LABELS: dict[str, str] = {
    "normal": "Modelo manda — mercado valida",
    "doubt": "Duda — mercado informa, modelo intacto",
    "extreme": "Desacople alto — arbitraje confianza (no bloqueo automático)",
}


@dataclass
class MarketAdjustment:
    """Metadatos de contexto mercado; nunca altera probabilidades del modelo."""

    home: float
    draw: float
    away: float
    model_weight: float
    market_weight: float
    blend_applied: bool
    layer: DecisionLayer
    layer_reason: str
    blend_reason: str | None
    raw_home: float
    raw_draw: float
    raw_away: float
    market_home: float
    market_draw: float
    market_away: float
    max_raw_divergence: float
    model_confidence_tier: str

    @property
    def shrink_applied(self) -> bool:
        return self.blend_applied


@dataclass
class DiscrepancyDiagnosis:
    primary_type: str
    label: str
    description: str
    secondary_type: str | None = None
    secondary_label: str | None = None
    secondary_description: str | None = None
    result: str = "NO BET"


@dataclass
class OutcomeDominanceSnapshot:
    selection: str
    model_prob: float
    market_implied: float | None
    divergence: float | None


@dataclass
class MarketDominanceResult:
    max_raw_divergence: float
    max_aux_divergence: float | None
    layer: DecisionLayer
    layer_reason: str
    is_market_dominant: bool
    dominance_level: DominanceLevel
    model_reliability: float
    market_reliability: float
    classification: str
    diagnosis: DiscrepancyDiagnosis | None
    adjustment: MarketAdjustment | None
    adjusted_market: MarketContext1X2 | None
    outcome_snapshots: list[OutcomeDominanceSnapshot]
    uncertainty: MarketUncertaintyResult | None = None


def model_confidence_tier(model: ModelMarkets) -> str:
    spread = max(model.home_win, model.draw, model.away_win) - min(
        model.home_win, model.draw, model.away_win
    )
    max_p = max(model.home_win, model.draw, model.away_win)
    if model.confidence == "high" or max_p >= 0.55 or spread >= 0.18:
        return "high"
    if model.confidence == "low" or max_p < 0.42 or spread < 0.08:
        return "low"
    return "medium"


def market_agreement_score(max_divergence: float) -> float:
    return max(0.0, min(1.0, 1.0 - max_divergence / 0.35))


def market_confidence_weights(tier: str) -> tuple[float, float]:
    return {
        "high": (0.80, 0.20),
        "medium": (0.60, 0.40),
        "low": (0.30, 0.70),
    }.get(tier, (0.60, 0.40))


def classify_decision_layer(
    max_divergence: float,
    *,
    extreme_threshold: float = 0.20,
    doubt_threshold: float = 0.12,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    model_tier: str = "medium",
) -> tuple[DecisionLayer, str]:
    if max_divergence >= extreme_threshold:
        return (
            "extreme",
            f"Δ {max_divergence*100:.0f}% ≥ umbral {extreme_threshold*100:.0f}% — arbitraje",
        )

    weak_data = hist_played < 8 or data_quality_pct < 70
    low_conf = model_tier == "low"
    moderate_div = max_divergence >= doubt_threshold

    if moderate_div or weak_data or low_conf:
        parts: list[str] = []
        if moderate_div:
            parts.append("divergencia moderada")
        if weak_data:
            parts.append("datos débiles")
        if low_conf:
            parts.append("baja confianza del modelo")
        return "doubt", "; ".join(parts)

    return "normal", "modelo y mercado alineados"


def _dominance_level(max_div: float, extreme_threshold: float) -> DominanceLevel:
    if max_div >= extreme_threshold:
        return "high"
    if max_div >= 0.12:
        return "moderate"
    return "aligned"


def _compute_model_reliability(
    *,
    model_tier: str,
    data_quality_pct: float,
    agreement: float,
) -> float:
    tier_score = {"high": 0.85, "medium": 0.65, "low": 0.45}.get(model_tier, 0.65)
    dq = max(0.0, min(1.0, data_quality_pct / 100.0))
    return round(min(1.0, max(0.0, 0.40 * tier_score + 0.35 * dq + 0.25 * agreement)), 2)


def _compute_market_reliability(market_ctx: MarketContext1X2) -> float:
    if not market_ctx.has_market:
        return 0.0
    n_books = market_ctx.n_books or 1
    book_score = min(1.0, n_books / 5.0)
    implieds = [o.market_implied for o in market_ctx.outcomes if o.market_implied]
    conviction = 0.5
    if implieds:
        conviction = max(implieds)
    return round(min(1.0, max(0.0, 0.50 * book_score + 0.50 * conviction)), 2)


def normalized_market_probs(
    market_ctx: MarketContext1X2,
    team1: str,
    team2: str,
) -> dict[str, float] | None:
    if not market_ctx.has_market:
        return None
    raw: dict[str, float] = {}
    for o in market_ctx.outcomes:
        if o.market_implied and o.market_implied > 0:
            raw[o.selection] = o.market_implied
    if len(raw) < 2:
        return None
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in raw.items()}


def blend_1x2_probs(
    home: float,
    draw: float,
    away: float,
    market: dict[str, float],
    team1: str,
    team2: str,
    model_weight: float,
) -> tuple[float, float, float]:
    """Legacy — NO usar en el pipeline de decisión. Solo compat/tests externos."""
    mkt_home = market.get(team1, home)
    mkt_draw = market.get("Empate", draw)
    mkt_away = market.get(team2, away)
    mw = model_weight
    kw = 1.0 - mw
    blended = (
        mw * home + kw * mkt_home,
        mw * draw + kw * mkt_draw,
        mw * away + kw * mkt_away,
    )
    total = sum(blended)
    if total <= 0:
        return home, draw, away
    return blended[0] / total, blended[1] / total, blended[2] / total


def recalculate_outcome_edges(
    outcomes: list[OutcomeEdge],
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    team1: str,
    team2: str,
) -> list[OutcomeEdge]:
    prob_map = {team1: home_prob, "Empate": draw_prob, team2: away_prob}
    updated: list[OutcomeEdge] = []
    for o in outcomes:
        p = prob_map.get(o.selection, o.model_prob)
        fair = round(1 / p, 2) if p > 0 else o.model_fair_odds
        edge = (
            round(expected_value_raw(p, o.market_odds) * 100, 1)
            if o.market_odds and o.market_odds > 1
            else 0.0
        )
        impl = o.market_implied
        div = abs(p - impl) if impl is not None else o.divergence
        updated.append(
            OutcomeEdge(
                selection=o.selection,
                model_prob=p,
                model_fair_odds=fair,
                market_odds=o.market_odds,
                edge_pct=edge,
                market_implied=impl,
                divergence=round(div, 4) if div is not None else None,
            )
        )
    return updated


def diagnose_discrepancy(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    *,
    max_divergence: float,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    layer: DecisionLayer = "normal",
    has_injury_news: bool = False,
    has_suspensions: bool = False,
) -> DiscrepancyDiagnosis | None:
    if not market_ctx or not market_ctx.has_market or max_divergence < 0.12:
        return None

    m = analysis.model
    if not m:
        return None

    tier = model_confidence_tier(m)
    weak_data = hist_played < 8 or data_quality_pct < 70

    market_dominant_flag = max_divergence >= 0.20
    if not market_dominant_flag:
        for o in market_ctx.outcomes:
            if (o.market_implied or 0) > 0.70 and (o.divergence or 0) > 0.15:
                market_dominant_flag = True
                break

    elo1 = float(analysis.elo.get(analysis.team1, {}).get("rating") or 1500)
    elo2 = float(analysis.elo.get(analysis.team2, {}).get("rating") or 1500)
    model_fav = analysis.team1 if m.home_win >= m.away_win else analysis.team2
    market_fav = max(market_ctx.outcomes, key=lambda o: o.market_implied or 0).selection
    has_elo_drift = (
        abs(elo1 - elo2) > 80
        and model_fav != market_fav
        and max_divergence > 0.15
    )

    has_poisson_bias = False
    if m.lambda_away > 0 and m.away_win > 0:
        lambda_ratio = m.lambda_home / m.lambda_away
        prob_ratio = m.home_win / m.away_win
        if lambda_ratio > 0 and prob_ratio > 0:
            log_gap = abs(math.log(lambda_ratio) - math.log(prob_ratio))
            has_poisson_bias = log_gap > 0.35 and max_divergence > 0.15

    if market_dominant_flag:
        if max_divergence >= 0.20 and not weak_data:
            if has_injury_news or has_suspensions:
                primary = "information_asymmetry"
            else:
                primary = "market_dominant"
        else:
            primary = "market_dominant"
    elif has_elo_drift:
        primary = "elo_drift"
    elif has_poisson_bias:
        primary = "poisson_bias"
    elif max_divergence >= 0.12:
        gap_pp = max_divergence * 100.0
        align_key, _, _ = alignment_status(gap_pp)
        primary = f"alignment_{align_key}"
    elif weak_data:
        gap_pp = max_divergence * 100.0
        align_key, _, _ = alignment_status(gap_pp)
        primary = f"alignment_{align_key}"
    else:
        return None

    secondary: str | None = None
    if primary == "market_dominant":
        top_div = max((o.divergence or 0) for o in market_ctx.outcomes)
        if top_div > 0.15 or tier in ("low", "medium"):
            secondary = "model_underconfidence"
    elif primary == "elo_drift" and tier == "low":
        secondary = "model_underconfidence"
    elif primary == "poisson_bias" and tier in ("low", "medium"):
        secondary = "model_underconfidence"
    elif primary.startswith("alignment_") and tier == "low":
        secondary = "model_underconfidence"

    p_label, p_desc = DISCREPANCY_LABELS.get(primary, ("Desacople", "Modelo y mercado divergen"))
    s_label, s_desc = (None, None)
    if secondary:
        s_label, s_desc = DISCREPANCY_LABELS.get(secondary, ("", ""))

    result = "REVISAR" if layer == "extreme" else "REVISAR"

    return DiscrepancyDiagnosis(
        primary_type=primary,
        label=p_label,
        description=p_desc,
        secondary_type=secondary,
        secondary_label=s_label,
        secondary_description=s_desc,
        result=result,
    )


def detect_market_dominance(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    *,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    extreme_threshold: float = 0.20,
    doubt_threshold: float = 0.12,
    has_injury_news: bool = False,
    has_suspensions: bool = False,
) -> MarketDominanceResult:
    """Detecta dominancia de mercado y fiabilidad relativa modelo vs mercado."""
    m = analysis.model
    if not m or not market_ctx or not market_ctx.has_market:
        return MarketDominanceResult(
            max_raw_divergence=0.0,
            max_aux_divergence=None,
            layer="normal",
            layer_reason="sin mercado",
            is_market_dominant=False,
            dominance_level="aligned",
            model_reliability=0.5,
            market_reliability=0.0,
            classification="aligned",
            diagnosis=None,
            adjustment=None,
            adjusted_market=None,
            outcome_snapshots=[],
            uncertainty=None,
        )

    uncertainty = compute_market_uncertainty(
        market_ctx,
        has_injury_news=has_injury_news,
        has_suspensions=has_suspensions,
    )

    max_raw = max_market_divergence(market_ctx)
    tier = model_confidence_tier(m)
    layer, layer_reason = classify_decision_layer(
        max_raw,
        extreme_threshold=extreme_threshold,
        doubt_threshold=doubt_threshold,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        model_tier=tier,
    )
    dom_level = _dominance_level(max_raw, extreme_threshold)
    agreement = market_agreement_score(max_raw)
    model_rel = _compute_model_reliability(
        model_tier=tier,
        data_quality_pct=data_quality_pct,
        agreement=agreement,
    )
    market_rel = _compute_market_reliability(market_ctx)

    diagnosis = diagnose_discrepancy(
        analysis,
        market_ctx,
        max_divergence=max_raw,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        layer=layer,
        has_injury_news=has_injury_news,
        has_suspensions=has_suspensions,
    )

    is_dominant = layer == "extreme" or (
        dom_level == "high"
        and diagnosis is not None
        and diagnosis.primary_type == "market_dominant"
    )

    if is_dominant:
        classification = "market_dominant"
    elif layer == "doubt" or (
        diagnosis is not None and diagnosis.primary_type.startswith("alignment_")
    ):
        classification = "model_degraded"
    else:
        classification = "aligned"

    market_probs = normalized_market_probs(market_ctx, analysis.team1, analysis.team2)
    # Sin blend: el mercado es filtro, no reemplazo del modelo (antes del decision tree).
    adjustment: MarketAdjustment | None = None
    if market_probs:
        adjustment = MarketAdjustment(
            home=m.home_win,
            draw=m.draw,
            away=m.away_win,
            model_weight=1.0,
            market_weight=0.0,
            blend_applied=False,
            layer=layer,
            layer_reason=layer_reason,
            blend_reason=None,
            raw_home=m.home_win,
            raw_draw=m.draw,
            raw_away=m.away_win,
            market_home=round(market_probs.get(analysis.team1, 0), 4),
            market_draw=round(market_probs.get("Empate", 0), 4),
            market_away=round(market_probs.get(analysis.team2, 0), 4),
            max_raw_divergence=max_raw,
            model_confidence_tier=tier,
        )

    snapshots = [
        OutcomeDominanceSnapshot(
            selection=o.selection,
            model_prob=o.model_prob,
            market_implied=o.market_implied,
            divergence=o.divergence,
        )
        for o in market_ctx.outcomes
    ]

    return MarketDominanceResult(
        max_raw_divergence=max_raw,
        max_aux_divergence=None,
        layer=layer,
        layer_reason=layer_reason,
        is_market_dominant=is_dominant,
        dominance_level=dom_level,
        model_reliability=model_rel,
        market_reliability=market_rel,
        classification=classification,
        diagnosis=diagnosis,
        adjustment=adjustment,
        adjusted_market=None,
        outcome_snapshots=snapshots,
        uncertainty=uncertainty,
    )
