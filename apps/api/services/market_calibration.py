"""
3-layer market architecture:
  1. Modelo = verdad estadística (Poisson + ELO)
  2. Mercado = filtro de anomalías (NO reemplazo)
  3. Blend auxiliar SOLO en capa duda (datos débiles / baja confianza modelo)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from apps.api.services.odds_context import MarketContext1X2, OutcomeEdge, max_market_divergence
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.worker.ml.odds_math import expected_value_raw

DecisionLayer = Literal["normal", "doubt", "extreme"]

DISCREPANCY_LABELS: dict[str, tuple[str, str]] = {
    "market_dominant": (
        "Market dominant signal",
        "Mercado domina el pricing; modelo no alineado en este partido",
    ),
    "model_underconfidence": (
        "Model underconfidence",
        "El modelo no refleja la convicción implícita del mercado",
    ),
    "elo_drift": ("ELO drift", "Rating ELO desalineado con el precio de mercado"),
    "poisson_bias": ("Poisson bias", "λ ofensivos mal estimados vs implied del mercado"),
    "data_noise": ("Data noise", "Muestra histórica limitada — confianza del modelo reducida"),
}

LAYER_LABELS: dict[str, str] = {
    "normal": "Modelo manda — mercado valida",
    "doubt": "Duda — referencia auxiliar (no sustituye modelo)",
    "extreme": "Extremo — bloqueo total, sin mezcla",
}


@dataclass
class MarketAdjustment:
    """Blend auxiliar opcional; nunca reemplaza al modelo en decisiones de EV."""

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
        """Compat con código que usaba shrink_applied."""
        return self.blend_applied


@dataclass
class DiscrepancyDiagnosis:
    """Un diagnóstico dominante + secundario opcional; sin mezcla contradictoria."""

    primary_type: str
    label: str
    description: str
    secondary_type: str | None = None
    secondary_label: str | None = None
    secondary_description: str | None = None
    result: str = "NO BET"

    @property
    def primary_type_legacy(self) -> str:
        return self.primary_type


def model_confidence_tier(model: ModelMarkets) -> str:
    """Alta / media / baja confianza interna del modelo."""
    spread = max(model.home_win, model.draw, model.away_win) - min(
        model.home_win, model.draw, model.away_win
    )
    max_p = max(model.home_win, model.draw, model.away_win)
    if model.confidence == "high" or max_p >= 0.55 or spread >= 0.18:
        return "high"
    if model.confidence == "low" or max_p < 0.42 or spread < 0.08:
        return "low"
    return "medium"


def market_confidence_weights(tier: str) -> tuple[float, float]:
    """
    Peso modelo/mercado según confianza del MODELO (no según divergencia).
    Alta confianza → modelo domina; baja → mercado como prior auxiliar.
    """
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
    """
    Capa 1 normal: modelo manda, mercado valida.
    Capa 2 duda: blend auxiliar permitido (solo referencia).
    Capa 3 extremo: bloqueo — mercado es filtro, NO blender.
    """
    if max_divergence >= extreme_threshold:
        return (
            "extreme",
            f"Δ {max_divergence*100:.0f}% ≥ umbral {extreme_threshold*100:.0f}% — sin mezcla",
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
    """
    Mercado como filtro. Blend auxiliar SOLO en capa 'doubt'.
    EV y picks siempre usan probabilidades del modelo base.
    """
    if not market_ctx or not market_ctx.has_market:
        return None, market_ctx

    market_probs = normalized_market_probs(market_ctx, team1, team2)
    if not market_probs:
        return None, market_ctx

    max_raw = max_market_divergence(market_ctx)
    tier = model_confidence_tier(model)
    layer, layer_reason = classify_decision_layer(
        max_raw,
        extreme_threshold=extreme_threshold,
        doubt_threshold=doubt_threshold,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        model_tier=tier,
    )

    blend_applied = layer == "doubt" and max_raw < extreme_threshold
    blend_reason: str | None = None
    mw, kw = market_confidence_weights(tier)

    if blend_applied:
        blend_reason = f"capa duda ({layer_reason}); pesos por confianza modelo {tier}"
        adj_home, adj_draw, adj_away = blend_1x2_probs(
            model.home_win,
            model.draw,
            model.away_win,
            market_probs,
            team1,
            team2,
            mw,
        )
    else:
        adj_home, adj_draw, adj_away = model.home_win, model.draw, model.away_win

    adjusted_ctx: MarketContext1X2 | None = None
    if blend_applied:
        adjusted_outcomes = recalculate_outcome_edges(
            market_ctx.outcomes,
            adj_home,
            adj_draw,
            adj_away,
            team1,
            team2,
        )
        adjusted_ctx = MarketContext1X2(
            has_market=True,
            outcomes=adjusted_outcomes,
            n_books=market_ctx.n_books,
        )

    adjustment = MarketAdjustment(
        home=round(adj_home, 4),
        draw=round(adj_draw, 4),
        away=round(adj_away, 4),
        model_weight=round(mw if blend_applied else 1.0, 2),
        market_weight=round(kw if blend_applied else 0.0, 2),
        blend_applied=blend_applied,
        layer=layer,
        layer_reason=layer_reason,
        blend_reason=blend_reason,
        raw_home=model.home_win,
        raw_draw=model.draw,
        raw_away=model.away_win,
        market_home=round(market_probs.get(team1, 0), 4),
        market_draw=round(market_probs.get("Empate", 0), 4),
        market_away=round(market_probs.get(team2, 0), 4),
        max_raw_divergence=max_raw,
        model_confidence_tier=tier,
    )
    return adjustment, adjusted_ctx


def diagnose_discrepancy(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    *,
    max_divergence: float,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    layer: DecisionLayer = "normal",
) -> DiscrepancyDiagnosis | None:
    """
    Diagnóstico limpio: UN primary dominante + UN secondary opcional.
    Nunca mezcla data_noise con market_dominant en el mismo caso.
    """
    if not market_ctx or not market_ctx.has_market or max_divergence < 0.12:
        return None

    m = analysis.model
    if not m:
        return None

    tier = model_confidence_tier(m)
    weak_data = hist_played < 8 or data_quality_pct < 70

    market_dominant = max_divergence >= 0.20
    if not market_dominant:
        for o in market_ctx.outcomes:
            if (o.market_implied or 0) > 0.70 and (o.divergence or 0) > 0.15:
                market_dominant = True
                break

    elo1 = float(analysis.elo.get(analysis.team1, {}).get("rating") or 1500)
    elo2 = float(analysis.elo.get(analysis.team2, {}).get("rating") or 1500)
    model_fav = analysis.team1 if m.home_win >= m.away_win else analysis.team2
    market_fav = max(
        market_ctx.outcomes,
        key=lambda o: o.market_implied or 0,
    ).selection
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

    # PRIMARY — exactamente uno, por prioridad
    if market_dominant:
        primary = "market_dominant"
    elif has_elo_drift:
        primary = "elo_drift"
    elif has_poisson_bias:
        primary = "poisson_bias"
    elif weak_data and max_divergence < 0.20:
        primary = "data_noise"
    elif max_divergence >= 0.15:
        primary = "market_dominant"
    else:
        return None

    # SECONDARY — como máximo uno; nunca data_noise si primary es market_dominant
    secondary: str | None = None
    if primary == "market_dominant":
        top_div = max((o.divergence or 0) for o in market_ctx.outcomes)
        if top_div > 0.15 or tier in ("low", "medium"):
            secondary = "model_underconfidence"
    elif primary == "elo_drift" and tier == "low":
        secondary = "model_underconfidence"
    elif primary == "poisson_bias" and tier in ("low", "medium"):
        secondary = "model_underconfidence"
    elif primary == "data_noise" and tier == "low":
        secondary = "model_underconfidence"

    p_label, p_desc = DISCREPANCY_LABELS.get(
        primary, ("Desacople", "Modelo y mercado divergen")
    )
    s_label, s_desc = (None, None)
    if secondary:
        s_label, s_desc = DISCREPANCY_LABELS.get(
            secondary, ("", "")
        )

    result = "NO BET" if layer == "extreme" else "REVISAR"

    return DiscrepancyDiagnosis(
        primary_type=primary,
        label=p_label,
        description=p_desc,
        secondary_type=secondary,
        secondary_label=s_label,
        secondary_description=s_desc,
        result=result,
    )


def compute_recalibrated_confidence(
    *,
    data_quality_pct: float,
    market_agreement: float,
    historical_accuracy: float | None = None,
    injury_penalty: float = 0.0,
    model_tier: str = "medium",
    layer: DecisionLayer = "normal",
) -> int:
    """
    Confianza = precisión histórica + acuerdo mercado (filtro) + calidad datos.
    En capa extrema penaliza acuerdo pero NO reemplaza modelo.
    """
    hist = historical_accuracy if historical_accuracy is not None else 0.52
    agreement = max(0.0, min(1.0, market_agreement))
    dq = max(0.0, min(100.0, data_quality_pct))

    tier_bonus = {"high": 8, "medium": 0, "low": -6}.get(model_tier, 0)
    layer_penalty = {"normal": 0, "doubt": -5, "extreme": -15}.get(layer, 0)

    score = (
        0.40 * hist * 100
        + 0.25 * agreement * 100
        + 0.35 * dq
        + tier_bonus
        + layer_penalty
        - injury_penalty
    )
    return int(max(0, min(100, round(score))))


def market_agreement_score(max_divergence: float) -> float:
    """Acuerdo modelo↔mercado (filtro), siempre sobre probs del modelo base."""
    return max(0.0, min(1.0, 1.0 - max_divergence / 0.35))
