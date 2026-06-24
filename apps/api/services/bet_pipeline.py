"""
Pipeline de apuestas en 3 capas estrictas.

    MODEL   → fuente de verdad (Poisson + ELO); nunca se muta
    MARKET  → contexto y filtros; informa, no corrige
    DECISION → único consumidor de señales crudas (modelo + contexto)

Invariante: «El mercado puede informar, pero no puede corregir antes de decidir.»
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.api.services.bet_decision_tree import BetDecisionResult, run_bet_decision_tree
from apps.api.services.injury_news import InjuryReport
from apps.api.services.market_dominance import MarketDominanceResult, detect_market_dominance
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2, compute_market_context
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import Settings, get_settings


class PipelineInvariantError(RuntimeError):
    """Violación del orden de capas (p. ej. blend antes del árbol)."""


@dataclass(frozen=True)
class ModelLayer:
    """Capa 1 — verdad del modelo; solo lectura."""

    analysis: MatchAnalysis

    @property
    def markets(self) -> ModelMarkets:
        assert self.analysis.model is not None
        return self.analysis.model


@dataclass(frozen=True)
class MarketLayer:
    """Capa 2 — contexto de mercado; sin alterar probabilidades del modelo."""

    context: MarketContext1X2
    dominance: MarketDominanceResult


@dataclass(frozen=True)
class BetPipelineResult:
    """Salida del pipeline: las tres capas + decisión final."""

    model: ModelLayer
    market: MarketLayer
    decision: BetDecisionResult


def validate_market_layer_invariants(
    model: ModelMarkets,
    dominance: MarketDominanceResult,
) -> None:
    """Garantiza que MARKET no haya corregido al modelo antes de DECISION."""
    if dominance.adjusted_market is not None:
        raise PipelineInvariantError(
            "adjusted_market no debe existir: el mercado no corrige probabilidades"
        )
    adj = dominance.adjustment
    if adj is None:
        return
    if adj.blend_applied:
        raise PipelineInvariantError(
            "blend_applied debe ser False: sin mezcla modelo-mercado pre-decisión"
        )
    if (
        adj.home != model.home_win
        or adj.draw != model.draw
        or adj.away != model.away_win
    ):
        raise PipelineInvariantError(
            "adjustment debe reflejar probabilidades del modelo sin modificar"
        )


def run_bet_pipeline(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity] | None = None,
    *,
    market_ctx: MarketContext1X2 | None = None,
    injury_report: InjuryReport | None = None,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    historical_accuracy: float | None = None,
    settings: Settings | None = None,
) -> BetPipelineResult:
    settings = settings or get_settings()
    m = analysis.model
    if not m:
        raise ValueError("analysis sin modelo")

    # ── Capa 1: MODEL (truth source) ─────────────────────────────────────
    model_layer = ModelLayer(analysis=analysis)

    # ── Capa 2: MARKET (context only) ────────────────────────────────────
    if market_ctx is None:
        market_ctx = compute_market_context(m, analysis.team1, analysis.team2, None)

    dominance = detect_market_dominance(
        analysis,
        market_ctx,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        extreme_threshold=settings.ev_max_model_market_divergence,
        has_injury_news=bool(injury_report and injury_report.has_injuries),
        has_suspensions=bool(injury_report and injury_report.has_suspensions),
    )
    validate_market_layer_invariants(m, dominance)
    market_layer = MarketLayer(context=market_ctx, dominance=dominance)

    # ── Capa 3: DECISION (raw signals only) ──────────────────────────────
    injury_penalty = 0.0
    if injury_report and (injury_report.has_injuries or injury_report.has_suspensions):
        injury_penalty = 8.0

    decision = run_bet_decision_tree(
        analysis,
        market_ctx,
        dominance,
        ev_opps or [],
        injury_penalty=injury_penalty,
        historical_accuracy=historical_accuracy,
        data_quality_pct=data_quality_pct,
        settings=settings,
    )

    return BetPipelineResult(
        model=model_layer,
        market=market_layer,
        decision=decision,
    )
