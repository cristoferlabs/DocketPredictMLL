"""
SHARP Engine — OUTPUT A: single value bets (edge hunting).

Reglas estrictas:
  EV_final (fair) >= 3%  AND  composite >= 68  → BET SINGLE
  WATCH → micro-stake si trust=model y EV pesimista > 0
  else → NO BET

Consume MODEL + MARKET vía bet_pipeline; no mezcla con PARLAY.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.api.services.bet_pipeline import BetPipelineResult, run_bet_pipeline
from apps.api.services.injury_news import InjuryReport
from apps.api.services.confidence_score import compute_mds, sharp_composite_passes
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2
from apps.api.services.risk_stake import allocate_sharp_stake
from apps.api.services.trust_arbitration import TrustArbitration
from apps.api.services.worldcup_engine import MatchAnalysis
from apps.shared.config import Settings, get_settings


@dataclass
class SharpBetResult:
    pipeline: BetPipelineResult
    mds: int
    ev_final: float
    confidence_norm: float
    sharp_allowed: bool
    sharp_reason: str
    decision: BetDecisionResult
    stake_pct: float


def compute_mds(
    dominance: MarketDominanceResult,
    trust: TrustArbitration | None = None,
) -> int:
    """Re-export — definición en confidence_score."""
    from apps.api.services.confidence_score import compute_mds as _compute_mds

    return _compute_mds(dominance, trust)


def apply_sharp_gate(
    pipeline: BetPipelineResult,
    *,
    settings: Settings | None = None,
) -> SharpBetResult:
    """Aplica filtro SHARP final sobre la decisión del árbol suave."""
    settings = settings or get_settings()
    dom = pipeline.market.dominance
    dec = pipeline.decision
    mds = compute_mds(dom, dec.trust)
    ev_final = dec.ev_band.base if dec.ev_band else dec.ev_market
    conf = dec.confidence_score / 100.0

    min_ev = settings.ev_min_edge_fair
    min_composite = settings.sharp_min_composite

    if dec.soft_action == "WATCH":
        gated = _apply_watch_gate(dec, mds=mds, ev_final=ev_final, settings=settings)
        return SharpBetResult(
            pipeline=pipeline,
            mds=mds,
            ev_final=ev_final,
            confidence_norm=conf,
            sharp_allowed=False,
            sharp_reason="WATCH — no single SHARP (stake exploratorio si aplica)",
            decision=gated,
            stake_pct=gated.stake_pct,
        )

    if dec.no_bet or dec.soft_action == "NO_BET":
        reason = dec.blocked_reason or "sin señal de apuesta"
        gated = _gate_no_bet(dec, reason, preserve_watch=False)
        return SharpBetResult(
            pipeline=pipeline,
            mds=mds,
            ev_final=ev_final,
            confidence_norm=conf,
            sharp_allowed=False,
            sharp_reason=reason,
            decision=gated,
            stake_pct=0.0,
        )

    if ev_final < min_ev:
        gated = _gate_no_bet(dec, f"EV fair {ev_final:.1%} < {min_ev:.0%}")
        return SharpBetResult(
            pipeline=pipeline,
            mds=mds,
            ev_final=ev_final,
            confidence_norm=conf,
            sharp_allowed=False,
            sharp_reason=gated.blocked_reason or "",
            decision=gated,
            stake_pct=0.0,
        )

    if not sharp_composite_passes(dec.confidence_score, settings=settings):
        if dec.soft_action == "WATCH":
            gated = _apply_watch_gate(dec, mds=mds, ev_final=ev_final, settings=settings)
            return SharpBetResult(
                pipeline=pipeline,
                mds=mds,
                ev_final=ev_final,
                confidence_norm=conf,
                sharp_allowed=False,
                sharp_reason="WATCH — composite bajo umbral SHARP",
                decision=gated,
                stake_pct=gated.stake_pct,
            )
        gated = _gate_no_bet(
            dec,
            f"composite {dec.confidence_score} < {min_composite}",
        )
        return SharpBetResult(
            pipeline=pipeline,
            mds=mds,
            ev_final=ev_final,
            confidence_norm=conf,
            sharp_allowed=False,
            sharp_reason=gated.blocked_reason or "",
            decision=gated,
            stake_pct=0.0,
        )

    stake = allocate_sharp_stake(dec, ev_final=ev_final, mds=mds, confidence_norm=conf)
    path = list(dec.tree_path) + [f"sharp_gate:PASS mds={mds} ev={ev_final:.2%}"]
    allowed_dec = replace(
        dec,
        tree_path=path,
        stake_pct=stake,
        no_bet=False,
        classification="Apuesta SHARP recomendada",
    )
    return SharpBetResult(
        pipeline=pipeline,
        mds=mds,
        ev_final=ev_final,
        confidence_norm=conf,
        sharp_allowed=True,
        sharp_reason="SHARP single aprobado",
        decision=allowed_dec,
        stake_pct=stake,
    )


def _watch_executable(
    dec: BetDecisionResult,
    *,
    trust: TrustArbitration | None,
    settings: Settings,
) -> bool:
    """Micro-stake ejecutable: trust modelo + EV pesimista fair > 0."""
    if dec.soft_action != "WATCH" or dec.stake_pct <= 0:
        return False
    if not dec.ev_band or dec.ev_band.pessimistic <= 0:
        return False
    if trust and trust.trust_side == "model":
        return True
    return dec.ev_band.base >= settings.watch_micro_ev_threshold


def _apply_watch_gate(
    dec: BetDecisionResult,
    *,
    mds: int,
    ev_final: float,
    settings: Settings,
) -> BetDecisionResult:
    """WATCH: SHARP no aprueba single pleno; micro-stake si criterios cumplen."""
    opt_ev = dec.ev_band.optimistic if dec.ev_band else ev_final
    stake = dec.stake_pct
    executable = _watch_executable(dec, trust=dec.trust, settings=settings)
    extra = f" stake_exploratorio={stake:g}%" if stake > 0 else " paper_trade"
    tag = "WATCH_EXEC" if executable else "WATCH"
    path = list(dec.tree_path) + [
        f"sharp_gate:{tag} mds={mds} ev_opt={opt_ev:.2%}{extra}",
    ]
    classification = dec.classification
    if executable and stake > 0:
        classification = f"Micro-stake WATCH {stake:g}% — edge fair detectado"
    return replace(
        dec,
        no_bet=not executable,
        soft_action="WATCH",
        action="WATCH",
        tree_path=path,
        stake_pct=stake,
        light="amarillo",
        light_emoji="👁️",
        classification=classification,
        blocked_reason=dec.blocked_reason or "WATCH — edge detectado, sin single SHARP",
    )


def _gate_no_bet(
    dec: BetDecisionResult,
    reason: str,
    *,
    preserve_watch: bool = False,
) -> BetDecisionResult:
    if preserve_watch and dec.soft_action == "WATCH":
        return _apply_watch_gate(
            dec,
            mds=0,
            ev_final=dec.ev_band.base if dec.ev_band else dec.ev_market,
            settings=get_settings(),
        )
    path = list(dec.tree_path) + [f"sharp_gate:NO_BET ({reason})"]
    return replace(
        dec,
        no_bet=True,
        soft_action="NO_BET",
        action="NO_BET",
        blocked_reason=reason,
        tree_path=path,
        stake_pct=0.0,
        light="rojo",
        light_emoji="🔴",
        classification="Sin valor SHARP — no apostar",
    )


def run_sharp_engine(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity] | None = None,
    *,
    market_ctx: MarketContext1X2 | None = None,
    injury_report: InjuryReport | None = None,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    historical_accuracy: float | None = None,
    settings: Settings | None = None,
) -> SharpBetResult:
    """MODEL → MARKET → DECISION → SHARP gate → RISK/STAKE."""
    pipeline = run_bet_pipeline(
        analysis,
        ev_opps,
        market_ctx=market_ctx,
        injury_report=injury_report,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        historical_accuracy=historical_accuracy,
        settings=settings,
    )
    return apply_sharp_gate(pipeline, settings=settings)
