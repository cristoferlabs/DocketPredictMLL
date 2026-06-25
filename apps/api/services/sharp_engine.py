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
from typing import Literal

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.api.services.bet_pipeline import BetPipelineResult, run_bet_pipeline
from apps.api.services.injury_news import InjuryReport
from apps.api.services.confidence_score import compute_mds, sharp_composite_passes
from apps.api.services.sharp_portfolio import portfolio_tier_for_confidence, sharp_rank_score
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2
from apps.api.services.risk_stake import allocate_sharp_stake
from apps.api.services.trust_arbitration import TrustArbitration
from apps.api.services.worldcup_engine import MatchAnalysis
from apps.api.services.ev_policy import ev_for_decision
from apps.shared.config import Settings, get_settings
from apps.worker.ml.model_learning import LearningState, load_learning_state

SharpPhase = Literal["cold", "warm", "mature"]

_SHARP_REGIME_MAX_DIV_PP: dict[str, float] = {
    "aligned": 12.0,
    "moderate": 15.0,
    "high": 18.0,
    "extreme": 20.0,
}


def _sharp_max_div_pp(cal_meta: dict, settings: Settings) -> float:
    regime = cal_meta.get("alpha_regime")
    if regime in _SHARP_REGIME_MAX_DIV_PP:
        return _SHARP_REGIME_MAX_DIV_PP[regime]
    return settings.sharp_max_divergence_pp


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
    portfolio_tier: str = "X"
    rank_score: float = 0.0


def compute_mds(
    dominance: MarketDominanceResult,
    trust: TrustArbitration | None = None,
) -> int:
    """Re-export — definición en confidence_score."""
    from apps.api.services.confidence_score import compute_mds as _compute_mds

    return _compute_mds(dominance, trust)


def resolve_sharp_phase(n_picks: int, settings: Settings) -> SharpPhase:
    if n_picks < settings.sharp_phase_cold_n:
        return "cold"
    if n_picks < settings.sharp_phase_mature_n:
        return "warm"
    return "mature"


def _calibration_meta(pipeline: BetPipelineResult) -> dict:
    blend = pipeline.model.markets.blend_meta or {}
    return dict(blend.get("calibration") or {})


def _apply_sharp_phase_gate(
    dec: BetDecisionResult,
    *,
    mds: int,
    phase: SharpPhase,
    state: LearningState,
    settings: Settings,
    cal_meta: dict,
) -> BetDecisionResult:
    """Fases cold/warm/mature — antes del pass SHARP final."""
    div_cal = float(cal_meta.get("divergence_cal_pp") or 0.0)
    alpha = float(cal_meta.get("alpha") or 0.0)
    regime = str(cal_meta.get("alpha_regime") or "moderate")
    shrink = bool(cal_meta.get("shrink_applied"))
    cal_active = alpha > 0 or shrink
    max_div_regime = _sharp_max_div_pp(cal_meta, settings)
    path = list(dec.tree_path)
    path.append(
        f"sharp_phase:{phase}|regime={regime}|α={alpha:.2f}|Δcal={div_cal:.0f}pp"
    )

    if dec.soft_action in ("NO_BET", "WATCH"):
        return replace(dec, tree_path=path)

    soft = dec.soft_action
    reason = dec.blocked_reason

    if phase == "cold":
        max_div = max_div_regime
        if mds < settings.sharp_cold_weak_mds_min:
            soft = "NO_BET"
            reason = f"cold: MDS {mds} < {settings.sharp_cold_weak_mds_min}"
        elif mds < settings.sharp_cold_strong_mds:
            if soft == "STRONG_BET":
                soft = "WEAK_BET"
                reason = f"cold: MDS {mds} < {settings.sharp_cold_strong_mds}"
        elif soft == "STRONG_BET":
            if div_cal >= max_div and div_cal > 15.0:
                soft = "WEAK_BET"
                reason = f"cold: Δ_cal {div_cal:.0f}pp ≥ {max_div:.0f}pp"
            elif settings.sharp_require_shrink_active and not cal_active:
                soft = "WEAK_BET"
                reason = "cold: calibración inactiva (α=0 sin shrink)"
        if soft == "STRONG_BET" and div_cal > 20.0:
            soft = "WEAK_BET"
            reason = f"cold: Δ_cal {div_cal:.0f}pp > 20pp downgrade"

    elif phase == "warm":
        if soft == "STRONG_BET" and (mds < 65 or div_cal >= max_div_regime):
            soft = "WEAK_BET"
            reason = f"warm: MDS {mds} o Δ_cal {div_cal:.0f}pp (regime≤{max_div_regime:.0f})"
        clv = state.rolling_clv
        if soft == "STRONG_BET" and clv is not None and clv < 0:
            soft = "WEAK_BET"
            reason = f"warm: rolling CLV {clv:.3f} < 0"
            path.append("clv_obs:negative")

    else:  # mature
        max_div = min(max_div_regime, settings.sharp_mature_max_divergence_pp)
        clv = state.rolling_clv
        if soft == "STRONG_BET":
            if clv is not None and clv <= 0:
                pes = dec.ev_band.pessimistic if dec.ev_band else 0.0
                soft = "WEAK_BET" if pes > 0 else "NO_BET"
                reason = f"mature: CLV gate fail ({clv:.3f})"
            elif mds < 65 or div_cal >= max_div:
                soft = "WEAK_BET"
                reason = f"mature: MDS {mds} o Δ_cal {div_cal:.0f}pp"

    if soft != dec.soft_action:
        path.append(f"phase_gate:{dec.soft_action}→{soft}")
    action = soft if soft in ("STRONG_BET", "WEAK_BET", "WATCH", "NO_BET") else dec.action
    no_bet = soft in ("NO_BET", "WATCH")
    return replace(
        dec,
        soft_action=soft,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        no_bet=no_bet,
        blocked_reason=reason,
        tree_path=path,
    )


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
    cal_meta = _calibration_meta(pipeline)
    regime = str(cal_meta.get("alpha_regime") or "moderate")
    ev_raw = dec.ev_band.base if dec.ev_band else dec.ev_market
    ev_final = ev_for_decision(ev_fair=ev_raw, alpha_regime=regime)
    conf = dec.confidence_score / 100.0
    tier = portfolio_tier_for_confidence(dec.confidence_score, settings=settings)
    rank_score = sharp_rank_score(
        ev_fair=ev_final,
        confidence=float(dec.confidence_score),
        mds=float(mds),
    )

    min_ev = settings.ev_min_edge_fair
    min_composite = settings.sharp_min_composite
    portfolio_mode = getattr(settings, "sharp_mode", "portfolio") == "portfolio"

    learning = load_learning_state()
    phase = resolve_sharp_phase(learning.rolling_clv_n, settings)
    dec = _apply_sharp_phase_gate(
        dec,
        mds=mds,
        phase=phase,
        state=learning,
        settings=settings,
        cal_meta=cal_meta,
    )
    if cal_meta:
        analysis = pipeline.model.analysis
        if analysis.model is not None:
            bm = dict(analysis.model.blend_meta or {})
            cal = dict(bm.get("calibration") or {})
            cal["sharp_phase"] = phase
            bm["calibration"] = cal
            analysis.model.blend_meta = bm

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
                portfolio_tier=tier,
                rank_score=rank_score,
            )
        reason = (
            f"composite {dec.confidence_score} < {settings.sharp_portfolio_min_composite}"
            if portfolio_mode
            else f"composite {dec.confidence_score} < {min_composite}"
        )
        gated = _gate_no_bet(dec, reason)
        return SharpBetResult(
            pipeline=pipeline,
            mds=mds,
            ev_final=ev_final,
            confidence_norm=conf,
            sharp_allowed=False,
            sharp_reason=gated.blocked_reason or "",
            decision=gated,
            stake_pct=0.0,
            portfolio_tier=tier,
            rank_score=rank_score,
        )

    # Portfolio: tier A/B + EV → single permitido; tier C → solo ranking/parlay pool
    single_ok = tier == "A" or (
        portfolio_mode and tier == "B" and ev_final >= min_ev
    ) or (
        not portfolio_mode and dec.confidence_score >= min_composite
    )
    if not single_ok and dec.soft_action not in ("WATCH",):
        gated = _gate_no_bet(
            dec,
            f"tier {tier} — ranking pool (sin single; usable en parlay top-K)",
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
            portfolio_tier=tier,
            rank_score=rank_score,
        )

    stake = allocate_sharp_stake(dec, ev_final=ev_final, mds=mds, confidence_norm=conf)
    path = list(dec.tree_path) + [
        f"sharp_gate:PASS tier={tier} mds={mds} ev={ev_final:.2%} score={rank_score:.1f}",
    ]
    allowed_dec = replace(
        dec,
        tree_path=path,
        stake_pct=stake,
        no_bet=False,
        classification=f"Apuesta SHARP tier {tier}",
    )
    return SharpBetResult(
        pipeline=pipeline,
        mds=mds,
        ev_final=ev_final,
        confidence_norm=conf,
        sharp_allowed=True,
        sharp_reason=f"SHARP single tier {tier}",
        decision=allowed_dec,
        stake_pct=stake,
        portfolio_tier=tier,
        rank_score=rank_score,
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
