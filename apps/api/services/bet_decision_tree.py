"""Bet Decision Tree — motor de decisión suave con MUS y bandas EV."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from apps.api.services.confidence_score import compute_mds, compute_unified_confidence
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.market_uncertainty import (
    EvBand,
    SoftBetAction,
    compute_ev_band,
    compute_ev_band_from_pick,
    resolve_soft_decision,
)
from apps.api.services.trust_arbitration import TrustArbitration, arbitrate_pick_trust
from apps.api.services.odds_context import (
    EvOpportunity,
    MarketContext1X2,
    OutcomeEdge,
    best_market_ev,
    check_market_outcome_allowed,
)
from apps.api.services.trading_types import TradingPick
from apps.api.services.worldcup_engine import MatchAnalysis
from apps.shared.config import Settings, get_settings

BetAction = Literal[
    "NO_BET",
    "WATCH",
    "WEAK_BET",
    "STRONG_BET",
    "BET_CAUTIOUS",
    "BET_RECOMMENDED",
]


@dataclass
class BetDecisionResult:
    action: BetAction
    soft_action: SoftBetAction
    no_bet: bool
    blocked_reason: str | None
    tree_path: list[str] = field(default_factory=list)
    pick: TradingPick | None = None
    extra_picks: list[TradingPick] = field(default_factory=list)
    ev_market: float = 0.0
    ev_band: EvBand | None = None
    mus: float = 0.0
    trust: TrustArbitration | None = None
    light: str = "rojo"
    light_emoji: str = "🔴"
    classification: str = "Sin valor, no apostar"
    pick_rating: int = 1
    pick_rating_emoji: str = "🔴"
    confidence_score: int = 0
    stake_pct: float = 0.0
    risk: str = "Alto"
    risk_emoji: str = "⚠️"
    min_odds: float | None = None


def _pick_from_model(analysis: MatchAnalysis) -> TradingPick:
    m = analysis.model
    assert m is not None
    candidates = [
        ("1X2", analysis.team1, m.home_win),
        ("1X2", "Empate", m.draw),
        ("1X2", analysis.team2, m.away_win),
    ]
    market, selection, prob = max(candidates, key=lambda x: x[2])
    fair = round(1 / prob, 2) if prob > 0 else 0.0
    return TradingPick(
        market=market,
        selection=selection,
        model_prob=prob,
        fair_odds=fair,
        from_ev=False,
    )


def _pick_from_ev(opp: EvOpportunity) -> TradingPick:
    kelly = (opp.metadata or {}).get("kelly_stake", 0.0)
    return TradingPick(
        market=opp.market,
        selection=opp.selection,
        model_prob=opp.model_prob,
        ev_fair=opp.expected_value,
        edge_fair=opp.edge_fair,
        fair_odds=opp.fair_odds,
        raw_odds=opp.raw_odds,
        kelly_stake=kelly,
        from_ev=True,
    )


def _pick_from_market_outcome(outcome: OutcomeEdge) -> TradingPick:
    ev = outcome.ev_fair_pct / 100.0
    edge = outcome.edge_fair_pct / 100.0
    return TradingPick(
        market="1X2",
        selection=outcome.selection,
        model_prob=outcome.model_prob,
        ev_fair=ev,
        edge_fair=edge,
        fair_odds=outcome.fair_odds or outcome.model_fair_odds,
        raw_odds=outcome.market_odds or 0.0,
        from_ev=ev > 0,
    )


def _outcome_for_pick(
    pick: TradingPick,
    market_ctx: MarketContext1X2 | None,
    team1: str,
    team2: str,
) -> OutcomeEdge | None:
    if not market_ctx:
        return None
    for o in market_ctx.outcomes:
        if o.selection == pick.selection:
            return o
    return None


def _best_fair_edge_outcome(
    market_ctx: MarketContext1X2,
    settings: Settings | None = None,
) -> OutcomeEdge | None:
    settings = settings or get_settings()
    positive: list[OutcomeEdge] = []
    for o in market_ctx.outcomes:
        if o.ev_fair_pct <= 0 or not o.market_odds:
            continue
        ok, flags = check_market_outcome_allowed(
            o,
            max_divergence=settings.ev_max_model_market_divergence,
            max_ev=settings.ev_max_edge_fair,
        )
        if not ok:
            div = o.divergence or 1.0
            ev = o.ev_fair_pct / 100.0
            if div < 0.16 and ev >= settings.ev_min_edge_fair and ev <= 0.45:
                ok = True
        if ok:
            positive.append(o)
    if not positive:
        return None
    return max(positive, key=lambda x: x.ev_fair_pct)


def _risk_label(prob: float, draw_prob: float, is_underdog: bool) -> tuple[str, str]:
    if draw_prob >= 0.28:
        return "Alto", "⚠️"
    if prob < 0.45 or is_underdog:
        return "Alto", "⚠️"
    if prob < 0.52:
        return "Medio", "🔶"
    return "Bajo", "✅"


def _pick_rating(
    soft_action: SoftBetAction,
    ev_band: EvBand,
    confidence_score: int,
) -> tuple[int, str]:
    if soft_action in ("NO_BET", "WATCH"):
        return 1 if soft_action == "NO_BET" else 2, "🔴" if soft_action == "NO_BET" else "🟡"
    if confidence_score < 35:
        return 1, "🔴"
    if soft_action == "STRONG_BET" and ev_band.base >= 0.06:
        return 5, "🟢"
    if soft_action == "STRONG_BET":
        return 4, "🟢"
    if soft_action == "WEAK_BET" and ev_band.base >= 0.03:
        return 3, "🟡"
    return 2, "🔴"


def _presentation(
    soft_action: SoftBetAction,
    ev_band: EvBand,
    confidence_score: int,
) -> tuple[str, str, str, BetAction]:
    if soft_action == "STRONG_BET":
        return "verde", "🟢", "Apuesta recomendada", "STRONG_BET"
    if soft_action == "WEAK_BET":
        return "amarillo", "🟡", "Apuesta cauta (MUS/EV)", "WEAK_BET"
    if soft_action == "WATCH":
        return "amarillo", "👁️", "Vigilar — edge detectado, stake reducido", "WATCH"
    return "rojo", "🔴", "Sin valor, no apostar", "NO_BET"


def _stake_for_action(
    soft_action: SoftBetAction,
    primary: TradingPick,
    ev_band: EvBand,
    light: str,
    *,
    settings: Settings | None = None,
) -> float:
    settings = settings or get_settings()
    if soft_action == "WATCH":
        opt = ev_band.optimistic
        if opt >= settings.watch_exploratory_ev_threshold:
            return settings.watch_exploratory_stake_pct
        if opt >= settings.watch_micro_ev_threshold or ev_band.base >= 0.08:
            return settings.watch_micro_stake_pct
        return 0.0
    if soft_action == "NO_BET":
        return 0.0
    base = round(primary.kelly_stake * 100, 1) if primary.kelly_stake else 0.0
    if soft_action == "WEAK_BET":
        base = min(base, 0.5) if base else 0.25
        if ev_band.pessimistic < 0.04:
            base = min(base, 0.35)
    elif soft_action == "STRONG_BET":
        if light == "verde" and base < 0.5:
            base = max(base, 0.5)
    return base


def _no_bet_result(
    path: list[str],
    reason: str,
    analysis: MatchAnalysis,
    *,
    soft_action: SoftBetAction = "NO_BET",
    ev_band: EvBand | None = None,
    mus: float = 0.0,
    pick: TradingPick | None = None,
    confidence_score: int = 0,
    trust: TrustArbitration | None = None,
    settings: Settings | None = None,
) -> BetDecisionResult:
    settings = settings or get_settings()
    resolved_pick = pick or _pick_from_model(analysis)
    band = ev_band or EvBand("", 0, 0, 0)
    _, light_emoji, classification, action = _presentation(soft_action, band, confidence_score)
    stake_pct = (
        _stake_for_action(soft_action, resolved_pick, band, "amarillo", settings=settings)
        if soft_action == "WATCH"
        else 0.0
    )
    if soft_action == "WATCH" and stake_pct > 0:
        classification = f"Vigilar — stake exploratorio {stake_pct:g}%"
    return BetDecisionResult(
        action=action,
        soft_action=soft_action,
        no_bet=True,
        blocked_reason=reason,
        tree_path=path + [f"result:{soft_action} ({reason})"],
        pick=resolved_pick,
        light="rojo" if soft_action == "NO_BET" else "amarillo",
        light_emoji=light_emoji,
        classification=classification,
        pick_rating=2 if soft_action == "WATCH" else 1,
        pick_rating_emoji="🟡" if soft_action == "WATCH" else "🔴",
        confidence_score=confidence_score,
        stake_pct=stake_pct,
        risk="Alto",
        risk_emoji="⚠️",
        ev_band=ev_band,
        mus=mus,
        trust=trust,
        ev_market=band.base,
        min_odds=round(resolved_pick.fair_odds, 2) if resolved_pick.fair_odds > 1 else None,
    )


def run_bet_decision_tree(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    dominance: MarketDominanceResult,
    ev_opps: list[EvOpportunity],
    *,
    injury_penalty: float = 0.0,
    historical_accuracy: float | None = None,
    data_quality_pct: float = 100.0,
    settings: Settings | None = None,
) -> BetDecisionResult:
    """
    Árbol suave: MUS + bandas EV deciden cuánto confiar en el mercado.

    Sin bloqueo rígido Δ>20%. El modelo permanece fuente de verdad para EV.
    """
    settings = settings or get_settings()
    path: list[str] = ["start"]

    m = analysis.model
    assert m is not None

    uncertainty = dominance.uncertainty
    mus = uncertainty.mus if uncertainty else 1.0
    mc = uncertainty.confidence_market if uncertainty else 0.0
    path.append(f"mus:{mus:.2f}")

    has_market = bool(market_ctx and market_ctx.has_market)
    path.append(f"has_market:{has_market}")

    if not has_market:
        return _no_bet_result(path, "sin cuotas mercado", analysis, mus=mus)

    structural = dominance.layer == "extreme"
    path.append(f"structural_mismatch:{'yes' if structural else 'no'}")

    ev_opps = ev_opps or []
    primary: TradingPick
    extras: list[TradingPick] = []
    source_outcome: OutcomeEdge | None = None

    if ev_opps:
        primary = _pick_from_ev(ev_opps[0])
        extras = [_pick_from_ev(o) for o in ev_opps[1:4]]
        source_outcome = _outcome_for_pick(primary, market_ctx, analysis.team1, analysis.team2)
        path.append("pick:ev_opportunity")
    else:
        best_o = _best_fair_edge_outcome(market_ctx, settings) if market_ctx else None
        path.append(f"check_edge:{'pass' if best_o else 'fail'}")
        if not best_o:
            if structural and market_ctx:
                primary = _pick_from_model(analysis)
                source_outcome = _outcome_for_pick(
                    primary, market_ctx, analysis.team1, analysis.team2
                )
                path.append("pick:model_favorite_watch")
            else:
                return _no_bet_result(path, "sin valor", analysis, mus=mus)
        else:
            primary = _pick_from_market_outcome(best_o)
            source_outcome = best_o
            path.append("pick:best_fair_edge")

    if source_outcome:
        ev_band = compute_ev_band(source_outcome, mus=mus, market_confidence=mc)
    else:
        div = dominance.max_raw_divergence if structural else 0.0
        ev_band = compute_ev_band_from_pick(
            selection=primary.selection,
            model_prob=primary.model_prob,
            market_odds=primary.raw_odds,
            divergence=div,
            mus=mus,
            market_confidence=mc,
            ev_base=primary.ev_fair,
        )

    pick_div = source_outcome.divergence if source_outcome else 0.0
    path.append(
        f"ev_band:opt={ev_band.optimistic:.2f}|base={ev_band.base:.2f}|pes={ev_band.pessimistic:.2f}"
    )

    pick_mkt_impl = source_outcome.market_implied if source_outcome else None
    fav_impl = max(
        (o.market_implied or 0.0 for o in market_ctx.outcomes),
        default=0.0,
    ) if market_ctx else 0.0
    trust = arbitrate_pick_trust(
        pick_model_prob=primary.model_prob,
        pick_market_implied=pick_mkt_impl,
        pick_divergence=pick_div or 0.0,
        model=m,
        dominance=dominance,
        data_quality_pct=data_quality_pct,
        market_favorite_implied=fav_impl,
    )
    path.append(
        f"trust:{trust.trust_side} "
        f"m={trust.model_confidence:.2f} mkt={trust.market_confidence:.2f}"
    )

    mds = compute_mds(dominance, trust)
    conf_score = compute_unified_confidence(
        mds=mds,
        model_reliability=dominance.model_reliability,
        trust=trust,
        cold_start=historical_accuracy is None,
    )
    if historical_accuracy is None:
        path.append("cold_start:cap58")
    path.append(f"mds:{mds}|confidence:{conf_score}")

    diag_primary = dominance.diagnosis.primary_type if dominance.diagnosis else None
    soft_action, soft_reason = resolve_soft_decision(
        ev_band=ev_band,
        mus=mus,
        max_divergence=dominance.max_raw_divergence,
        pick_divergence=pick_div or 0.0,
        confidence_score=conf_score,
        diagnosis_primary=diag_primary,
        trust=trust,
    )
    if (
        soft_action == "NO_BET"
        and structural
        and "pick:model_favorite_watch" in path
    ):
        max_raw = max((o.ev_raw_pct for o in market_ctx.outcomes), default=0.0) if market_ctx else 0.0
        if max_raw >= 5.0 or dominance.max_raw_divergence >= 0.20:
            soft_action, soft_reason = (
                "WATCH",
                "mismatch estructural — fair sin valor; edge raw informativo",
            )
    path.append(f"soft_gate:{soft_action} ({soft_reason})")

    if soft_action in ("NO_BET", "WATCH"):
        return _no_bet_result(
            path,
            soft_reason,
            analysis,
            soft_action=soft_action,
            ev_band=ev_band,
            mus=mus,
            pick=primary,
            confidence_score=conf_score,
            trust=trust,
            settings=settings,
        )

    light, light_emoji, classification, action = _presentation(
        soft_action, ev_band, conf_score
    )
    pick_rating, pick_rating_emoji = _pick_rating(soft_action, ev_band, conf_score)

    is_dog = primary.selection not in (analysis.team1, "Empate") and primary.model_prob < 0.40
    if primary.selection == analysis.team2 and primary.model_prob < 0.45:
        is_dog = True
    risk, risk_emoji = _risk_label(primary.model_prob, m.draw, is_dog)

    stake_pct = _stake_for_action(soft_action, primary, ev_band, light, settings=settings)
    ev_market = max(ev_band.base, best_market_ev(market_ctx))

    path.append(f"result:{action}")

    return BetDecisionResult(
        action=action,
        soft_action=soft_action,
        no_bet=False,
        blocked_reason=None,
        tree_path=path,
        pick=primary,
        extra_picks=extras,
        ev_market=ev_market,
        ev_band=ev_band,
        mus=mus,
        trust=trust,
        light=light,
        light_emoji=light_emoji,
        classification=classification,
        pick_rating=pick_rating,
        pick_rating_emoji=pick_rating_emoji,
        confidence_score=conf_score,
        stake_pct=stake_pct,
        risk=risk,
        risk_emoji=risk_emoji,
        min_odds=round(primary.fair_odds, 2) if primary.fair_odds > 1 else None,
    )
