"""
Parlay Engine — OUTPUT B: portfolio / combinadas.

MODEL → PARLAY (entry flexible + pick score) → RISK/STAKE

Objetivo: maximizar probabilidad conjunta, no edge individual.
Separado del SHARP Engine (sharp_engine.py).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from apps.api.services.injury_news import InjuryReport
from apps.api.services.market_dominance import MarketDominanceResult, detect_market_dominance
from apps.api.services.odds_context import MarketContext1X2
from apps.api.services.engine_constants import ENGINE_VERSION_TAG
from apps.api.services.risk_stake import allocate_parlay_stake
from apps.api.services.worldcup_engine import MatchAnalysis
from apps.shared.config import Settings, get_settings


@dataclass
class ParlayLeg:
    team1: str
    team2: str
    fecha: str
    ronda: str
    selection: str
    model_prob: float
    market_prob: float | None
    effective_prob: float
    odds: float | None
    ev_adjusted: float
    pick_score: float = 0.0
    stability: float = 1.0
    market_agreement: float = 1.0
    volatility_factor: float = 1.0
    news_penalty: float = 1.0
    stable: bool = True
    exclude_reason: str | None = None


@dataclass
class ParlayTicket:
    legs: list[ParlayLeg]
    combined_prob: float
    combined_odds: float
    ev_parlay: float
    combo_score: float
    correlation_penalty: float
    stake_pct: float
    n_legs: int


@dataclass
class ParlayBuildResult:
    eligible_legs: list[ParlayLeg] = field(default_factory=list)
    rejected_legs: list[ParlayLeg] = field(default_factory=list)
    tickets: list[ParlayTicket] = field(default_factory=list)
    message_hint: str = ""


def _pick_components(
    model_p: float,
    market_p: float | None,
    max_divergence: float,
    has_news: bool,
) -> tuple[float, float, float, float]:
    """stability, agreement, volatility_factor, news_penalty."""
    mkt = market_p or 0.0
    div = abs(model_p - mkt) if mkt > 0 else max_divergence
    stability = max(0.35, 1.0 - div / 0.35)
    if mkt > 0:
        agreement = 1.0 - min(1.0, div / 0.25)
    else:
        agreement = 0.7
    volatility = max(0.5, 1.0 - max(0.0, max_divergence - 0.12) / 0.30)
    news_penalty = 0.75 if has_news else 1.0
    return (
        round(stability, 3),
        round(max(0.0, agreement), 3),
        round(volatility, 3),
        news_penalty,
    )


def compute_pick_score(leg: ParlayLeg) -> float:
    """
    Pick Score = P(win) × Stability × Market Agreement × Low Volatility × news.
    """
    prob = leg.effective_prob
    score = (
        prob
        * leg.stability
        * leg.market_agreement
        * leg.volatility_factor
        * leg.news_penalty
    )
    return round(score, 5)


def _parlay_effective_prob(
    model_p: float,
    mkt_p: float | None,
    *,
    max_divergence: float,
) -> float:
    """
    Prob conjunta = modelo calibrado (no max(model, market)).

    Haircut conservador si el mercado es mucho más optimista que el modelo.
    """
    mkt = mkt_p or 0.0
    if mkt > 0 and mkt > model_p + 0.06:
        div = abs(model_p - mkt)
        haircut = min(0.22, div * 0.45 + max(0.0, max_divergence - 0.12) * 0.35)
        return round(max(0.05, model_p * (1.0 - haircut)), 4)
    return round(model_p, 4)


def _entry_candidate(
    model_p: float,
    market_p: float | None,
    settings: Settings,
) -> bool:
    """ENTRY: P_model >= 55% OR P_market >= 60%."""
    mkt = market_p or 0.0
    return model_p >= settings.parlay_min_win_prob or mkt >= settings.parlay_market_min_prob


def _is_match_unstable_for_parlay(
    dominance: MarketDominanceResult,
    *,
    model_prob: float,
    market_prob: float | None,
) -> str | None:
    if dominance.layer != "extreme":
        return None
    if dominance.max_raw_divergence < 0.22:
        return None
    mkt = market_prob or 0.0
    if mkt >= 0.70 and model_prob < 0.45:
        return "mismatch estructural sin respaldo modelo"
    if dominance.diagnosis and dominance.diagnosis.primary_type == "information_asymmetry":
        if model_prob < 0.50 and mkt > 0.65:
            return "asimetría informativa — inestable"
    return None


def _critical_news(injury: InjuryReport | None, selection: str, t1: str, t2: str) -> bool:
    if not injury or not (injury.has_injuries or injury.has_suspensions):
        return False
    return selection in (t1, t2) and len(injury.articles) >= 2


def evaluate_parlay_leg(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    dominance: MarketDominanceResult | None = None,
    *,
    injury_report: InjuryReport | None = None,
    settings: Settings | None = None,
) -> ParlayLeg:
    """ENTRY LAYER — flexible; sin EV fuerte obligatorio."""
    settings = settings or get_settings()
    t1, t2 = analysis.team1, analysis.team2

    base = ParlayLeg(
        team1=t1, team2=t2, fecha=analysis.fecha, ronda=analysis.ronda,
        selection="", model_prob=0.0, market_prob=None, effective_prob=0.0,
        odds=None, ev_adjusted=0.0, stable=False,
    )

    if not analysis.model or not market_ctx or not market_ctx.has_market:
        base.exclude_reason = "sin mercado"
        return base

    if dominance is None:
        dominance = detect_market_dominance(analysis, market_ctx)

    best: tuple[str, float, float | None, float | None, float] | None = None
    for o in market_ctx.outcomes:
        if o.selection == "Empate":
            continue
        mkt = o.market_implied
        if not _entry_candidate(o.model_prob, mkt, settings):
            continue
        eff = _parlay_effective_prob(
            o.model_prob, mkt, max_divergence=dominance.max_raw_divergence
        )
        if best is None or eff > best[4]:
            best = (o.selection, o.model_prob, mkt, o.market_odds, eff)

    if best is None:
        base.exclude_reason = "no cumple entry (P modelo <55% y mercado <60%)"
        return base

    selection, model_p, mkt_p, odds, eff_p = best
    edge = 0.0
    for o in market_ctx.outcomes:
        if o.selection == selection:
            edge = max(0.0, o.ev_fair_pct / 100.0)
            break

    has_news = _critical_news(injury_report, selection, t1, t2)
    stab, agr, vol, news_p = _pick_components(
        model_p, mkt_p, dominance.max_raw_divergence, has_news
    )

    leg = ParlayLeg(
        team1=t1,
        team2=t2,
        fecha=analysis.fecha,
        ronda=analysis.ronda,
        selection=selection,
        model_prob=model_p,
        market_prob=mkt_p,
        effective_prob=eff_p,
        odds=odds,
        ev_adjusted=edge,
        stability=stab,
        market_agreement=agr,
        volatility_factor=vol,
        news_penalty=news_p,
        stable=True,
    )
    leg.pick_score = compute_pick_score(leg)

    unstable = _is_match_unstable_for_parlay(
        dominance, model_prob=model_p, market_prob=mkt_p
    )
    if unstable:
        leg.stable = False
        leg.exclude_reason = unstable
        return leg

    if _critical_news(injury_report, selection, t1, t2):
        leg.stable = False
        leg.exclude_reason = "noticias críticas en equipo pick"
        return leg

    if eff_p > 0.75 and stab < 0.5:
        mkt = mkt_p or 0.0
        if mkt >= 0.65 and mkt >= model_p:
            leg.pick_score = compute_pick_score(leg)
            return leg
        leg.stable = False
        leg.exclude_reason = "prob alta pero inestable (discrepancia)"
        return leg

    return leg


def correlation_penalty(legs: list[ParlayLeg]) -> float:
    if len(legs) < 2:
        return 1.0
    penalty = 1.0
    probs = [leg.effective_prob for leg in legs]
    if all(p >= 0.68 for p in probs):
        penalty *= 0.80
    rounds = {leg.ronda for leg in legs if leg.ronda}
    if len(rounds) == 1 and len(legs) >= 3:
        penalty *= 0.90
    fechas = {leg.fecha[:10] for leg in legs if leg.fecha}
    if len(fechas) == 1 and len(legs) >= 4:
        penalty *= 0.88
    avg = sum(probs) / len(probs)
    if avg >= 0.70 and max(probs) - min(probs) < 0.06:
        penalty *= 0.85
    return round(penalty, 3)


def _ticket_from_legs(legs: list[ParlayLeg]) -> ParlayTicket:
    combined_prob = 1.0
    combined_odds = 1.0
    for leg in legs:
        combined_prob *= leg.effective_prob
        if leg.odds and leg.odds > 1:
            combined_odds *= leg.odds
        elif leg.effective_prob > 0:
            combined_odds *= 1 / leg.effective_prob
    corr = correlation_penalty(legs)
    avg_pick_score = sum(leg.pick_score for leg in legs) / len(legs)
    combo_score = combined_prob * avg_pick_score * corr
    ev_parlay = combined_prob * combined_odds - 1.0
    stake = allocate_parlay_stake(
        combined_prob=combined_prob,
        ev_parlay=ev_parlay,
        combo_score=combo_score,
        n_legs=len(legs),
        correlation_penalty=corr,
    )
    return ParlayTicket(
        legs=list(legs),
        combined_prob=round(combined_prob, 5),
        combined_odds=round(combined_odds, 2),
        ev_parlay=round(ev_parlay, 4),
        combo_score=round(combo_score, 5),
        correlation_penalty=corr,
        stake_pct=stake,
        n_legs=len(legs),
    )


def build_parlay_tickets(
    legs: list[ParlayLeg],
    *,
    min_legs: int | None = None,
    max_legs: int | None = None,
    top_n: int = 3,
) -> list[ParlayTicket]:
    settings = get_settings()
    min_legs = min_legs or settings.parlay_min_legs
    max_legs = max_legs or settings.parlay_max_legs
    eligible = [l for l in legs if l.stable and not l.exclude_reason]
    if len(eligible) < min_legs:
        return []

    eligible.sort(key=lambda x: x.pick_score, reverse=True)
    pool = eligible[: settings.parlay_max_pool_legs]

    tickets: list[ParlayTicket] = []
    for n in range(min_legs, min(max_legs, len(pool)) + 1):
        for combo in itertools.combinations(pool, n):
            ticket = _ticket_from_legs(list(combo))
            if ticket.ev_parlay >= settings.parlay_min_ev:
                tickets.append(ticket)

    tickets.sort(key=lambda t: t.combo_score, reverse=True)
    seen: set[frozenset[str]] = set()
    unique: list[ParlayTicket] = []
    for t in tickets:
        key = frozenset(f"{l.team1}|{l.team2}" for l in t.legs)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
        if len(unique) >= top_n:
            break
    return unique


def build_parlays_from_legs(legs: list[ParlayLeg]) -> ParlayBuildResult:
    eligible = [l for l in legs if l.stable and not l.exclude_reason]
    rejected = [l for l in legs if not l.stable or l.exclude_reason]
    tickets = build_parlay_tickets(eligible)
    settings = get_settings()
    hint = ""
    if len(eligible) < settings.parlay_min_legs:
        hint = (
            f"Solo {len(eligible)} pierna(s) elegibles "
            f"(mínimo {settings.parlay_min_legs})."
        )
    return ParlayBuildResult(
        eligible_legs=eligible,
        rejected_legs=rejected,
        tickets=tickets,
        message_hint=hint,
    )


def format_parlay_message(result: ParlayBuildResult) -> str:
    lines: list[str] = [
        f"🟢 {ENGINE_VERSION_TAG}",
        "🔵 PARLAY ENGINE — portfolio (2º cerebro)",
        "Maximiza P₁×P₂×P₃ — no usa lógica NO BET de SHARP.",
        "",
    ]

    if result.tickets:
        best = result.tickets[0]
        lines.append(f"🎯 Mejor combinada ({best.n_legs} legs)")
        lines.append(f"   P combo: {best.combined_prob*100:.2f}%")
        lines.append(f"   Cuota ~{best.combined_odds:.2f}")
        lines.append(f"   EV parlay: {best.ev_parlay*100:+.1f}%")
        lines.append(f"   Combo Score: {best.combo_score:.4f}")
        lines.append(f"   Correlación: {best.correlation_penalty:.2f}")
        lines.append(f"   💵 Stake sugerido: {best.stake_pct}% bankroll")
        lines.append("")
        for i, leg in enumerate(best.legs, 1):
            label = leg.selection
            if leg.selection == leg.team1:
                label = f"{leg.team1} gana"
            elif leg.selection == leg.team2:
                label = f"{leg.team2} gana"
            odds_s = f" @ {leg.odds:.2f}" if leg.odds else ""
            lines.append(f"   {i}. {leg.team1} vs {leg.team2} → {label}{odds_s}")
            lines.append(
                f"      P {leg.effective_prob*100:.0f}% | "
                f"score {leg.pick_score:.3f} | "
                f"estab {leg.stability:.2f}"
            )
        lines.append("")
    else:
        lines.append("❌ Sin combinada viable.")
        if result.message_hint:
            lines.append(result.message_hint)
        lines.append("")

    if result.eligible_legs:
        lines.append(f"✅ Candidatos ({len(result.eligible_legs)})")
        for leg in sorted(result.eligible_legs, key=lambda x: -x.pick_score)[:8]:
            lines.append(
                f"• {leg.team1} vs {leg.team2} → {leg.selection} "
                f"(score {leg.pick_score:.3f})"
            )
        lines.append("")

    if result.rejected_legs:
        lines.append(f"⛔ Descartados ({len(result.rejected_legs)})")
        for leg in result.rejected_legs[:6]:
            lines.append(f"• {leg.team1} vs {leg.team2}: {leg.exclude_reason}")
        lines.append("")

    lines.append("🟢 SHARP ENGINE: singles en /alta o Team vs Team.")
    return "\n".join(lines)
