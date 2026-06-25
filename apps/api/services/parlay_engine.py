"""
Parlay Engine v3 — Portfolio Optimization (quant riguroso).

INPUT ÚNICO: picks validados vía SHARP ENGINE (extract_sharp_parlay_pick).
Sin heurísticas de EV, sin max(model, market), sin tiers safe/balanced/risk.

MODEL → MARKET → SHARP → PARLAY (portfolio)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

from apps.api.services.engine_constants import ENGINE_VERSION_TAG
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2
from apps.api.services.risk_stake import allocate_parlay_stake
from apps.api.services.sharp_engine import SharpBetResult
from apps.api.services.worldcup_engine import MatchAnalysis
from apps.shared.config import Settings, get_settings

CorrelationRisk = Literal["low", "medium-low", "medium", "high"]


@dataclass(frozen=True)
class SharpParlayPick:
    """Pick validado para portfolio — única fuente de entrada al optimizador."""

    match_id: str
    team1: str
    team2: str
    fecha: str
    ronda: str
    outcome: str
    market: str
    p_model: float
    odds: float
    ev_fair: float
    confidence: float
    mds: float
    correlation_group: str
    reject_reason: str | None = None

    @property
    def eligible(self) -> bool:
        return self.reject_reason is None

    @property
    def team_set(self) -> frozenset[str]:
        return frozenset({self.team1, self.team2})


@dataclass
class ParlayLeg:
    """Adapter legacy para bet_profile / trading_card (solo lectura)."""

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
    sharp_pick: SharpParlayPick | None = None


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
    correlation_score: float = 0.0
    confidence_avg: float = 0.0
    correlation_risk: CorrelationRisk = "medium"
    risk_label: str = "CONTROLLED"
    sharp_picks: list[SharpParlayPick] = field(default_factory=list)


@dataclass
class ParlayBuildResult:
    eligible_picks: list[SharpParlayPick] = field(default_factory=list)
    rejected_picks: list[SharpParlayPick] = field(default_factory=list)
    eligible_legs: list[ParlayLeg] = field(default_factory=list)
    rejected_legs: list[ParlayLeg] = field(default_factory=list)
    tickets: list[ParlayTicket] = field(default_factory=list)
    message_hint: str = ""
    reject_reasons: list[str] = field(default_factory=list)


# ── SHARP hard filter ─────────────────────────────────────────────────────────

def passes_sharp_parlay_filter(
    pick: SharpParlayPick,
    dominance: MarketDominanceResult | None,
    *,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    settings = settings or get_settings()
    min_ev = max(settings.ev_min_edge_fair, settings.parlay_sharp_min_ev)
    min_conf = settings.parlay_min_confidence
    min_mds = settings.parlay_min_mds

    if pick.ev_fair < min_ev:
        return False, f"ev_fair {pick.ev_fair:.1%} < {min_ev:.0%}"
    if pick.confidence < min_conf:
        return False, f"confidence {pick.confidence:.0f} < {min_conf}"
    if pick.mds < min_mds:
        return False, f"mds {pick.mds:.0f} < {min_mds}"
    if dominance and dominance.is_market_dominant:
        return False, "market_dominant"
    if dominance and dominance.diagnosis and dominance.diagnosis.primary_type == "market_dominant":
        return False, "market_dominant"
    return True, ""


def _match_id(team1: str, team2: str, fecha: str) -> str:
    return f"{team1}|{team2}|{fecha[:10]}"


def _correlation_group(team1: str, team2: str, fecha: str) -> str:
    return _match_id(team1, team2, fecha)


def extract_sharp_parlay_pick(
    analysis: MatchAnalysis,
    sharp: SharpBetResult,
    market_ctx: MarketContext1X2 | None,
    ev_opps: list[EvOpportunity] | None = None,
    *,
    settings: Settings | None = None,
) -> SharpParlayPick | None:
    """Construye pick desde SHARP output; marca reject_reason si no pasa filtro."""
    settings = settings or get_settings()
    dec = sharp.decision
    dom = sharp.pipeline.market.dominance
    t1, t2 = analysis.team1, analysis.team2
    fecha = (analysis.fecha or "")[:10]
    mid = _match_id(t1, t2, fecha)

    if not dec.pick:
        return SharpParlayPick(
            match_id=mid,
            team1=t1,
            team2=t2,
            fecha=fecha,
            ronda=analysis.ronda or "",
            outcome="",
            market="",
            p_model=0.0,
            odds=0.0,
            ev_fair=0.0,
            confidence=float(dec.confidence_score),
            mds=float(sharp.mds),
            correlation_group=mid,
            reject_reason="sin pick SHARP",
        )

    pick = dec.pick
    ev_fair = dec.ev_band.base if dec.ev_band else sharp.ev_final
    market = "1X2"
    odds = pick.fair_odds or dec.min_odds or 0.0

    if market_ctx and market_ctx.has_market:
        for o in market_ctx.outcomes:
            if o.selection == pick.selection and o.market_odds and o.market_odds > 1:
                odds = o.market_odds
                break

    if ev_opps:
        for o in ev_opps:
            if o.selection == pick.selection:
                market = o.market
                if o.raw_odds and o.raw_odds > 1:
                    odds = o.raw_odds
                elif o.fair_odds and o.fair_odds > 1 and odds <= 1:
                    odds = o.fair_odds
                break

    if odds <= 1.0:
        if pick.model_prob > 0:
            odds = round(1.0 / pick.model_prob, 2)

    mkt_p = None
    if market_ctx and market_ctx.has_market:
        for o in market_ctx.outcomes:
            if o.selection == pick.selection:
                mkt_p = o.market_implied
                break

    sp = SharpParlayPick(
        match_id=mid,
        team1=t1,
        team2=t2,
        fecha=fecha,
        ronda=analysis.ronda or "",
        outcome=pick.selection,
        market=market,
        p_model=round(pick.model_prob, 6),
        odds=round(odds, 4),
        ev_fair=round(ev_fair, 6),
        confidence=float(dec.confidence_score),
        mds=float(sharp.mds),
        correlation_group=_correlation_group(t1, t2, fecha),
    )
    ok, reason = passes_sharp_parlay_filter(sp, dom, settings=settings)
    if not ok:
        return SharpParlayPick(
            match_id=sp.match_id,
            team1=sp.team1,
            team2=sp.team2,
            fecha=sp.fecha,
            ronda=sp.ronda,
            outcome=sp.outcome,
            market=sp.market,
            p_model=sp.p_model,
            odds=sp.odds,
            ev_fair=sp.ev_fair,
            confidence=sp.confidence,
            mds=sp.mds,
            correlation_group=sp.correlation_group,
            reject_reason=reason,
        )
    return sp


def sharp_pick_to_parlay_leg(
    pick: SharpParlayPick,
    *,
    market_prob: float | None = None,
) -> ParlayLeg:
    """Adapter bet_profile — p_model puro, sin effective_prob heurístico."""
    stable = pick.eligible
    return ParlayLeg(
        team1=pick.team1,
        team2=pick.team2,
        fecha=pick.fecha,
        ronda=pick.ronda,
        selection=pick.outcome,
        model_prob=pick.p_model,
        market_prob=market_prob,
        effective_prob=pick.p_model,
        odds=pick.odds if pick.odds > 1 else None,
        ev_adjusted=pick.ev_fair,
        pick_score=round(pick.p_model * pick.ev_fair, 5),
        stability=0.85 if stable else 0.4,
        market_agreement=0.8,
        stable=stable,
        exclude_reason=pick.reject_reason,
        sharp_pick=pick,
    )


def evaluate_parlay_leg(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    dominance: MarketDominanceResult | None = None,
    *,
    injury_report=None,
    sharp: SharpBetResult | None = None,
    ev_opps: list[EvOpportunity] | None = None,
    settings: Settings | None = None,
) -> ParlayLeg:
    """ENTRY v3 — solo desde SHARP; dominance/injury ignorados (filtro en pick)."""
    t1, t2 = analysis.team1, analysis.team2
    base = ParlayLeg(
        team1=t1,
        team2=t2,
        fecha=analysis.fecha,
        ronda=analysis.ronda,
        selection="",
        model_prob=0.0,
        market_prob=None,
        effective_prob=0.0,
        odds=None,
        ev_adjusted=0.0,
        stable=False,
        exclude_reason="requiere output SHARP",
    )
    if not analysis.model or not sharp:
        return base
    if not market_ctx or not market_ctx.has_market:
        base.exclude_reason = "sin mercado"
        return base

    sp = extract_sharp_parlay_pick(
        analysis, sharp, market_ctx, ev_opps, settings=settings
    )
    if sp is None:
        return base

    mkt_p = None
    for o in market_ctx.outcomes:
        if o.selection == sp.outcome:
            mkt_p = o.market_implied
            break
    return sharp_pick_to_parlay_leg(sp, market_prob=mkt_p)


# ── Correlation engine ────────────────────────────────────────────────────────

def pairwise_correlation(a: SharpParlayPick, b: SharpParlayPick) -> float:
    """
    Correlación determinística entre dos picks SHARP.

    same_match + same outcome → redundante (0.85)
    same_match + diff outcome/market → 0.55
    shared team across matches → 0.35
    independent → 0.125
    """
    if a.match_id == b.match_id:
        if a.outcome == b.outcome and a.market == b.market:
            return 0.85
        return 0.55
    if a.team_set & b.team_set:
        return 0.35
    return 0.125


def _pairwise_penalty_sum(picks: list[SharpParlayPick]) -> float:
    total = 0.0
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            total += pairwise_correlation(picks[i], picks[j])
    return total


def _avg_correlation_score(picks: list[SharpParlayPick]) -> float:
    n = len(picks)
    if n < 2:
        return 0.0
    return _pairwise_penalty_sum(picks) / (n * (n - 1) / 2)


def _correlation_risk_label(score: float) -> CorrelationRisk:
    if score <= 0.25:
        return "low"
    if score <= 0.45:
        return "medium-low"
    if score <= 0.65:
        return "medium"
    return "high"


# ── Parlay probability & EV (real) ──────────────────────────────────────────

def compute_parlay_metrics(
    picks: list[SharpParlayPick],
) -> tuple[float, float, float, float, float]:
    """
    Returns:
        p_parlay, odds_parlay, ev_parlay, correlation_adjustment, avg_correlation
    """
    p_joint = 1.0
    odds_joint = 1.0
    for p in picks:
        p_joint *= p.p_model
        odds_joint *= max(p.odds, 1.01)

    penalty_sum = _pairwise_penalty_sum(picks)
    correlation_adjustment = math.exp(-penalty_sum)
    p_parlay = p_joint * correlation_adjustment
    ev_parlay = p_parlay * odds_joint - 1.0
    avg_corr = _avg_correlation_score(picks)
    return (
        round(p_parlay, 6),
        round(odds_joint, 4),
        round(ev_parlay, 6),
        round(correlation_adjustment, 6),
        round(avg_corr, 4),
    )


def _acceptable_drawdown_risk(p_parlay: float, n_legs: int) -> bool:
    """Heurística conservadora de varianza del portfolio."""
    if p_parlay < 0.005:
        return False
    if p_parlay < 0.02 and n_legs >= 4:
        return False
    if p_parlay < 0.01 and n_legs >= 5:
        return False
    return True


def _is_redundant_combo(picks: list[SharpParlayPick]) -> bool:
    seen: set[tuple[str, str, str]] = set()
    for p in picks:
        key = (p.match_id, p.outcome, p.market)
        if key in seen:
            return True
        seen.add(key)
    return False


def _combo_passes_pruning(
    picks: list[SharpParlayPick],
    *,
    max_pairwise: float,
) -> tuple[bool, str]:
    if _is_redundant_combo(picks):
        return False, "picks redundantes (misma señal)"
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            c = pairwise_correlation(picks[i], picks[j])
            if c > max_pairwise:
                return False, f"correlación pairwise {c:.2f} > {max_pairwise:.2f}"
    return True, ""


def _ticket_from_picks(
    picks: list[SharpParlayPick],
    *,
    settings: Settings,
) -> ParlayTicket:
    p_parlay, odds_parlay, ev_parlay, corr_adj, avg_corr = compute_parlay_metrics(picks)
    legs = [sharp_pick_to_parlay_leg(p) for p in picks]
    conf_avg = sum(p.confidence for p in picks) / len(picks)
    corr_risk = _correlation_risk_label(avg_corr)
    risk_label = "CONTROLLED" if corr_risk in ("low", "medium-low") else "ELEVATED"

    combo_score = round(p_parlay * (conf_avg / 100.0) * corr_adj, 6)
    stake = allocate_parlay_stake(
        combined_prob=p_parlay,
        ev_parlay=ev_parlay,
        combo_score=combo_score,
        n_legs=len(picks),
        correlation_penalty=corr_adj,
    )

    return ParlayTicket(
        legs=legs,
        combined_prob=p_parlay,
        combined_odds=odds_parlay,
        ev_parlay=ev_parlay,
        combo_score=combo_score,
        correlation_penalty=corr_adj,
        stake_pct=stake,
        n_legs=len(picks),
        correlation_score=avg_corr,
        confidence_avg=round(conf_avg, 1),
        correlation_risk=corr_risk,
        risk_label=risk_label,
        sharp_picks=list(picks),
    )


def _passes_portfolio_rules(
    ticket: ParlayTicket,
    *,
    settings: Settings,
) -> tuple[bool, str]:
    if ticket.ev_parlay < settings.parlay_min_ev:
        return False, f"EV {ticket.ev_parlay:.1%} < {settings.parlay_min_ev:.0%}"
    if ticket.confidence_avg < settings.parlay_min_confidence:
        return False, "confidence_avg below threshold"
    if ticket.correlation_score > settings.parlay_max_correlation_score:
        return False, "correlation too high"
    if not _acceptable_drawdown_risk(ticket.combined_prob, ticket.n_legs):
        return False, "drawdown risk excessive"
    return True, ""


# ── Portfolio optimizer ───────────────────────────────────────────────────────

def build_parlays_from_sharp_picks(
    picks: list[SharpParlayPick],
    *,
    min_legs: int | None = None,
    max_legs: int | None = None,
    top_n: int = 3,
    settings: Settings | None = None,
) -> ParlayBuildResult:
    """Optimizador principal v3 — ranking portfolio + SharpParlayPick elegibles."""
    settings = settings or get_settings()
    min_legs = min_legs or settings.parlay_min_legs
    max_legs = max_legs or settings.parlay_max_legs
    max_pairwise = settings.parlay_max_pairwise_correlation

    if getattr(settings, "sharp_mode", "portfolio") == "portfolio":
        from apps.api.services.sharp_portfolio import promote_portfolio_picks

        picks = promote_portfolio_picks(
            picks,
            top_pct=settings.sharp_portfolio_top_pct,
            top_k=settings.sharp_portfolio_top_k,
        )

    eligible = [p for p in picks if p.eligible]
    rejected = [p for p in picks if not p.eligible]
    eligible_legs = [sharp_pick_to_parlay_leg(p) for p in eligible]
    rejected_legs = [sharp_pick_to_parlay_leg(p) for p in rejected]

    reject_reasons: list[str] = []
    if len(eligible) < min_legs:
        hint = (
            f"Solo {len(eligible)} pick(s) SHARP elegible(s) "
            f"(mínimo {min_legs})."
        )
        for p in rejected[:5]:
            if p.reject_reason:
                reject_reasons.append(f"{p.team1} vs {p.team2}: {p.reject_reason}")
        return ParlayBuildResult(
            eligible_picks=eligible,
            rejected_picks=rejected,
            eligible_legs=eligible_legs,
            rejected_legs=rejected_legs,
            tickets=[],
            message_hint=hint,
            reject_reasons=reject_reasons or ["SHARP picks insufficient quality"],
        )

    pool = sorted(eligible, key=lambda p: (-p.ev_fair, -p.mds, -p.p_model))[
        : settings.parlay_max_pool_legs
    ]

    candidates: list[ParlayTicket] = []
    for n in range(min_legs, min(max_legs, len(pool)) + 1):
        for combo in itertools.combinations(pool, n):
            plist = list(combo)
            ok, prune_reason = _combo_passes_pruning(
                plist, max_pairwise=max_pairwise
            )
            if not ok:
                continue
            ticket = _ticket_from_picks(plist, settings=settings)
            passes, fail_reason = _passes_portfolio_rules(ticket, settings=settings)
            if passes:
                candidates.append(ticket)
            elif fail_reason not in reject_reasons:
                reject_reasons.append(fail_reason)

    candidates.sort(key=lambda t: (t.ev_parlay, t.combo_score), reverse=True)

    seen: set[frozenset[str]] = set()
    unique: list[ParlayTicket] = []
    for t in candidates:
        key = frozenset(p.match_id for p in t.sharp_picks)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
        if len(unique) >= top_n:
            break

    hint = ""
    if not unique:
        hint = "No parlay — insufficient statistical edge"
        if not reject_reasons:
            reject_reasons = [
                "EV below threshold",
                "correlation too high",
                "SHARP picks insufficient quality",
            ]

    return ParlayBuildResult(
        eligible_picks=eligible,
        rejected_picks=rejected,
        eligible_legs=eligible_legs,
        rejected_legs=rejected_legs,
        tickets=unique,
        message_hint=hint,
        reject_reasons=reject_reasons,
    )


def build_parlays_from_legs(legs: list[ParlayLeg]) -> ParlayBuildResult:
    """Legacy adapter — requiere sharp_pick en cada leg."""
    picks: list[SharpParlayPick] = []
    for leg in legs:
        if leg.sharp_pick:
            picks.append(leg.sharp_pick)
        elif leg.stable and not leg.exclude_reason and leg.selection:
            picks.append(
                SharpParlayPick(
                    match_id=_match_id(leg.team1, leg.team2, leg.fecha),
                    team1=leg.team1,
                    team2=leg.team2,
                    fecha=leg.fecha,
                    ronda=leg.ronda,
                    outcome=leg.selection,
                    market="1X2",
                    p_model=leg.model_prob,
                    odds=leg.odds or 2.0,
                    ev_fair=leg.ev_adjusted,
                    confidence=70.0,
                    mds=70.0,
                    correlation_group=_match_id(leg.team1, leg.team2, leg.fecha),
                )
            )
    return build_parlays_from_sharp_picks(picks)


def build_parlay_tickets(
    legs: list[ParlayLeg],
    *,
    min_legs: int | None = None,
    max_legs: int | None = None,
    top_n: int = 3,
) -> list[ParlayTicket]:
    return build_parlays_from_legs(legs).tickets


# Backward-compat aliases
def compute_pick_score(leg: ParlayLeg) -> float:
    if leg.sharp_pick:
        p = leg.sharp_pick
        return round(p.p_model * (1.0 + p.ev_fair), 5)
    return round(leg.effective_prob * (1.0 + leg.ev_adjusted), 5)


def correlation_penalty(legs: list[ParlayLeg]) -> float:
    picks = [l.sharp_pick for l in legs if l.sharp_pick]
    if len(picks) < 2:
        return 1.0
    _, _, _, corr_adj, _ = compute_parlay_metrics(picks)
    return corr_adj


# ── Presentation (separable from core) ────────────────────────────────────────

def _display_outcome(pick: SharpParlayPick) -> str:
    if pick.outcome == pick.team1:
        return f"{pick.team1} gana"
    if pick.outcome == pick.team2:
        return f"{pick.team2} gana"
    if pick.outcome == "Empate":
        return "Empate"
    return f"{pick.outcome} ({pick.market})"


def format_parlay_message(result: ParlayBuildResult) -> str:
    lines: list[str] = [
        f"🟢 {ENGINE_VERSION_TAG}",
        "💎 PARLAY VALIDATED (QUANT ENGINE v3)",
        "Solo picks SHARP | correlación real | EV matemático",
        "",
    ]

    if result.tickets:
        for idx, ticket in enumerate(result.tickets, 1):
            lines.append(f"Combo {idx}")
            lines.append("Legs:")
            for p in ticket.sharp_picks:
                odds_s = f" @ {p.odds:.2f}" if p.odds > 1 else ""
                lines.append(f"- {_display_outcome(p)}{odds_s}")
            lines.append("")
            lines.append(f"📊 p_combo: {ticket.combined_prob*100:.1f}%")
            lines.append(f"📊 EV real: {ticket.ev_parlay*100:+.1f}%")
            lines.append(f"📊 Correlation risk: {ticket.correlation_risk}")
            lines.append(f"📊 Confidence: {ticket.confidence_avg:.0f}")
            lines.append(f"Risk: {ticket.risk_label}")
            lines.append(f"💵 Stake: {ticket.stake_pct:g}%")
            lines.append("")
    else:
        lines.append("❌ NO PARLAY — insufficient statistical edge")
        lines.append("")
        lines.append("Reason:")
        reasons = result.reject_reasons or (
            [result.message_hint] if result.message_hint else []
        )
        for r in reasons[:6]:
            lines.append(f"- {r}")
        if result.rejected_picks:
            lines.append("")
            lines.append(f"Rejected SHARP picks ({len(result.rejected_picks)}):")
            for p in result.rejected_picks[:5]:
                lines.append(f"• {p.team1} vs {p.team2}: {p.reject_reason}")
        lines.append("")

    if result.eligible_picks:
        lines.append(f"✅ SHARP pool ({len(result.eligible_picks)})")
        for p in result.eligible_picks[:6]:
            lines.append(
                f"• {p.team1} vs {p.team2} → {p.outcome} "
                f"EV {p.ev_fair*100:+.1f}% MDS {p.mds:.0f}"
            )
        lines.append("")

    return "\n".join(lines)
