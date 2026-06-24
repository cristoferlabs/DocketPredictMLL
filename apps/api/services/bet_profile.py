"""
Bet Profile Layer — separa predicción del partido de la decisión de apuesta.

Solo lectura sobre salidas de MODEL, MARKET, SHARP y PARLAY.
No modifica Poisson, ELO, EV, MUS, árbol ni motores de riesgo.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.api.services.injury_news import InjuryReport
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.odds_context import MarketContext1X2, OutcomeEdge
from apps.api.services.parlay_engine import ParlayLeg
from apps.api.services.worldcup_engine import ModelMarkets
from apps.shared.config import Settings, get_settings

LONGSHOT_CAP = 0.30
HIGH_PROB = 0.60
MEDIUM_PROB = 0.40


@dataclass(frozen=True)
class ProfileSide:
    """Una cara del partido (favorito, valor, parlay, sharp)."""

    selection: str
    display: str
    probability: float | None = None
    ev_pct: float | None = None
    confidence: int | None = None
    stake_pct: float | None = None
    action: str | None = None
    prob_class: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class BetProfile:
    most_likely: ProfileSide | None
    value_side: ProfileSide | None
    parlay_side: ProfileSide | None
    sharp_side: ProfileSide | None


def classify_probability(p: float) -> str:
    if p >= HIGH_PROB:
        return "HIGH"
    if p >= MEDIUM_PROB:
        return "MEDIUM"
    if p < LONGSHOT_CAP:
        return "LONGSHOT"
    return "MEDIUM"


def _prob_label(cls: str) -> str:
    return {
        "HIGH": "alta probabilidad",
        "MEDIUM": "probabilidad media",
        "LONGSHOT": "longshot",
    }.get(cls, "")


def _model_outcomes(
    model: ModelMarkets,
    team1: str,
    team2: str,
) -> list[tuple[str, float]]:
    return [
        (team1, model.home_win),
        ("Empate", model.draw),
        (team2, model.away_win),
    ]


def _display_selection(selection: str, team1: str, team2: str) -> str:
    if selection == team1:
        return f"{team1} gana"
    if selection == team2:
        return f"{team2} gana"
    if selection == "Empate":
        return "Empate"
    return selection


def _is_longshot(p: float) -> bool:
    return p < LONGSHOT_CAP


def _resolve_most_likely(
    model: ModelMarkets,
    team1: str,
    team2: str,
) -> ProfileSide | None:
    """Favorito por probabilidad pura del modelo (sin EV ni cuotas)."""
    candidates = _model_outcomes(model, team1, team2)
    eligible = [(s, p) for s, p in candidates if not _is_longshot(p)]
    pool = eligible if eligible else candidates
    selection, prob = max(pool, key=lambda x: x[1])
    cls = classify_probability(prob)
    note = None
    if _is_longshot(prob):
        note = "P < 30% — sin favorito claro por regla de longshot"
        return ProfileSide(
            selection=selection,
            display=_display_selection(selection, team1, team2),
            probability=prob,
            prob_class=cls,
            note=note,
        )
    return ProfileSide(
        selection=selection,
        display=_display_selection(selection, team1, team2),
        probability=prob,
        prob_class=cls,
    )


def _resolve_value_side(
    market_ctx: MarketContext1X2 | None,
    team1: str,
    team2: str,
    *,
    settings: Settings | None = None,
) -> ProfileSide | None:
    """Mejor EV informativo (raw); decisión usa fair en motores."""
    settings = settings or get_settings()
    if not market_ctx or not market_ctx.has_market:
        return None
    positive = [o for o in market_ctx.outcomes if o.ev_raw_pct > 0]
    if not positive:
        return None
    best: OutcomeEdge = max(positive, key=lambda o: o.ev_raw_pct)
    if best.ev_fair_pct / 100.0 < settings.ev_min_edge_fair and best.ev_raw_pct < 5.0:
        return None
    note = "EV raw informativo"
    if abs(best.ev_raw_pct - best.ev_fair_pct) > 1.0:
        note += f" (fair {best.ev_fair_pct:+.1f}%)"
    return ProfileSide(
        selection=best.selection,
        display=_display_selection(best.selection, team1, team2),
        probability=best.model_prob,
        ev_pct=best.ev_raw_pct,
        prob_class=classify_probability(best.model_prob),
        note=note,
    )


def _parlay_confidence(
    leg: ParlayLeg,
    *,
    mus: float,
    mds: int,
    injury: InjuryReport | None,
) -> int:
    news_pen = 0 if (injury and (injury.has_injuries or injury.has_suspensions)) else 8
    raw = (
        leg.effective_prob * 55
        + leg.stability * 20
        + leg.market_agreement * 15
        + (1.0 - mus) * 10
        + min(mds, 100) / 100 * 10
        - news_pen
    )
    return int(max(0, min(100, round(raw))))


def _resolve_parlay_side(
    parlay_leg: ParlayLeg | None,
    model: ModelMarkets,
    team1: str,
    team2: str,
    *,
    mus: float = 0.5,
    mds: int = 0,
    injury: InjuryReport | None = None,
) -> ProfileSide | None:
    """Candidato parlay: P alta, riesgo bajo (desde motor PARLAY, solo lectura)."""
    if parlay_leg is None:
        return None
    model_p = parlay_leg.model_prob
    if _is_longshot(model_p) and parlay_leg.selection != "Empate":
        return ProfileSide(
            selection=parlay_leg.selection,
            display=_display_selection(parlay_leg.selection, team1, team2),
            probability=model_p,
            action="N/A",
            prob_class="LONGSHOT",
            note="P < 30% — no elegible como parlay",
        )
    if not parlay_leg.stable or parlay_leg.exclude_reason:
        if parlay_leg.exclude_reason and model_p >= LONGSHOT_CAP:
            return ProfileSide(
                selection=parlay_leg.selection,
                display=_display_selection(parlay_leg.selection, team1, team2),
                probability=model_p,
                action="RECHAZADO",
                note=parlay_leg.exclude_reason,
            )
        return None
    conf = _parlay_confidence(parlay_leg, mus=mus, mds=mds, injury=injury)
    return ProfileSide(
        selection=parlay_leg.selection,
        display=_display_selection(parlay_leg.selection, team1, team2),
        probability=parlay_leg.effective_prob,
        confidence=conf,
        action="ELIGIBLE",
        prob_class=classify_probability(parlay_leg.effective_prob),
    )


def _resolve_sharp_side(
    decision: BetDecisionResult | None,
    *,
    sharp_allowed: bool,
    sharp_gate_label: str,
    mds: int,
    team1: str,
    team2: str,
) -> ProfileSide | None:
    """Single SHARP: EV + confianza + MDS (lectura del gate SHARP)."""
    if decision is None or decision.pick is None:
        return None
    pick = decision.pick
    ev_pct = None
    if decision.ev_band:
        ev_pct = decision.ev_band.base * 100
    elif pick.ev_fair:
        ev_pct = pick.ev_fair * 100

    if sharp_allowed:
        action = "BET"
    elif decision.soft_action == "WATCH":
        action = "WATCH"
    else:
        action = "NO_BET"

    if _is_longshot(pick.model_prob) and pick.selection != "Empate":
        if action == "BET":
            action = "WATCH"
        note = "longshot — solo VALUE / WATCH"
    else:
        note = None

    return ProfileSide(
        selection=pick.selection,
        display=_display_selection(pick.selection, team1, team2),
        probability=pick.model_prob,
        ev_pct=ev_pct,
        confidence=decision.confidence_score,
        stake_pct=decision.stake_pct if decision.stake_pct > 0 else None,
        action=action,
        prob_class=classify_probability(pick.model_prob),
        note=note or f"MDS {mds}",
    )


def build_bet_profile(
    *,
    model: ModelMarkets,
    team1: str,
    team2: str,
    market_ctx: MarketContext1X2 | None,
    dominance: MarketDominanceResult | None,
    decision: BetDecisionResult | None,
    parlay_leg: ParlayLeg | None,
    injury_report: InjuryReport | None = None,
    sharp_allowed: bool = False,
    sharp_gate_label: str = "",
    mds: int = 0,
    settings: Settings | None = None,
) -> BetProfile:
    """Construye las 4 caras del partido sin alterar motores upstream."""
    settings = settings or get_settings()
    mus = dominance.uncertainty.mus if dominance and dominance.uncertainty else 0.5

    return BetProfile(
        most_likely=_resolve_most_likely(model, team1, team2),
        value_side=_resolve_value_side(market_ctx, team1, team2, settings=settings),
        parlay_side=_resolve_parlay_side(
            parlay_leg,
            model,
            team1,
            team2,
            mus=mus,
            mds=mds,
            injury=injury_report,
        ),
        sharp_side=_resolve_sharp_side(
            decision,
            sharp_allowed=sharp_allowed,
            sharp_gate_label=sharp_gate_label,
            mds=mds,
            team1=team1,
            team2=team2,
        ),
    )


def format_bet_profile_block(profile: BetProfile) -> list[str]:
    """Bloque Telegram obligatorio — predicción vs apuesta."""
    lines: list[str] = [
        "─────────────────",
        "📋 BET PROFILE (predicción ≠ apuesta)",
        "",
    ]

    ml = profile.most_likely
    if ml and not (ml.note and "sin favorito" in (ml.note or "")):
        lines.append("🎯 FAVORITO DEL PARTIDO")
        lines.append(ml.display)
        if ml.probability is not None:
            lines.append(f"   Probabilidad: {ml.probability*100:.1f}% ({_prob_label(ml.prob_class or '')})")
        lines.append("")
    elif ml:
        lines.append("🎯 FAVORITO DEL PARTIDO")
        lines.append(f"   Sin favorito claro ({ml.display} {ml.probability*100:.0f}%)")
        lines.append(f"   {ml.note}")
        lines.append("")

    vs = profile.value_side
    if vs:
        lines.append("💎 VALUE SIDE")
        lines.append(vs.display)
        if vs.ev_pct is not None:
            sign = "+" if vs.ev_pct > 0 else ""
            lines.append(f"   EV: {sign}{vs.ev_pct:.1f}%")
        lines.append("")
    else:
        lines.append("💎 VALUE SIDE")
        lines.append("   — sin valor por encima del umbral")
        lines.append("")

    ps = profile.parlay_side
    if ps and ps.action == "ELIGIBLE":
        lines.append("🎲 PARLAY SIDE")
        lines.append(ps.display)
        if ps.confidence is not None:
            lines.append(f"   Confianza: {ps.confidence}/100")
        lines.append("")
    elif ps and ps.action == "RECHAZADO":
        lines.append("🎲 PARLAY SIDE")
        lines.append(f"   Rechazado ({ps.display})")
        lines.append(f"   {ps.note}")
        lines.append("")
    elif ps and ps.action == "N/A":
        lines.append("🎲 PARLAY SIDE")
        lines.append(f"   No elegible — {ps.note}")
        lines.append("")
    else:
        lines.append("🎲 PARLAY SIDE")
        lines.append("   — sin candidato parlay")
        lines.append("")

    ss = profile.sharp_side
    if ss:
        lines.append("⚡ SHARP SIDE")
        if ss.action == "WATCH":
            lines.append(f"   WATCH {ss.display}")
        elif ss.action == "BET":
            lines.append(ss.display)
        else:
            lines.append(f"   {ss.action} — {ss.display}")
        if ss.ev_pct is not None:
            sign = "+" if ss.ev_pct > 0 else ""
            lines.append(f"   EV: {sign}{ss.ev_pct:.1f}%")
        if ss.stake_pct is not None and ss.stake_pct > 0:
            lines.append(f"   Stake: {ss.stake_pct:g}%")
        elif ss.action == "WATCH":
            lines.append("   Stake: 0% (exploratorio si aplica)")
        else:
            lines.append("   Stake: 0%")
        lines.append("")
    else:
        lines.append("⚡ SHARP SIDE")
        lines.append("   — sin señal SHARP")
        lines.append("")

    return lines
