"""Trading-style presentation for Telegram picks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.odds_context import EvOpportunity
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


@dataclass
class TradingPick:
    market: str
    selection: str
    model_prob: float
    ev_fair: float = 0.0
    edge_fair: float = 0.0
    fair_odds: float = 0.0
    raw_odds: float = 0.0
    kelly_stake: float = 0.0
    from_ev: bool = False


@dataclass
class TradingCard:
    team1: str
    team2: str
    fecha: str
    ronda: str
    model: ModelMarkets
    pick: TradingPick
    light: str  # verde | amarillo | rojo
    light_emoji: str
    classification: str
    confidence: str
    confidence_emoji: str
    stars: str
    stake_pct: float
    risk: str
    risk_emoji: str
    min_odds: float | None
    rationale: str
    no_bet: bool = False
    extra_picks: list[TradingPick] = field(default_factory=list)


def prob_risk_emoji(probability: float) -> str:
    """Emoji de riesgo por nivel de probabilidad del modelo."""
    if probability >= 0.55:
        return "🟢"
    if probability >= 0.40:
        return "🟡"
    return "🔴"


def _confidence_label(prob: float) -> tuple[str, str]:
    if prob >= 0.58:
        return "Alta", "📈"
    if prob >= 0.45:
        return "Media", "📊"
    return "Baja", "📉"


def _risk_label(prob: float, draw_prob: float, is_underdog: bool) -> tuple[str, str]:
    if draw_prob >= 0.28:
        return "Alto", "⚠️"
    if prob < 0.45 or is_underdog:
        return "Alto", "⚠️"
    if prob < 0.52:
        return "Medio", "🔶"
    return "Bajo", "✅"


def _star_rating(ev: float, edge: float, prob: float) -> str:
    score = 0
    if ev >= 0.08:
        score += 2
    elif ev >= 0.05:
        score += 1.5
    elif ev >= 0.03:
        score += 1
    elif ev > 0:
        score += 0.5
    if edge >= 0.06:
        score += 1
    elif edge >= 0.03:
        score += 0.5
    if prob >= 0.55:
        score += 1
    elif prob >= 0.48:
        score += 0.5
    stars = max(1, min(5, round(score)))
    filled = "★" * stars
    empty = "☆" * (5 - stars)
    return filled + empty


def _traffic_light(ev: float, edge: float, prob: float, has_odds: bool) -> tuple[str, str, str]:
    if not has_odds or ev <= 0:
        return "rojo", "🔴", "Sin valor, no apostar"
    if ev >= 0.04 and edge >= 0.03 and prob >= 0.45:
        return "verde", "🟢", "Apuesta recomendada"
    if ev > 0:
        return "amarillo", "🟡", "Solo si la cuota supera la cuota justa"
    return "rojo", "🔴", "Sin valor, no apostar"


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


def _rationale(
    pick: TradingPick,
    analysis: MatchAnalysis,
    light: str,
) -> str:
    m = analysis.model
    if not m:
        return "Datos insuficientes para recomendar."

    if light == "rojo":
        if not pick.from_ev:
            best = max(m.home_win, m.draw, m.away_win)
            if best < 0.50:
                fav = pick.selection
                return (
                    f"{fav} no supera el 50% en el modelo; "
                    f"alta probabilidad de empate o sorpresa."
                )
            return "Sin ventaja vs mercado fair; no hay edge positivo."
        return "EV no supera umbrales o cuota actual por debajo de la justa."

    if pick.from_ev and pick.ev_fair > 0:
        return (
            f"Modelo {pick.model_prob*100:.1f}% vs mercado fair "
            f"~{(1/pick.fair_odds*100) if pick.fair_odds > 1 else 0:.1f}%; "
            f"edge {pick.edge_fair*100:+.1f}%."
        )

    return f"Pick del modelo ({pick.model_prob*100:.1f}%); confirma cuota antes de apostar."


def build_trading_card(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity] | None = None,
    *,
    odds_available: bool = True,
) -> TradingCard:
    m = analysis.model
    if not m:
        raise ValueError("analysis sin modelo")

    ev_opps = ev_opps or []
    if ev_opps:
        primary = _pick_from_ev(ev_opps[0])
        extras = [_pick_from_ev(o) for o in ev_opps[1:4]]
    else:
        primary = _pick_from_model(analysis)
        extras = []

    light, light_emoji, classification = _traffic_light(
        primary.ev_fair,
        primary.edge_fair,
        primary.model_prob,
        odds_available,
    )
    conf_label, conf_emoji = _confidence_label(primary.model_prob)
    is_dog = primary.selection not in (analysis.team1, "Empate") and primary.model_prob < 0.40
    if primary.selection == analysis.team2 and primary.model_prob < 0.45:
        is_dog = True
    risk, risk_emoji = _risk_label(primary.model_prob, m.draw, is_dog)

    min_odds = round(primary.fair_odds, 2) if primary.fair_odds > 1 else None
    stake_pct = round(primary.kelly_stake * 100, 1) if primary.kelly_stake else 0.0
    if light == "verde" and stake_pct < 0.5:
        stake_pct = max(stake_pct, 0.5)
    elif light == "amarillo" and stake_pct < 0.25:
        stake_pct = 0.5 if primary.ev_fair >= 0.02 else 0.25

    return TradingCard(
        team1=analysis.team1,
        team2=analysis.team2,
        fecha=analysis.fecha,
        ronda=analysis.ronda,
        model=m,
        pick=primary,
        light=light,
        light_emoji=light_emoji,
        classification=classification,
        confidence=conf_label,
        confidence_emoji=conf_emoji,
        stars=_star_rating(primary.ev_fair, primary.edge_fair, primary.model_prob),
        stake_pct=stake_pct,
        risk=risk,
        risk_emoji=risk_emoji,
        min_odds=min_odds,
        rationale=_rationale(primary, analysis, light),
        no_bet=light == "rojo",
        extra_picks=extras,
    )


def _format_pick_label(pick: TradingPick, team1: str, team2: str) -> str:
    if pick.selection == team1:
        return f"{team1} gana"
    if pick.selection == team2:
        return f"{team2} gana"
    if pick.selection == "Empate":
        return "Empate"
    if pick.market.startswith("Over"):
        return "Over 2.5"
    if pick.market.startswith("Under"):
        return "Under 2.5"
    return pick.selection


def format_trading_message(
    card: TradingCard,
    *,
    quality_note: str = "",
    alta_header: bool = False,
) -> str:
    m = card.model
    t1, t2 = card.team1, card.team2

    lines: list[str] = []
    if alta_header:
        lines.append("💎 Alta probabilidad / valor fair")
    lines.append(f"⚽ {t1} vs {t2}")
    if card.fecha:
        lines.append(f"📅 {card.fecha}" + (f" | {card.ronda}" if card.ronda else ""))

    lines.append("")
    lines.append("📐 Probabilidades")
    lines.append(f"{prob_risk_emoji(m.home_win)} {t1}: {m.home_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.draw)} Empate: {m.draw*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.away_win)} {t2}: {m.away_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.over_25)} Over 2.5: {m.over_25*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.btts_yes)} BTTS Sí: {m.btts_yes*100:.1f}%")

    lines.append("─────────────────")

    if card.no_bet:
        lines.append("🛑 NO APOSTAR")
        lines.append("Sin ventaja clara vs mercado fair.")
        lines.append(f"{card.light_emoji} {card.classification}")
        lines.append(f"📉 Motivo: {card.rationale}")
    else:
        pick_label = _format_pick_label(card.pick, t1, t2)
        lines.append("🎯 PICK PRINCIPAL")
        lines.append(pick_label)
        lines.append("")
        if card.pick.from_ev:
            lines.append(f"💰 EV: {card.pick.ev_fair*100:+.1f}%")
        else:
            lines.append("💰 EV: — (solo modelo, sin cuota +EV)")
        lines.append(f"{card.confidence_emoji} Confianza: {card.confidence}")
        lines.append(f"⭐ Rating: {card.stars}")
        if card.stake_pct > 0:
            lines.append(f"💵 Stake: {card.stake_pct:g}% bankroll")
        lines.append(f"🚦 Estado: {card.light_emoji} {card.light.upper()}")
        lines.append(f"{card.classification}.")
        lines.append("")
        if card.min_odds:
            sel = card.pick.selection if card.pick.selection in (t1, t2) else card.pick.selection
            lines.append("📊 Cuota mínima:")
            lines.append(f"   {sel} > {card.min_odds:.2f}")
        lines.append("")
        lines.append(f"{card.risk_emoji} Riesgo: {card.risk}")
        lines.append(f"📉 Motivo: {card.rationale}")

    if card.extra_picks:
        lines.append("─────────────────")
        lines.append("📋 Otros mercados +EV")
        for ep in card.extra_picks:
            label = _format_pick_label(ep, t1, t2)
            lines.append(
                f"• {label} | EV {ep.ev_fair*100:+.1f}% | "
                f"cuota min {ep.fair_odds:.2f} | stake {ep.kelly_stake*100:.1f}%"
            )

    if quality_note:
        lines.append(f"\n📊 Calidad datos: {quality_note}")

    lines.append("\n⚠️ Predicciones probabilísticas, no garantías.")
    return "\n".join(lines)
