"""Trading-style presentation for Telegram picks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.engine_constants import ENGINE_VERSION_TAG
from apps.api.services.ev_policy import format_ev_display

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.api.services.bet_profile import BetProfile, build_bet_profile, format_bet_profile_block
from apps.api.services.parlay_engine import ParlayLeg, evaluate_parlay_leg
from apps.api.services.sharp_engine import run_sharp_engine
from apps.api.services.injury_news import InjuryReport
from apps.api.services.market_dominance import (
    LAYER_LABELS,
    DiscrepancyDiagnosis,
    MarketAdjustment,
    MarketDominanceResult,
)
from apps.api.services.odds_context import (
    EvOpportunity,
    MarketContext1X2,
    OutcomeEdge,
    max_market_divergence,
)
from apps.api.services.trading_types import TradingPick
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings

# Re-export for backward compatibility
__all__ = ["TradingPick", "TradingCard", "build_trading_card", "format_trading_message"]


@dataclass
class TradingCard:
    team1: str
    team2: str
    fecha: str
    ronda: str
    model: ModelMarkets
    pick: TradingPick
    light: str
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
    market: MarketContext1X2 | None = None
    confidence_score: int = 0
    pick_rating: int = 1
    pick_rating_emoji: str = "🔴"
    injury: InjuryReport | None = None
    market_divergence_flag: bool = False
    max_divergence: float = 0.0
    divergence_lines: list[str] = field(default_factory=list)
    market_adjustment: MarketAdjustment | None = None
    adjusted_market: MarketContext1X2 | None = None
    diagnosis: DiscrepancyDiagnosis | None = None
    decision_layer: str = "normal"
    dominance: MarketDominanceResult | None = None
    decision: BetDecisionResult | None = None
    mds: int = 0
    sharp_allowed: bool = False
    sharp_gate_label: str = ""
    parlay_leg: ParlayLeg | None = None
    bet_profile: BetProfile | None = None


def prob_risk_emoji(probability: float) -> str:
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
    return "★" * stars + "☆" * (5 - stars)


def _divergence_alert_lines(market_ctx: MarketContext1X2 | None) -> list[str]:
    if not market_ctx or not market_ctx.has_market:
        return []
    flagged = sorted(
        (o for o in market_ctx.outcomes if (o.divergence or 0) > 0.15),
        key=lambda x: x.divergence or 0,
        reverse=True,
    )
    lines: list[str] = []
    for o in flagged[:3]:
        mkt = f"{(o.market_implied or 0)*100:.1f}%" if o.market_implied else "n/d"
        lines.append(f"{o.selection}: modelo {o.model_prob*100:.1f}% | mercado {mkt}")
    return lines


def _rationale(
    pick: TradingPick,
    analysis: MatchAnalysis,
    decision: BetDecisionResult,
    *,
    divergence_lines: list[str] | None = None,
    diagnosis: DiscrepancyDiagnosis | None = None,
    dominance: MarketDominanceResult | None = None,
) -> str:
    m = analysis.model
    if not m:
        return "Datos insuficientes para recomendar."

    if decision.soft_action == "WATCH":
        band = ""
        if decision.ev_band:
            b = decision.ev_band
            band = (
                f" EV optimista {b.optimistic*100:+.1f}% | "
                f"base {b.base*100:+.1f}% | pesimista {b.pessimistic*100:+.1f}%."
            )
        return (
            f"VIGILAR: edge detectado pero stake 0%.{band} "
            f"MUS {decision.mus:.2f} — {decision.blocked_reason or 'contexto mixto'}."
        )

    if decision.no_bet and dominance and dominance.layer == "extreme":
        diag = ""
        if diagnosis:
            diag = f" {diagnosis.label}: {diagnosis.description}."
        return (
            f"MISMATCH ESTRUCTURAL (Δ {dominance.max_raw_divergence*100:.0f}%).{diag} "
            f"Gate MUS: {decision.blocked_reason or 'mercado confiado'}. Stake 0%."
        )

    if decision.no_bet:
        reason = decision.blocked_reason or "sin valor"
        return f"NO APOSTAR ({reason}). Modelo base intacto."

    if decision.soft_action == "WEAK_BET":
        return (
            f"Apuesta cauta — MUS {decision.mus:.2f}, "
            f"EV base {pick.ev_fair*100:+.1f}%."
        )

    if pick.from_ev and pick.ev_fair > 0:
        return (
            f"Modelo {pick.model_prob*100:.1f}% alineado con mercado; "
            f"edge {pick.edge_fair*100:+.1f}%."
        )

    return f"Pick del modelo ({pick.model_prob*100:.1f}%); confirma cuota antes de apostar."


def build_trading_card(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity] | None = None,
    *,
    odds_available: bool = True,
    market_ctx: MarketContext1X2 | None = None,
    injury_report: InjuryReport | None = None,
    data_quality_pct: float = 100.0,
    hist_played: int = 20,
    historical_accuracy: float | None = None,
) -> TradingCard:
    m = analysis.model
    if not m:
        raise ValueError("analysis sin modelo")

    sharp = run_sharp_engine(
        analysis,
        ev_opps,
        market_ctx=market_ctx,
        injury_report=injury_report,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        historical_accuracy=historical_accuracy,
    )
    pipeline = sharp.pipeline
    dominance = pipeline.market.dominance
    decision = sharp.decision
    market_ctx = pipeline.market.context

    parlay_leg = evaluate_parlay_leg(
        analysis,
        market_ctx,
        dominance,
        injury_report=injury_report,
    )
    if sharp.sharp_allowed:
        sharp_gate_label = "BET"
    elif decision.soft_action == "WATCH":
        sharp_gate_label = "WATCH"
    else:
        sharp_gate_label = "NO_BET"

    assert decision.pick is not None
    primary = decision.pick
    divergence_flag = dominance.layer == "extreme"
    divergence_lines = _divergence_alert_lines(market_ctx) if divergence_flag else []
    conf_label, conf_emoji = _confidence_label(primary.model_prob)
    if divergence_flag:
        conf_label, conf_emoji = "Baja", "📉"

    bet_profile = build_bet_profile(
        model=m,
        team1=analysis.team1,
        team2=analysis.team2,
        market_ctx=market_ctx,
        dominance=dominance,
        decision=decision,
        parlay_leg=parlay_leg,
        injury_report=injury_report,
        sharp_allowed=sharp.sharp_allowed,
        sharp_gate_label=sharp_gate_label,
        mds=sharp.mds,
    )

    return TradingCard(
        team1=analysis.team1,
        team2=analysis.team2,
        fecha=analysis.fecha,
        ronda=analysis.ronda,
        model=m,
        pick=primary,
        light=decision.light,
        light_emoji=decision.light_emoji,
        classification=decision.classification,
        confidence=conf_label,
        confidence_emoji=conf_emoji,
        stars=_star_rating(primary.ev_fair, primary.edge_fair, primary.model_prob),
        stake_pct=decision.stake_pct,
        risk=decision.risk,
        risk_emoji=decision.risk_emoji,
        min_odds=decision.min_odds,
        rationale=_rationale(
            primary,
            analysis,
            decision,
            divergence_lines=divergence_lines,
            diagnosis=dominance.diagnosis,
            dominance=dominance,
        ),
        no_bet=decision.no_bet,
        extra_picks=decision.extra_picks,
        market=market_ctx,
        confidence_score=decision.confidence_score,
        pick_rating=decision.pick_rating,
        pick_rating_emoji=decision.pick_rating_emoji,
        injury=injury_report,
        market_divergence_flag=divergence_flag,
        max_divergence=dominance.max_raw_divergence,
        divergence_lines=divergence_lines,
        market_adjustment=dominance.adjustment,
        adjusted_market=dominance.adjusted_market,
        diagnosis=dominance.diagnosis,
        decision_layer=dominance.layer,
        dominance=dominance,
        decision=decision,
        mds=sharp.mds,
        sharp_allowed=sharp.sharp_allowed,
        sharp_gate_label=sharp_gate_label,
        parlay_leg=parlay_leg,
        bet_profile=bet_profile,
    )


def build_trading_card_from_dict(data: dict[str, Any]) -> TradingCard:
    partido = data.get("partido", "")
    teams = partido.split(" vs ") if " vs " in partido else ["Local", "Visitante"]
    t1 = teams[0].strip() if teams else "Local"
    t2 = teams[1].strip() if len(teams) > 1 else "Visitante"
    modelo = data.get("modelo", {})
    x12 = modelo.get("1x2", {})

    def _prob(key: str, default: float = 0.33) -> float:
        val = x12.get(key)
        return float(val) if val is not None else default

    m = ModelMarkets(
        home_win=_prob(t1),
        draw=_prob("empate"),
        away_win=_prob(t2),
        over_25=float(modelo.get("over_25") or 0.5),
        under_25=float(modelo.get("under_25") or 0.5),
        btts_yes=float(modelo.get("btts_si") or 0.5),
        btts_no=float(1 - float(modelo.get("btts_si") or 0.5)),
        lambda_home=float(modelo.get("lambda_home") or 1.2),
        lambda_away=float(modelo.get("lambda_away") or 1.2),
        confidence=str(modelo.get("confianza") or "medium"),
    )
    analysis = MatchAnalysis(
        team1=t1,
        team2=t2,
        fecha=data.get("fecha", ""),
        ronda=data.get("ronda", ""),
        grupo=data.get("grupo", ""),
        estadio="",
        model=m,
    )
    ev_opps: list[EvOpportunity] = []
    for o in data.get("oportunidades_ev", []):
        fair = float(o.get("cuota_fair") or 0)
        ev_opps.append(
            EvOpportunity(
                market=o.get("mercado", "1X2"),
                selection=o.get("seleccion", ""),
                model_prob=float(o.get("prob_modelo") or 0),
                book_odds=float(o.get("cuota_bruta") or fair or 2.0),
                implied_prob=1 / fair if fair > 1 else 0.5,
                expected_value=float(o.get("ev_fair") or 0),
                edge_pct=float(o.get("ev_fair") or 0) * 100,
                priority=o.get("prioridad", "low"),
                raw_odds=float(o.get("cuota_bruta") or 0),
                fair_odds=fair,
                edge_fair=float(o.get("ev_fair") or 0),
                expected_value_raw=float(o.get("ev_bruto") or 0),
                metadata={"kelly_stake": 0.01},
            )
        )
    odds_ok = bool(data.get("mercado_casas", {}).get("disponible", True))
    return build_trading_card(analysis, ev_opps, odds_available=odds_ok)


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


def _format_edge_line(o: OutcomeEdge) -> str:
    sign = "+" if o.edge_pct > 0 else ""
    return f"{o.selection}: {sign}{o.edge_pct:.1f}%"


def _format_tree_path(path: list[str]) -> list[str]:
    """Colapsa tree_path a líneas legibles para Telegram."""
    skip = {"start"}
    lines: list[str] = []
    for step in path:
        if step in skip:
            continue
        lines.append(f"   → {step}")
    return lines[-5:]


def format_trading_message(
    card: TradingCard,
    *,
    quality_note: str = "",
    alta_header: bool = False,
) -> str:
    m = card.model
    t1, t2 = card.team1, card.team2
    market = card.market
    dom = card.dominance
    dec = card.decision

    lines: list[str] = []
    lines.append(f"🟢 {ENGINE_VERSION_TAG}")
    if alta_header:
        lines.append("💎 Alta probabilidad / valor fair (SHARP scan)")
    lines.append(f"⚽ {t1} vs {t2}")
    if card.fecha:
        lines.append(f"📅 {card.fecha}" + (f" | {card.ronda}" if card.ronda else ""))

    lines.append("")
    lines.append("📊 Nivel 1 — MODEL (fuente de verdad)")
    lines.append(f"{prob_risk_emoji(m.home_win)} {t1}: {m.home_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.draw)} Empate: {m.draw*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.away_win)} {t2}: {m.away_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.over_25)} Over 2.5: {m.over_25*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.btts_yes)} BTTS Sí: {m.btts_yes*100:.1f}%")

    if card.bet_profile:
        lines.extend(format_bet_profile_block(card.bet_profile))

    if market and market.has_market:
        lines.append("")
        lines.append("📊 Nivel 2 — MARKET (solo contexto, sin corrección)")
        for o in market.outcomes:
            if o.market_implied:
                lines.append(
                    f"   {o.selection}: implied {o.market_implied*100:.1f}%"
                    + (f" | Δ {o.divergence*100:.0f}%" if o.divergence else "")
                )

    if dom and dom.uncertainty:
        u = dom.uncertainty
        lines.append("")
        lines.append("📊 Market Uncertainty (MUS)")
        lines.append(f"   Confianza mercado: {u.confidence_market:.2f}")
        lines.append(f"   MUS: {u.mus:.2f} (alto = mercado menos confiable)")
        lines.append(f"   {u.rationale}")

    if dom and dom.layer != "normal":
        lines.append("")
        lines.append("🧠 Market Dominance")
        lines.append(f"   Δ raw: {dom.max_raw_divergence*100:.1f}%")
        lines.append(f"   Clasificación: {dom.classification}")
        lines.append(f"   Fiabilidad modelo: {dom.model_reliability:.2f}")
        lines.append(f"   Fiabilidad mercado: {dom.market_reliability:.2f}")
        if dom.is_market_dominant:
            lines.append("   → MARKET DOMINANT — modelo degradado en este partido")

    layer = card.decision_layer
    layer_label = LAYER_LABELS.get(layer, layer)
    lines.append("")
    lines.append(f"📊 Nivel 3 — DECISION ({layer_label})")
    if dom and dom.layer_reason and layer != "normal":
        lines.append(f"   ({dom.layer_reason})")
    if dec and dec.tree_path:
        lines.append("   Árbol:")
        for step in _format_tree_path(dec.tree_path):
            lines.append(step)
    if card.mds:
        lines.append(f"   MDS (SHARP): {card.mds}/100")
    if card.sharp_gate_label:
        lines.append(f"   SHARP gate: {card.sharp_gate_label}")
    if dec and dec.trust:
        t = dec.trust
        lines.append(
            f"   Arbitraje: {t.trust_side} "
            f"(modelo {t.model_confidence:.0%} vs mercado {t.market_confidence:.0%})"
        )

    if dec and dec.ev_band:
        b = dec.ev_band
        lines.append("")
        lines.append("📊 Banda EV (decisión)")
        lines.append(f"   Optimista: {b.optimistic*100:+.1f}%")
        lines.append(f"   Base: {b.base*100:+.1f}%")
        lines.append(f"   Pesimista: {b.pessimistic*100:+.1f}%")

    pl = card.parlay_leg
    if pl is not None:
        lines.append("")
        lines.append("📊 PARLAY ENGINE (independiente de SHARP)")
        if pl.stable and not pl.exclude_reason:
            label = pl.selection
            if pl.selection == t1:
                label = f"{t1} gana"
            elif pl.selection == t2:
                label = f"{t2} gana"
            lines.append(f"   Estado: ELIGIBLE → {label}")
            lines.append(
                f"   P efectiva {pl.effective_prob*100:.0f}% | score {pl.pick_score:.3f}"
            )
        elif pl.exclude_reason:
            lines.append(f"   Estado: RECHAZADO — {pl.exclude_reason}")
        else:
            lines.append("   Estado: NO ELEGIBLE")

    lines.append("")
    lines.append("─────────────────")
    lines.append("📈 Cuotas fair")
    if market and market.outcomes:
        for o in market.outcomes:
            lines.append(f"{o.selection}: {o.model_fair_odds:.2f}")
    else:
        for sel, prob in [(t1, m.home_win), ("Empate", m.draw), (t2, m.away_win)]:
            if prob > 0:
                lines.append(f"{sel}: {round(1 / prob, 2):.2f}")

    if market and market.has_market:
        lines.append("")
        lines.append("🏦 Cuotas mercado")
        for o in market.outcomes:
            if o.market_odds and o.market_odds > 1:
                lines.append(f"{o.selection}: {o.market_odds:.2f}")
        lines.append("")
        lines.append("💎 Edge (modelo base)")
        for o in market.outcomes:
            lines.append(_format_edge_line(o))

    if card.market_divergence_flag:
        lines.append("")
        lines.append("⚠️ MISMATCH ESTRUCTURAL — gate MUS")
        if card.diagnosis:
            d = card.diagnosis
            lines.append(f"PRIMARY: {d.label}")
            lines.append(f"   {d.description}")
            if d.secondary_label:
                lines.append(f"SECONDARY: {d.secondary_label}")
                if d.secondary_description:
                    lines.append(f"   {d.secondary_description}")
            lines.append(f"RESULT: {d.result}")
        for line in card.divergence_lines:
            lines.append(f"• {line}")
        lines.append(f"Δ modelo vs mercado: {card.max_divergence*100:.1f}%")

    injury = card.injury
    if injury and injury.articles:
        lines.append("")
        lines.append("🩺 Alertas plantilla")
        for line in injury.headline_lines(3):
            lines.append(f"• {line}")
        flags = []
        if injury.has_injuries:
            flags.append("lesiones")
        if injury.has_suspensions:
            flags.append("sanciones")
        if flags:
            lines.append(f"⚠️ Noticias recientes: {', '.join(flags)} (revisar alineación)")

    lines.append("")
    lines.append("─────────────────")

    if card.no_bet:
        if dec and dec.soft_action == "WATCH":
            if card.stake_pct > 0:
                lines.append("👁️ MICRO-STAKE WATCH — sin single SHARP")
            else:
                lines.append("👁️ VIGILAR — sin single SHARP")
        else:
            lines.append("🛑 NO APOSTAR")
        lines.append(f"⭐ Rating: {card.pick_rating_emoji} {card.pick_rating}/5")
        if dec and dec.soft_action == "WATCH":
            pick_label = _format_pick_label(card.pick, t1, t2)
            lines.append(f"🎯 Edge detectado: {pick_label}")
            if card.stake_pct > 0:
                lines.append(f"💵 Micro-stake: {card.stake_pct:g}% bankroll")
        lines.append("")
        lines.append("📉 Motivo:")
        lines.append(card.rationale)
    else:
        pick_label = _format_pick_label(card.pick, t1, t2)
        if dec and dec.soft_action == "WATCH" and card.stake_pct > 0:
            lines.append("👁️ MICRO-STAKE WATCH")
        else:
            lines.append("🎯 PICK PRINCIPAL")
        lines.append(pick_label)
        lines.append(f"⭐ Rating: {card.pick_rating_emoji} {card.pick_rating}/5")
        lines.append("")
        if card.pick.from_ev or (market and market.has_market):
            pick_outcome = next(
                (o for o in (market.outcomes if market else []) if o.selection == card.pick.selection),
                None,
            )
            if pick_outcome:
                lines.append(
                    "💰 " + format_ev_display(
                        ev_fair_pct=pick_outcome.ev_fair_pct,
                        ev_raw_pct=pick_outcome.ev_raw_pct,
                    )
                )
            elif card.pick.ev_fair:
                lines.append(f"💰 EV fair {card.pick.ev_fair*100:+.1f}%")
        if card.pick.raw_odds > 1:
            lines.append(f"🏦 Cuota mercado: {card.pick.raw_odds:.2f}")
        if card.min_odds:
            sel = card.pick.selection if card.pick.selection in (t1, t2, "Empate") else card.pick.selection
            lines.append(f"📊 Cuota mínima: {sel} > {card.min_odds:.2f}")
        lines.append(f"🚦 Estado: {card.light_emoji} {card.light.upper()}")
        lines.append(card.classification + ".")
        lines.append("")
        lines.append("📉 Motivo:")
        lines.append(card.rationale)

    lines.append("")
    lines.append(f"🎯 Confianza: {card.confidence_score}/100")
    lines.append(f"{card.risk_emoji} Riesgo: {card.risk}")
    if dec and dec.soft_action == "WATCH" and card.stake_pct > 0:
        lines.append(f"💵 Stake exploratorio: {card.stake_pct:g}% bankroll")
    elif card.no_bet:
        lines.append("💵 Stake: 0%")
    elif card.stake_pct > 0:
        lines.append(f"💵 Stake: {card.stake_pct:g}% bankroll")
    else:
        lines.append("💵 Stake: 0%")

    if card.extra_picks:
        lines.append("─────────────────")
        lines.append("📋 Otros mercados +EV")
        for ep in card.extra_picks:
            label = _format_pick_label(ep, t1, t2)
            lines.append(
                f"• {label} | {format_ev_display(ev_fair_pct=ep.ev_fair*100, ev_raw_pct=None)} | "
                f"cuota min {ep.fair_odds:.2f} | stake {ep.kelly_stake*100:.1f}%"
            )

    if quality_note:
        lines.append("")
        lines.append("📊 Calidad:")
        lines.append(quality_note)

    lines.append("")
    lines.append("⚠️ Predicciones probabilísticas, no garantías.")
    return "\n".join(lines)
