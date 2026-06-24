"""Trading-style presentation for Telegram picks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.odds_context import (
    EvOpportunity,
    MarketContext1X2,
    OutcomeEdge,
    best_bettable_market_ev,
    check_market_outcome_allowed,
    max_market_divergence,
)
from apps.api.services.injury_news import InjuryReport
from apps.api.services.market_calibration import (
    DISCREPANCY_LABELS,
    LAYER_LABELS,
    DiscrepancyDiagnosis,
    MarketAdjustment,
    apply_market_calibration,
    compute_recalibrated_confidence,
    diagnose_discrepancy,
    market_agreement_score,
    model_confidence_tier,
)
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings


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


def _pick_rating(
    ev_market: float,
    has_market: bool,
    no_bet: bool,
    *,
    market_divergence_flag: bool = False,
    confidence_score: int = 0,
) -> tuple[int, str]:
    """1–5 estrellas de fuerza del pick (jerarquía trading)."""
    if no_bet or not has_market or market_divergence_flag:
        return 1, "🔴"
    if confidence_score < 35:
        return 1, "🔴"
    if ev_market >= 0.06:
        return 5, "🟢"
    if ev_market >= 0.04:
        return 4, "🟢"
    if ev_market >= 0.02:
        return 3, "🟡"
    if ev_market > 0:
        return 2, "🔴"
    return 1, "🔴"


def _traffic_light(
    ev_market: float,
    has_market: bool,
    *,
    market_divergence_flag: bool = False,
    confidence_score: int = 0,
) -> tuple[str, str, str]:
    if market_divergence_flag:
        return "rojo", "🔴", "Modelo desacoplado del mercado — no apostar"
    if not has_market or ev_market <= 0:
        return "rojo", "🔴", "Sin valor, no apostar"
    if confidence_score < 35:
        return "rojo", "🔴", "Confianza insuficiente vs mercado"
    if ev_market >= 0.04:
        return "verde", "🟢", "Apuesta recomendada"
    if ev_market >= 0.02:
        return "amarillo", "🟡", "Apuesta moderada"
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
    *,
    market_divergence_flag: bool = False,
    divergence_lines: list[str] | None = None,
    diagnosis: DiscrepancyDiagnosis | None = None,
    adjustment: MarketAdjustment | None = None,
) -> str:
    m = analysis.model
    if not m:
        return "Datos insuficientes para recomendar."

    if market_divergence_flag:
        diag = ""
        if diagnosis:
            diag = f" Tipo: {diagnosis.label} — {diagnosis.description}."
        if adjustment and adjustment.layer == "extreme":
            return (
                f"BLOQUEO EXTREMO: divergencia modelo vs mercado.{diag} "
                "El modelo conserva su edge; el mercado actúa solo como filtro "
                "(no se mezclan probabilidades). Stake 0%."
            )
        lines = divergence_lines or []
        detail = lines[0] if lines else "divergencia modelo vs mercado"
        return (
            f"FILTRO MERCADO: {detail}.{diag} "
            "Modelo base intacto; sin apuesta hasta alinear señales."
        )

    if light == "rojo":
        if not pick.from_ev:
            best = max(m.home_win, m.draw, m.away_win)
            if best < 0.50:
                fav = pick.selection
                return (
                    f"{fav} no supera el 50% en el modelo; "
                    f"alta probabilidad de empate o sorpresa."
                )
            return "No existe ventaja estadística frente al mercado."
        return "EV no supera umbrales o cuota actual por debajo de la justa."

    if pick.from_ev and pick.ev_fair > 0:
        return (
            f"Modelo {pick.model_prob*100:.1f}% alineado con mercado; "
            f"edge {pick.edge_fair*100:+.1f}%."
        )

    return f"Pick del modelo ({pick.model_prob*100:.1f}%); confirma cuota antes de apostar."


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


def _pick_from_market_outcome(outcome: OutcomeEdge) -> TradingPick:
    ev = outcome.edge_pct / 100.0
    return TradingPick(
        market="1X2",
        selection=outcome.selection,
        model_prob=outcome.model_prob,
        ev_fair=ev,
        edge_fair=ev,
        fair_odds=outcome.model_fair_odds,
        raw_odds=outcome.market_odds or 0.0,
        from_ev=ev > 0,
    )


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

    from apps.api.services.odds_context import compute_market_context

    settings = get_settings()
    max_div = settings.ev_max_model_market_divergence
    max_ev = settings.ev_max_fair

    if market_ctx is None:
        market_ctx = compute_market_context(m, analysis.team1, analysis.team2, None)

    adjustment, adjusted_ctx = apply_market_calibration(
        m,
        market_ctx,
        analysis.team1,
        analysis.team2,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        extreme_threshold=max_div,
    )

    ev_opps = ev_opps or []
    max_div_val = max_market_divergence(market_ctx)
    decision_layer = adjustment.layer if adjustment else "normal"
    divergence_flag = decision_layer == "extreme"
    divergence_lines = _divergence_alert_lines(market_ctx) if divergence_flag else []

    diagnosis = diagnose_discrepancy(
        analysis,
        market_ctx,
        max_divergence=max_div_val,
        data_quality_pct=data_quality_pct,
        hist_played=hist_played,
        layer=decision_layer,  # type: ignore[arg-type]
    )

    if decision_layer == "extreme":
        adjusted_ctx = None

    # EV y picks SIEMPRE sobre modelo base — mercado es filtro, no reemplazo
    decision_ctx = market_ctx

    if ev_opps and not divergence_flag:
        primary = _pick_from_ev(ev_opps[0])
        extras = [_pick_from_ev(o) for o in ev_opps[1:4]]
    elif decision_ctx and decision_ctx.has_market and not divergence_flag:
        bettable: list[OutcomeEdge] = []
        for o in decision_ctx.outcomes:
            ok, _ = check_market_outcome_allowed(
                o, max_divergence=max_div, max_ev=max_ev
            )
            if ok:
                bettable.append(o)
        if bettable:
            best_o = max(bettable, key=lambda x: x.edge_pct)
            primary = _pick_from_market_outcome(best_o)
        else:
            primary = _pick_from_model(analysis)
        extras = []
    else:
        primary = _pick_from_model(analysis)
        extras = []

    has_market = bool(market_ctx and market_ctx.has_market)
    ev_market = (
        0.0
        if divergence_flag
        else best_bettable_market_ev(
            decision_ctx, max_divergence=max_div, max_ev=max_ev
        )
    )
    if ev_opps and not divergence_flag:
        ev_market = max(ev_market, ev_opps[0].expected_value)

    agreement = market_agreement_score(max_div_val)
    injury_penalty = 0.0
    if injury_report and (injury_report.has_injuries or injury_report.has_suspensions):
        injury_penalty = 8.0

    tier = model_confidence_tier(m)
    conf_score = compute_recalibrated_confidence(
        data_quality_pct=data_quality_pct,
        market_agreement=agreement,
        historical_accuracy=historical_accuracy,
        injury_penalty=injury_penalty,
        model_tier=tier,
        layer=decision_layer,  # type: ignore[arg-type]
    )

    light, light_emoji, classification = _traffic_light(
        ev_market,
        has_market,
        market_divergence_flag=divergence_flag,
        confidence_score=conf_score,
    )
    no_bet = light == "rojo" or divergence_flag
    pick_rating, pick_rating_emoji = _pick_rating(
        ev_market,
        has_market,
        no_bet,
        market_divergence_flag=divergence_flag,
        confidence_score=conf_score,
    )
    conf_label, conf_emoji = _confidence_label(primary.model_prob)
    is_dog = primary.selection not in (analysis.team1, "Empate") and primary.model_prob < 0.40
    if primary.selection == analysis.team2 and primary.model_prob < 0.45:
        is_dog = True
    risk, risk_emoji = _risk_label(primary.model_prob, m.draw, is_dog)
    if divergence_flag:
        risk, risk_emoji = "Alto", "⚠️"
        conf_label, conf_emoji = "Baja", "📉"

    min_odds = round(primary.fair_odds, 2) if primary.fair_odds > 1 else None
    stake_pct = round(primary.kelly_stake * 100, 1) if primary.kelly_stake else 0.0
    if no_bet:
        stake_pct = 0.0
    elif light == "verde" and stake_pct < 0.5:
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
        rationale=_rationale(
            primary,
            analysis,
            light,
            market_divergence_flag=divergence_flag,
            divergence_lines=divergence_lines,
            diagnosis=diagnosis,
            adjustment=adjustment,
        ),
        no_bet=no_bet,
        extra_picks=extras,
        market=market_ctx,
        confidence_score=conf_score,
        pick_rating=pick_rating,
        pick_rating_emoji=pick_rating_emoji,
        injury=injury_report,
        market_divergence_flag=divergence_flag,
        max_divergence=max_div_val,
        divergence_lines=divergence_lines,
        market_adjustment=adjustment,
        adjusted_market=adjusted_ctx,
        diagnosis=diagnosis,
        decision_layer=decision_layer,
    )


def build_trading_card_from_dict(data: dict[str, Any]) -> TradingCard:
    """Rebuild trading card from serialized analysis dict (fallback sin LLM)."""
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


def format_trading_message(
    card: TradingCard,
    *,
    quality_note: str = "",
    alta_header: bool = False,
) -> str:
    m = card.model
    t1, t2 = card.team1, card.team2
    market = card.market

    lines: list[str] = []
    if alta_header:
        lines.append("💎 Alta probabilidad / valor fair")
    lines.append(f"⚽ {t1} vs {t2}")
    if card.fecha:
        lines.append(f"📅 {card.fecha}" + (f" | {card.ronda}" if card.ronda else ""))

    lines.append("")
    lines.append("📊 Nivel 1 — Modelo puro (Poisson + ELO)")
    lines.append(f"{prob_risk_emoji(m.home_win)} {t1}: {m.home_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.draw)} Empate: {m.draw*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.away_win)} {t2}: {m.away_win*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.over_25)} Over 2.5: {m.over_25*100:.1f}%")
    lines.append(f"{prob_risk_emoji(m.btts_yes)} BTTS Sí: {m.btts_yes*100:.1f}%")

    adj = card.market_adjustment
    layer = card.decision_layer
    layer_label = LAYER_LABELS.get(layer, layer)

    if market and market.has_market:
        lines.append("")
        lines.append("📊 Nivel 2 — Filtro mercado (validez)")
        for o in market.outcomes:
            if o.market_implied:
                lines.append(
                    f"   {o.selection}: implied {o.market_implied*100:.1f}%"
                    + (f" | Δ {o.divergence*100:.0f}%" if o.divergence else "")
                )

    lines.append("")
    lines.append(f"📊 Nivel 3 — Motor decisión: {layer_label}")
    if adj and layer != "normal":
        lines.append(f"   ({adj.layer_reason})")

    if adj and adj.blend_applied and layer == "doubt" and card.max_divergence < 0.20:
        lines.append("")
        lines.append(
            f"📎 Referencia auxiliar ({adj.model_weight:.0%}/{adj.market_weight:.0%}) "
            "— informativa, NO usada en EV"
        )
        lines.append(f"   {t1}: {adj.home*100:.1f}% | Empate: {adj.draw*100:.1f}% | {t2}: {adj.away*100:.1f}%")

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
        lines.append("💎 Edge (solo modelo base — único válido para EV)")
        for o in market.outcomes:
            lines.append(_format_edge_line(o))
        if card.market_divergence_flag:
            lines.append("   ⚠️ Edge post-ajuste: IGNORADO (Δ > umbral)")

    if card.market_divergence_flag:
        lines.append("")
        lines.append("🚨 FILTRO MERCADO — BLOQUEO")
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
        lines.append(
            f"Δ modelo vs mercado: {card.max_divergence*100:.1f}% "
            f"(umbral {get_settings().ev_max_model_market_divergence*100:.0f}%)"
        )

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
        lines.append("🛑 NO APOSTAR")
        lines.append(f"⭐ Rating: {card.pick_rating_emoji} {card.pick_rating}/5")
        lines.append("")
        lines.append("📉 Motivo:")
        lines.append(card.rationale)
    else:
        pick_label = _format_pick_label(card.pick, t1, t2)
        lines.append("🎯 PICK PRINCIPAL")
        lines.append(pick_label)
        lines.append(f"⭐ Rating: {card.pick_rating_emoji} {card.pick_rating}/5")
        lines.append("")
        if card.pick.from_ev or (market and market.has_market):
            best_edge = max((o.edge_pct for o in market.outcomes), default=0) if market else 0
            pick_edge = next(
                (o.edge_pct for o in (market.outcomes if market else []) if o.selection == card.pick.selection),
                best_edge,
            )
            lines.append(f"💰 EV: {pick_edge:+.1f}%")
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
    if card.no_bet:
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
                f"• {label} | EV {ep.ev_fair*100:+.1f}% | "
                f"cuota min {ep.fair_odds:.2f} | stake {ep.kelly_stake*100:.1f}%"
            )

    if quality_note:
        lines.append("")
        lines.append("📊 Calidad:")
        lines.append(quality_note)

    lines.append("")
    lines.append("⚠️ Predicciones probabilísticas, no garantías.")
    return "\n".join(lines)
