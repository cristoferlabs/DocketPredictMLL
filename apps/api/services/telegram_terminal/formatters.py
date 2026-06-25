"""Formatters UI — solo presentación, sin decisión."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api.services.ev_policy import (
    format_ev_display,
    is_actionable_value,
    is_structural_mismatch,
)
from apps.api.services.pick_quality import format_pick_quality_lines
from apps.api.services.market_alignment import alignment_status, gap_pp, model_outlier_status
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2
from apps.api.services.sharp_engine import SharpBetResult
from apps.api.services.telegram_terminal.keyboards import terminal_header
from apps.api.services.trading_card import prob_risk_emoji
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


@dataclass
class RankedPick:
    label: str
    market: str
    ev_fair_pct: float
    ev_raw_pct: float | None
    confidence: int
    risk: str
    model_prob: float
    market_implied: float | None
    odds_decimal: float | None
    edge_pp: float | None
    actionable: bool
    note: str = ""


def format_exploration(matches: list[dict]) -> str:
    lines = [
        terminal_header(),
        "",
        "📅 PARTIDOS DE HOY",
        "",
    ]
    if not matches:
        lines.append("No hay partidos próximos.")
        return "\n".join(lines)
    for i, m in enumerate(matches[:12], 1):
        t1 = m.get("team1", "TBD")
        t2 = m.get("team2", "TBD")
        fecha = (m.get("fecha") or "")[:10]
        suffix = f" ({fecha})" if fecha else ""
        lines.append(f"{i}. {t1} vs {t2}{suffix}")
    lines.extend(["", "[Selecciona partido]", "", "⚠️ Terminal informativo — tú decides."])
    return "\n".join(lines)


def _market_summary(market_ctx: MarketContext1X2 | None, team1: str, team2: str) -> str:
    if not market_ctx or not market_ctx.has_market:
        return "Sin cuotas de mercado disponibles"
    best = max(market_ctx.outcomes, key=lambda o: o.model_prob)
    if best.selection in (team1, team2):
        fav = best.selection
        mkt = f"{(best.market_implied or 0) * 100:.0f}%" if best.market_implied else "n/d"
        return f"{fav} favorito en mercado ({mkt} implícito)"
    if best.selection == "Empate":
        return "Mercado equilibrado — empate líder en modelo"
    peak = max(
        ((o.selection, o.market_implied or 0) for o in market_ctx.outcomes),
        key=lambda x: x[1],
    )
    return f"Mercado: {peak[0]} ~{peak[1]*100:.0f}% implícito"


def format_match_dashboard(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
) -> str:
    m = analysis.model
    lines = [
        terminal_header(),
        "",
        f"⚽ {analysis.team1} vs {analysis.team2}",
        f"📅 {(analysis.fecha or '')[:10]} | {analysis.ronda or '—'}",
        "",
        "📊 MODEL",
    ]
    if not m:
        lines.append("Sin modelo disponible.")
        return "\n".join(lines)
    lines.extend(
        [
            f"{prob_risk_emoji(m.home_win)} {analysis.team1}: {m.home_win*100:.1f}%",
            f"{prob_risk_emoji(m.draw)} Empate: {m.draw*100:.1f}%",
            f"{prob_risk_emoji(m.away_win)} {analysis.team2}: {m.away_win*100:.1f}%",
            "",
            "📈 MARKET CONTEXT",
            _market_summary(market_ctx, analysis.team1, analysis.team2),
            "",
            "Estado: listo para explorar opciones.",
        ]
    )
    return "\n".join(lines)


def _pick_label(selection: str, team1: str, team2: str) -> str:
    if selection == team1:
        return f"{team1} gana"
    if selection == team2:
        return f"{team2} gana"
    if selection == "Empate":
        return "Empate"
    return selection


def _risk_from_prob(p: float) -> str:
    if p >= 0.55:
        return "Bajo"
    if p >= 0.40:
        return "Medio"
    return "Alto"


def build_ranked_picks(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity],
    sharp: SharpBetResult | None,
    market_ctx: MarketContext1X2 | None,
) -> list[RankedPick]:
    """Tabla EV por outcome — fair primero, cuota explícita, todos los desenlaces 1X2."""
    picks: list[RankedPick] = []
    conf_base = (
        sharp.decision.confidence_score
        if sharp and sharp.decision
        else 0
    )
    mds = sharp.mds if sharp else 0
    t1, t2 = analysis.team1, analysis.team2
    seen: set[str] = set()

    if market_ctx and market_ctx.has_market:
        for o in market_ctx.outcomes:
            label = _pick_label(o.selection, t1, t2)
            seen.add(label)
            edge_pp = None
            if o.market_implied is not None:
                edge_pp = round((o.model_prob - o.market_implied) * 100.0, 1)
            actionable = is_actionable_value(
                o.ev_fair_pct,
                o.model_prob,
                o.market_implied,
                divergence=o.divergence,
            )
            note = format_ev_display(
                ev_fair_pct=o.ev_fair_pct,
                ev_raw_pct=o.ev_raw_pct,
                odds_decimal=o.market_odds,
                model_prob=o.model_prob,
                market_implied=o.market_implied,
                divergence=o.divergence,
                market="1X2",
            )
            picks.append(
                RankedPick(
                    label=label,
                    market="1X2",
                    ev_fair_pct=o.ev_fair_pct,
                    ev_raw_pct=o.ev_raw_pct,
                    confidence=conf_base or mds or int(o.model_prob * 100),
                    risk=_risk_from_prob(o.model_prob),
                    model_prob=o.model_prob,
                    market_implied=o.market_implied,
                    odds_decimal=o.market_odds,
                    edge_pp=edge_pp,
                    actionable=actionable,
                    note=note,
                )
            )

    for o in ev_opps:
        label = _pick_label(o.selection, t1, t2)
        if label in seen:
            continue
        ev_f = o.expected_value * 100.0
        ev_r = o.expected_value_raw * 100.0 if o.expected_value_raw else None
        picks.append(
            RankedPick(
                label=label,
                market=o.market,
                ev_fair_pct=ev_f,
                ev_raw_pct=ev_r,
                confidence=conf_base or mds or int(o.model_prob * 100),
                risk=_risk_from_prob(o.model_prob),
                model_prob=o.model_prob,
                market_implied=o.implied_prob,
                odds_decimal=o.raw_odds or o.fair_odds,
                edge_pp=round(o.edge_fair * 100.0, 1) if o.edge_fair else None,
                actionable=ev_f >= 3.0,
                note=format_ev_display(
                    ev_fair_pct=ev_f,
                    ev_raw_pct=ev_r,
                    odds_decimal=o.raw_odds or o.fair_odds,
                    model_prob=o.model_prob,
                    market_implied=o.implied_prob,
                    market=o.market,
                ),
            )
        )

    picks.sort(key=lambda x: (-x.ev_fair_pct, -x.model_prob))
    return picks


def format_opportunities(
    analysis: MatchAnalysis,
    picks: list[RankedPick],
) -> str:
    lines = [
        terminal_header(),
        "",
        f"🎯 TOP PICKS — {analysis.team1} vs {analysis.team2}",
        "",
        "Ranking informativo (sin decisión final).",
        "",
    ]
    if not picks:
        lines.append("Sin cuotas o sin mercado para calcular EV.")
        lines.append("Revisa análisis técnico.")
        return "\n".join(lines)

    lines.append("EV = (p_model × cuota) − 1 | fair = devig")
    lines.append("")
    for i, p in enumerate(picks[:6], 1):
        tag = "✓" if p.actionable else "·"
        odds_s = f"@ {p.odds_decimal:.2f}" if p.odds_decimal and p.odds_decimal > 1 else "sin cuota"
        mkt_s = f"mkt {p.market_implied*100:.1f}%" if p.market_implied else "mkt n/d"
        lines.append(f"{tag} {i}. {p.label}")
        lines.append(
            f"   model {p.model_prob*100:.1f}% | {mkt_s} | {odds_s}"
        )
        if p.note:
            warn = (
                "estructural" in p.note
                or "INVESTIGATE" in p.note
                or "NO BET" in p.note
                or "tope visual" in p.note
            )
            prefix = "⚠️ " if warn else ""
            lines.append(f"   {prefix}{p.note}")
        else:
            lines.append(
                f"   EV fair {p.ev_fair_pct:+.1f}%"
                + (f" | raw {p.ev_raw_pct:+.1f}%" if p.ev_raw_pct is not None else "")
                + (f" | Δ {p.edge_pp:+.1f}pp" if p.edge_pp is not None else "")
            )
        lines.append("")
    lines.append("⚠️ Ranking ≠ recomendación de apuesta.")
    return "\n".join(lines)


def _format_blend_engine_lines(
    model: ModelMarkets,
    market_ctx: MarketContext1X2 | None,
) -> list[str]:
    """Muestra pesos Poisson/ELO/mercado del combiner v3."""
    meta = model.blend_meta or {}
    if not meta:
        return []
    wc = meta.get("weights_config") or {}
    wp = wc.get("poisson", 0.5)
    we = wc.get("elo", 0.3)
    wm = wc.get("market", 0.2)
    lines = [
        "── Model blend (v3) ──",
        f"Pesos: Poisson {wp*100:.0f}% + ELO {we*100:.0f}%",
        f"Mercado calibración: {wm*100:.0f}% (no entra en EV)",
    ]
    stat = meta.get("blended_statistical") or {}
    if stat:
        lines.append(
            f"1X2 blend (pre-cal): {stat.get('home_win', 0)*100:.1f}% / "
            f"{stat.get('draw', 0)*100:.1f}% / {stat.get('away_win', 0)*100:.1f}%"
        )
    lines.append(
        f"1X2 FINAL (decisión): {model.home_win*100:.1f}% / "
        f"{model.draw*100:.1f}% / {model.away_win*100:.1f}%"
    )
    sanity = meta.get("sanity_adjustments") or []
    if sanity:
        lines.append(f"Ajustes sanity: {', '.join(sanity)}")
    if market_ctx and market_ctx.has_market and wm > 0:
        from apps.worker.ml.model_combiner import (
            ModelCombinationWeights,
            Probabilities1X2,
            apply_market_calibration_layer,
        )

        outs = market_ctx.outcomes
        if len(outs) >= 3 and all(o.fair_implied for o in outs[:3]):
            fair = {
                "home_win": outs[0].fair_implied or 0,
                "draw": outs[1].fair_implied or 0,
                "away_win": outs[2].fair_implied or 0,
            }
            decision = Probabilities1X2(
                model.home_win, model.draw, model.away_win
            )
            anchored = apply_market_calibration_layer(
                decision,
                fair,
                weights=ModelCombinationWeights(poisson=wp, elo=we, market=wm),
            )
            a = anchored.as_dict()
            lines.append(
                f"Ancla mercado (ref): {a['home_win']*100:.1f}% / "
                f"{a['draw']*100:.1f}% / {a['away_win']*100:.1f}%"
            )
    return lines


def format_full_analysis(
    analysis: MatchAnalysis,
    market_ctx: MarketContext1X2 | None,
    sharp: SharpBetResult | None,
    ev_opps: list[EvOpportunity],
) -> str:
    m = analysis.model
    lines = [
        terminal_header(),
        "",
        f"🔬 ANÁLISIS TÉCNICO — {analysis.team1} vs {analysis.team2}",
        "",
    ]
    if not m:
        lines.append("Sin modelo.")
        return "\n".join(lines)

    lines.extend(
        [
            "── Poisson ──",
            f"λ home: {m.lambda_home:.3f} | λ away: {m.lambda_away:.3f}",
            f"Over 2.5: {m.over_25*100:.1f}% | BTTS: {m.btts_yes*100:.1f}%",
            "",
            "── ELO ──",
        ]
    )
    for team in (analysis.team1, analysis.team2):
        elo = analysis.elo.get(team, {})
        lines.append(f"{team}: {elo.get('rating', '—')} (rank {elo.get('rank', '—')})")

    blend_lines = _format_blend_engine_lines(m, market_ctx)
    if blend_lines:
        lines.extend(["", *blend_lines])

    if sharp and sharp.pipeline.market.dominance:
        dom = sharp.pipeline.market.dominance
        lines.extend(
            [
                "",
                "── Market dominance ──",
                f"Capa: {LAYER_LABELS.get(dom.layer, dom.layer)}",
                f"Δ max: {dom.max_raw_divergence*100:.1f}%",
                f"Model rel: {dom.model_reliability:.2f} | Market rel: {dom.market_reliability:.2f}",
            ]
        )
        if dom.diagnosis:
            lines.append(f"Diagnóstico: {dom.diagnosis.label}")
            if dom.diagnosis.description:
                lines.append(f"  {dom.diagnosis.description}")

    if sharp and sharp.decision and sharp.decision.pick and market_ctx and market_ctx.has_market:
        pick = sharp.decision.pick
        impl = None
        for o in market_ctx.outcomes:
            if o.selection == pick.selection:
                impl = o.market_implied
                break
        g = gap_pp(pick.model_prob, impl)
        _, align_label, align_desc = alignment_status(g)
        _, out_label, out_desc, _, _ = model_outlier_status(g, market=pick.market)
        lines.extend(
            [
                "",
                "── Alineación modelo vs mercado (pick) ──",
                f"Δ pick: {g:.1f}pp — {align_label}",
                align_desc,
            ]
        )
        if out_label != "OK":
            lines.append(f"⚠️ {out_label}: {out_desc}")

    lines.extend(["", "── EV breakdown (fair) ──"])
    if market_ctx and market_ctx.has_market:
        lines.append("1X2:")
        for o in market_ctx.outcomes:
            raw_s = f" | raw {o.ev_raw_pct:+.1f}%" if o.ev_raw_pct else ""
            lines.append(
                f"  {o.selection}: EV {o.ev_fair_pct:+.1f}%{raw_s} | "
                f"model {o.model_prob*100:.1f}%"
            )
    if ev_opps:
        lines.append("Totales / otros:")
        for o in ev_opps[:5]:
            lines.append(
                f"  {o.market} {o.selection}: EV {o.expected_value*100:+.1f}% | "
                f"model {o.model_prob*100:.1f}%"
            )
    if not (market_ctx and market_ctx.has_market) and not ev_opps:
        lines.append("Sin mercado — EV no calculable.")

    if sharp:
        dec = sharp.decision
        pick = dec.pick
        pick_label = pick.selection if pick else "—"
        if pick and pick.selection == analysis.team1:
            pick_label = f"{analysis.team1} gana"
        elif pick and pick.selection == analysis.team2:
            pick_label = f"{analysis.team2} gana"

        pick_impl = None
        if pick and market_ctx and market_ctx.has_market and pick.market == "1X2":
            for o in market_ctx.outcomes:
                if o.selection == pick.selection:
                    pick_impl = o.market_implied
                    break

        lines.extend(
            [
                "",
                "── Risk metrics (SHARP output) ──",
                f"Pick decisión: {pick.market if pick else '—'} — {pick_label}",
                f"EV base (pick): {sharp.ev_final*100:+.1f}%",
            ]
        )
        if dec.ev_band and pick and dec.ev_band.selection != pick_label:
            b = dec.ev_band
            lines.append(
                f"  Banda EV: opt {b.optimistic*100:+.1f}% | "
                f"pes {b.pessimistic*100:+.1f}%"
            )
        if pick:
            g_pick = gap_pp(pick.model_prob, pick_impl) if pick_impl else None
            lines.extend(
                format_pick_quality_lines(
                    model_prob=pick.model_prob,
                    market_implied=pick_impl,
                    ev_fair=sharp.ev_final,
                    gap_pp=g_pick,
                    market=pick.market,
                )
            )
        lines.extend(
            [
                f"MDS: {sharp.mds}/100",
                f"MUS: {dec.mus:.2f}",
                f"Confianza: {dec.confidence_score}/100",
            ]
        )
        if dec.tree_path:
            lines.append(f"Árbol: {' → '.join(dec.tree_path[-5:])}")

    lines.extend(["", "⚠️ Modo debug — sin stake ni decisión final."])
    return "\n".join(lines)
