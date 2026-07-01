"""
Menú unificado de apuestas — opción E del terminal.

Muestra en un solo mensaje:
 1. Mercados individuales (todos con EV)
 2. Combinaciones seguras (joint Poisson)
 3. Comparativa EV + recomendación final
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api.services.dc_engine import DCPick, evaluate_dc
from apps.api.services.odds_context import EvOpportunity, MarketContext1X2
from apps.api.services.safe_combo_engine import SafeCombo, build_live_combinations, build_safe_combinations
from apps.api.services.sharp_engine import SharpBetResult
from apps.api.services.telegram_terminal.keyboards import terminal_header
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.worker.ml.odds_math import fair_dc_market, fair_totals_market

# TYPE_CHECKING import to avoid circular deps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apps.worker.ml.poisson_live import LivePoissonResult


_SEP = "─" * 39
_STAKE_MAP = {"STRONG_BET": 2.0, "MODERATE_BET": 1.0, "WEAK_BET": 0.5}


# ─── Row dataclass ────────────────────────────────────────────────────────────

@dataclass
class MarketRow:
    rank: int
    label: str
    model_prob: float
    market_odds: float | None   # raw odds from bookmakers (0 = no quote)
    fair_odds: float            # devigged or model fair
    market_implied: float | None  # devigged market implied prob (1/fair_odds)
    ev_pct: float               # EV% decision (from ev_opps, may be regime-capped)
    ev_raw_pct: float | None    # EV% real = model_prob × market_odds − 1 (uncapped, display only)
    has_market: bool
    market_type: str            # "1X2", "DC", "OU", "BTTS"


# ─── Market rows builder ───────────────────────────────────────────────────────

def _build_market_rows(
    model: ModelMarkets,
    team1: str,
    team2: str,
    odds_event: dict | None,
    ev_opps: list[EvOpportunity],
) -> list[MarketRow]:
    """
    Build one row per selectable outcome across all markets.
    EV is computed against fair (devigged) market price when available,
    otherwise against the model's own fair odds (edge = 0 by definition, shown as 0%).
    """
    # Index ev_opps by (market, selection) for fast lookup
    ev_idx: dict[tuple[str, str], EvOpportunity] = {}
    for opp in ev_opps:
        ev_idx[(opp.market, opp.selection)] = opp

    dc_fair = fair_dc_market(odds_event) if odds_event else {}
    ou15 = fair_totals_market(odds_event, 1.5) if odds_event else {}
    ou25 = fair_totals_market(odds_event, 2.5) if odds_event else {}
    ou35 = fair_totals_market(odds_event, 3.5) if odds_event else {}

    def _row(label: str, model_prob: float, market_key: str, fair_dict: dict,
             mtype: str, ev_market: str, ev_sel: str) -> MarketRow | None:
        if model_prob <= 0:
            return None
        fm = fair_dict.get(market_key, {})
        raw_o = fm.get("raw_odds", 0.0) or 0.0
        fair_o = fm.get("fair_odds") or round(1.0 / model_prob, 2)
        has_mkt = raw_o > 1.0

        # Look up pre-computed EV from ev_opps (regime-capped decision EV)
        opp = ev_idx.get((ev_market, ev_sel))
        if opp:
            ev_pct = round(opp.expected_value * 100, 1)
        elif has_mkt and raw_o > 1:
            # No pre-computed opp → compute real EV vs raw market odds
            ev_pct = round((model_prob * raw_o - 1) * 100, 1)
        else:
            ev_pct = 0.0

        # Real uncapped EV vs market odds (for display only)
        ev_raw_pct: float | None = None
        if has_mkt and raw_o > 1:
            ev_raw_pct = round((model_prob * raw_o - 1) * 100, 1)

        # Devigged market implied (from fair_odds which comes from devig function)
        market_implied: float | None = None
        if has_mkt and fair_o > 1:
            market_implied = round(1.0 / fair_o, 4)

        # No real market quote → no edge by definition
        if not has_mkt:
            ev_pct = 0.0
            ev_raw_pct = None
            market_implied = None

        return MarketRow(
            rank=0,
            label=label,
            model_prob=model_prob,
            market_odds=raw_o if has_mkt else None,
            fair_odds=fair_o,
            market_implied=market_implied,
            ev_pct=ev_pct,
            ev_raw_pct=ev_raw_pct,
            has_market=has_mkt,
            market_type=mtype,
        )

    from apps.worker.ml.odds_math import fair_h2h_market
    h2h = fair_h2h_market(odds_event) if odds_event else {}
    btts_ou_fair = {}  # BTTS has no direct market, derive from model
    # BTTS fair odds are purely model-based
    btts_yes_fo = round(1.0 / model.btts_yes, 2) if model.btts_yes > 0 else 0.0
    btts_no_fo = round(1.0 / model.btts_no, 2) if model.btts_no > 0 else 0.0

    candidates: list[tuple] = [
        # (label, model_prob, market_key, fair_dict, mtype, ev_market, ev_sel)
        (team1,                    model.home_win,    "home",      h2h,    "1X2", "1X2",          team1),
        ("Empate",                 model.draw,        "draw",      h2h,    "1X2", "1X2",          "Empate"),
        (team2,                    model.away_win,    "away",      h2h,    "1X2", "1X2",          team2),
        (f"1X ({team1}/Emp)",      model.dc_home_draw,"home_draw", dc_fair,"DC",  "Doble Oportunidad", f"1X ({team1}/Empate)"),
        (f"X2 (Emp/{team2})",      model.dc_away_draw,"away_draw", dc_fair,"DC",  "Doble Oportunidad", f"X2 (Empate/{team2})"),
        (f"12 ({team1}/{team2})",  model.dc_home_away,"home_away", dc_fair,"DC",  "Doble Oportunidad", f"12 ({team1}/{team2})"),
        ("Over 1.5",               model.over_15,     "over",      ou15,   "OU",  "Over/Under 1.5","Over"),
        ("Under 1.5",              model.under_15,    "under",     ou15,   "OU",  "Over/Under 1.5","Under"),
        ("Over 2.5",               model.over_25,     "over",      ou25,   "OU",  "Over/Under 2.5","Over"),
        ("Under 2.5",              model.under_25,    "under",     ou25,   "OU",  "Over/Under 2.5","Under"),
        ("Over 3.5",               model.over_35,     "over",      ou35,   "OU",  "Over/Under 3.5","Over"),
        ("Under 3.5",              model.under_35,    "under",     ou35,   "OU",  "Over/Under 3.5","Under"),
        ("BTTS Sí",                model.btts_yes,    "",          {},     "BTTS","",              ""),
        ("BTTS No",                model.btts_no,     "",          {},     "BTTS","",              ""),
    ]

    rows: list[MarketRow] = []
    for args in candidates:
        label, model_prob, mkt_key, fair_dict, mtype, ev_mkt, ev_sel = args

        if mtype == "BTTS":
            fo = btts_yes_fo if label == "BTTS Sí" else btts_no_fo
            # BTTS has no direct market quote — EV = 0 by definition
            rows.append(MarketRow(
                rank=0, label=label, model_prob=model_prob,
                market_odds=None, fair_odds=fo, market_implied=None,
                ev_pct=0.0, ev_raw_pct=None, has_market=False, market_type=mtype,
            ))
        else:
            r = _row(*args)
            if r:
                rows.append(r)

    # Rank by EV descending (when market available), then model prob descending
    rows.sort(key=lambda r: (-r.ev_pct if r.has_market else 0, -r.model_prob))
    for i, r in enumerate(rows, 1):
        r.rank = i
    return rows


# ─── Stake helper ─────────────────────────────────────────────────────────────

def _stake_for_combo(combo: SafeCombo) -> float:
    return _STAKE_MAP.get(combo.decision, 0.5)


def _stake_for_sharp(sharp: SharpBetResult | None) -> float:
    # stake_pct is already in % (e.g. 2.0 = 2%)
    # Display cap is configurable via max_stake_display_pct (default 5%)
    if sharp and sharp.stake_pct > 0:
        from apps.shared.config import get_settings as _gs
        cap = getattr(_gs(), "max_stake_display_pct", 5.0)
        return round(min(sharp.stake_pct, cap), 1)
    return 0.0


# ─── Formatter ────────────────────────────────────────────────────────────────

def _stars(combo: SafeCombo) -> str:
    if combo.decision == "STRONG_BET":
        return "⭐⭐⭐"
    if combo.decision == "MODERATE_BET":
        return "⭐⭐"
    return "⭐"


def _ev_tag(ev_pct: float, has_market: bool, ev_raw_pct: float | None = None) -> str:
    if not has_market:
        return "  EV n/d"
    # Show real EV when regime cap reduced decision EV significantly
    if ev_raw_pct is not None and abs(ev_raw_pct - ev_pct) >= 2.0:
        return f"  EV {ev_raw_pct:+.1f}% (cap EV_MAX_FAIR→{ev_pct:+.1f}%)"
    return f"  EV {ev_pct:+.1f}%"


def _odds_tag(row: MarketRow) -> str:
    if row.market_odds and row.market_odds > 1:
        return f"@{row.market_odds:.2f}"
    return f"[f{row.fair_odds:.2f}]"   # fair model odds, no market quote


def _format_dc_section(dc_picks: list[DCPick]) -> list[str]:
    """Compact 🛡️ section for the 2 main DC picks (X2 and 1X)."""
    lines = ["🛡️ APUESTAS SEGURAS (DOBLE OPORTUNIDAD)", ""]
    for p in dc_picks[:2]:
        has_mkt = p.market_odds > 1.0
        odds_s = f"@{p.market_odds:.2f}" if has_mkt else f"[f{p.fair_odds:.2f}]"
        ev_s = f"EV {p.ev_pct:+.1f}%" if has_mkt else "sin cuota mkt"
        stake_s = f"Stake: {p.stake_pct:.1f}%" if p.stake_pct > 0 else "sin stake (sin mkt)"
        if p.is_primary:
            role_line = f"  {p.risk_emoji} RECOMENDACIÓN PRINCIPAL"
        else:
            role_line = f"  {p.risk_emoji} ALTERNATIVA SEGURA"
        lines += [
            f"✅ {p.label}   {p.model_prob*100:.1f}%  {odds_s}  {ev_s}",
            f"   Riesgo: {p.risk} | {stake_s}",
            role_line,
            "─" * 39,
        ]
    return lines


def _format_risk_table(
    analysis: MatchAnalysis,
    dc_picks: list[DCPick],
) -> list[str]:
    """Comparison table: individual 1X2 vs DC picks by risk level."""
    m = analysis.model
    if not m:
        return []
    t1, t2 = analysis.team1, analysis.team2

    def _risk_label(prob: float) -> str:
        if prob >= 0.65:
            return "MUY BAJO"
        if prob >= 0.55:
            return "BAJO"
        if prob >= 0.40:
            return "MEDIO"
        return "ALTO"

    rows = [
        (f"{t2} solo",  m.away_win,       _risk_label(m.away_win)),
        (f"{t1} solo",  m.home_win,        _risk_label(m.home_win)),
        ("Empate",      m.draw,            _risk_label(m.draw)),
    ]
    for p in dc_picks:
        rows.append((p.label, p.model_prob, p.risk))

    rows.sort(key=lambda r: -r[1])  # sort by prob descending

    lines = ["📊 COMPARATIVA DE RIESGO", ""]
    lines.append(f"  {'Apuesta':<24} {'Prob':>6}  Riesgo")
    lines.append(f"  {'─'*24} {'─'*6}  {'─'*8}")
    for label, prob, risk in rows:
        lines.append(f"  {label:<24} {prob*100:>5.1f}%  {risk}")
    return lines


def format_betting_menu(
    analysis: MatchAnalysis,
    market_rows: list[MarketRow],
    combos: list[SafeCombo],
    dc_picks: list[DCPick] | None = None,
    sharp: SharpBetResult | None = None,
    max_individual: int = 8,
    max_combos: int = 3,
    live_result: "LivePoissonResult | None" = None,
) -> str:
    """Full betting menu — DC section + individual markets + safe combos + EV recommendation."""
    t1 = analysis.team1
    t2 = analysis.team2
    fecha = (analysis.fecha or "")[:10]
    ronda = analysis.ronda or ""

    lines: list[str] = [
        terminal_header(),
        "",
        f"📊 MENÚ APUESTAS — {t1} vs {t2}",
        f"📅 {fecha}" + (f" | {ronda}" if ronda else ""),
        "",
    ]

    # ── Live banner ────────────────────────────────────────────────────────────
    if live_result:
        g_h, g_a = live_result.home_goals, live_result.away_goals
        min_rem = live_result.minutes_remaining
        state_label = {
            "first_half": "1ª Parte",
            "halftime": "Descanso",
            "second_half": "2ª Parte",
            "extra_time": "Prórroga",
        }.get(live_result.game_state_label, "En Vivo")
        rc_line = ""
        if hasattr(live_result, "lambda_home_prematch"):
            pass  # diagnostic available if needed
        rc_parts = []
        lines += [
            f"🔴 EN VIVO — {state_label} | {min_rem} min restantes",
            f"   Marcador: {t1} {g_h} — {g_a} {t2}",
            f"   λ restante: {live_result.lambda_home_remaining:.2f} (casa) · {live_result.lambda_away_remaining:.2f} (visit.)",
            "",
        ]
        if live_result.intensity_home != 1.0 or live_result.intensity_away != 1.0:
            lines.append(
                f"   Intensidad: casa {live_result.intensity_home:.2f}x · visit. {live_result.intensity_away:.2f}x"
            )

    lines += [_SEP, ""]

    # ── 0. DC section (priority) ───────────────────────────────────────────────
    if dc_picks:
        lines += _format_dc_section(dc_picks)
        lines += ["", _SEP, ""]
        lines += _format_risk_table(analysis, dc_picks)
        lines += ["", _SEP, ""]

    # ── 1. Individual markets ─────────────────────────────────────────────────
    lines.append("🎯 MERCADOS INDIVIDUALES")
    lines.append("")

    top_rows = market_rows[:max_individual]
    if not top_rows:
        lines.append("Sin datos de mercado disponibles.")
    else:
        has_any_mkt = any(r.has_market for r in top_rows)
        for r in top_rows:
            odds_s = _odds_tag(r)
            ev_s = _ev_tag(r.ev_pct, r.has_market, r.ev_raw_pct)
            ev_flag = "✓" if (r.ev_raw_pct or r.ev_pct) > 0 else "·"
            prob_s = f"{r.model_prob * 100:.1f}%"
            lines.append(
                f"{ev_flag} {r.rank:2}. {r.label:<20} {prob_s:<7} {odds_s:<9}{ev_s}"
            )

        if not has_any_mkt:
            lines.append("")
            lines.append("  (sin cuotas de mercado — EV basado en fair odds modelo)")

    lines += ["", _SEP, ""]

    # ── 2. Safe combinations ─────────────────────────────────────────────────
    combo_label = "🔄 COMBINACIONES EN VIVO" if live_result else "🔄 COMBINACIONES SEGURAS"
    combo_sublabel = (
        "  Poisson condicionado al marcador actual · tiempo restante"
        if live_result else
        "  Prob conjunta exacta (Poisson) · mismo partido"
    )
    lines.append(combo_label)
    lines.append(combo_sublabel)
    lines.append("")

    top_combos = combos[:max_combos]
    if not top_combos:
        lines.append("  Sin combinaciones con ventaja suficiente.")
    else:
        for i, c in enumerate(top_combos, 1):
            dec_e = "🟢" if c.decision == "STRONG_BET" else "🟡" if c.decision == "MODERATE_BET" else "🔴"
            stake = _stake_for_combo(c)
            stars = _stars(c)
            lines.append(f"⭐ COMBINACIÓN {i}: {c.leg1_label} + {c.leg2_label} {stars}")
            lines.append(f"   {c.label1_display}: {c.leg1_prob*100:.1f}%")
            lines.append(f"   {c.label2_display}: {c.leg2_prob*100:.1f}%")
            lines.append(f"   Prob combinada: {c.combo_prob*100:.1f}%")
            lines.append(f"   Fair odds: {c.fair_odds:.2f} | min. mkt: {c.market_min_odds:.2f}")
            lines.append(f"   Riesgo: {c.risk}")
            bd = getattr(c, "score_breakdown", None)
            if bd is not None:
                for dl in bd.detail_lines():
                    lines.append(f"  {dl}")
            else:
                lines.append(f"   Score: {c.score}/100")
            lines.append(f"   {dec_e} {c.decision} | Stake: {stake:.1f}%")
            lines.append("")

    lines += [_SEP, ""]

    # ── 3. EV comparison + final recommendation ───────────────────────────────
    lines.append("📈 COMPARATIVA EV")
    lines.append("")

    best_ind = market_rows[0] if market_rows else None
    best_combo = combos[0] if combos else None

    if best_ind:
        display_ev = best_ind.ev_raw_pct if best_ind.ev_raw_pct is not None else best_ind.ev_pct
        ev_s = f"EV {display_ev:+.1f}%"
        if best_ind.ev_raw_pct is not None and abs(best_ind.ev_raw_pct - best_ind.ev_pct) >= 2.0:
            ev_s += f" (dec. cap {best_ind.ev_pct:+.1f}%)"
        ev_s += f" | Prob {best_ind.model_prob*100:.1f}%"
        if not best_ind.has_market:
            ev_s += " (sin mercado)"
        lines.append(f"  Mejor individual:  {best_ind.label} ({ev_s})")

    if best_combo:
        combo_ev_s = f"Prob {best_combo.combo_prob*100:.1f}% | score {best_combo.score}/100"
        lines.append(f"  Mejor combinación: {best_combo.leg1_label} + {best_combo.leg2_label} ({combo_ev_s})")

    lines.append("")

    # Recommendation logic — DC STRONG_BET > combo STRONG_BET > individual EV > combo fallback
    rec_divergence_lines: list[str] = []
    rec_filter_lines: list[str] = []
    best_dc = next((p for p in (dc_picks or []) if p.is_primary), None)

    if best_dc and best_dc.decision == "STRONG_BET" and best_dc.market_odds > 1:
        rec_label = best_dc.label
        rec_reasons = [
            f"Probabilidad: {best_dc.model_prob*100:.1f}%",
            f"EV: {best_dc.ev_pct:+.1f}%",
            f"Riesgo: {best_dc.risk}",
            f"Stake: {best_dc.stake_pct:.1f}% del bankroll",
            "Razón: Mayor seguridad con EV positivo (cubre 2 de 3 resultados)",
        ]
    elif best_combo and best_combo.decision == "STRONG_BET":
        rec_label = f"{best_combo.leg1_label} + {best_combo.leg2_label}"
        rec_prob = best_combo.combo_prob
        rec_stake = _stake_for_combo(best_combo)
        rec_reasons = [
            f"Mayor prob conjunta ({rec_prob*100:.1f}%) con correlación real (Poisson)",
            f"Riesgo {best_combo.risk} — ambas piernas >60% individualmente",
            f"Stake sugerido: {rec_stake:.1f}% del bankroll",
        ]
        if not best_ind or not best_ind.has_market:
            rec_reasons.append("Sin cuotas de mercado — apuesta cuando las tengas")
    elif best_ind and (best_ind.ev_raw_pct or best_ind.ev_pct) > 3.0 and best_ind.has_market:
        from apps.worker.ml.ev_anomaly import kelly_full as _kelly_full, fractional_kelly as _kelly_frac
        from apps.shared.config import get_settings as _get_settings
        _cfg = _get_settings()
        rec_label = best_ind.label
        rec_prob = best_ind.model_prob
        rec_stake = _stake_for_sharp(sharp) or 1.0
        display_ev = best_ind.ev_raw_pct if best_ind.ev_raw_pct is not None else best_ind.ev_pct
        ev_line = f"EV {display_ev:+.1f}%"
        if best_ind.ev_raw_pct is not None and abs(best_ind.ev_raw_pct - best_ind.ev_pct) >= 2.0:
            ev_line += f" (cap EV_MAX_FAIR → decisión {best_ind.ev_pct:+.1f}%)"
        _raw_o = best_ind.market_odds or 0.0
        _kf = _kq = 0.0
        kelly_line = f"Stake sugerido: {rec_stake:.1f}% del bankroll"
        if _raw_o > 1 and rec_prob > 0:
            _kf = _kelly_full(rec_prob, _raw_o)
            _kq = _kelly_frac(rec_prob, _raw_o, _cfg.kelly_fraction)
            kelly_line = (
                f"Kelly {_kf*100:.1f}% → "
                f"¼Kelly {_kq*100:.1f}% → "
                f"stake ajustado {rec_stake:.1f}% bankroll"
            )
        rec_reasons = [
            ev_line,
            f"Cuota mercado: {_odds_tag(best_ind)} | Fair: {best_ind.fair_odds:.2f}",
            kelly_line,
        ]
        # Divergence block
        _mkt_impl = best_ind.market_implied
        _div_limit = getattr(_cfg, "ev_max_model_market_divergence", 0.20)
        rec_divergence_lines = []
        if _mkt_impl is not None:
            _delta_pp = (rec_prob - _mkt_impl) * 100.0
            _within = abs(_delta_pp) <= _div_limit * 100
            _div_icon = "✔" if _within else "⚠"
            rec_divergence_lines = [
                "── Δ modelo–mercado ──",
                f"  Modelo:   {rec_prob*100:.1f}%",
                f"  Mercado:  {_mkt_impl*100:.1f}% (devigged)",
                f"  Δ:        {_delta_pp:+.1f}pp  {_div_icon} (límite {_div_limit*100:.0f}pp)",
            ]
        # Filter checklist
        _ev_ok = display_ev > 0
        _kelly_ok = _kf > 0
        _mkt_ok = best_ind.has_market
        _div_ok = _mkt_impl is None or abs((rec_prob - _mkt_impl) * 100) <= _div_limit * 100
        _sharp_ok = (sharp.sharp_allowed if sharp else None)
        rec_filter_lines = [
            "── Filtros ──",
            f"  {'✓' if _ev_ok else '✗'} EV positivo",
            f"  {'✓' if _kelly_ok else '✗'} Kelly > 0",
            f"  {'✓' if _mkt_ok else '✗'} Cuota mercado disponible",
            f"  {'✓' if _div_ok else '✗'} Divergencia dentro del límite",
            f"  {'✓' if _sharp_ok else ('✗' if _sharp_ok is False else '·')} Sharp gate",
        ]
    elif best_dc:
        rec_label = best_dc.label
        rec_reasons = [
            f"Probabilidad: {best_dc.model_prob*100:.1f}% | Riesgo: {best_dc.risk}",
            "Sin cuotas de mercado — apuesta X2/1X cuando tengas precio",
            f"Fair odds modelo: [{best_dc.fair_odds:.2f}]",
        ]
    elif best_combo:
        rec_label = f"{best_combo.leg1_label} + {best_combo.leg2_label}"
        rec_reasons = [
            f"Combinación moderada: {best_combo.combo_prob*100:.1f}% prob conjunta",
            f"Riesgo {best_combo.risk} | Stake: {_stake_for_combo(best_combo):.1f}%",
        ]
    else:
        lines.append("🎯 RECOMENDACIÓN: Sin ventaja clara — espera mejor precio o siguiente partido.")
        lines.append("")
        lines.append("⚠️ Prob Poisson · cuotas devigged · no garantiza retorno.")
        return "\n".join(lines)

    lines.append(f"🎯 RECOMENDACIÓN FINAL: {rec_label}")
    for r in rec_reasons:
        lines.append(f"   ✅ {r}")

    if rec_divergence_lines:
        lines.append("")
        lines.extend(rec_divergence_lines)

    if rec_filter_lines:
        lines.append("")
        lines.extend(rec_filter_lines)

    # ── Model confidence block (always shown when sharp result available) ──────
    if sharp:
        conf = sharp.decision.confidence_score
        mds = sharp.mds
        trust = sharp.decision.trust
        # When the decision tree exits before computing confidence (early NO_BET
        # or sharp gate), conf=0 is meaningless — show N/D instead of "0/100 (Baja)"
        if conf == 0 and not sharp.sharp_allowed:
            confidence_lines = [
                "",
                "── Confianza del modelo ──",
                f"  Confianza:  N/D (gate bloqueó pre-scoring)",
                f"  MDS:        {mds}/100",
            ]
        else:
            conf_label = (
                "Muy alta" if conf >= 80 else
                "Alta"     if conf >= 65 else
                "Media"    if conf >= 50 else
                "Baja"
            )
            trust_line = ""
            if trust:
                trust_line = f" · fuente {trust.trust_side}"
            confidence_lines = [
                "",
                "── Confianza del modelo ──",
                f"  Confianza:  {conf}/100 ({conf_label}){trust_line}",
                f"  MDS:        {mds}/100",
            ]
        try:
            from apps.worker.ml.model_learning import load_learning_state as _lls
            ls = _lls()
            brier = ls.rolling_brier
            if brier is not None:
                brier_label = (
                    "Excelente" if brier < 0.20 else
                    "Buena"     if brier < 0.25 else
                    "Aceptable" if brier < 0.30 else
                    "A mejorar"
                )
                confidence_lines.append(
                    f"  Brier vivo: {brier:.3f} ({brier_label}) · N={ls.rolling_brier_n}"
                )
            if ls.rolling_clv is not None:
                clv_icon = "✔" if ls.rolling_clv >= 0 else "⚠"
                confidence_lines.append(
                    f"  CLV vivo:   {ls.rolling_clv:+.3f} {clv_icon} · N={ls.rolling_clv_n}"
                )
        except Exception:
            pass
        lines.extend(confidence_lines)

    lines += [
        "",
        "⚠️ Prob Poisson · cuotas devigged · no garantiza retorno.",
    ]

    return "\n".join(lines)


# ─── Main entry ───────────────────────────────────────────────────────────────

def build_betting_menu(
    analysis: MatchAnalysis,
    ev_opps: list[EvOpportunity],
    sharp: SharpBetResult | None,
    odds_event: dict | None,
    live_result: "LivePoissonResult | None" = None,
) -> str:
    """
    Build the full betting menu text for a given match bundle.

    When live_result is provided (match currently in progress):
    - Markets are recomputed conditioned on current score + time remaining
    - Combinations use build_live_combinations() (score-aware Poisson conditions)
    - A live banner shows current score, minute, and intensity signals
    """
    model = analysis.model
    if not model:
        return f"{terminal_header()}\n\nModelo no disponible para {analysis.team1} vs {analysis.team2}."

    # Use live-conditioned markets when match is in progress
    display_model = model
    if live_result is not None:
        from apps.api.services.worldcup_engine import live_result_to_model_markets
        display_model = live_result_to_model_markets(live_result, blend_meta=model.blend_meta)

    dc_picks = evaluate_dc(display_model, ev_opps, analysis.team1, analysis.team2)
    market_rows = _build_market_rows(
        display_model, analysis.team1, analysis.team2, odds_event, ev_opps
    )

    if live_result is not None:
        combos = build_live_combinations(live_result, analysis.team1, analysis.team2)
    else:
        combos = build_safe_combinations(display_model, analysis.team1, analysis.team2)

    # Audit log — runs silently, never raises
    from apps.api.services.discard_log import log_discards
    log_discards(
        match=f"{analysis.team1} vs {analysis.team2}",
        fecha=(analysis.fecha or "")[:10] or None,
        ronda=analysis.ronda or None,
        market_rows=market_rows,
        sharp=sharp,
    )

    return format_betting_menu(
        analysis=analysis,
        market_rows=market_rows,
        combos=combos,
        dc_picks=dc_picks,
        sharp=sharp,
        live_result=live_result,
    )
