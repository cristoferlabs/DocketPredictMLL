"""
Política EV — una sola fuente de verdad.

  Raw EV  → informativo (cuota bruta con vig).  EV = p×odds − 1
  Fair EV → decisión (cuota fair devig, gates SHARP/árbol).
"""

from __future__ import annotations

from typing import Any, Literal

EvKind = Literal["raw", "fair"]

# EV fair > 50% con cuota < 4 suele ser Δ modelo-mercado, no value real
MAX_DISPLAY_EV_FAIR_PCT = 35.0
MAX_DISPLAY_EV_RAW_PCT = 45.0
STRUCTURAL_DIVERGENCE_PP = 12.0


def ev_decimal(model_prob: float, odds_decimal: float) -> float:
    """EV canónico: (p_model × odds) − 1."""
    if odds_decimal <= 1.0 or model_prob <= 0:
        return 0.0
    return round(model_prob * odds_decimal - 1.0, 6)


def ev_calibrated(prob: float, odds: float) -> float:
    """EV con P_calibrated — única fuente para decisión post live_calibration."""
    return ev_decimal(prob, odds)


def ev_percent(model_prob: float, odds_decimal: float) -> float:
    return round(ev_decimal(model_prob, odds_decimal) * 100.0, 2)


def ev_for_decision(
    *,
    ev_fair: float,
    ev_raw: float | None = None,
    alpha_regime: str | None = None,
    settings: Any | None = None,
) -> float:
    """EV usado en gates, stake y árbol — con clamp estructural por régimen."""
    from apps.shared.config import Settings, get_settings

    s = settings or get_settings()
    if not getattr(s, "ev_regime_clamp_enabled", True):
        return ev_fair
    cap = regime_ev_cap(alpha_regime, settings=s)
    return min(ev_fair, cap)


def regime_ev_cap(alpha_regime: str | None, *, settings: Any | None = None) -> float:
    """Tope EV decimal por régimen de calibración."""
    from apps.shared.config import Settings, get_settings

    s = settings or get_settings()
    caps = {
        "aligned": s.ev_cap_regime_aligned,
        "moderate": s.ev_cap_regime_moderate,
        "high": s.ev_cap_regime_high,
        "extreme": s.ev_cap_regime_extreme,
    }
    return float(caps.get(alpha_regime or "", s.ev_cap_regime_default))


def clamp_ev_by_regime(
    ev_decimal: float,
    alpha_regime: str | None,
    *,
    settings: Any | None = None,
) -> tuple[float, bool]:
    """Devuelve (ev_clamped, was_capped)."""
    cap = regime_ev_cap(alpha_regime, settings=settings)
    if ev_decimal > cap:
        return cap, True
    return ev_decimal, False


def edge_for_decision(*, edge_fair: float, edge_raw: float | None = None) -> float:
    """Edge usado en gates y ranking de picks (decimal)."""
    return edge_fair


def is_structural_mismatch(
    model_prob: float,
    market_implied: float | None,
    *,
    divergence: float | None = None,
    threshold_pp: float = STRUCTURAL_DIVERGENCE_PP,
) -> bool:
    """Modelo y mercado demasiado separados — EV raw/fair engañoso."""
    if market_implied is None or market_implied <= 0:
        return False
    gap_pp = abs(model_prob - market_implied) * 100.0
    if gap_pp >= threshold_pp:
        return True
    if divergence is not None and divergence >= threshold_pp / 100.0:
        return True
    return False


def is_actionable_value(
    ev_fair_pct: float,
    model_prob: float,
    market_implied: float | None,
    *,
    min_ev_fair_pct: float = 3.0,
    divergence: float | None = None,
) -> bool:
    """Value mostrable para decisión — no underdog inflado vs mercado."""
    if ev_fair_pct < min_ev_fair_pct:
        return False
    if is_structural_mismatch(model_prob, market_implied, divergence=divergence):
        return False
    if ev_fair_pct > MAX_DISPLAY_EV_FAIR_PCT:
        return False
    return True


def is_ev_display_capped(ev_pct: float, *, is_raw: bool = False) -> bool:
    cap = MAX_DISPLAY_EV_RAW_PCT if is_raw else MAX_DISPLAY_EV_FAIR_PCT
    return abs(ev_pct) > cap + 0.01


def clamp_display_ev(ev_pct: float, *, is_raw: bool = False) -> float:
    cap = MAX_DISPLAY_EV_RAW_PCT if is_raw else MAX_DISPLAY_EV_FAIR_PCT
    return max(-cap, min(cap, ev_pct))


def format_ev_display(
    *,
    ev_fair_pct: float,
    ev_raw_pct: float | None = None,
    odds_decimal: float | None = None,
    model_prob: float | None = None,
    market_implied: float | None = None,
    divergence: float | None = None,
    market: str | None = None,
) -> str:
    """Texto Telegram: EV calculado; si hay cap o Δ estructural, no ocultar el valor real."""
    structural = is_structural_mismatch(
        model_prob or 0, market_implied, divergence=divergence
    )
    fair_capped = clamp_display_ev(ev_fair_pct)
    fair_capped_flag = is_ev_display_capped(ev_fair_pct)

    if structural or fair_capped_flag:
        parts = [f"EV calc {ev_fair_pct:+.1f}%"]
        if fair_capped_flag:
            parts.append(f"(tope visual {fair_capped:+.1f}%)")
    else:
        parts = [f"EV fair {fair_capped:+.1f}%"]

    if odds_decimal and odds_decimal > 1:
        parts.append(f"@ {odds_decimal:.2f}")
    if model_prob is not None and market_implied is not None:
        edge_pp = (model_prob - market_implied) * 100.0
        parts.append(f"Δ mkt {edge_pp:+.1f}pp")

    if structural:
        if market == "1X2" or market is None:
            gap = gap_pp_from_probs(model_prob or 0, market_implied)
            if gap > 30:
                parts.append("(1X2 Δ>30pp — NO BET)")
            elif gap > 25:
                parts.append("(1X2 Δ>25pp — INVESTIGATE)")
            else:
                parts.append("(Δ estructural — no value)")
        else:
            parts.append("(Δ estructural — no value 1X2)")
        return " ".join(parts)

    if ev_raw_pct is not None and abs(ev_raw_pct - ev_fair_pct) > 0.5:
        raw_capped = clamp_display_ev(ev_raw_pct, is_raw=True)
        if is_ev_display_capped(ev_raw_pct, is_raw=True):
            parts.append(f"raw calc {ev_raw_pct:+.1f}% (tope {raw_capped:+.1f}%)")
        else:
            parts.append(f"raw {raw_capped:+.1f}% ref.")
    return " ".join(parts)


def gap_pp_from_probs(model_prob: float, market_implied: float | None) -> float:
    if market_implied is None or market_implied <= 0:
        return 0.0
    return abs(model_prob - market_implied) * 100.0
