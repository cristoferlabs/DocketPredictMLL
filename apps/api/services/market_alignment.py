"""Alineación modelo vs mercado — etiquetas y penalización de confianza."""

from __future__ import annotations

AlignmentKey = str

ALIGNMENT_TIERS: dict[AlignmentKey, tuple[str, str]] = {
    "aligned": ("Alineado", "<5pp entre modelo y mercado en el pick"),
    "mild": ("Ligera discrepancia", "5–10pp — vigilar línea"),
    "divergence": ("Divergencia", "10–15pp — no STRONG_BET automático"),
    "alert": ("Alerta", ">15pp — confiar más en mercado; investigar"),
}


def gap_pp(model_prob: float, market_implied: float | None) -> float:
    if market_implied is None or market_implied <= 0:
        return 0.0
    return abs(model_prob - market_implied) * 100.0


def alignment_status(gap_percentage_points: float) -> tuple[AlignmentKey, str, str]:
    """Devuelve (key, label_es, descripción)."""
    g = gap_percentage_points
    if g < 5.0:
        key = "aligned"
    elif g < 10.0:
        key = "mild"
    elif g < 15.0:
        key = "divergence"
    else:
        key = "alert"
    label, desc = ALIGNMENT_TIERS[key]
    return key, label, desc


def confidence_divergence_penalty(
    gap_percentage_points: float,
    *,
    model_prob: float | None = None,
    market_implied: float | None = None,
) -> int:
    """Resta puntos de confidence_score por Δ modelo-mercado en el pick."""
    g = gap_percentage_points
    if g < 5.0:
        return 0

    if model_prob is not None and market_implied is not None and market_implied > 0:
        model_inflates_favorite = model_prob > market_implied and model_prob >= 0.52
        market_more_convinced = market_implied > model_prob and market_implied >= 0.58
        if not model_inflates_favorite and not market_more_convinced:
            return 0

    if g < 10.0:
        return 8
    if g < 12.0:
        return 14
    if g < 15.0:
        return 20
    return min(35, int(20 + (g - 15.0) * 1.5))


def max_soft_action_for_gap(gap_percentage_points: float) -> str | None:
    """Tope de acción suave según Δ (None = sin tope extra)."""
    if gap_percentage_points >= 15.0:
        return "WATCH"
    if gap_percentage_points >= 12.0:
        return "WEAK_BET"
    return None


OutlierKey = str

OUTLIER_TIERS: dict[OutlierKey, tuple[str, str]] = {
    "ok": ("OK", "Δ dentro de rango operativo"),
    "outlier": ("MODEL OUTLIER", "Δ >20pp — modelo desacoplado del mercado"),
    "investigate": ("INVESTIGATE", "Δ >25pp — revisar antes de apostar 1X2"),
    "error": ("MODEL ERROR", "Δ >30pp — probable error de calibración 1X2"),
}


def model_outlier_status(
    gap_percentage_points: float,
    *,
    market: str | None = None,
) -> tuple[OutlierKey, str, str, float, str | None]:
    """
    Devuelve (key, label, descripción, confidence_multiplier, forced_action).

    1X2:
      >30pp → NO_BET (MODEL ERROR)
      >25pp → WATCH (INVESTIGATE)
      >20pp → confidence ×0.5
    """
    g = gap_percentage_points
    is_1x2 = market is None or market.upper() in ("1X2", "H2H", "MATCH_WINNER")

    if g > 30.0 and is_1x2:
        label, desc = OUTLIER_TIERS["error"]
        return "error", label, desc, 0.0, "NO_BET"
    if g > 25.0 and is_1x2:
        label, desc = OUTLIER_TIERS["investigate"]
        return "investigate", label, desc, 0.5, "WATCH"
    if g > 30.0:
        label, desc = OUTLIER_TIERS["error"]
        return "error", label, desc, 0.0, "NO_BET"
    if g > 25.0:
        label, desc = OUTLIER_TIERS["investigate"]
        return "investigate", label, desc, 0.5, "WATCH"
    if g > 20.0:
        label, desc = OUTLIER_TIERS["outlier"]
        return "outlier", label, desc, 0.5, None
    label, desc = OUTLIER_TIERS["ok"]
    return "ok", label, desc, 1.0, None


def one_x_two_gap_blocks_bet(gap_percentage_points: float) -> bool:
    """1X2 con Δ>30pp no es apostable."""
    return gap_percentage_points > 30.0
