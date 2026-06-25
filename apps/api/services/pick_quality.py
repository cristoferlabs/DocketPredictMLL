"""Métricas de calidad del pick — CLV esperado y calibration score (display)."""

from __future__ import annotations

from apps.worker.ml.calibration_metrics import load_fitted_model_weights


def calibration_score_for_prob(
    model_prob: float,
    *,
    market: str = "1X2",
) -> float | None:
    """
    Score 0–1: qué tan bien calibrado está el modelo en este rango de prob (histórico WC).

    Usa reliability bins del fit Poisson/ELO (train 2018+2022).
    """
    if model_prob <= 0 or model_prob >= 1:
        return None

    weights = load_fitted_model_weights() or {}
    report = weights.get("test") or weights.get("train") or {}
    bins = report.get("reliability_bins") or []
    if not bins:
        ece = report.get("ece_max_prob")
        if ece is not None:
            return round(max(0.0, min(1.0, 1.0 - float(ece))), 2)
        return None

    hit_bin = None
    for b in bins:
        lo = float(b.get("lo", 0))
        hi = float(b.get("hi", 1))
        if lo <= model_prob < hi or (model_prob >= hi and hi >= 0.99):
            hit_bin = b
            break
    if not hit_bin:
        return None

    count = int(hit_bin.get("count") or 0)
    mean_pred = float(hit_bin.get("mean_pred") or 0)
    mean_out = float(hit_bin.get("mean_outcome") or 0)
    if count < 2:
        ece = report.get("ece_max_prob")
        if ece is not None:
            return round(max(0.0, min(1.0, 1.0 - float(ece))), 2)
        return None

    gap = abs(mean_pred - mean_out)
    local = max(0.0, min(1.0, 1.0 - gap * 2.5))
    global_ece = float(report.get("ece_max_prob") or 0.08)
    global_score = max(0.0, min(1.0, 1.0 - global_ece))
    weight = min(1.0, count / 12.0)
    return round(local * weight + global_score * (1.0 - weight), 2)


def expected_clv_movement_pp(
    *,
    model_prob: float,
    market_implied: float | None,
    ev_fair: float,
    gap_pp: float | None = None,
) -> float | None:
    """
    CLV esperado vs cierre (heurística).

    Δ≈0 → movimiento de línea bajo; edge moderado → CLV positivo acotado.
    """
    if market_implied is None or market_implied <= 0:
        return None

    gap = gap_pp if gap_pp is not None else abs(model_prob - market_implied) * 100.0
    edge_pp = (model_prob - market_implied) * 100.0

    if gap < 3.0:
        return round(max(-0.5, min(1.5, ev_fair * 100 * 0.12)), 2)

    if edge_pp > 0 and ev_fair > 0:
        raw = edge_pp * 0.22 + ev_fair * 100 * 0.18
        return round(max(0.0, min(8.0, raw)), 2)

    if edge_pp < -5:
        return round(max(-6.0, edge_pp * 0.12), 2)

    return round(ev_fair * 100 * 0.08, 2)


def format_pick_quality_lines(
    *,
    model_prob: float,
    market_implied: float | None,
    ev_fair: float,
    gap_pp: float | None = None,
    market: str = "1X2",
) -> list[str]:
    """Líneas Telegram para CLV esperado + calibration score."""
    lines: list[str] = []
    clv = expected_clv_movement_pp(
        model_prob=model_prob,
        market_implied=market_implied,
        ev_fair=ev_fair,
        gap_pp=gap_pp,
    )
    cal = calibration_score_for_prob(model_prob, market=market)
    if clv is not None:
        sign = "+" if clv >= 0 else ""
        lines.append(f"CLV esperado (vs cierre): {sign}{clv:.1f}pp")
    if cal is not None:
        lines.append(f"Calibration score: {cal:.2f} (histórico WC, prob ~{model_prob*100:.0f}%)")
    return lines
