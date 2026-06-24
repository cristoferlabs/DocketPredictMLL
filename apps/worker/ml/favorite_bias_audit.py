"""
Favorite Bias Audit — P_model vs P_market.

Detecta:
- favorite compression (modelo subestima favoritos >60%)
- draw inflation (modelo sobreestima empates)
- underdog inflation (modelo sobreestima underdogs)

No modifica motores; solo análisis diagnóstico.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FAVORITE_THRESHOLD = 0.60
DRAW_BAND = "draw"
UNDERDOG_MAX = 0.40


@dataclass(frozen=True)
class OutcomeBias:
    selection: str
    p_model: float
    p_market: float
    delta: float  # model - market
    bucket: str  # favorite | medium | underdog | draw


@dataclass
class MatchBiasRow:
    team1: str
    team2: str
    outcomes: list[OutcomeBias]
    favorite_compression: float = 0.0
    draw_inflation: float = 0.0
    underdog_inflation: float = 0.0


@dataclass
class FavoriteBiasAudit:
    n_matches: int
    n_outcomes: int
    favorite_bias_score: float
    favorite_compression_avg: float
    draw_inflation_avg: float
    underdog_inflation_avg: float
    bias_detected: str
    magnitude: str
    probable_cause: str
    calibration_proposal: str
    rows: list[MatchBiasRow] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


def _bucket(selection: str, p_model: float, p_market: float) -> str:
    if selection == "Empate" or selection.lower() == "draw":
        return DRAW_BAND
    p_ref = max(p_model, p_market)
    if p_ref >= FAVORITE_THRESHOLD:
        return "favorite"
    if p_ref < UNDERDOG_MAX:
        return "underdog"
    return "medium"


def _match_row(
    team1: str,
    team2: str,
    outcomes: list[tuple[str, float, float | None]],
) -> MatchBiasRow | None:
    """outcomes: (selection, p_model, p_market|None)"""
    obs: list[OutcomeBias] = []
    fav_comp: list[float] = []
    draw_inf: list[float] = []
    dog_inf: list[float] = []

    for sel, pm, pmk in outcomes:
        if pmk is None or pmk <= 0:
            continue
        delta = pm - pmk
        bucket = _bucket(sel, pm, pmk)
        obs.append(OutcomeBias(sel, pm, pmk, delta, bucket))

        if bucket == "favorite":
            # compression: market > model on favorites
            fav_comp.append(pmk - pm)
        elif bucket == DRAW_BAND:
            draw_inf.append(pm - pmk)
        elif bucket == "underdog":
            dog_inf.append(pm - pmk)

    if not obs:
        return None

    return MatchBiasRow(
        team1=team1,
        team2=team2,
        outcomes=obs,
        favorite_compression=sum(fav_comp) / len(fav_comp) if fav_comp else 0.0,
        draw_inflation=sum(draw_inf) / len(draw_inf) if draw_inf else 0.0,
        underdog_inflation=sum(dog_inf) / len(dog_inf) if dog_inf else 0.0,
    )


def aggregate_favorite_bias(rows: list[MatchBiasRow]) -> FavoriteBiasAudit:
    if not rows:
        return FavoriteBiasAudit(
            n_matches=0,
            n_outcomes=0,
            favorite_bias_score=0.0,
            favorite_compression_avg=0.0,
            draw_inflation_avg=0.0,
            underdog_inflation_avg=0.0,
            bias_detected="sin datos",
            magnitude="n/a",
            probable_cause="Sin partidos con cuotas mercado",
            calibration_proposal="Recolectar snapshots de mercado",
        )

    fav_comp = [r.favorite_compression for r in rows if r.favorite_compression != 0]
    draw_inf = [r.draw_inflation for r in rows if r.draw_inflation != 0]
    dog_inf = [r.underdog_inflation for r in rows if r.underdog_inflation != 0]

    fav_avg = sum(fav_comp) / len(fav_comp) if fav_comp else 0.0
    draw_avg = sum(draw_inf) / len(draw_inf) if draw_inf else 0.0
    dog_avg = sum(dog_inf) / len(dog_inf) if dog_inf else 0.0

    # favorite_bias_score: positivo = mercado más confiado en favoritos que modelo
    # rango aprox [-1, 1]
    score = fav_avg - 0.5 * draw_avg - 0.5 * dog_avg
    score = max(-1.0, min(1.0, score * 2.5))

    n_out = sum(len(r.outcomes) for r in rows)

    biases: list[str] = []
    if fav_avg > 0.05:
        biases.append("favorite_compression")
    if draw_avg > 0.03:
        biases.append("draw_inflation")
    if dog_avg > 0.04:
        biases.append("underdog_inflation")
    if fav_avg < -0.03:
        biases.append("favorite_overconfidence")

    bias_detected = " + ".join(biases) if biases else "alineado (sin sesgo fuerte)"

    abs_max = max(abs(fav_avg), abs(draw_avg), abs(dog_avg))
    if abs_max >= 0.12:
        magnitude = "alta"
    elif abs_max >= 0.06:
        magnitude = "moderada"
    elif abs_max >= 0.03:
        magnitude = "baja"
    else:
        magnitude = "despreciable"

    cause, proposal = _diagnose(fav_avg, draw_avg, dog_avg, score)

    all_deltas = [o.delta for r in rows for o in r.outcomes]
    fav_deltas = [o.delta for r in rows for o in r.outcomes if o.bucket == "favorite"]
    draw_deltas = [o.delta for r in rows for o in r.outcomes if o.bucket == DRAW_BAND]
    dog_deltas = [o.delta for r in rows for o in r.outcomes if o.bucket == "underdog"]

    return FavoriteBiasAudit(
        n_matches=len(rows),
        n_outcomes=n_out,
        favorite_bias_score=round(score, 4),
        favorite_compression_avg=round(fav_avg, 4),
        draw_inflation_avg=round(draw_avg, 4),
        underdog_inflation_avg=round(dog_avg, 4),
        bias_detected=bias_detected,
        magnitude=magnitude,
        probable_cause=cause,
        calibration_proposal=proposal,
        rows=rows,
        details={
            "mean_delta_all": round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0,
            "mean_delta_favorite": round(sum(fav_deltas) / len(fav_deltas), 4) if fav_deltas else 0,
            "mean_delta_draw": round(sum(draw_deltas) / len(draw_deltas), 4) if draw_deltas else 0,
            "mean_delta_underdog": round(sum(dog_deltas) / len(dog_deltas), 4) if dog_deltas else 0,
            "n_favorite_outcomes": len(fav_deltas),
            "n_draw_outcomes": len(draw_deltas),
            "n_underdog_outcomes": len(dog_deltas),
        },
    )


def _diagnose(fav_avg: float, draw_avg: float, dog_avg: float, score: float) -> tuple[str, str]:
    parts: list[str] = []
    if fav_avg > 0.05:
        parts.append(
            "Blend Poisson+ELO (60/40) + draw base Poisson comprimen favoritos fuertes; "
            "el mercado incorpora información táctica/informacional que el modelo no ve."
        )
    if draw_avg > 0.03:
        parts.append(
            "Poisson independiente sobreestima empates en partidos desequilibrados; "
            "ELO draw ~28% fijo empuja masa probabilística al empate."
        )
    if dog_avg > 0.04:
        parts.append(
            "Lambdas/xG de selecciones débiles infladas por histórico WC parcial o "
            "host boost; el modelo asigna más P a underdogs de lo que el mercado liquida."
        )
    if not parts:
        if score > 0.02:
            parts.append("Ligera compresión de favoritos vs mercado.")
        elif score < -0.02:
            parts.append("Modelo ligeramente más agresivo que mercado en favoritos.")
        else:
            parts.append("Distribución 1X2 alineada con mercado en la muestra.")

    proposal_parts: list[str] = []
    if fav_avg > 0.05:
        proposal_parts.append(
            "Calibrar 1X2: factor home_win/away_win >1.0 en favoritos (P_mkt≥60%) "
            "vía isotonic post-hoc o shrink hacia mercado solo en capa calibración."
        )
    if draw_avg > 0.03:
        proposal_parts.append(
            "Reducir draw: factor calibración draw <1.0 (ej. 0.88–0.92) o "
            "draw dampening cuando max(1,X2)>0.55."
        )
    if dog_avg > 0.04:
        proposal_parts.append(
            "Penalizar underdog: factor away_win/home_win <1.0 cuando P_model<40% "
            "y Δ vs mercado >10pp."
        )
    if not proposal_parts:
        proposal_parts.append("Mantener calibración identidad; re-auditar con n≥30 partidos.")

    return " ".join(parts), " | ".join(proposal_parts)


def format_audit_report(audit: FavoriteBiasAudit) -> str:
    lines = [
        "📊 AUDIT FAVORITE BIAS",
        f"Muestra: {audit.n_matches} partidos | {audit.n_outcomes} outcomes",
        "",
        f"favorite_bias_score: {audit.favorite_bias_score:+.3f}",
        f"  (>0 = mercado más confiado en favoritos que modelo)",
        "",
        "── Magnitudes medias (P_model − P_market) ──",
        f"  Favorite compression: {audit.favorite_compression_avg*100:+.1f} pp "
        f"(mercado − modelo en favoritos >60%)",
        f"  Draw inflation:       {audit.draw_inflation_avg*100:+.1f} pp "
        f"(modelo − mercado en empates)",
        f"  Underdog inflation:   {audit.underdog_inflation_avg*100:+.1f} pp "
        f"(modelo − mercado en underdogs <40%)",
        "",
        f"1. Sesgo detectado: {audit.bias_detected}",
        f"2. Magnitud: {audit.magnitude}",
        f"3. Causa probable: {audit.probable_cause}",
        f"4. Calibración propuesta: {audit.calibration_proposal}",
    ]
    if audit.details:
        lines.append("")
        lines.append("── Detalle agregado ──")
        for k, v in audit.details.items():
            if "n_" in k:
                lines.append(f"  {k}: {int(v)}")
            else:
                lines.append(f"  {k}: {v*100:+.2f} pp" if "delta" in k else f"  {k}: {v}")

    if audit.rows:
        lines.append("")
        lines.append("── Top desvíos por partido ──")
        ranked = sorted(
            audit.rows,
            key=lambda r: abs(r.favorite_compression) + abs(r.draw_inflation) + abs(r.underdog_inflation),
            reverse=True,
        )
        for row in ranked[:8]:
            lines.append(f"  {row.team1} vs {row.team2}")
            for o in row.outcomes:
                if abs(o.delta) >= 0.08:
                    lines.append(
                        f"    {o.selection}: model {o.p_model*100:.1f}% | "
                        f"mkt {o.p_market*100:.1f}% | Δ {o.delta*100:+.1f}pp [{o.bucket}]"
                    )
    return "\n".join(lines)
