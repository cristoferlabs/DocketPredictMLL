"""
xG-based Lambda Estimator — reemplaza goles históricos como proxy.

Fórmula base (configurable):
    λ = 0.55 * xG_for + 0.25 * shots_adj + 0.20 * possession_adj

    xG_for       = xG generado por partido (StatsBomb events)
    shots_adj    = shots_per_game * WC_XG_PER_SHOT  (convierte volumen a xG equiv)
    possession_adj = xG_for * (possession_pct / 0.50)  (ajuste posesión)

WC Tournament Scaler:
    xG de torneos WC es generalmente 10-15% menor que ligas de clubes
    (defensas más organizadas, prioridad táctica más alta)
    WC_SCALER = 0.92  (configurable)
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Constantes calibradas en WC2018+2022 (StatsBomb) ─────────────────────────
WC_AVG_XG_PER_GAME: float = 1.18    # promedio xG generado por equipo por partido WC
WC_AVG_SHOTS_PER_GAME: float = 12.5 # tiros por equipo por partido WC
WC_XG_PER_SHOT: float = WC_AVG_XG_PER_GAME / WC_AVG_SHOTS_PER_GAME  # ≈ 0.094
WC_SCALER: float = 1.00             # xG StatsBomb WC ya es WC-level, sin ajuste adicional

# Pesos de la fórmula λ
W_XG_FOR: float = 0.55
W_SHOTS: float = 0.25
W_POSSESSION: float = 0.20


@dataclass(frozen=True)
class LambdaXGEstimate:
    """Resultado del estimador xG-based."""
    lambda_value: float
    xg_for: float
    shots_adj: float
    possession_adj: float
    source: str          # "statsbomb_wc", "statsbomb_blend", "fallback_goals"
    n_matches: int


def estimate_lambda_from_xg_profile(
    xg_profile: dict[str, float] | None,
    *,
    fallback_goals: float | None = None,
    understat_xg: float | None = None,
    wc_scaler: float = WC_SCALER,
    w_xg: float = W_XG_FOR,
    w_shots: float = W_SHOTS,
    w_possession: float = W_POSSESSION,
    min_matches_for_primary: int = 2,
) -> LambdaXGEstimate:
    """
    Estima λ usando el perfil xG de StatsBomb.

    Cascada de fuentes (de mayor a menor confianza):
    1. StatsBomb WC xG (si n_matches >= min_matches_for_primary)
    2. Understat season xG (club, si disponible)
    3. Goles históricos WC (fallback actual)

    Args:
        xg_profile: salida de compute_team_xg_profile() — puede ser None
        fallback_goals: promedio de goles WC históricos del equipo
        understat_xg: xG/partido de Understat (temporada de clubes)
        wc_scaler: factor de escala WC vs club football
        min_matches_for_primary: mínimo de partidos para confiar en xG de WC

    Returns:
        LambdaXGEstimate con λ calculado y metadata de source
    """
    # ── Fuente 1: StatsBomb WC xG (in-tournament, más fiable) ────────────────
    if xg_profile and xg_profile.get("n_matches", 0) >= min_matches_for_primary:
        xg_for = xg_profile["xg_per_game"]
        shots = xg_profile["shots_per_game"]
        possession = xg_profile["possession_pct"]

        # Componente de volumen de tiros (normalizado a escala xG)
        shots_adj = shots * WC_XG_PER_SHOT  # tiros × xG_promedio_por_tiro

        # Componente de posesión ajustada
        # Más posesión = más oportunidades, pero con rendimiento decreciente
        poss_ratio = possession / 0.50  # ratio vs posesión neutra (50%)
        possession_adj = xg_for * poss_ratio

        # Fórmula compuesta
        raw_lambda = (
            w_xg * xg_for +
            w_shots * shots_adj +
            w_possession * possession_adj
        )

        lam = round(raw_lambda * wc_scaler, 3)
        return LambdaXGEstimate(
            lambda_value=max(0.3, lam),
            xg_for=xg_for,
            shots_adj=shots_adj,
            possession_adj=possession_adj,
            source="statsbomb_wc",
            n_matches=xg_profile["n_matches"],
        )

    # ── Fuente 2: Understat season xG (pre-torneo, datos de clubes) ──────────
    if understat_xg is not None and understat_xg > 0:
        # Club xG → WC: aplicar scaler (WC más defensivo)
        lam = round(understat_xg * wc_scaler, 3)

        # Si hay 1 partido WC, blendear suavemente
        if xg_profile and xg_profile.get("n_matches", 0) >= 1:
            wc_xg = xg_profile["xg_per_game"]
            lam = round(wc_xg * 0.4 * wc_scaler + understat_xg * 0.6 * wc_scaler, 3)
            source = "statsbomb_blend"
        else:
            source = "understat_season"

        return LambdaXGEstimate(
            lambda_value=max(0.3, lam),
            xg_for=understat_xg,
            shots_adj=0.0,
            possession_adj=0.0,
            source=source,
            n_matches=xg_profile.get("n_matches", 0) if xg_profile else 0,
        )

    # ── Fuente 3: Fallback — goles históricos WC (comportamiento anterior) ────
    lam = fallback_goals if fallback_goals and fallback_goals > 0 else WC_AVG_XG_PER_GAME
    return LambdaXGEstimate(
        lambda_value=max(0.3, lam),
        xg_for=lam,
        shots_adj=0.0,
        possession_adj=0.0,
        source="fallback_goals",
        n_matches=0,
    )


def xg_based_defense_strength(
    xg_profile: dict[str, float] | None,
    *,
    fallback_ga: float | None = None,
    wc_avg_xg_against: float = WC_AVG_XG_PER_GAME,
    elo_adj: float = 1.0,
) -> float:
    """
    Factor de resistencia defensiva basado en xG concedido.

    Reemplaza rival_defense_strength cuando hay datos StatsBomb.
    Valor > 1.0 = defensa débil (concede más que promedio WC)
    Valor < 1.0 = defensa fuerte (concede menos que promedio WC)
    Clamp: [0.70, 1.40]
    """
    if xg_profile and xg_profile.get("n_matches", 0) >= 2:
        xg_against = xg_profile["xg_against_per_game"]
        base = xg_against / wc_avg_xg_against
        return round(max(0.70, min(1.40, base * elo_adj)), 3)

    # Fallback: usa goles concedidos históricos (comportamiento actual)
    if fallback_ga and fallback_ga > 0:
        base = fallback_ga / wc_avg_xg_against
        return round(max(0.70, min(1.40, base * elo_adj)), 3)

    return 1.0
