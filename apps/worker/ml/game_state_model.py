"""
Game State Model — Ajuste pre-partido de λ para dinámicas esperadas de juego.

Problema central: Poisson asume tasas de gol CONSTANTES e INDEPENDIENTES.
Fútbol real: equipo que lidera BAJA su λ_ataque (gestión), el que persigue
la SUBE (presión). Integrar sobre este efecto produce un λ corregido
más realista sin necesitar datos en tiempo real.

Correcciones de sesgo confirmadas experimentalmente:
  1. Away win sobreestimado (-29pp): favoritos visitantes tienen λ_away MENOR
     de lo que Poisson predice porque gestionan el juego tras marcar.
  2. Over/Under sobreconfiado: partidos cerrados tienen más varianza real
     que Poisson → comprimir probabilidades extremas.

Uso en pre-partido:
    lam_h, lam_a = adjust_lambdas_for_flow(lambda_home, lambda_away)
    # Pasar lam_h, lam_a a compute_match_lambdas / build_score_matrix
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Parámetros calibrables ──────────────────────────────────────────────
# Todos los factores tienen valores conservadores por defecto.
# Recalibrar con ablation sobre WC2018+2022 cuando haya suficiente data.

_MGMT_FACTOR: float = 0.88    # el equipo que lidera reduce λ_ataque 12%
_CHASE_FACTOR: float = 1.14   # el equipo que persigue aumenta λ_ataque 14%
_BLEND: float = 0.35          # peso del ajuste vs λ base (35% ajuste, 65% base)
_MIN_IMBALANCE: float = 0.15  # ajuste solo si |lam_h - lam_a| > umbral


@dataclass(frozen=True)
class GameStateAdjustment:
    """Resultado del ajuste de λ con metadata de diagnóstico."""
    lam_home_adj: float
    lam_away_adj: float
    lam_home_base: float
    lam_away_base: float
    p_home_leads_first: float
    p_away_leads_first: float
    applied: bool

    @property
    def delta_home(self) -> float:
        return round(self.lam_home_adj - self.lam_home_base, 4)

    @property
    def delta_away(self) -> float:
        return round(self.lam_away_adj - self.lam_away_base, 4)

    @property
    def delta_total(self) -> float:
        total_adj = self.lam_home_adj + self.lam_away_adj
        total_base = self.lam_home_base + self.lam_away_base
        return round(total_adj - total_base, 4)


def adjust_lambdas_for_flow(
    lam_home: float,
    lam_away: float,
    *,
    mgmt_factor: float = _MGMT_FACTOR,
    chase_factor: float = _CHASE_FACTOR,
    blend: float = _BLEND,
    min_imbalance: float = _MIN_IMBALANCE,
) -> GameStateAdjustment:
    """
    Ajusta λ pre-partido integrando sobre escenarios de estado de juego.

    Mecanismo:
    ──────────
    1. P(local marca primero) = lam_home / (lam_home + lam_away)   [Poisson carrera]
    2. Si local lidera:  lam_home × mgmt_factor, lam_away × chase_factor
       Si visitante lidera: lam_away × mgmt_factor, lam_home × chase_factor
    3. λ ajustado = promedio ponderado de los dos escenarios
    4. λ final = blend × λ_ajustado + (1-blend) × λ_base  (conservador)

    Por qué funciona:
    ─────────────────
    Cuando lam_away >> lam_home (favorito visitante):
      - P(visitante lidera primero) es alta (~0.7+)
      - En ese estado, lam_away × 0.88 (gestiona) y lam_home × 1.14 (presiona)
      - Resultado: λ_away baja, λ_home sube → P(visitante gana) baja
      - Corrige el sesgo de -29pp en victorias visitante

    Cuando lam_total alto (partido abierto):
      - Ambos escenarios tienen λ similares → cambio mínimo
      - Total puede subir ligeramente (ambos equipos más activos tras gol)

    Args:
        lam_home: Lambda goles local (base, de wc_features.py)
        lam_away: Lambda goles visitante (base)
        mgmt_factor: Reducción λ del equipo que lidera
        chase_factor: Aumento λ del equipo que persigue
        blend: Peso del escenario ajustado en el mix final [0, 1]
        min_imbalance: Solo aplica si |lam_h - lam_a| > este valor
                       (no ajusta partidos muy equilibrados)

    Returns:
        GameStateAdjustment con λ ajustados y metadata
    """
    lam_total = lam_home + lam_away
    imbalance = abs(lam_home - lam_away)

    # No aplicar en partidos muy equilibrados (señal demasiado débil)
    if lam_total <= 0.01 or imbalance < min_imbalance:
        return GameStateAdjustment(
            lam_home_adj=lam_home,
            lam_away_adj=lam_away,
            lam_home_base=lam_home,
            lam_away_base=lam_away,
            p_home_leads_first=lam_home / max(0.01, lam_total),
            p_away_leads_first=lam_away / max(0.01, lam_total),
            applied=False,
        )

    # P(equipo X es el primero en marcar) — modelo de carrera exponencial
    p_home_first = lam_home / lam_total
    p_away_first = lam_away / lam_total

    # Escenario A: local marca primero → local gestiona, visitante presiona
    lam_h_if_home_leads = lam_home * mgmt_factor
    lam_a_if_home_leads = lam_away * chase_factor

    # Escenario B: visitante marca primero → visitante gestiona, local presiona
    lam_h_if_away_leads = lam_home * chase_factor
    lam_a_if_away_leads = lam_away * mgmt_factor

    # Expectativa ponderada por probabilidad de cada escenario
    lam_h_expected = p_home_first * lam_h_if_home_leads + p_away_first * lam_h_if_away_leads
    lam_a_expected = p_home_first * lam_a_if_home_leads + p_away_first * lam_a_if_away_leads

    # Mix conservador: blend del ajustado + (1-blend) del original
    lam_h_final = lam_home * (1.0 - blend) + lam_h_expected * blend
    lam_a_final = lam_away * (1.0 - blend) + lam_a_expected * blend

    return GameStateAdjustment(
        lam_home_adj=round(max(0.5, lam_h_final), 3),
        lam_away_adj=round(max(0.5, lam_a_final), 3),
        lam_home_base=lam_home,
        lam_away_base=lam_away,
        p_home_leads_first=round(p_home_first, 3),
        p_away_leads_first=round(p_away_first, 3),
        applied=True,
    )


def compress_extreme_ou_prob(
    prob: float,
    lambda_total: float,
    *,
    compression_alpha: float = 0.12,
    threshold_lambda: float = 2.2,
) -> float:
    """
    Comprime probabilidades extremas de Over/Under hacia 0.50.

    Motivación: Poisson es subdispersado vs fútbol real. Los partidos
    tienen más varianza real que la distribución de Poisson predice,
    lo que significa que probabilidades muy altas (>65%) de Over o Under
    están sistémicamente sobreestimadas.

    Auditoría experimental confirmó:
      Under 2.5 a prob>0.60: modelo dice 70.8% pero real es 56.1% → -14.7pp
      Over  2.5 a prob>0.60: modelo dice 67.9% pero real es 52.4% → -15.6pp

    La compresión corrige esto de forma proporcional: mayor desviación
    de 0.50 → mayor corrección (no afecta probs ya cercanas a 0.50).

    Args:
        prob: Probabilidad O/U original del modelo [0, 1]
        lambda_total: lam_home + lam_away (mayor λ → más confianza del modelo)
        compression_alpha: Fracción de compresión [0, 0.3]
        threshold_lambda: λ_total mínimo para aplicar compresión (partidos abiertos)

    Returns:
        Probabilidad comprimida
    """
    # No comprimir si λ total es bajo (partidos cerrados son genuinamente predecibles)
    if lambda_total < threshold_lambda:
        return prob

    # Solo comprimir si la prob es extrema (lejos de 0.5)
    deviation = abs(prob - 0.5)
    if deviation < 0.10:
        return prob

    # Compresión proporcional a la desviación de 0.5
    direction = 1.0 if prob > 0.5 else -1.0
    compressed = 0.5 + direction * deviation * (1.0 - compression_alpha)
    return round(max(0.01, min(0.99, compressed)), 4)
