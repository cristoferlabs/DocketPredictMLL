"""Diagnósticos de partido — filtros pre-decisión para evitar apuestas falsas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.worldcup_engine import ModelMarkets


@dataclass
class MatchDiagnostics:
    """Resultado de todos los chequeos de calidad para un partido."""

    max_ev_pct: float  # EV% real (post-devig), NO gap de probabilidad en pp
    edge_below_threshold: bool  # max EV vs market < min_ev_pct
    is_balanced: bool  # sin favorito claro (p_máx < 45%, gap favorito-away < 15pp)
    btts_ou_inconsistent: bool  # BTTS vs O/U inconsistency detected
    btts_ou_penalty: float  # uncertainty multiplier (0-1)
    variance_class: str  # "high" | "medium" | "low"
    no_bet_signal: bool  # señal NO_BET por calidad general
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _check_balanced_match(
    model: ModelMarkets,
    *,
    max_prob_threshold: float = 0.45,
    min_fav_gap_pp: float = 15.0,
) -> tuple[bool, str, list[str]]:
    """Detecta partidos balanceados tipo 39/33/29 — alta varianza, sin edge.
    
    Criterios:
    - Max probability among home/draw/away < 45%
    - Gap between highest and lowest < 15pp
    """
    probs = [model.home_win, model.draw, model.away_win]
    max_p = max(probs)
    min_p = min(p for p in probs if p > 0)
    gap = (max_p - min_p) * 100

    is_balanced = max_p < max_prob_threshold and gap < min_fav_gap_pp
    variance = "high" if is_balanced else "low"

    flags: list[str] = []
    if is_balanced:
        flags.append(f"balanced_match_pmax={max_p:.0%}_gap={gap:.0f}pp")
    return is_balanced, variance, flags


def _check_btts_ou_consistency(
    model: ModelMarkets,
    *,
    tolerance: float = 0.03,
) -> tuple[bool, float, list[str]]:
    """Verifica consistencia entre BTTS y Over/Under.
    
    BTTS y O/U 2.5 están correlacionados. Si ambos están ~50/50
    simultáneamente, es posible que el modelo no resuelva bien la 
    distribución de goles.
    
    Penalización alta si:
    - BTTS_yes ≈ BTTS_no (diferencia < 10pp) AND
    - Over_25 ≈ Under_25 (diferencia < 10pp)
    - Esto indica ruido modelado — no una distribución real de goles
    
    Returns: (inconsistent, penalty_multiplier, flags)
    """
    flags: list[str] = []

    btts_yes = model.btts_yes
    btts_no = model.btts_no
    over_25 = model.over_25
    under_25 = model.under_25

    btts_gap = abs(btts_yes - btts_no)
    ou_gap = abs(over_25 - under_25)

    # Check if both markets are near 50/50 (equilibrium)
    btts_flat = btts_gap < 0.10
    ou_flat = ou_gap < 0.10

    inconsistent = btts_flat and ou_flat

    # Compute penalty based on inconsistency severity
    penalty = 1.0  # 1.0 = no penalty, lower = more penalty
    if inconsistent:
        avg_flatness = (btts_gap + ou_gap) / 2
        if avg_flatness < 0.05:
            penalty = 0.80  # 20% confidence reduction — mild
            flags.append("btts_ou_both_flat_mild")
        elif avg_flatness < 0.08:
            penalty = 0.90  # 10% reduction
            flags.append("btts_ou_both_flat_mild")
        else:
            penalty = 0.95  # 5% reduction
            flags.append("btts_ou_both_flat_mild")

    # Also check: BTTS_yes + BTTS_no approximately = 1
    prob_sum = btts_yes + btts_no
    if abs(prob_sum - 1.0) > tolerance:
        flags.append(f"btts_sum_{prob_sum:.3f}_off_by_{abs(prob_sum-1):.3f}")
        penalty = min(penalty, 0.90)

    return inconsistent, penalty, flags


def run_match_diagnostics(
    model: ModelMarkets,
    *,
    max_market_ev_pct: float | None = None,
    min_ev_pct: float = 3.0,
) -> MatchDiagnostics:
    """Ejecuta todos los chequeos de diagnóstico para un partido.

    Args:
        model: ModelMarkets del partido
        max_market_ev_pct: EV% real máximo entre outcomes (post-devig, shrink,
            regime cap) — NO un gap de probabilidad en pp. None = sin mercado.
        min_ev_pct: Umbral mínimo de EV% para rechazar (default 3%, alineado
            con settings.ev_min_edge_fair — misma unidad que sharp_engine.py).

    NOTA HISTÓRICA: hasta esta versión, este gate comparaba un gap de
    probabilidad en puntos porcentuales (`model_prob - fair_prob`) contra un
    umbral fijo de 5pp. Esto mezclaba unidades con el resto del pipeline
    (que usa EV% en sharp_engine.py) y penalizaba desproporcionadamente a
    underdogs: para el mismo gap en pp, el EV real es mucho mayor cuanto
    menor es fair_prob (EV = edge_pp / fair_prob). Auditoría confirmó que
    ~81% de los picks con EV%>=3% reales eran rechazados por el gate viejo.
    """
    all_flags: list[str] = []
    penalties: list[float] = []

    # 1. EV threshold check (against market if available) — unidad: EV%
    if max_market_ev_pct is not None:
        max_ev = max_market_ev_pct
        edge_below = max_ev < min_ev_pct
        if edge_below:
            all_flags.append(f"max_market_ev_{max_ev:.1f}pct_below_{min_ev_pct:.0f}pct")
    else:
        max_ev = 0.0
        edge_below = False  # can't determine EV without market
        all_flags.append("no_market_data")

    # 2. Balanced match check
    is_balanced, variance, balanced_flags = _check_balanced_match(model)
    all_flags.extend(balanced_flags)
    if is_balanced and not edge_below:
        all_flags.append("balanced_match_edge_likely_noise")

    # 3. BTTS vs O/U consistency check
    btts_inconsistent, btts_penalty, btts_flags = _check_btts_ou_consistency(model)
    penalties.append(btts_penalty)
    all_flags.extend(btts_flags)

    # Combined no_bet signal — only EV threshold triggers hard NO_BET
    no_bet = edge_below

    return MatchDiagnostics(
        max_ev_pct=round(max_ev, 1),
        edge_below_threshold=edge_below,
        is_balanced=is_balanced,
        btts_ou_inconsistent=btts_inconsistent,
        btts_ou_penalty=round(btts_penalty, 4),
        variance_class=variance,
        no_bet_signal=no_bet,
        flags=all_flags,
        meta={
            "uncertainty_penalty": round(
                min(penalties) if penalties else 1.0, 4
            ),
        },
    )
