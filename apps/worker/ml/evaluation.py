"""Evaluation utilities for self-improving loop."""

import math
import re
from typing import Any


def resolve_1x2_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def resolve_over_under(home_goals: int, away_goals: int, line: float = 2.5) -> str:
    total = home_goals + away_goals
    return "over" if total > line else "under"


def resolve_btts(home_goals: int, away_goals: int) -> str:
    return "yes" if home_goals > 0 and away_goals > 0 else "no"


def _extract_line(market_type_lower: str, default: float = 2.5) -> float:
    m = re.search(r"(\d+\.?\d*)", market_type_lower)
    return float(m.group(1)) if m else default


def resolve_actual_outcome(market_type: str, home_goals: int, away_goals: int) -> str:
    """
    Acepta tanto claves normalizadas ("1X2", "over_under_2.5") como las
    etiquetas reales que usa producción ("Over/Under 2.5", "Doble Oportunidad").
    """
    mt = (market_type or "").strip().lower()
    if mt == "1x2":
        return resolve_1x2_outcome(home_goals, away_goals)
    if mt.startswith("over/under") or mt.startswith("over_under") or mt == "totals":
        return resolve_over_under(home_goals, away_goals, _extract_line(mt))
    if mt == "btts" or "ambos anotan" in mt:
        return resolve_btts(home_goals, away_goals)
    if "doble oportunidad" in mt or mt in ("dc", "double chance"):
        # DC no tiene una sola etiqueta ganadora — se resuelve en evaluate_prediction()
        return resolve_1x2_outcome(home_goals, away_goals)
    return "unknown"


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def resolve_predicted_key(
    market_type: str,
    predicted_outcome: str,
    *,
    team_home: str | None = None,
    team_away: str | None = None,
) -> str:
    """
    Normaliza el texto de selección guardado en producción (nombre de equipo,
    "Over"/"Under", strings formateados de Doble Oportunidad) a una clave
    comparable con resolve_actual_outcome().

    BUG HISTÓRICO: hasta esta versión, evaluate_prediction() comparaba
    predicted_outcome (texto de display, ej. "Turkey") directamente contra
    actual_outcome (etiqueta normalizada, ej. "home_win") — nunca podían
    coincidir, así que TODA predicción 1X2 se marcaba incorrecta sin importar
    el resultado real. Esta función cierra ese gap.
    """
    mt = _norm(market_type)
    sel = (predicted_outcome or "").strip()
    sel_l = _norm(sel)

    if mt == "1x2":
        if sel_l in ("empate", "draw", "x"):
            return "draw"
        if team_home and sel_l == _norm(team_home):
            return "home_win"
        if team_away and sel_l == _norm(team_away):
            return "away_win"
        if sel_l in ("home_win", "away_win", "draw"):
            return sel_l  # ya normalizado (callers legacy)
        return "unknown"

    if mt.startswith("over/under") or mt.startswith("over_under") or mt == "totals":
        if sel_l.startswith("over"):
            return "over"
        if sel_l.startswith("under"):
            return "under"
        return "unknown"

    if mt == "btts" or "ambos anotan" in mt:
        if sel_l in ("si", "sí", "yes", "btts_yes"):
            return "yes"
        if sel_l in ("no", "btts_no"):
            return "no"
        return "unknown"

    if "doble oportunidad" in mt or mt in ("dc", "double chance"):
        if sel_l.startswith("1x"):
            return "home_draw"
        if sel_l.startswith("x2"):
            return "away_draw"
        if sel_l.startswith("12"):
            return "home_away"
        return "unknown"

    return sel_l or "unknown"


_DC_WINNING_LABELS: dict[str, tuple[str, ...]] = {
    "home_draw": ("home_win", "draw"),
    "away_draw": ("draw", "away_win"),
    "home_away": ("home_win", "away_win"),
}


def brier_score(predicted_prob: float, is_correct: bool) -> float:
    actual = 1.0 if is_correct else 0.0
    return (predicted_prob - actual) ** 2


def log_loss_value(predicted_prob: float, is_correct: bool, eps: float = 1e-15) -> float:
    p = max(eps, min(1.0 - eps, predicted_prob))
    if is_correct:
        return -math.log(p)
    return -math.log(1.0 - p)


def evaluate_prediction(
    market_type: str,
    predicted_outcome: str,
    probability: float,
    home_goals: int,
    away_goals: int,
    *,
    team_home: str | None = None,
    team_away: str | None = None,
) -> dict[str, Any]:
    """
    team_home/team_away son opcionales pero CRÍTICOS para evaluar 1X2
    correctamente cuando predicted_outcome es un nombre de equipo (el caso
    real en producción) en vez de una etiqueta "home_win"/"away_win"/"draw".
    Sin ellos, el comportamiento cae a comparación directa (legacy).
    """
    mt = _norm(market_type)
    predicted_key = resolve_predicted_key(
        market_type, predicted_outcome, team_home=team_home, team_away=team_away
    )

    if "doble oportunidad" in mt or mt in ("dc", "double chance"):
        actual_1x2 = resolve_1x2_outcome(home_goals, away_goals)
        is_correct = actual_1x2 in _DC_WINNING_LABELS.get(predicted_key, ())
        actual = actual_1x2
    else:
        actual = resolve_actual_outcome(market_type, home_goals, away_goals)
        is_correct = predicted_key == actual and actual != "unknown"

    return {
        "actual_outcome": actual,
        "predicted_key": predicted_key,
        "is_correct": is_correct,
        "brier_score": round(brier_score(probability, is_correct), 6),
        "log_loss": round(log_loss_value(probability, is_correct), 6),
    }
