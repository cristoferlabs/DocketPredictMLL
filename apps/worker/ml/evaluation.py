"""Evaluation utilities for self-improving loop."""

import math
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


def resolve_actual_outcome(market_type: str, home_goals: int, away_goals: int) -> str:
    if market_type == "1X2":
        return resolve_1x2_outcome(home_goals, away_goals)
    if market_type in ("over_under_2.5", "over_under"):
        return resolve_over_under(home_goals, away_goals)
    if market_type == "btts":
        return resolve_btts(home_goals, away_goals)
    return "unknown"


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
) -> dict[str, Any]:
    actual = resolve_actual_outcome(market_type, home_goals, away_goals)
    is_correct = predicted_outcome == actual
    return {
        "actual_outcome": actual,
        "is_correct": is_correct,
        "brier_score": round(brier_score(probability, is_correct), 6),
        "log_loss": round(log_loss_value(probability, is_correct), 6),
    }
