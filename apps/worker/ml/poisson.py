"""Poisson goal distribution model for football matches."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, factorial
from typing import Any

import numpy as np


@dataclass
class PoissonConfig:
    max_goals: int = 10
    dixon_coles_rho: float | None = None
    use_dixon_coles: bool = False


@dataclass
class PoissonPrediction:
    lambda_home: float
    lambda_away: float
    score_matrix: np.ndarray
    over_25: float
    under_25: float
    btts_yes: float
    btts_no: float
    most_likely_score: tuple[int, int]
    lambda_base_home: float = 0.0
    lambda_base_away: float = 0.0
    match_context: str = "balanced"
    rho: float = 0.0
    dixon_coles_applied: bool = False
    dc_meta: dict[str, Any] = field(default_factory=dict)


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k) * exp(-lam) / factorial(k)


def build_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Factor τ Dixon-Coles — solo scores bajos (0,1)."""
    if home_goals > 1 or away_goals > 1:
        return 1.0
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def build_score_matrix_dixon_coles(
    lambda_home: float,
    lambda_away: float,
    *,
    rho: float = -0.13,
    max_goals: int = 10,
) -> np.ndarray:
    """Matriz de scores con corrección Dixon-Coles (mejor cola baja / empates)."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            base = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
            matrix[i, j] = base * dixon_coles_tau(i, j, lambda_home, lambda_away, rho)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def estimate_lambdas(
    home_xg: float | None = None,
    away_xg: float | None = None,
    home_goals_avg: float | None = None,
    away_goals_avg: float | None = None,
    league_avg_goals: float = 2.6,
) -> tuple[float, float]:
    """Estimate Poisson lambdas from xG or historical averages."""
    if home_xg is not None and away_xg is not None:
        return max(0.3, home_xg), max(0.3, away_xg)

    home_avg = home_goals_avg if home_goals_avg is not None else league_avg_goals / 2
    away_avg = away_goals_avg if away_goals_avg is not None else league_avg_goals / 2
    return max(0.3, home_avg), max(0.3, away_avg)


def predict_match(
    lambda_home: float,
    lambda_away: float,
    config: PoissonConfig | None = None,
    *,
    elo_home: float | None = None,
    elo_away: float | None = None,
) -> PoissonPrediction:
    cfg = config or PoissonConfig()
    from apps.worker.ml.dixon_coles import prepare_dixon_coles_lambdas

    dc = prepare_dixon_coles_lambdas(
        lambda_home,
        lambda_away,
        elo_home=elo_home,
        elo_away=elo_away,
    )
    lh, la = dc.lambda_corrected_home, dc.lambda_corrected_away

    if cfg.dixon_coles_rho is not None:
        rho = cfg.dixon_coles_rho
        use_dc = cfg.use_dixon_coles
    else:
        rho = dc.rho
        use_dc = dc.dixon_coles_applied

    if use_dc and rho != 0.0:
        matrix = build_score_matrix_dixon_coles(lh, la, rho=rho, max_goals=cfg.max_goals)
    else:
        matrix = build_score_matrix(lh, la, cfg.max_goals)

    over_25 = float(sum(matrix[i, j] for i in range(cfg.max_goals + 1) for j in range(cfg.max_goals + 1) if i + j > 2))
    btts_yes = float(sum(matrix[i, j] for i in range(1, cfg.max_goals + 1) for j in range(1, cfg.max_goals + 1)))

    max_idx = np.unravel_index(matrix.argmax(), matrix.shape)

    return PoissonPrediction(
        lambda_home=lh,
        lambda_away=la,
        score_matrix=matrix,
        over_25=round(over_25, 6),
        under_25=round(1.0 - over_25, 6),
        btts_yes=round(btts_yes, 6),
        btts_no=round(1.0 - btts_yes, 6),
        most_likely_score=(int(max_idx[0]), int(max_idx[1])),
        lambda_base_home=dc.lambda_base_home,
        lambda_base_away=dc.lambda_base_away,
        match_context=dc.match_context,
        rho=rho if use_dc else 0.0,
        dixon_coles_applied=use_dc and rho != 0.0,
        dc_meta=dc.to_dict(),
    )


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """1X2 from score matrix."""
    max_g = matrix.shape[0] - 1
    home_win = sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i > j)
    draw = sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i == j)
    away_win = sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i < j)
    return {
        "home_win": round(float(home_win), 6),
        "draw": round(float(draw), 6),
        "away_win": round(float(away_win), 6),
    }
