"""ELO rating model for football match outcome probabilities."""

from dataclasses import dataclass
from math import pow

DEFAULT_RATING = 1500.0
DEFAULT_K = 32.0
HOME_ADVANTAGE = 100.0


@dataclass
class EloConfig:
    k_factor: float = DEFAULT_K
    home_advantage: float = HOME_ADVANTAGE
    default_rating: float = DEFAULT_RATING


@dataclass
class EloProbabilities:
    home_win: float
    draw: float
    away_win: float
    home_rating: float
    away_rating: float


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + pow(10, (rating_b - rating_a) / 400.0))


def outcome_to_score(home_goals: int, away_goals: int) -> tuple[float, float]:
  if home_goals > away_goals:
      return 1.0, 0.0
  if home_goals < away_goals:
      return 0.0, 1.0
  return 0.5, 0.5


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    config: EloConfig | None = None,
) -> tuple[float, float]:
    cfg = config or EloConfig()
    home_adj = home_rating + cfg.home_advantage
    exp_home = expected_score(home_adj, away_rating)
    exp_away = 1.0 - exp_home
    act_home, act_away = outcome_to_score(home_goals, away_goals)
    new_home = home_rating + cfg.k_factor * (act_home - exp_home)
    new_away = away_rating + cfg.k_factor * (act_away - exp_away)
    return new_home, new_away


def predict_match(
    home_rating: float,
    away_rating: float,
    config: EloConfig | None = None,
) -> EloProbabilities:
    """Estimate 1X2 probabilities from ELO ratings with draw adjustment."""
    cfg = config or EloConfig()
    home_adj = home_rating + cfg.home_advantage
    p_home_not_draw = expected_score(home_adj, away_rating)
    p_away_not_draw = expected_score(away_rating, home_adj)

    # Draw probability decreases as rating gap grows
    rating_diff = abs(home_adj - away_rating)
    draw_base = 0.28 * pow(0.995, rating_diff / 10.0)
    draw_base = max(0.08, min(0.32, draw_base))

    remaining = 1.0 - draw_base
    total = p_home_not_draw + p_away_not_draw
    if total <= 0:
        p_home = remaining / 2
        p_away = remaining / 2
    else:
        p_home = remaining * (p_home_not_draw / total)
        p_away = remaining * (p_away_not_draw / total)

    return EloProbabilities(
        home_win=round(p_home, 6),
        draw=round(draw_base, 6),
        away_win=round(p_away, 6),
        home_rating=home_rating,
        away_rating=away_rating,
    )
