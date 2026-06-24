"""Goalkeeper metrics adjustment for defensive strength."""

from dataclasses import dataclass


@dataclass
class GoalkeeperProfile:
    name: str
    save_pct: float | None = None
    xga_90: float | None = None
    is_starter: bool = True


@dataclass
class GKAdjustment:
    home_factor: float
    away_factor: float
    home_xga_adjusted: float
    away_xga_adjusted: float


DEFAULT_SAVE_PCT = 0.70
DEFAULT_XGA_90 = 1.25
ADJUSTMENT_CAP = 0.15


def _gk_quality(gk: GoalkeeperProfile | None) -> float:
    """Higher = better goalkeeper (0-1 scale)."""
    if gk is None:
        return 0.5
    save = gk.save_pct if gk.save_pct is not None else DEFAULT_SAVE_PCT
    xga = gk.xga_90 if gk.xga_90 is not None else DEFAULT_XGA_90
    save_score = min(1.0, max(0.0, (save - 0.55) / 0.25))
    xga_score = min(1.0, max(0.0, (2.0 - xga) / 1.0))
    return 0.6 * save_score + 0.4 * xga_score


def adjustment_factor(gk: GoalkeeperProfile | None, cap: float = ADJUSTMENT_CAP) -> float:
    """
    Multiplicative factor on opponent's scoring lambda.
    Better GK -> factor < 1 (reduces conceded goals).
  """
    quality = _gk_quality(gk)
    # quality 0.5 -> factor 1.0; quality 1.0 -> factor (1-cap); quality 0 -> factor (1+cap)
    return 1.0 - (quality - 0.5) * 2 * cap


def apply_gk_adjustment(
    lambda_home: float,
    lambda_away: float,
    home_gk: GoalkeeperProfile | None = None,
    away_gk: GoalkeeperProfile | None = None,
    cap: float = ADJUSTMENT_CAP,
) -> GKAdjustment:
    """
    Home team scores against away GK; away team scores against home GK.
    """
    away_gk_factor = adjustment_factor(away_gk, cap)
    home_gk_factor = adjustment_factor(home_gk, cap)

    adj_lambda_home = lambda_home * away_gk_factor
    adj_lambda_away = lambda_away * home_gk_factor

    return GKAdjustment(
        home_factor=round(away_gk_factor, 4),
        away_factor=round(home_gk_factor, 4),
        home_xga_adjusted=round(adj_lambda_home, 4),
        away_xga_adjusted=round(adj_lambda_away, 4),
    )
