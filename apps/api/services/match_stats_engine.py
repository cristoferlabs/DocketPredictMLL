"""
Match statistics market predictions — corners, shots on target, cards.

Calibrated against WC 2018 + 2022 averages:
  - Corners:  ~10.1 per match (scale factor ≈ 3.77 × total λ)
  - SoT:      ~8.2 per match  (scale factor ≈ 3.1 × total λ)
  - Cards:    ~3.2 per match  (baseline + ELO mismatch adjustment)

When real per-team WC 2026 stats are provided via home_team_stats /
away_team_stats, those averages replace the generic Poisson scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.api.services.team_wc_stats_service import TeamWCStats


# WC historical averages (2018 + 2022)
# Calibrated against observed data: England 2-1 DRC had 10 SoT (8+2), 8 corners, 2 cards
_WC_AVG_GOALS = 2.66      # avg total goals/match
_WC_AVG_CORNERS = 9.5     # avg total corners/match (revised down from 10.1)
_WC_AVG_SOT = 9.0         # avg total SoT/match (revised up: 3.9x goals, not 3.1x)
_WC_AVG_CARDS = 2.8       # avg yellow+red cards/match (revised down from 3.2)


@dataclass
class StatsMarkets:
    """Predicted stats markets for a match (goal-independent Poisson)."""
    # Corners
    lambda_corners: float
    corners_over_85: float     # P(total > 8.5)
    corners_over_95: float     # P(total > 9.5)  ← main bookmaker line
    corners_over_105: float    # P(total > 10.5)
    corners_under_85: float
    corners_under_95: float
    corners_under_105: float
    # Shots on target (total both teams)
    lambda_sot: float
    sot_over_75: float         # P(total SoT > 7.5)
    sot_over_85: float         # P(total SoT > 8.5)  ← main line
    sot_over_95: float         # P(total SoT > 9.5)
    # Yellow+red cards
    lambda_cards: float
    cards_over_25: float       # P(total > 2.5)
    cards_over_35: float       # P(total > 3.5)
    cards_over_45: float       # P(total > 4.5)
    # Individual team shots on target estimate
    shots_on_target_home: float   # expected SoT for home team
    shots_on_target_away: float   # expected SoT for away team


def _poisson_over(lam: float, line: float) -> float:
    """P(X > line) for Poisson(lam). line must be a .5 value (e.g. 9.5 → k_max=9)."""
    k_max = int(math.floor(line))
    p_le = sum(
        math.exp(-lam) * (lam ** k) / math.factorial(k)
        for k in range(k_max + 1)
    )
    return round(max(0.0, min(1.0, 1.0 - p_le)), 4)


def predict_stats_markets(
    lambda_home: float,
    lambda_away: float,
    elo_home: float = 1500.0,
    elo_away: float = 1500.0,
    home_team_stats: "TeamWCStats | None" = None,
    away_team_stats: "TeamWCStats | None" = None,
) -> StatsMarkets:
    """
    Predict corners, shots on target, and cards using Poisson.

    When home_team_stats / away_team_stats are provided (real WC 2026 data
    fetched from API-Football), those per-team averages are used directly.
    Otherwise falls back to generic WC historical scaling.
    """
    elo_diff = abs(elo_home - elo_away)

    if home_team_stats is not None and away_team_stats is not None:
        # Real per-team WC 2026 averages from API-Football
        sot_home = round(home_team_stats.avg_shots_on_target, 2)
        sot_away = round(away_team_stats.avg_shots_on_target, 2)
        lam_sot = round(sot_home + sot_away, 2)
        lam_corners = round(home_team_stats.avg_corners + away_team_stats.avg_corners, 2)
        # Base cards from actual averages + small ELO frustration adjustment
        base_cards = (
            home_team_stats.avg_yellow_cards + home_team_stats.avg_red_cards
            + away_team_stats.avg_yellow_cards + away_team_stats.avg_red_cards
        )
        card_adj = min(0.4, elo_diff / 800.0)
        lam_cards = round(base_cards + card_adj, 2)
    else:
        # Fallback: generic Poisson scaling from WC 2018+2022 averages
        total_lam = max(0.5, lambda_home + lambda_away)
        scale = total_lam / _WC_AVG_GOALS
        lam_corners = round(_WC_AVG_CORNERS * scale, 2)
        sot_home = round(lambda_home * (_WC_AVG_SOT / (2 * _WC_AVG_GOALS / 2)), 2)
        sot_away = round(lambda_away * (_WC_AVG_SOT / (2 * _WC_AVG_GOALS / 2)), 2)
        lam_sot = round(sot_home + sot_away, 2)
        card_adj = min(0.6, elo_diff / 600.0)
        lam_cards = round(_WC_AVG_CARDS + card_adj, 2)

    return StatsMarkets(
        lambda_corners=lam_corners,
        corners_over_85=_poisson_over(lam_corners, 8.5),
        corners_over_95=_poisson_over(lam_corners, 9.5),
        corners_over_105=_poisson_over(lam_corners, 10.5),
        corners_under_85=round(1.0 - _poisson_over(lam_corners, 8.5), 4),
        corners_under_95=round(1.0 - _poisson_over(lam_corners, 9.5), 4),
        corners_under_105=round(1.0 - _poisson_over(lam_corners, 10.5), 4),
        lambda_sot=lam_sot,
        sot_over_75=_poisson_over(lam_sot, 7.5),
        sot_over_85=_poisson_over(lam_sot, 8.5),
        sot_over_95=_poisson_over(lam_sot, 9.5),
        lambda_cards=lam_cards,
        cards_over_25=_poisson_over(lam_cards, 2.5),
        cards_over_35=_poisson_over(lam_cards, 3.5),
        cards_over_45=_poisson_over(lam_cards, 4.5),
        shots_on_target_home=sot_home,
        shots_on_target_away=sot_away,
    )


def live_stats_display(live_stats: object, t1: str, t2: str) -> list[str]:
    """
    Format live stats from LiveStats object into display lines.
    Returns empty list if no stats available.
    """
    lines = []
    ls = live_stats
    if not ls:
        return lines

    def _fmt(home_val, away_val, label: str, unit: str = "") -> str | None:
        if home_val is None and away_val is None:
            return None
        h = f"{home_val}{unit}" if home_val is not None else "—"
        a = f"{away_val}{unit}" if away_val is not None else "—"
        t1s = t1[:7]
        t2s = t2[:7]
        return f"  {label:<16} {t1s}: {h:<5} {t2s}: {a}"

    if (ls.home_shots is not None or ls.away_shots is not None):
        lines.append("📊 ESTADÍSTICAS LIVE")
        row = _fmt(ls.home_shots, ls.away_shots, "Tiros totales")
        if row:
            lines.append(row)

    sot_row = _fmt(ls.home_shots_on_target, ls.away_shots_on_target, "Tiros al arco")
    if sot_row:
        lines.append(sot_row)

    corn_row = _fmt(ls.home_corners, ls.away_corners, "Corners")
    if corn_row:
        lines.append(corn_row)

    poss_row = _fmt(
        f"{ls.home_possession:.0f}%" if ls.home_possession else None,
        f"{ls.away_possession:.0f}%" if ls.away_possession else None,
        "Posesión"
    )
    if poss_row:
        lines.append(poss_row)

    if ls.home_xg is not None or ls.away_xg is not None:
        xg_row = _fmt(
            f"{ls.home_xg:.2f}" if ls.home_xg else None,
            f"{ls.away_xg:.2f}" if ls.away_xg else None,
            "xG (API-Football)"
        )
        if xg_row:
            lines.append(xg_row)

    if lines and lines[0] != "📊 ESTADÍSTICAS LIVE":
        lines.insert(0, "📊 ESTADÍSTICAS LIVE")

    return lines
