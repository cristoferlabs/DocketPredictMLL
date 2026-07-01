"""
Poisson Live Engine — conditioned predictions on live game state.

Given score (g_h, g_a) at minute M with remaining lambdas (λr_h, λr_a):
  P(home wins) = Σ_{i,j} P(Poisson(λr_h)=i) × P(Poisson(λr_a)=j) × I(g_h+i > g_a+j)

Remaining lambdas are scaled from pre-match:
  λr_home = λ_pre × time_remaining × intensity_home × momentum_home × red_card_adj
  λr_away = λ_pre × time_remaining × intensity_away × momentum_away × red_card_adj

All markets (1X2, O/U 0.5–4.5, BTTS, DC, combos) are recomputed from scratch
conditioned on the exact current score — not extrapolated from pre-match calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# WC historical: ~5.5 shots per goal (used to infer intensity from shot count)
_WC_SHOTS_PER_GOAL = 5.5
# Max goals to simulate for remaining time (8 covers >99.9% of Poisson mass)
_MAX_REMAINING_GOALS = 8
# Red card: -22% attack lambda per card for penalized team, +12% for opponent
_RC_OFFENSE = 0.78
_RC_OPPONENT_BOOST = 1.12


@dataclass
class GameState:
    """Current state of a live match."""
    minutes_elapsed: int          # 0=kick-off, 45=halftime, 90=end regulation
    home_goals: int = 0
    away_goals: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    is_extra_time: bool = False


@dataclass
class LiveStats:
    """Live statistics from API-Football fixtures/statistics endpoint."""
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_possession: float | None = None   # 0.0–100.0
    away_possession: float | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_xg: float | None = None           # expected goals from API-Football Pro
    away_xg: float | None = None
    home_attacks: int | None = None
    away_attacks: int | None = None
    home_dangerous_attacks: int | None = None
    away_dangerous_attacks: int | None = None


@dataclass
class LivePoissonResult:
    """Full market recomputation conditioned on current game state."""
    # Remaining lambdas
    lambda_home_remaining: float
    lambda_away_remaining: float
    # Live 1X2
    home_win: float
    draw: float
    away_win: float
    # Live O/U (all thresholds)
    over_05: float
    under_05: float
    over_15: float
    under_15: float
    over_25: float
    under_25: float
    over_35: float
    under_35: float
    over_45: float
    under_45: float
    # Live BTTS
    btts_yes: float
    btts_no: float
    # Live DC
    dc_home_draw: float
    dc_away_draw: float
    dc_home_away: float
    # Diagnostics
    minutes_remaining: int
    intensity_home: float
    intensity_away: float
    momentum_home: float
    momentum_away: float
    game_state_label: str
    home_goals: int
    away_goals: int
    lambda_home_prematch: float
    lambda_away_prematch: float


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _cdf(n: int, lam: float) -> float:
    """P(Poisson(lam) <= n). Returns 0 for n < 0."""
    if n < 0:
        return 0.0
    return sum(_pmf(k, lam) for k in range(n + 1))


def _time_fraction_remaining(minutes_elapsed: int, is_extra_time: bool) -> float:
    if is_extra_time:
        return max(0.0, (120 - minutes_elapsed) / 120)
    return max(0.0, min(1.0, (90 - minutes_elapsed) / 90))


def _intensity_multiplier(
    shots: int | None,
    xg: float | None,
    pre_lambda: float,
    minutes_elapsed: int,
) -> float:
    """Ratio of actual attack rate to expected attack rate. Clamped to [0.5, 2.0]."""
    frac_played = max(1, min(90, minutes_elapsed)) / 90.0
    if frac_played <= 0:
        return 1.0

    # xG is the most direct signal — prefer it over shot count
    if xg is not None and xg > 0:
        expected_xg = pre_lambda * frac_played
        if expected_xg > 0:
            return round(max(0.5, min(2.0, xg / expected_xg)), 4)

    if shots is not None and shots > 0:
        expected_shots = pre_lambda * _WC_SHOTS_PER_GOAL * frac_played
        if expected_shots > 0:
            return round(max(0.5, min(2.0, shots / expected_shots)), 4)

    return 1.0


def _momentum_multiplier(home_possession: float | None, for_home: bool) -> float:
    """Possession-based momentum. +/-20% of lambda, clamped to [0.80, 1.20]."""
    if home_possession is None:
        return 1.0
    bias = (home_possession - 50.0) / 50.0   # −1 to +1
    delta = 0.20 * bias
    return round(max(0.80, min(1.20, 1.0 + delta if for_home else 1.0 - delta)), 4)


def _red_card_factors(home_rc: int, away_rc: int) -> tuple[float, float]:
    """
    Each red card reduces the penalized team's attack lambda by 22%.
    The opponent gets a 12% boost per card advantage.
    Results clamped to [0.60, 1.25].
    """
    h_adj = max(0.60, _RC_OFFENSE ** home_rc)
    a_adj = max(0.60, _RC_OFFENSE ** away_rc)
    net = home_rc - away_rc
    if net > 0:
        a_adj = min(1.25, a_adj * (_RC_OPPONENT_BOOST ** net))
    elif net < 0:
        h_adj = min(1.25, h_adj * (_RC_OPPONENT_BOOST ** (-net)))
    return round(h_adj, 4), round(a_adj, 4)


def _game_state_label(minutes_elapsed: int, is_extra_time: bool) -> str:
    if minutes_elapsed == 0:
        return "pre_match"
    if minutes_elapsed < 45:
        return "first_half"
    if minutes_elapsed == 45:
        return "halftime"
    if minutes_elapsed < 90:
        return "second_half"
    if is_extra_time:
        return "extra_time"
    return "full_time"


# ─── Public API ────────────────────────────────────────────────────────────────

def compute_live_markets(
    lambda_home: float,
    lambda_away: float,
    game_state: GameState,
    live_stats: LiveStats | None = None,
) -> LivePoissonResult:
    """
    Recompute all markets conditioned on the current score and remaining time.

    Args:
        lambda_home: Pre-match expected goals for home team (full 90 min).
        lambda_away: Pre-match expected goals for away team (full 90 min).
        game_state: Current match state (minute, score, cards).
        live_stats: Optional live statistics from API-Football (improves λ estimates).

    Returns:
        LivePoissonResult with all markets recomputed for the remaining game.
    """
    ls = live_stats or LiveStats()

    fr = _time_fraction_remaining(game_state.minutes_elapsed, game_state.is_extra_time)
    min_rem = (
        max(0, 120 - game_state.minutes_elapsed)
        if game_state.is_extra_time
        else max(0, 90 - game_state.minutes_elapsed)
    )

    int_h = _intensity_multiplier(ls.home_shots, ls.home_xg, lambda_home, game_state.minutes_elapsed)
    int_a = _intensity_multiplier(ls.away_shots, ls.away_xg, lambda_away, game_state.minutes_elapsed)
    mom_h = _momentum_multiplier(ls.home_possession, for_home=True)
    mom_a = _momentum_multiplier(ls.home_possession, for_home=False)
    rc_h, rc_a = _red_card_factors(game_state.home_red_cards, game_state.away_red_cards)

    lr_h = round(max(0.01, lambda_home * fr * int_h * mom_h * rc_h), 4)
    lr_a = round(max(0.01, lambda_away * fr * int_a * mom_a * rc_a), 4)

    g_h = game_state.home_goals
    g_a = game_state.away_goals
    total_goals = g_h + g_a

    # ── 1X2 via exact Poisson sum over remaining score matrix ──────────────────
    p_hw = p_d = p_aw = 0.0
    for di in range(_MAX_REMAINING_GOALS + 1):
        p_di = _pmf(di, lr_h)
        if p_di < 1e-10 and di > 0:
            break
        for dj in range(_MAX_REMAINING_GOALS + 1):
            p_dj = _pmf(dj, lr_a)
            if p_dj < 1e-10 and dj > 0:
                break
            p = p_di * p_dj
            fh, fa = g_h + di, g_a + dj
            if fh > fa:
                p_hw += p
            elif fh == fa:
                p_d += p
            else:
                p_aw += p

    # Normalize for floating-point drift
    s = p_hw + p_d + p_aw
    if s > 0:
        p_hw /= s
        p_d /= s
        p_aw /= s

    # ── O/U markets ────────────────────────────────────────────────────────────
    lr_total = lr_h + lr_a
    ou: dict[str, float] = {}
    for t10 in (5, 15, 25, 35, 45):
        threshold = t10 / 10            # 0.5, 1.5, 2.5, 3.5, 4.5
        limit = threshold - total_goals  # remaining goals allowed for Under
        if limit < 0:
            under_p = 0.0              # already exceeded threshold
        else:
            # Under N.5 means total <= N, so remaining <= N - total_goals
            under_p = _cdf(int(limit), lr_total)
        ou[f"u{t10}"] = round(max(0.0, min(1.0, under_p)), 4)
        ou[f"o{t10}"] = round(max(0.0, min(1.0, 1.0 - under_p)), 4)

    # ── BTTS ───────────────────────────────────────────────────────────────────
    if g_h > 0 and g_a > 0:
        btts_yes, btts_no = 1.0, 0.0
    elif g_h > 0 and g_a == 0:
        btts_no = round(math.exp(-lr_a), 4)         # Away must score 0 more
        btts_yes = round(1.0 - btts_no, 4)
    elif g_h == 0 and g_a > 0:
        btts_no = round(math.exp(-lr_h), 4)         # Home must score 0 more
        btts_yes = round(1.0 - btts_no, 4)
    else:
        p_h0 = math.exp(-lr_h)
        p_a0 = math.exp(-lr_a)
        btts_no = round(max(0.0, p_h0 + p_a0 - math.exp(-(lr_h + lr_a))), 4)
        btts_yes = round(1.0 - btts_no, 4)

    return LivePoissonResult(
        lambda_home_remaining=lr_h,
        lambda_away_remaining=lr_a,
        home_win=round(p_hw, 4),
        draw=round(p_d, 4),
        away_win=round(p_aw, 4),
        over_05=ou["o5"],
        under_05=ou["u5"],
        over_15=ou["o15"],
        under_15=ou["u15"],
        over_25=ou["o25"],
        under_25=ou["u25"],
        over_35=ou["o35"],
        under_35=ou["u35"],
        over_45=ou["o45"],
        under_45=ou["u45"],
        btts_yes=btts_yes,
        btts_no=btts_no,
        dc_home_draw=round(p_hw + p_d, 4),
        dc_away_draw=round(p_aw + p_d, 4),
        dc_home_away=round(p_hw + p_aw, 4),
        minutes_remaining=min_rem,
        intensity_home=int_h,
        intensity_away=int_a,
        momentum_home=mom_h,
        momentum_away=mom_a,
        game_state_label=_game_state_label(game_state.minutes_elapsed, game_state.is_extra_time),
        home_goals=g_h,
        away_goals=g_a,
        lambda_home_prematch=lambda_home,
        lambda_away_prematch=lambda_away,
    )


def build_live_score_matrix(lr_h: float, lr_a: float, max_goals: int = 8) -> np.ndarray:
    """Remaining-goals probability matrix for live combo calculations."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = _pmf(i, lr_h) * _pmf(j, lr_a)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def live_stats_from_api_football(raw_stats: list[dict]) -> LiveStats:
    """
    Parse API-Football fixtures/statistics response into LiveStats.

    raw_stats is the list returned by ApiFootballClient.get_fixture_statistics().
    Each element: {"team": {"id": ..., "name": ...}, "statistics": [{"type": ..., "value": ...}]}
    """
    home_raw: dict[str, object] = {}
    away_raw: dict[str, object] = {}

    # API-Football returns home team first, away team second — order is authoritative
    for idx, team_data in enumerate(raw_stats[:2]):
        stats_list = team_data.get("statistics", [])
        parsed = {s.get("type", ""): s.get("value") for s in stats_list if s.get("type")}
        if idx == 0:
            home_raw = parsed
        else:
            away_raw = parsed

    def _int(d: dict, key: str) -> int | None:
        v = d.get(key)
        if v is None:
            return None
        try:
            return int(str(v).replace("%", "").strip())
        except (ValueError, TypeError):
            return None

    def _float(d: dict, key: str) -> float | None:
        v = d.get(key)
        if v is None:
            return None
        try:
            return float(str(v).replace("%", "").strip())
        except (ValueError, TypeError):
            return None

    return LiveStats(
        home_shots=_int(home_raw, "Total Shots"),
        away_shots=_int(away_raw, "Total Shots"),
        home_shots_on_target=_int(home_raw, "Shots on Goal"),
        away_shots_on_target=_int(away_raw, "Shots on Goal"),
        home_possession=_float(home_raw, "Ball Possession"),
        away_possession=_float(away_raw, "Ball Possession"),
        home_corners=_int(home_raw, "Corner Kicks"),
        away_corners=_int(away_raw, "Corner Kicks"),
        home_xg=_float(home_raw, "expected_goals"),
        away_xg=_float(away_raw, "expected_goals"),
        home_attacks=_int(home_raw, "Total attacks"),
        away_attacks=_int(away_raw, "Total attacks"),
        home_dangerous_attacks=_int(home_raw, "Dangerous Attacks"),
        away_dangerous_attacks=_int(away_raw, "Dangerous Attacks"),
    )


def live_game_state_from_api_football(fixture: dict) -> GameState:
    """
    Build a GameState from an API-Football fixtures response entry.
    fixture is one element of the list returned by get_fixtures() or get_live_fixtures().
    """
    fx = fixture.get("fixture", {})
    goals = fixture.get("goals", {})
    score = fixture.get("score", {})

    # Minute — API-Football returns {"elapsed": M, "extra": E}
    elapsed_obj = fx.get("status", {})
    minutes = int(elapsed_obj.get("elapsed") or 0)
    is_et = elapsed_obj.get("short", "") in ("ET", "P", "BT")

    home_goals = int(goals.get("home") or 0)
    away_goals = int(goals.get("away") or 0)

    # Red cards from the events list (not included in basic fixture — caller must merge)
    return GameState(
        minutes_elapsed=minutes,
        home_goals=home_goals,
        away_goals=away_goals,
        is_extra_time=is_et,
    )
