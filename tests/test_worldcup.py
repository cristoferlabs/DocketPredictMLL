"""Tests for World Cup engine and odds separation."""

from apps.api.services.live_calibration import apply_live_calibration, LiveCalibrationContext
from apps.api.services.odds_context import compute_ev_opportunities
from apps.api.services.worldcup_engine import (
    calc_elo_ratings,
    compute_model_markets,
    name_match,
)


def test_name_match_fuzzy():
    assert name_match("United States", "USA") or name_match("Colombia", "colombia")


def test_model_markets_sum_approx_one():
    m = compute_model_markets(1.4, 1.1, 1950, 1800)
    total = m.home_win + m.draw + m.away_win
    assert 0.99 <= total <= 1.01


def test_statistical_preserved_ev_uses_calibrated():
    """P_statistical no muta; EV con P_cal puede diferir."""
    m = compute_model_markets(1.5, 1.0, 2000, 1700)
    p_stat_home = m.home_win
    fake_event = {
        "home_team": "Team A",
        "away_team": "Team B",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Team A", "price": 3.5},
                            {"name": "Draw", "price": 3.2},
                            {"name": "Team B", "price": 2.1},
                        ],
                    }
                ]
            }
            for _ in range(3)
        ],
    }
    from apps.worker.ml.odds_math import fair_h2h_market

    h2h = fair_h2h_market(fake_event)
    market = {
        "home_win": h2h["home"]["fair_prob"],
        "draw": h2h["draw"]["fair_prob"],
        "away_win": h2h["away"]["fair_prob"],
    }
    ctx = LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=12.0, n_books=3)
    cal = apply_live_calibration(m, market, context=ctx)
    assert cal.statistical["home_win"] == p_stat_home
    assert cal.calibrated.home_win != p_stat_home or cal.alpha == 0
    opps = compute_ev_opportunities(cal.calibrated, "Team A", "Team B", fake_event)
    assert isinstance(opps, list)
    if opps:
        assert opps[0].metadata.get("prob_source") == "calibrated"


def test_ev_positive_when_model_beats_market():
    from apps.worker.ml.odds_math import expected_value_fair, implied_probability

    assert expected_value_fair(0.5, 2.2) > 0
    assert implied_probability(2.0) == 0.5


def test_home_dog_not_inflated_to_sixty():
    """Calibración no debe empujar perros locales a ~60%."""
    m = compute_model_markets(1.61, 1.18, 1468, 1467)
    assert m.home_win < 0.52
    assert abs(m.home_win - 0.47) < 0.05


def test_czech_style_close_match_stays_near_blend():
    m = compute_model_markets(1.25, 1.15, 1500, 1540)
    assert 0.35 <= m.home_win <= 0.45
    assert m.home_win < 0.50


def test_low_lambda_dampens_strong_favorite():
    """λ total ~2.3 no debería sostener >62% sin ajuste."""
    m = compute_model_markets(1.75, 0.55, 1494, 1500)
    assert m.home_win < 0.62
    meta = m.blend_meta or {}
    assert meta.get("sanity_adjustments") or m.home_win < 0.65


def test_elo_anchor_when_poisson_overrates_home():
    """ELO parejo: Poisson no debe inflar local sin tope (DC/λ o elo_anchor)."""
    m = compute_model_markets(1.75, 0.55, 1500, 1494)
    meta = m.blend_meta or {}
    sanity = meta.get("sanity_adjustments") or []
    assert m.home_win < 0.66
    assert sanity or m.home_win < 0.58


def test_elo_from_archives():
    d18 = {"rounds": [{"name": "Final", "matches": [{"date": "2018-07-15", "team1": {"name": "France"}, "team2": {"name": "Croatia"}, "score": {"ft": [4, 2]}}]}]}
    ratings = calc_elo_ratings(d18, {}, {})
    assert ratings.get("France", 1500) > 1500
