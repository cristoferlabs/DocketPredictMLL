"""Tests for World Cup engine and odds separation."""

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


def test_odds_do_not_change_model():
    m = compute_model_markets(1.5, 1.0, 2000, 1700)
    original_home = m.home_win
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
        ],
    }
    opps = compute_ev_opportunities(m, "Team A", "Team B", fake_event)
    assert m.home_win == original_home
    assert isinstance(opps, list)


def test_ev_positive_when_model_beats_market():
    from apps.worker.ml.odds_math import expected_value_fair, implied_probability

    assert expected_value_fair(0.5, 2.2) > 0
    assert implied_probability(2.0) == 0.5


def test_elo_from_archives():
    d18 = {"rounds": [{"name": "Final", "matches": [{"date": "2018-07-15", "team1": {"name": "France"}, "team2": {"name": "Croatia"}, "score": {"ft": [4, 2]}}]}]}
    ratings = calc_elo_ratings(d18, {}, {})
    assert ratings.get("France", 1500) > 1500
