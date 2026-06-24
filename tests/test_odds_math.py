"""Tests for fair odds, devig and EV inflation detection."""

from apps.api.services.odds_context import compute_ev_opportunities
from apps.api.services.worldcup_engine import ModelMarkets
from apps.worker.ml.odds_math import (
    devig_multiclass,
    devig_two_way,
    expected_value_fair,
    expected_value_raw,
    fair_h2h_market,
    fair_odds,
    overround,
)


def _sample_event(home_odds: float, draw_odds: float, away_odds: float, n_books: int = 3) -> dict:
    """Synthetic odds event with identical prices across bookmakers."""
    bookmakers = []
    for i in range(n_books):
        bookmakers.append(
            {
                "key": f"book{i}",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Team A", "price": home_odds},
                            {"name": "Draw", "price": draw_odds},
                            {"name": "Team B", "price": away_odds},
                        ],
                    }
                ],
            }
        )
    return {"home_team": "Team A", "away_team": "Team B", "bookmakers": bookmakers}


def test_devig_multiclass_sums_to_one():
    fair = devig_multiclass({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert abs(sum(fair.values()) - 1.0) < 1e-6


def test_overround_detects_margin():
    # ~5% overround typical market
    margin = overround({"home": 1.95, "draw": 3.4, "away": 4.2})
    assert margin > 0.03


def test_max_odds_creates_false_positive_ev():
    """Mejor cuota cross-book puede mostrar +EV cuando fair mediana no."""
    model_prob = 0.42
    median_raw = 2.10
    best_raw = 2.50
    ev_median = expected_value_raw(model_prob, median_raw)
    ev_best = expected_value_raw(model_prob, best_raw)
    assert ev_best > 0
    assert ev_median <= 0
    assert ev_best > ev_median


def test_model_equals_fair_market_near_zero_ev():
    event = _sample_event(2.0, 3.5, 4.0)
    fair = fair_h2h_market(event)
    home_fair_p = fair["home"]["fair_prob"]
    ev = expected_value_fair(home_fair_p, fair["home"]["fair_odds"])
    assert abs(ev) < 0.02


def test_best_odds_vs_median_fewer_false_positives():
    """Max odds across books inflates EV vs median fair aggregation."""
    event = {
        "home_team": "Team A",
        "away_team": "Team B",
        "bookmakers": [
            {
                "key": "b1",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Team A", "price": 2.10},
                            {"name": "Draw", "price": 3.30},
                            {"name": "Team B", "price": 3.80},
                        ],
                    }
                ],
            },
            {
                "key": "b2",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Team A", "price": 2.50},
                            {"name": "Draw", "price": 3.20},
                            {"name": "Team B", "price": 3.50},
                        ],
                    }
                ],
            },
        ],
    }
    fair = fair_h2h_market(event)
    model_prob = 0.42
    ev_fair = expected_value_fair(model_prob, fair["home"]["fair_odds"])
    ev_best_raw = expected_value_raw(model_prob, 2.50)
    assert ev_best_raw > ev_fair


def test_devig_two_way():
    over_p, under_p = devig_two_way(1.90, 1.95)
    assert abs(over_p + under_p - 1.0) < 1e-6


def test_compute_ev_single_best_opportunity():
    model = ModelMarkets(
        home_win=0.55,
        draw=0.25,
        away_win=0.20,
        over_25=0.5,
        under_25=0.5,
        btts_yes=0.5,
        btts_no=0.5,
        lambda_home=1.4,
        lambda_away=1.1,
    )
    event = _sample_event(2.20, 3.40, 4.50)
    opps = compute_ev_opportunities(model, "Team A", "Team B", event, single_best=True)
    assert len(opps) <= 1
    if opps:
        assert opps[0].fair_odds > 0
        assert opps[0].vig_pct >= 0
