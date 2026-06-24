"""Tests for market fair odds and edge computation."""

from apps.api.services.odds_context import compute_market_context
from apps.api.services.worldcup_engine import ModelMarkets
from apps.worker.ml.odds_math import expected_value_fair, expected_value_raw


def _model() -> ModelMarkets:
    return ModelMarkets(
        home_win=0.235,
        draw=0.262,
        away_win=0.503,
        over_25=0.462,
        under_25=0.538,
        btts_yes=0.469,
        btts_no=0.531,
        lambda_home=1.1,
        lambda_away=1.4,
        confidence="medium",
    )


def _odds_event() -> dict:
    return {
        "home_team": "Scotland",
        "away_team": "Brazil",
        "bookmakers": [
            {
                "key": "book1",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Scotland", "price": 4.10},
                            {"name": "Draw", "price": 3.65},
                            {"name": "Brazil", "price": 1.80},
                        ],
                    }
                ],
            },
            {
                "key": "book2",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Scotland", "price": 4.20},
                            {"name": "Draw", "price": 3.70},
                            {"name": "Brazil", "price": 1.85},
                        ],
                    }
                ],
            },
        ],
    }


def test_market_context_from_model_only():
    ctx = compute_market_context(_model(), "Scotland", "Brazil", None)
    assert not ctx.has_market
    assert len(ctx.outcomes) == 3
    brazil = next(o for o in ctx.outcomes if o.selection == "Brazil")
    assert brazil.model_fair_odds == round(1 / 0.503, 2)
    assert brazil.market_odds is None
    assert brazil.edge_pct == 0.0


def test_market_context_ev_vs_market():
    ctx = compute_market_context(_model(), "Scotland", "Brazil", _odds_event())
    assert ctx.has_market
    brazil = next(o for o in ctx.outcomes if o.selection == "Brazil")
    assert brazil.market_odds is not None
    assert brazil.market_implied is not None
    assert brazil.divergence is not None
    assert brazil.fair_odds is not None
    ev_fair = round(expected_value_fair(0.503, brazil.fair_odds) * 100, 1)
    ev_raw = round(expected_value_raw(0.503, brazil.market_odds) * 100, 1)
    assert brazil.edge_pct == ev_fair
    assert brazil.ev_fair_pct == ev_fair
    assert brazil.ev_raw_pct == ev_raw
    assert brazil.edge_pct < 0
    assert abs(brazil.ev_raw_pct) >= abs(brazil.ev_fair_pct)
