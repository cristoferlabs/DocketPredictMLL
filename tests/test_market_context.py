"""Tests for market fair odds and edge computation."""

from apps.api.services.odds_context import compute_market_context
from apps.api.services.worldcup_engine import ModelMarkets


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


def test_market_context_from_model_only():
    ctx = compute_market_context(_model(), "Scotland", "Brazil", None)
    assert not ctx.has_market
    assert len(ctx.outcomes) == 3
    brazil = next(o for o in ctx.outcomes if o.selection == "Brazil")
    assert brazil.fair_odds == round(1 / 0.503, 2)
    assert brazil.edge_pct == 0.0


def test_market_context_with_odds_event():
    event = {
        "home_team": "Scotland",
        "away_team": "Brazil",
        "bookmakers": [
            {
                "key": "book1",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Scotland", "price": 4.5},
                            {"name": "Draw", "price": 3.8},
                            {"name": "Brazil", "price": 1.95},
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
                            {"name": "Scotland", "price": 4.2},
                            {"name": "Draw", "price": 3.9},
                            {"name": "Brazil", "price": 2.0},
                        ],
                    }
                ],
            },
        ],
    }
    ctx = compute_market_context(_model(), "Scotland", "Brazil", event)
    assert ctx.has_market
    brazil = next(o for o in ctx.outcomes if o.selection == "Brazil")
    assert brazil.fair_odds > 1.5
    assert brazil.raw_odds > 1.5
    assert isinstance(brazil.edge_pct, float)
