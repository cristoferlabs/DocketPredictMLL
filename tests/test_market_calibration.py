"""Tests for market_calibration facade (backward compatibility)."""

from apps.api.services.market_calibration import (
    apply_market_calibration,
    classify_decision_layer,
    detect_market_dominance,
    market_confidence_weights,
)
from apps.api.services.odds_context import compute_market_context
from apps.api.services.worldcup_engine import ModelMarkets


def _scotland_brazil_model() -> ModelMarkets:
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


def _scotland_brazil_odds() -> dict:
    return {
        "home_team": "Scotland",
        "away_team": "Brazil",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Scotland", "price": 9.50},
                {"name": "Draw", "price": 5.60},
                {"name": "Brazil", "price": 1.32},
            ]}],
        }],
    }


def test_facade_apply_market_calibration_extreme():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    adj, adj_ctx = apply_market_calibration(model, ctx, "Scotland", "Brazil")
    assert adj is not None
    assert adj.layer == "extreme"
    assert adj.blend_applied is False
    assert adj.home == model.home_win
    assert adj_ctx is None


def test_facade_detect_market_dominance_reexport():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    from apps.api.services.worldcup_engine import MatchAnalysis

    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    dom = detect_market_dominance(analysis, ctx)
    assert dom.is_market_dominant is True


def test_market_confidence_weights():
    assert market_confidence_weights("high") == (0.80, 0.20)


def test_normal_layer():
    layer, _ = classify_decision_layer(0.08, data_quality_pct=95, hist_played=30, model_tier="high")
    assert layer == "normal"
