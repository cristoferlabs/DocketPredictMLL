"""Tests for 3-layer market architecture and clean diagnosis."""

from apps.api.services.market_calibration import (
    apply_market_calibration,
    classify_decision_layer,
    diagnose_discrepancy,
    market_confidence_weights,
    model_confidence_tier,
)
from apps.api.services.odds_context import compute_market_context
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


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


def test_extreme_layer_no_blend_scotland_brazil():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    adj, adj_ctx = apply_market_calibration(model, ctx, "Scotland", "Brazil")
    assert adj is not None
    assert adj.layer == "extreme"
    assert adj.blend_applied is False
    assert adj_ctx is None
    assert adj.away == model.away_win


def test_diagnosis_single_primary_market_dominant():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="2026-06-24",
        ronda="",
        grupo="",
        estadio="",
        model=model,
        elo={"Scotland": {"rating": 1848}, "Brazil": {"rating": 2068}},
    )
    max_div = max(o.divergence or 0 for o in ctx.outcomes)
    diag = diagnose_discrepancy(
        analysis, ctx, max_divergence=max_div, hist_played=20, layer="extreme"
    )
    assert diag is not None
    assert diag.primary_type == "market_dominant"
    assert diag.secondary_type == "model_underconfidence"
    assert diag.result == "NO BET"
    assert diag.primary_type != "data_noise"


def test_no_data_noise_with_market_dominant():
    """Con Δ alto, data_noise no debe ser primary aunque hist_played sea bajo."""
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    max_div = max(o.divergence or 0 for o in ctx.outcomes)
    diag = diagnose_discrepancy(
        analysis,
        ctx,
        max_divergence=max_div,
        data_quality_pct=50,
        hist_played=2,
        layer="extreme",
    )
    assert diag is not None
    assert diag.primary_type == "market_dominant"
    assert diag.primary_type != "data_noise"


def test_market_confidence_weights_by_model_tier():
    assert market_confidence_weights("high") == (0.80, 0.20)
    assert market_confidence_weights("medium") == (0.60, 0.40)
    assert market_confidence_weights("low") == (0.30, 0.70)


def test_blend_only_in_doubt_layer_weak_data():
    model = ModelMarkets(
        home_win=0.40,
        draw=0.28,
        away_win=0.32,
        over_25=0.5,
        under_25=0.5,
        btts_yes=0.5,
        btts_no=0.5,
        lambda_home=1.1,
        lambda_away=1.1,
        confidence="low",
    )
    odds = {
        "home_team": "TeamA",
        "away_team": "TeamB",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "TeamA", "price": 2.40},
                {"name": "Draw", "price": 3.30},
                {"name": "TeamB", "price": 3.00},
            ]}],
        }],
    }
    ctx = compute_market_context(model, "TeamA", "TeamB", odds)
    adj, adj_ctx = apply_market_calibration(
        model, ctx, "TeamA", "TeamB", data_quality_pct=55, hist_played=4
    )
    assert adj is not None
    assert adj.layer == "doubt"
    assert adj.blend_applied is True
    assert adj_ctx is not None


def test_normal_layer_no_blend():
    layer, reason = classify_decision_layer(
        0.08, data_quality_pct=95, hist_played=30, model_tier="high"
    )
    assert layer == "normal"
    assert "alineados" in reason
