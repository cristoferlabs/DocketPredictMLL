"""Tests for formal Market Dominance module."""

from apps.api.services.market_dominance import detect_market_dominance
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


def _analysis(model: ModelMarkets) -> MatchAnalysis:
    return MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="2026-06-24",
        ronda="",
        grupo="",
        estadio="",
        model=model,
        elo={"Scotland": {"rating": 1848}, "Brazil": {"rating": 2068}},
    )


def test_scotland_brazil_market_dominant():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    dom = detect_market_dominance(_analysis(model), ctx)
    assert dom.layer == "extreme"
    assert dom.is_market_dominant is True
    assert dom.classification == "market_dominant"
    assert dom.adjustment is None or (
        dom.adjustment.blend_applied is False and dom.adjustment.home == model.home_win
    )
    assert 0.0 <= dom.model_reliability <= 1.0
    assert 0.0 <= dom.market_reliability <= 1.0
    assert dom.diagnosis is not None
    assert dom.diagnosis.primary_type == "market_dominant"
    assert dom.uncertainty is not None
    assert dom.uncertainty is not None


def test_scotland_brazil_information_asymmetry_with_news():
    model = _scotland_brazil_model()
    ctx = compute_market_context(model, "Scotland", "Brazil", _scotland_brazil_odds())
    dom = detect_market_dominance(_analysis(model), ctx, has_injury_news=True)
    assert dom.diagnosis is not None
    assert dom.diagnosis.primary_type == "information_asymmetry"


def test_morocco_haiti_information_asymmetry_not_noise():
    model = ModelMarkets(
        home_win=0.374,
        draw=0.284,
        away_win=0.342,
        over_25=0.371,
        under_25=0.629,
        btts_yes=0.380,
        btts_no=0.620,
        lambda_home=1.3,
        lambda_away=1.1,
        confidence="medium",
    )
    odds = {
        "home_team": "Morocco",
        "away_team": "Haiti",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Morocco", "price": 1.19},
                {"name": "Draw", "price": 7.50},
                {"name": "Haiti", "price": 17.00},
            ]}],
        }],
    }
    ctx = compute_market_context(model, "Morocco", "Haiti", odds)
    analysis = MatchAnalysis(
        team1="Morocco",
        team2="Haiti",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    dom = detect_market_dominance(
        analysis,
        ctx,
        has_injury_news=True,
    )
    assert dom.diagnosis is not None
    assert dom.diagnosis.primary_type == "information_asymmetry"
    assert dom.adjustment is not None
    assert dom.adjustment.blend_applied is False
    assert dom.adjustment.home == model.home_win
    assert dom.adjusted_market is None
    assert dom.max_aux_divergence is None


def test_doubt_layer_never_blends_probs():
    """Cualquier capa (doubt/normal/extreme): sin mezcla modelo-mercado."""
    model = ModelMarkets(
        home_win=0.42,
        draw=0.28,
        away_win=0.30,
        over_25=0.45,
        under_25=0.55,
        btts_yes=0.48,
        btts_no=0.52,
        lambda_home=1.2,
        lambda_away=1.1,
        confidence="low",
    )
    odds = {
        "home_team": "TeamA",
        "away_team": "TeamB",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "TeamA", "price": 2.10},
                {"name": "Draw", "price": 3.40},
                {"name": "TeamB", "price": 3.80},
            ]}],
        }],
    }
    ctx = compute_market_context(model, "TeamA", "TeamB", odds)
    analysis = MatchAnalysis(
        team1="TeamA",
        team2="TeamB",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    dom = detect_market_dominance(analysis, ctx, data_quality_pct=55.0, hist_played=5)
    assert dom.adjustment is not None
    assert dom.adjustment.blend_applied is False
    assert dom.adjustment.home == model.home_win
    assert dom.adjusted_market is None
