"""Tests for Market Uncertainty Score (MUS) and soft decision."""

from apps.api.services.market_uncertainty import (
    compute_ev_band,
    compute_market_uncertainty,
    resolve_soft_decision,
)
from apps.api.services.odds_context import OutcomeEdge, compute_market_context
from apps.api.services.worldcup_engine import ModelMarkets


def test_mus_high_with_few_books():
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
    mus = compute_market_uncertainty(ctx)
    assert 0.0 <= mus.mus <= 1.0
    assert mus.book_consistency < 0.5


def test_ev_band_pessimistic_below_optimistic():
    outcome = OutcomeEdge(
        selection="Haiti",
        model_prob=0.342,
        model_fair_odds=2.92,
        market_odds=17.0,
        edge_pct=117.0,
        market_implied=0.0588,
        divergence=0.28,
    )
    band = compute_ev_band(outcome, mus=0.45, market_confidence=0.55)
    assert band.optimistic > band.pessimistic
    assert band.base > 0


def test_soft_decision_watch_on_sharp_market_extreme_edge():
    band = compute_ev_band(
        OutcomeEdge(
            selection="Haiti",
            model_prob=0.342,
            model_fair_odds=2.92,
            market_odds=17.0,
            edge_pct=117.0,
            divergence=0.28,
        ),
        mus=0.35,
        market_confidence=0.65,
    )
    action, _ = resolve_soft_decision(
        ev_band=band,
        mus=0.35,
        max_divergence=0.28,
        pick_divergence=0.28,
        confidence_score=40,
        diagnosis_primary="information_asymmetry",
    )
    assert action in ("WATCH", "NO_BET", "WEAK_BET")


def test_soft_decision_strong_on_aligned_low_mus():
    band = compute_ev_band(
        OutcomeEdge(
            selection="Brazil",
            model_prob=0.487,
            model_fair_odds=2.05,
            market_odds=2.3,
            edge_pct=6.0,
            divergence=0.05,
        ),
        mus=0.25,
        market_confidence=0.75,
    )
    action, _ = resolve_soft_decision(
        ev_band=band,
        mus=0.25,
        max_divergence=0.05,
        pick_divergence=0.05,
        confidence_score=50,
    )
    assert action in ("STRONG_BET", "WEAK_BET")
