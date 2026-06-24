"""Tests for SHARP Engine gate."""

from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import EvOpportunity, compute_market_context
from apps.api.services.sharp_engine import apply_sharp_gate, compute_mds, run_sharp_engine
from apps.api.services.bet_pipeline import run_bet_pipeline
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings


def _aligned_match():
    model = ModelMarkets(
        home_win=0.255,
        draw=0.259,
        away_win=0.487,
        over_25=0.475,
        under_25=0.525,
        btts_yes=0.491,
        btts_no=0.509,
        lambda_home=1.1,
        lambda_away=1.4,
        confidence="medium",
    )
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    ctx = compute_market_context(model, "Scotland", "Brazil", {
        "home_team": "Scotland",
        "away_team": "Brazil",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Scotland", "price": 4.5},
                {"name": "Draw", "price": 3.5},
                {"name": "Brazil", "price": 2.3},
            ]}],
        }],
    })
    return analysis, ctx, model


def test_compute_mds_range():
    analysis, ctx, _ = _aligned_match()
    dom = detect_market_dominance(analysis, ctx)
    mds = compute_mds(dom)
    assert 0 <= mds <= 100


def test_sharp_gate_blocks_low_mds_extreme():
    model = ModelMarkets(
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
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    ctx = compute_market_context(model, "Scotland", "Brazil", {
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
    })
    pipeline = run_bet_pipeline(analysis, [], market_ctx=ctx, settings=get_settings())
    sharp = apply_sharp_gate(pipeline, settings=get_settings())
    assert sharp.sharp_allowed is False
    path_str = " ".join(sharp.decision.tree_path)
    assert "sharp_gate:WATCH" in path_str or "sharp_gate:NO_BET" in path_str


def test_run_sharp_engine_with_ev_opportunity():
    analysis, ctx, _ = _aligned_match()
    opp = EvOpportunity(
        market="1X2",
        selection="Brazil",
        model_prob=0.487,
        book_odds=2.2,
        implied_prob=0.45,
        expected_value=0.062,
        edge_pct=5.0,
        priority="high",
        fair_odds=2.05,
        edge_fair=0.058,
        raw_odds=2.2,
        metadata={"kelly_stake": 0.01},
    )
    sharp = run_sharp_engine(analysis, [opp], market_ctx=ctx, settings=get_settings())
    assert sharp.mds > 0
    assert "sharp_gate" in " ".join(sharp.decision.tree_path)
