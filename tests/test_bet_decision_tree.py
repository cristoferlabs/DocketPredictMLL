"""Tests for Bet Decision Tree module."""

from apps.api.services.bet_decision_tree import run_bet_decision_tree
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import EvOpportunity, compute_market_context
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings


def _scotland_brazil():
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
    odds = {
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
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="2026-06-24",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    dom = detect_market_dominance(analysis, ctx)
    return analysis, ctx, dom


def test_scotland_brazil_tree_watch_or_no_bet_on_structural():
    analysis, ctx, dom = _scotland_brazil()
    dec = run_bet_decision_tree(analysis, ctx, dom, [], settings=get_settings())
    assert dec.soft_action in ("WATCH", "NO_BET", "WEAK_BET")
    assert "structural_mismatch:yes" in dec.tree_path
    assert "soft_gate:" in " ".join(dec.tree_path)
    assert dec.ev_band is not None


def test_ev_positive_bet_recommended():
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
    dom = detect_market_dominance(analysis, ctx)
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
    dec = run_bet_decision_tree(analysis, ctx, dom, [opp], settings=get_settings())
    assert dec.no_bet is False
    assert dec.soft_action in ("STRONG_BET", "WEAK_BET")
    assert "pick:ev_opportunity" in dec.tree_path
    assert dec.pick is not None
    assert dec.pick.selection == "Brazil"
