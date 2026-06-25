"""Integration: árbol no emite STRONG_BET con divergencia >12pp."""

from apps.api.services.bet_decision_tree import run_bet_decision_tree
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import EvOpportunity, compute_market_context
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings

def test_south_africa_outlier_forces_investigate_watch():
    """Δ ~29pp en 1X2 → INVESTIGATE / WATCH, no STRONG_BET ni SHARP single."""
    model = ModelMarkets(
        home_win=0.45,
        draw=0.24,
        away_win=0.31,
        over_25=0.51,
        under_25=0.49,
        btts_yes=0.54,
        btts_no=0.46,
        lambda_home=1.61,
        lambda_away=1.18,
        confidence="medium",
    )
    analysis = MatchAnalysis(
        team1="South Africa",
        team2="South Korea",
        fecha="2026-06-24",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    odds = {
        "home_team": "South Africa",
        "away_team": "South Korea",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "South Africa", "price": 6.10},
                {"name": "Draw", "price": 4.00},
                {"name": "South Korea", "price": 1.59},
            ]}],
        }],
    }
    ctx = compute_market_context(model, "South Africa", "South Korea", odds)
    dom = detect_market_dominance(analysis, ctx)
    sa_impl = next(
        o.market_implied for o in ctx.outcomes if o.selection == "South Africa"
    )
    opp = EvOpportunity(
        market="1X2",
        selection="South Africa",
        model_prob=0.45,
        book_odds=6.10,
        implied_prob=sa_impl,
        expected_value=0.45 * 6.10 - 1.0,
        edge_pct=(0.45 - sa_impl) * 100,
        priority="high",
        fair_odds=1 / 0.45,
        edge_fair=0.45 - sa_impl,
        raw_odds=6.10,
    )
    dec = run_bet_decision_tree(analysis, ctx, dom, [opp], settings=get_settings())
    assert dec.soft_action == "WATCH"
    assert any("investigate" in p or "1x2_investigate" in p for p in dec.tree_path)


def test_bosnia_scenario_no_strong_bet_on_13pp_gap():
    """Mercado 73% vs modelo 60% — máximo WEAK_BET o WATCH."""
    model = ModelMarkets(
        home_win=0.60,
        draw=0.22,
        away_win=0.18,
        over_25=0.38,
        under_25=0.62,
        btts_yes=0.35,
        btts_no=0.65,
        lambda_home=1.79,
        lambda_away=0.50,
        confidence="medium",
    )
    analysis = MatchAnalysis(
        team1="Bosnia",
        team2="Qatar",
        fecha="2026-06-24",
        ronda="",
        grupo="",
        estadio="",
        model=model,
        elo={"Bosnia": {"rating": 1680}, "Qatar": {"rating": 1420}},
    )
    odds = {
        "home_team": "Bosnia",
        "away_team": "Qatar",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Bosnia", "price": 1.37},
                {"name": "Draw", "price": 4.80},
                {"name": "Qatar", "price": 9.00},
            ]}],
        }],
    }
    ctx = compute_market_context(model, "Bosnia", "Qatar", odds)
    dom = detect_market_dominance(analysis, ctx)
    opp = EvOpportunity(
        market="1X2",
        selection="Bosnia",
        model_prob=0.60,
        book_odds=1.37,
        implied_prob=0.73,
        expected_value=0.05,
        edge_pct=5.0,
        priority="high",
        fair_odds=1.67,
        edge_fair=0.04,
        raw_odds=1.37,
    )
    dec = run_bet_decision_tree(analysis, ctx, dom, [opp], settings=get_settings())
    assert dec.soft_action != "STRONG_BET"
    assert any("align:" in p for p in dec.tree_path)
