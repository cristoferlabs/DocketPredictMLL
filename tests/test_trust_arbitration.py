"""Quant audit: Czech–Mexico and Switzerland–Canada trust arbitration."""

from apps.api.services.bet_decision_tree import run_bet_decision_tree
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.sharp_engine import apply_sharp_gate, run_sharp_engine
from apps.api.services.bet_pipeline import run_bet_pipeline
from apps.api.services.trust_arbitration import arbitrate_pick_trust
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings


def _czech_mexico():
    model = ModelMarkets(
        home_win=0.45,
        draw=0.25,
        away_win=0.30,
        over_25=0.48,
        under_25=0.52,
        btts_yes=0.50,
        btts_no=0.50,
        lambda_home=1.25,
        lambda_away=1.15,
        confidence="medium",
    )
    analysis = MatchAnalysis(
        team1="Czech Republic",
        team2="Mexico",
        fecha="2026-06-24",
        ronda="Group",
        grupo="",
        estadio="",
        model=model,
    )
    odds = {
        "home_team": "Czech Republic",
        "away_team": "Mexico",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Czech Republic", "price": 3.75},
                {"name": "Draw", "price": 3.40},
                {"name": "Mexico", "price": 1.95},
            ]}],
        }],
    }
    return analysis, model, odds


def _switzerland_canada():
    model = ModelMarkets(
        home_win=0.57,
        draw=0.24,
        away_win=0.19,
        over_25=0.50,
        under_25=0.50,
        btts_yes=0.48,
        btts_no=0.52,
        lambda_home=1.35,
        lambda_away=1.0,
        confidence="medium",
    )
    analysis = MatchAnalysis(
        team1="Switzerland",
        team2="Canada",
        fecha="2026-06-24",
        ronda="Group",
        grupo="",
        estadio="",
        model=model,
    )
    odds = {
        "home_team": "Switzerland",
        "away_team": "Canada",
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Switzerland", "price": 2.38},
                {"name": "Draw", "price": 3.30},
                {"name": "Canada", "price": 3.10},
            ]}],
        }],
    }
    return analysis, model, odds


def test_czech_mexico_trusts_model_not_symmetric_block():
    analysis, model, odds = _czech_mexico()
    ctx = compute_market_context(model, "Czech Republic", "Mexico", odds)
    dom = detect_market_dominance(analysis, ctx)
    czech = next(o for o in ctx.outcomes if o.selection == "Czech Republic")
    trust = arbitrate_pick_trust(
        pick_model_prob=czech.model_prob,
        pick_market_implied=czech.market_implied,
        pick_divergence=czech.divergence or 0,
        model=model,
        dominance=dom,
    )
    assert trust.trust_side == "model"
    dec = run_bet_decision_tree(
        analysis, ctx, dom, [], settings=get_settings(), historical_accuracy=0.55
    )
    assert dec.trust is not None
    assert dec.trust.trust_side == "model"
    assert dec.soft_action in ("WEAK_BET", "STRONG_BET", "WATCH")
    assert "trust:model" in " ".join(dec.tree_path)


def test_czech_mexico_sharp_gate_passes_with_model_trust():
    """Edge mid-range: árbol + SHARP alineados cuando confiamos en modelo."""
    from apps.api.services.odds_context import compute_market_context
    from apps.api.services.sharp_engine import run_sharp_engine

    analysis, model, odds = _czech_mexico()
    ctx = compute_market_context(model, analysis.team1, analysis.team2, odds)
    sharp = run_sharp_engine(
        analysis, market_ctx=ctx, settings=get_settings(), historical_accuracy=0.55
    )
    assert sharp.decision.trust is not None
    assert sharp.decision.trust.trust_side == "model"
    assert sharp.sharp_allowed is True


def test_switzerland_canada_mid_range_edge_allowed():
    analysis, model, odds = _switzerland_canada()
    ctx = compute_market_context(model, "Switzerland", "Canada", odds)
    dom = detect_market_dominance(analysis, ctx)
    swiss = next(o for o in ctx.outcomes if o.selection == "Switzerland")
    assert swiss.edge_pct > 0
    trust = arbitrate_pick_trust(
        pick_model_prob=swiss.model_prob,
        pick_market_implied=swiss.market_implied,
        pick_divergence=swiss.divergence or 0,
        model=model,
        dominance=dom,
    )
    assert trust.trust_side == "model"
    dec = run_bet_decision_tree(
        analysis, ctx, dom, [], settings=get_settings(), historical_accuracy=0.55
    )
    assert dec.soft_action in ("WEAK_BET", "STRONG_BET", "WATCH")
    if dec.soft_action in ("WEAK_BET", "STRONG_BET"):
        assert dec.no_bet is False


def test_morocco_haiti_still_trusts_market():
    """Extremo con mercado sharp sin respaldo modelo → seguir mercado."""
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
    analysis = MatchAnalysis(
        team1="Morocco",
        team2="Haiti",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
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
    dom = detect_market_dominance(analysis, ctx)
    haiti = next(o for o in ctx.outcomes if o.selection == "Haiti")
    trust = arbitrate_pick_trust(
        pick_model_prob=haiti.model_prob,
        pick_market_implied=haiti.market_implied,
        pick_divergence=haiti.divergence or 0,
        model=model,
        dominance=dom,
    )
    assert trust.trust_side in ("market", "ambiguous")
    dec = run_bet_decision_tree(
        analysis, ctx, dom, [], settings=get_settings(), historical_accuracy=0.55
    )
    assert dec.soft_action in ("NO_BET", "WATCH")
