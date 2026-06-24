"""Tests for 3-layer bet pipeline invariants."""

from apps.api.services.bet_pipeline import run_bet_pipeline, validate_market_layer_invariants
from apps.api.services.odds_context import compute_market_context
from apps.api.services.trading_card import build_trading_card, format_trading_message
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


def _morocco_haiti() -> tuple[MatchAnalysis, dict]:
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
        fecha="2026-06-24",
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
    return analysis, odds


def test_pipeline_layer_order_model_unchanged():
    analysis, odds = _morocco_haiti()
    model_before = (
        analysis.model.home_win,
        analysis.model.draw,
        analysis.model.away_win,
    )
    ctx = compute_market_context(analysis.model, "Morocco", "Haiti", odds)
    result = run_bet_pipeline(analysis, [], market_ctx=ctx)

    assert (
        analysis.model.home_win,
        analysis.model.draw,
        analysis.model.away_win,
    ) == model_before
    assert result.model.markets is analysis.model
    validate_market_layer_invariants(analysis.model, result.market.dominance)
    assert result.market.dominance.adjusted_market is None
    assert result.decision.soft_action in ("WATCH", "NO_BET", "WEAK_BET", "STRONG_BET")


def test_pipeline_decision_uses_fair_ev_guardrails():
    """Pick respeta fair EV; outliers raw (Haiti) no dominan la decisión."""
    analysis, odds = _morocco_haiti()
    ctx = compute_market_context(analysis.model, "Morocco", "Haiti", odds)
    haiti_edge = next(o for o in ctx.outcomes if o.selection == "Haiti")
    assert haiti_edge.model_prob == analysis.model.away_win
    result = run_bet_pipeline(analysis, [], market_ctx=ctx)
    assert result.decision.pick is not None
    assert result.decision.pick.selection == "Morocco"
    assert result.decision.pick.model_prob == analysis.model.home_win


def test_trading_message_reflects_three_layers():
    analysis, odds = _morocco_haiti()
    ctx = compute_market_context(analysis.model, "Morocco", "Haiti", odds)
    card = build_trading_card(analysis, [], market_ctx=ctx)
    msg = format_trading_message(card)
    assert "Nivel 1 — MODEL" in msg
    assert "Nivel 2 — MARKET" in msg
    assert "Nivel 3 — DECISION" in msg
    assert "Modelo ajustado" not in msg
    assert "Morocco: 37.4%" in msg
