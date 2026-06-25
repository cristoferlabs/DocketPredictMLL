"""WATCH coherence — confianza preservada, stake exploratorio, SHARP vs PARLAY."""

from apps.api.services.bet_decision_tree import run_bet_decision_tree
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.sharp_engine import apply_sharp_gate, run_sharp_engine
from apps.api.services.bet_pipeline import run_bet_pipeline
from apps.api.services.trading_card import build_trading_card, format_trading_message
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
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="2026-06-24",
        ronda="Group",
        grupo="",
        estadio="",
        model=model,
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
    return analysis, model, odds


def test_watch_preserves_confidence_not_zero():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    dom = detect_market_dominance(analysis, ctx)
    dec = run_bet_decision_tree(analysis, ctx, dom, [], settings=get_settings())
    assert dec.soft_action == "WATCH"
    assert dec.confidence_score >= 45
    assert dec.confidence_score <= 85


def test_sharp_watch_does_not_crush_confidence_or_state():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    dec = sharp.decision
    assert dec.soft_action == "WATCH"
    assert dec.confidence_score >= 45
    assert sharp.sharp_allowed is False
    assert "sharp_gate:WATCH" in " ".join(dec.tree_path)


def test_watch_high_ev_gets_exploratory_stake():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    pipeline = run_bet_pipeline(analysis, market_ctx=ctx, settings=get_settings())
    dec = pipeline.decision
    settings = get_settings()
    if dec.ev_band and dec.ev_band.optimistic >= settings.watch_exploratory_ev_threshold:
        sharp = apply_sharp_gate(pipeline, settings=settings)
        assert sharp.decision.stake_pct >= settings.watch_micro_stake_pct


def test_scotland_brazil_sharp_watch_parlay_rejected_v3():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    card = build_trading_card(analysis, [], market_ctx=ctx)
    assert card.sharp_gate_label == "WATCH"
    assert card.parlay_leg is not None
    assert card.parlay_leg.stable is False
    assert card.parlay_leg.exclude_reason is not None
    assert card.parlay_leg.selection == "Brazil"
    msg = format_trading_message(card)
    assert "PARLAY" in msg.upper()
    assert card.confidence_score >= 45
