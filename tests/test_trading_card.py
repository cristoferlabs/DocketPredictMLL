"""Tests for trading card formatting."""

from apps.api.services.odds_context import EvOpportunity, compute_market_context
from apps.api.services.trading_card import (
    build_trading_card,
    format_trading_message,
    prob_risk_emoji,
)
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


def _analysis() -> MatchAnalysis:
    return MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="2026-06-15",
        ronda="Grupo A",
        grupo="A",
        estadio="",
        model=ModelMarkets(
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
        ),
    )


def test_prob_risk_emoji():
    assert prob_risk_emoji(0.60) == "🟢"
    assert prob_risk_emoji(0.45) == "🟡"
    assert prob_risk_emoji(0.25) == "🔴"


def test_build_trading_card_with_ev():
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
    m = _analysis().model
    market = compute_market_context(m, "Scotland", "Brazil", {
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
    card = build_trading_card(
        _analysis(), [opp], odds_available=True, market_ctx=market, historical_accuracy=0.55
    )
    assert card.light in ("verde", "amarillo")
    assert card.no_bet is False
    assert card.pick.selection == "Brazil"
    msg = format_trading_message(card)
    assert "PICK PRINCIPAL" in msg
    assert "Cuotas mercado" in msg
    assert "Edge" in msg
    assert "Rating:" in msg
    assert "/5" in msg


def test_no_bet_red_light():
    card = build_trading_card(_analysis(), [], odds_available=True)
    assert card.no_bet is True
    assert card.pick_rating == 1
    msg = format_trading_message(card)
    assert "NO APOSTAR" in msg
    assert "Cuotas fair" in msg
    assert "Rating:" in msg
    assert "Stake: 0%" in msg


def _morocco_haiti_odds() -> dict:
    return {
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


def test_morocco_haiti_structural_mismatch_watch_or_gate():
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
        ronda="Matchday 14",
        grupo="",
        estadio="",
        model=model,
    )
    market = compute_market_context(model, "Morocco", "Haiti", _morocco_haiti_odds())
    card = build_trading_card(analysis, [], odds_available=True, market_ctx=market)
    assert card.market_divergence_flag is True
    assert card.decision_layer == "extreme"
    assert card.decision is not None
    assert card.decision.soft_action in ("WATCH", "NO_BET", "WEAK_BET", "STRONG_BET")
    assert card.decision.ev_band is not None
    msg = format_trading_message(card)
    assert "MISMATCH ESTRUCTURAL" in msg
    assert "Market Uncertainty" in msg
    assert "Banda EV" in msg
    assert "Morocco: 37.4%" in msg
    if card.decision.soft_action == "WATCH":
        assert "VIGILAR" in msg
    elif card.decision.soft_action in ("WEAK_BET", "STRONG_BET"):
        assert "PICK PRINCIPAL" in msg
    else:
        assert "NO APOSTAR" in msg


def test_scotland_ev_outlier_blocks_bet():
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
        ronda="Matchday 14",
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
    market = compute_market_context(model, "Scotland", "Brazil", odds)
    scot = next(o for o in market.outcomes if o.selection == "Scotland")
    assert scot.ev_raw_pct > 50
    assert scot.ev_fair_pct > 0
    card = build_trading_card(analysis, [], odds_available=True, market_ctx=market)
    assert card.decision is not None
    assert card.decision.soft_action in ("WATCH", "NO_BET")
    assert card.no_bet is True
    msg = format_trading_message(card)
    assert "NO APOSTAR" in msg or "VIGILAR" in msg
