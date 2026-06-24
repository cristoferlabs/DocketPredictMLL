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
    card = build_trading_card(_analysis(), [opp], odds_available=True, market_ctx=market)
    assert card.light == "verde"
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


def test_morocco_haiti_extreme_divergence_blocks_bet():
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
    assert card.no_bet is True
    assert card.decision_layer == "extreme"
    msg = format_trading_message(card)
    assert "NO APOSTAR" in msg
    assert "FILTRO MERCADO" in msg
    assert "PRIMARY:" in msg
    assert "Edge post-ajuste: IGNORADO" in msg
    assert "Modelo ajustado" not in msg
    assert "También:" not in msg


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
    assert scot.edge_pct > 100
    card = build_trading_card(analysis, [], odds_available=True, market_ctx=market)
    assert card.no_bet is True
    assert card.pick.selection != "Scotland" or card.no_bet
    msg = format_trading_message(card)
    assert "NO APOSTAR" in msg
