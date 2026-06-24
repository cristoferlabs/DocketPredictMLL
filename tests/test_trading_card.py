"""Tests for trading card formatting."""

from apps.api.services.odds_context import EvOpportunity
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
    card = build_trading_card(_analysis(), [opp], odds_available=True)
    assert card.light == "verde"
    assert card.pick.selection == "Brazil"
    msg = format_trading_message(card)
    assert "PICK PRINCIPAL" in msg
    assert "Brazil" in msg
    assert "EV:" in msg
    assert "🚦" in msg


def test_no_bet_red_light():
    card = build_trading_card(_analysis(), [], odds_available=True)
    assert card.no_bet is True
    msg = format_trading_message(card)
    assert "NO APOSTAR" in msg
    assert "🔴" in msg
