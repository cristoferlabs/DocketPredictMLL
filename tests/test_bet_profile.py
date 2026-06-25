"""Bet Profile Layer — separación predicción vs apuesta."""

from apps.api.services.bet_profile import build_bet_profile, classify_probability
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.sharp_engine import run_sharp_engine
from apps.api.services.parlay_engine import evaluate_parlay_leg
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


def test_classify_probability_bands():
    assert classify_probability(0.65) == "HIGH"
    assert classify_probability(0.50) == "MEDIUM"
    assert classify_probability(0.25) == "LONGSHOT"


def test_scotland_brazil_profile_four_sides():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, "Scotland", "Brazil", odds)
    dom = detect_market_dominance(analysis, ctx)
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    parlay = evaluate_parlay_leg(analysis, ctx, dom, sharp=sharp)

    profile = build_bet_profile(
        model=model,
        team1="Scotland",
        team2="Brazil",
        market_ctx=ctx,
        dominance=dom,
        decision=sharp.decision,
        parlay_leg=parlay,
        sharp_allowed=sharp.sharp_allowed,
        sharp_gate_label="WATCH",
        mds=sharp.mds,
    )

    assert profile.most_likely is not None
    assert profile.most_likely.selection == "Brazil"
    assert profile.most_likely.probability == 0.503

    assert profile.value_side is not None
    assert profile.value_side.selection == "Scotland"
    # Fair EV puede ser positivo pero no accionable por Δ estructural
    assert profile.value_side.ev_pct is not None
    assert profile.value_side.note is not None
    assert "estructural" in profile.value_side.note.lower() or "fair" in profile.value_side.note.lower()

    assert profile.parlay_side is not None
    assert profile.parlay_side.selection == "Brazil"
    # v3: Scotland vs Brazil — EV fair negativo → no elegible parlay SHARP
    assert profile.parlay_side.action in ("RECHAZADO", "N/A")

    assert profile.sharp_side is not None
    assert profile.sharp_side.action == "WATCH"
    assert profile.sharp_side.selection == "Brazil"


def test_longshot_not_parlay_favorite():
    from apps.api.services.parlay_engine import ParlayLeg

    profile = build_bet_profile(
        model=ModelMarkets(
            home_win=0.28,
            draw=0.30,
            away_win=0.42,
            over_25=0.5,
            under_25=0.5,
            btts_yes=0.5,
            btts_no=0.5,
            lambda_home=1.0,
            lambda_away=1.0,
            confidence="low",
        ),
        team1="A",
        team2="B",
        market_ctx=None,
        dominance=None,
        decision=None,
        parlay_leg=ParlayLeg(
            team1="A",
            team2="B",
            fecha="",
            ronda="",
            selection="A",
            model_prob=0.28,
            market_prob=0.15,
            effective_prob=0.28,
            odds=5.0,
            ev_adjusted=0.2,
            stable=True,
        ),
    )
    assert profile.most_likely is not None
    assert profile.most_likely.selection == "B"
    assert profile.parlay_side is not None
    assert profile.parlay_side.action == "N/A"


def test_trading_message_includes_bet_profile():
    analysis, model, odds = _scotland_brazil()
    ctx = compute_market_context(model, analysis.team1, analysis.team2, odds)
    card = build_trading_card(analysis, [], market_ctx=ctx)
    msg = format_trading_message(card)
    assert "BET PROFILE" in msg
    assert "FAVORITO DEL PARTIDO" in msg
    assert "VALUE SIDE" in msg
    assert "PARLAY SIDE" in msg
    assert "SHARP SIDE" in msg
    assert "Brazil" in msg
    assert "Scotland" in msg
