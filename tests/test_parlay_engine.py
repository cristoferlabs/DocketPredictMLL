"""Tests for Parlay Engine (OUTPUT B)."""

from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.parlay_engine import (
    build_parlay_tickets,
    build_parlays_from_legs,
    evaluate_parlay_leg,
)
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


def _analysis(
    t1: str,
    t2: str,
    model: ModelMarkets,
) -> MatchAnalysis:
    return MatchAnalysis(
        team1=t1,
        team2=t2,
        fecha="2026-06-24",
        ronda="Grupo A",
        grupo="A",
        estadio="",
        model=model,
    )


def _odds(t1: str, t2: str, home_p: float, draw_p: float, away_p: float) -> dict:
    return {
        "home_team": t1,
        "away_team": t2,
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": t1, "price": home_p},
                {"name": "Draw", "price": draw_p},
                {"name": t2, "price": away_p},
            ]}],
        }],
    }


def test_parlay_effective_prob_not_max_model_market():
    """effective_prob usa modelo calibrado, no max(model, market)."""
    model = ModelMarkets(
        home_win=0.62,
        draw=0.22,
        away_win=0.16,
        over_25=0.50,
        under_25=0.50,
        btts_yes=0.48,
        btts_no=0.52,
        lambda_home=1.4,
        lambda_away=0.9,
        confidence="high",
    )
    t1, t2 = "Bosnia", "Qatar"
    analysis = _analysis(t1, t2, model)
    ctx = compute_market_context(model, t1, t2, _odds(t1, t2, 1.55, 4.0, 6.0))
    leg = evaluate_parlay_leg(analysis, ctx)
    assert leg.stable is True
    assert leg.effective_prob <= leg.model_prob + 0.001
    assert leg.effective_prob >= leg.model_prob - 0.05


def test_stable_favorite_eligible():
    model = ModelMarkets(
        home_win=0.62,
        draw=0.22,
        away_win=0.16,
        over_25=0.50,
        under_25=0.50,
        btts_yes=0.48,
        btts_no=0.52,
        lambda_home=1.4,
        lambda_away=0.9,
        confidence="high",
    )
    t1, t2 = "Bosnia", "Qatar"
    analysis = _analysis(t1, t2, model)
    ctx = compute_market_context(model, t1, t2, _odds(t1, t2, 1.55, 4.0, 6.0))
    leg = evaluate_parlay_leg(analysis, ctx)
    assert leg.stable is True
    assert leg.pick_score > 0


def test_morocco_haiti_rejected_unstable():
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
    t1, t2 = "Morocco", "Haiti"
    analysis = _analysis(t1, t2, model)
    ctx = compute_market_context(model, t1, t2, _odds(t1, t2, 1.19, 7.5, 17.0))
    dom = detect_market_dominance(analysis, ctx)
    leg = evaluate_parlay_leg(analysis, ctx, dominance=dom)
    assert leg.stable is False
    assert leg.exclude_reason is not None


def test_build_parlay_ticket_from_eligible_legs():
    legs = []
    for i, (t1, t2, p) in enumerate([
        ("Bosnia", "Qatar", 0.62),
        ("Brazil", "Scotland", 0.58),
        ("Spain", "Japan", 0.61),
        ("France", "Canada", 0.57),
    ]):
        model = ModelMarkets(
            home_win=p,
            draw=0.22,
            away_win=1 - p - 0.22,
            over_25=0.5,
            under_25=0.5,
            btts_yes=0.5,
            btts_no=0.5,
            lambda_home=1.3,
            lambda_away=1.0,
            confidence="high",
        )
        analysis = _analysis(t1, t2, model)
        ctx = compute_market_context(
            model, t1, t2,
            _odds(t1, t2, 1.5 + i * 0.05, 4.0, 5.5),
        )
        leg = evaluate_parlay_leg(analysis, ctx)
        leg.stable = True
        leg.exclude_reason = None
        leg.effective_prob = p
        leg.odds = 2.5
        leg.pick_score = p * 0.8
        leg.ev_adjusted = 0.05
        legs.append(leg)

    tickets = build_parlay_tickets(legs, min_legs=3, max_legs=4)
    assert len(tickets) >= 1
    assert tickets[0].n_legs >= 3
    assert tickets[0].combined_prob > 0
    assert tickets[0].combo_score > 0


def test_build_parlays_from_legs_message_hint_when_few():
    result = build_parlays_from_legs([])
    assert not result.tickets
    assert "0 pierna" in result.message_hint or result.message_hint == ""
