"""Tests Parlay Engine v3 — adapter + portfolio (legacy file)."""

from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.parlay_engine import (
    SharpParlayPick,
    build_parlays_from_legs,
    build_parlays_from_sharp_picks,
    evaluate_parlay_leg,
    extract_sharp_parlay_pick,
)
from apps.api.services.sharp_engine import run_sharp_engine
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import get_settings


def _analysis(t1: str, t2: str, model: ModelMarkets) -> MatchAnalysis:
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


def test_evaluate_parlay_leg_requires_sharp():
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
    assert leg.stable is False
    assert "SHARP" in (leg.exclude_reason or "")


def test_sharp_pick_uses_p_model_not_max():
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
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    leg = evaluate_parlay_leg(analysis, ctx, sharp=sharp)
    if leg.sharp_pick and leg.sharp_pick.eligible:
        assert leg.effective_prob == leg.model_prob


def test_morocco_haiti_rejected_market_dominant():
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
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    sp = extract_sharp_parlay_pick(analysis, sharp, ctx)
    assert sp is not None
    assert not sp.eligible or sp.reject_reason is not None


def test_build_parlay_from_sharp_picks():
    picks = [
        SharpParlayPick(
            match_id="A|B|2026-06-01",
            team1="Bosnia",
            team2="Qatar",
            fecha="2026-06-01",
            ronda="G",
            outcome="Bosnia",
            market="1X2",
            p_model=0.62,
            odds=1.75,
            ev_fair=0.06,
            confidence=75.0,
            mds=72.0,
            correlation_group="A|B|2026-06-01",
        ),
        SharpParlayPick(
            match_id="C|D|2026-06-02",
            team1="Brazil",
            team2="Scotland",
            fecha="2026-06-02",
            ronda="G",
            outcome="Brazil",
            market="1X2",
            p_model=0.58,
            odds=1.85,
            ev_fair=0.05,
            confidence=74.0,
            mds=71.0,
            correlation_group="C|D|2026-06-02",
        ),
    ]
    result = build_parlays_from_sharp_picks(picks, min_legs=2)
    if result.tickets:
        t = result.tickets[0]
        assert t.n_legs >= 2
        assert abs(t.ev_parlay - (t.combined_prob * t.combined_odds - 1.0)) < 1e-4


def test_build_parlays_from_legs_message_hint_when_few():
    result = build_parlays_from_legs([])
    assert not result.tickets
    assert "SHARP" in result.message_hint or result.reject_reasons
