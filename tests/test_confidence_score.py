"""Tests for unified confidence score."""

from apps.api.services.confidence_score import (
    compute_mds,
    compute_unified_confidence,
    sharp_composite_passes,
)
from apps.api.services.market_dominance import detect_market_dominance
from apps.api.services.odds_context import compute_market_context
from apps.api.services.trust_arbitration import TrustArbitration
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets


def test_cold_start_caps_confidence():
    trust = TrustArbitration(
        model_confidence=0.90,
        market_confidence=0.55,
        trust_side="model",
        trust_ratio=1.64,
        w_model=0.62,
        w_market=0.38,
        rationale="model",
    )
    uncapped = compute_unified_confidence(
        mds=85, model_reliability=0.80, trust=trust, cold_start=False
    )
    capped = compute_unified_confidence(
        mds=85, model_reliability=0.80, trust=trust, cold_start=True
    )
    assert capped <= 58
    assert uncapped > capped


def test_sharp_composite_gate():
    from apps.shared.config import get_settings

    settings = get_settings()
    assert sharp_composite_passes(settings.sharp_min_composite, settings=settings)
    assert not sharp_composite_passes(settings.sharp_min_composite - 1, settings=settings)


def test_unified_confidence_weights():
    trust = TrustArbitration(
        model_confidence=0.80,
        market_confidence=0.60,
        trust_side="model",
        trust_ratio=1.33,
        w_model=0.57,
        w_market=0.43,
        rationale="test",
    )
    mds = 70
    score = compute_unified_confidence(
        mds=mds,
        model_reliability=0.75,
        trust=trust,
    )
    expected = int(round((0.4 * 0.70 + 0.3 * 0.75 + 0.3 * 0.80) * 100))
    assert score == expected


def test_mds_with_trust_model_bonus():
    model = ModelMarkets(
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
    )
    analysis = MatchAnalysis(
        team1="Scotland",
        team2="Brazil",
        fecha="",
        ronda="",
        grupo="",
        estadio="",
        model=model,
    )
    ctx = compute_market_context(model, "Scotland", "Brazil", {
        "bookmakers": [{
            "key": "b1",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Scotland", "price": 4.5},
                {"name": "Draw", "price": 3.5},
                {"name": "Brazil", "price": 2.3},
            ]}],
        }],
    })
    dom = detect_market_dominance(analysis, ctx)
    trust = TrustArbitration(
        model_confidence=0.90,
        market_confidence=0.55,
        trust_side="model",
        trust_ratio=1.64,
        w_model=0.62,
        w_market=0.38,
        rationale="model",
    )
    mds = compute_mds(dom, trust)
    assert mds >= 60
