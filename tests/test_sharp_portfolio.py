"""Tests SHARP v2 portfolio mode."""

from apps.api.services.confidence_score import compute_unified_confidence
from apps.api.services.parlay_engine import SharpParlayPick, build_parlays_from_sharp_picks
from apps.api.services.sharp_portfolio import promote_portfolio_picks, rank_sharp_picks
from apps.api.services.trust_arbitration import TrustArbitration


def _trust() -> TrustArbitration:
    return TrustArbitration(
        trust_side="model",
        model_confidence=0.72,
        market_confidence=0.55,
        trust_ratio=1.2,
        w_model=0.6,
        w_market=0.4,
        rationale="test",
    )


def test_cold_start_preserves_variance_not_cap_58():
    high = compute_unified_confidence(
        mds=78, model_reliability=0.7, trust=_trust(), cold_start=False
    )
    low = compute_unified_confidence(
        mds=52, model_reliability=0.5, trust=_trust(), cold_start=False
    )
    high_cs = compute_unified_confidence(
        mds=78, model_reliability=0.7, trust=_trust(), cold_start=True
    )
    low_cs = compute_unified_confidence(
        mds=52, model_reliability=0.5, trust=_trust(), cold_start=True
    )
    assert high > low
    assert high_cs > low_cs
    assert high_cs != 58 or low_cs != 58 or high_cs != low_cs


def test_portfolio_promotes_top_percentile():
    picks = [
        SharpParlayPick(
            match_id=f"m{i}",
            team1=f"T{i}a",
            team2=f"T{i}b",
            fecha="2026-06-15",
            ronda="G",
            outcome="T1a",
            market="1X2",
            p_model=0.55 + i * 0.02,
            odds=2.1,
            ev_fair=0.02 + i * 0.01,
            confidence=58.0 + i * 3,
            mds=60.0 + i * 2,
            correlation_group=f"m{i}",
            reject_reason="confidence 58 < 70",
        )
        for i in range(5)
    ]
    promoted = promote_portfolio_picks(picks, top_pct=0.4, top_k=3)
    eligible = [p for p in promoted if p.eligible]
    assert len(eligible) >= 2
    ranked = rank_sharp_picks(picks, top_pct=0.4, top_k=3)
    assert ranked[0].rank_score >= ranked[-1].rank_score


def test_parlay_build_with_portfolio_promotion():
    picks = [
        SharpParlayPick(
            match_id="a|b|2026-06-15",
            team1="Japan",
            team2="Sweden",
            fecha="2026-06-15",
            ronda="G",
            outcome="Japan",
            market="1X2",
            p_model=0.62,
            odds=1.95,
            ev_fair=0.06,
            confidence=64.0,
            mds=66.0,
            correlation_group="a|b|2026-06-15",
            reject_reason="mds 66 < 70",
        ),
        SharpParlayPick(
            match_id="c|d|2026-06-16",
            team1="Morocco",
            team2="Haiti",
            fecha="2026-06-16",
            ronda="G",
            outcome="Morocco",
            market="1X2",
            p_model=0.71,
            odds=1.75,
            ev_fair=0.08,
            confidence=72.0,
            mds=74.0,
            correlation_group="c|d|2026-06-16",
            reject_reason=None,
        ),
        SharpParlayPick(
            match_id="e|f|2026-06-17",
            team1="Brazil",
            team2="Scotland",
            fecha="2026-06-17",
            ronda="G",
            outcome="Brazil",
            market="1X2",
            p_model=0.68,
            odds=1.55,
            ev_fair=0.05,
            confidence=61.0,
            mds=63.0,
            correlation_group="e|f|2026-06-17",
            reject_reason="confidence 61 < 70",
        ),
    ]
    result = build_parlays_from_sharp_picks(picks, min_legs=2, max_legs=3, top_n=2)
    assert len(result.eligible_picks) >= 2
