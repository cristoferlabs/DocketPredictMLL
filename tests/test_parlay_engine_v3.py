"""Tests Parlay Engine v3 — quant portfolio."""

from apps.api.services.parlay_engine import (
    SharpParlayPick,
    build_parlays_from_sharp_picks,
    compute_parlay_metrics,
    pairwise_correlation,
    passes_sharp_parlay_filter,
)
from apps.api.services.market_dominance import MarketDominanceResult
from apps.shared.config import get_settings


def _pick(
    t1: str,
    t2: str,
    fecha: str,
    outcome: str,
    *,
    p: float = 0.55,
    odds: float = 2.0,
    ev: float = 0.05,
    conf: float = 75.0,
    mds: float = 72.0,
) -> SharpParlayPick:
    mid = f"{t1}|{t2}|{fecha}"
    return SharpParlayPick(
        match_id=mid,
        team1=t1,
        team2=t2,
        fecha=fecha,
        ronda="G",
        outcome=outcome,
        market="1X2",
        p_model=p,
        odds=odds,
        ev_fair=ev,
        confidence=conf,
        mds=mds,
        correlation_group=mid,
    )


def test_pairwise_correlation_independent_low():
    a = _pick("Brazil", "France", "2026-06-01", "Brazil")
    b = _pick("Spain", "Germany", "2026-06-02", "Spain")
    c = pairwise_correlation(a, b)
    assert 0.05 <= c <= 0.2


def test_pairwise_correlation_same_match_high():
    a = _pick("Brazil", "France", "2026-06-01", "Brazil")
    b = _pick("Brazil", "France", "2026-06-01", "France")
    assert pairwise_correlation(a, b) >= 0.4


def test_parlay_ev_formula():
    picks = [
        _pick("A", "B", "2026-06-01", "A", p=0.6, odds=2.0),
        _pick("C", "D", "2026-06-02", "C", p=0.55, odds=1.9),
    ]
    p_parlay, odds_parlay, ev_parlay, corr_adj, _avg_corr = compute_parlay_metrics(picks)
    assert p_parlay < picks[0].p_model * picks[1].p_model
    assert odds_parlay == round(2.0 * 1.9, 4)
    expected_ev = p_parlay * odds_parlay - 1.0
    assert abs(ev_parlay - expected_ev) < 1e-6
    assert 0 < corr_adj <= 1.0


def test_sharp_filter_rejects_low_ev():
    pick = _pick("X", "Y", "2026-06-01", "X", ev=0.01)
    ok, reason = passes_sharp_parlay_filter(pick, None)
    assert not ok
    assert "ev_fair" in reason


def test_sharp_filter_rejects_market_dominant():
    pick = _pick("X", "Y", "2026-06-01", "X")
    dom = MarketDominanceResult(
        max_raw_divergence=0.3,
        max_aux_divergence=None,
        layer="extreme",
        layer_reason="test",
        is_market_dominant=True,
        dominance_level="high",
        model_reliability=0.3,
        market_reliability=0.9,
        classification="market",
        diagnosis=None,
        adjustment=None,
        adjusted_market=None,
        outcome_snapshots=[],
    )
    ok, reason = passes_sharp_parlay_filter(pick, dom)
    assert not ok
    assert reason == "market_dominant"


def test_build_parlays_rare_but_valid():
    settings = get_settings()
    picks = [
        _pick("Brazil", "France", "2026-06-01", "Brazil", p=0.62, odds=1.75, ev=0.06),
        _pick("Spain", "Germany", "2026-06-02", "Spain", p=0.58, odds=1.85, ev=0.05),
        _pick("Argentina", "Mexico", "2026-06-03", "Argentina", p=0.57, odds=1.80, ev=0.04),
    ]
    result = build_parlays_from_sharp_picks(picks, min_legs=2, settings=settings)
    if result.tickets:
        t = result.tickets[0]
        assert t.ev_parlay >= settings.parlay_min_ev
        assert t.confidence_avg >= settings.parlay_min_confidence
        assert t.correlation_score <= settings.parlay_max_correlation_score
    else:
        assert result.message_hint or result.reject_reasons


def test_no_parlay_insufficient_picks():
    result = build_parlays_from_sharp_picks(
        [_pick("A", "B", "2026-06-01", "A")],
        min_legs=2,
    )
    assert not result.tickets
    assert "insufficient" in result.message_hint.lower() or result.reject_reasons
