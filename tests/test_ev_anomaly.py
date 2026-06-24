"""Tests for EV anomaly detection and Kelly sizing."""

from apps.worker.ml.ev_anomaly import check_ev_anomaly, evaluate_pick, fractional_kelly, kelly_full


def test_kelly_full_positive_edge():
    k = kelly_full(0.55, 2.0)
    assert k > 0


def test_fractional_kelly_uses_fraction():
    full = kelly_full(0.55, 2.0)
    frac = fractional_kelly(0.55, 2.0, kelly_fraction=0.25)
    assert frac <= full
    assert frac <= 0.25


def test_blocks_high_edge():
    allowed, flags = check_ev_anomaly(
        edge_fair=0.20,
        ev_fair=0.10,
        model_prob=0.60,
        fair_implied=0.40,
    )
    assert not allowed
    assert any("edge_fair" in f for f in flags)


def test_evaluate_pick_allowed():
    stake = evaluate_pick(
        model_prob=0.52,
        fair_odds=2.1,
        edge_fair=0.05,
        ev_fair=0.04,
        fair_implied=0.476,
    )
    assert stake.allowed
    assert stake.stake_units >= 0
