"""Tests for model-market alignment tiers and penalties."""

from apps.api.services.market_alignment import (
    alignment_status,
    confidence_divergence_penalty,
    gap_pp,
    max_soft_action_for_gap,
)


def test_gap_pp():
    assert gap_pp(0.60, 0.73) == 13.0
    assert gap_pp(0.55, None) == 0.0


def test_alignment_tiers():
    assert alignment_status(3.0)[0] == "aligned"
    assert alignment_status(7.0)[0] == "mild"
    assert alignment_status(12.0)[0] == "divergence"
    assert alignment_status(18.0)[0] == "alert"


def test_confidence_penalty_increases_with_gap():
    assert confidence_divergence_penalty(4.0) == 0
    assert confidence_divergence_penalty(11.0, model_prob=0.58, market_implied=0.47) == 14
    assert confidence_divergence_penalty(20.0, model_prob=0.62, market_implied=0.48) >= 27
    assert confidence_divergence_penalty(13.0, model_prob=0.61, market_implied=0.55) == 20
    assert confidence_divergence_penalty(13.0, model_prob=0.60, market_implied=0.73) == 20
    assert confidence_divergence_penalty(18.0, model_prob=0.45, market_implied=0.27) == 0


def test_max_soft_action_caps_strong_bet():
    from apps.api.services.market_alignment import model_outlier_status

    assert model_outlier_status(15.0)[0] == "ok"
    assert model_outlier_status(22.0)[3] == 0.5
    assert model_outlier_status(28.0, market="1X2")[0] == "investigate"
    assert model_outlier_status(28.0, market="1X2")[4] == "WATCH"
    assert model_outlier_status(35.0, market="1X2")[0] == "error"
    assert model_outlier_status(35.0, market="1X2")[4] == "NO_BET"

    assert max_soft_action_for_gap(11.0) is None
    assert max_soft_action_for_gap(13.0) == "WEAK_BET"
    assert max_soft_action_for_gap(16.0) == "WATCH"
