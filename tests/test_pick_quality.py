"""Tests pick quality metrics — CLV esperado y calibration score."""

from apps.api.services.pick_quality import (
    calibration_score_for_prob,
    expected_clv_movement_pp,
    format_pick_quality_lines,
)


def test_calibration_score_in_range():
    score = calibration_score_for_prob(0.42)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_expected_clv_low_when_aligned():
    clv = expected_clv_movement_pp(
        model_prob=0.378,
        market_implied=0.378,
        ev_fair=0.068,
        gap_pp=0.0,
    )
    assert clv is not None
    assert abs(clv) < 2.0


def test_expected_clv_positive_with_edge():
    clv = expected_clv_movement_pp(
        model_prob=0.55,
        market_implied=0.45,
        ev_fair=0.12,
        gap_pp=10.0,
    )
    assert clv is not None
    assert clv > 0


def test_format_pick_quality_lines():
    lines = format_pick_quality_lines(
        model_prob=0.37,
        market_implied=0.37,
        ev_fair=0.15,
        gap_pp=0.0,
    )
    assert any("CLV esperado" in ln for ln in lines)
    assert any("Calibration score" in ln for ln in lines)
