"""Tests for calibration metrics."""

import numpy as np

from apps.worker.ml.calibration import (
    IsotonicCalibrator,
    apply_scalar_calibration,
    brier_score_multiclass,
    calibrate_model_markets,
    expected_calibration_error,
    reliability_bins,
)


def test_ece_perfect_calibration_near_zero():
  # Well-calibrated synthetic: pred ≈ outcome rate per bin
    probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    outcomes = [0, 0, 0, 1, 1, 1]
    ece = expected_calibration_error(probs, outcomes, n_bins=3)
    assert ece < 0.35


def test_reliability_bins_structure():
    bins = reliability_bins([0.2, 0.8], [0, 1], n_bins=2)
    assert len(bins) == 2
    assert bins[0]["count"] + bins[1]["count"] == 2


def test_isotonic_reduces_overconfidence():
    cal = IsotonicCalibrator("test")
    # Model always predicts 0.9 but only 50% win
    probs = [0.9] * 20 + [0.1] * 20
    outcomes = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    cal.fit(probs, outcomes)
    adjusted = cal.transform(0.9)
    assert adjusted < 0.9


def test_scalar_calibration_shrinks_toward_half():
    assert apply_scalar_calibration(0.8, 0.8) < 0.8
    assert apply_scalar_calibration(0.8, 1.0) == 0.8


def test_calibrate_model_markets_renormalizes_1x2():
    result = calibrate_model_markets(0.5, 0.3, 0.2, 0.55, 0.45, 0.5, 0.5)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_brier_multiclass():
    probs = [[0.6, 0.2, 0.2], [0.1, 0.7, 0.2]]
    labels = [0, 1]
    score = brier_score_multiclass(probs, labels)
    assert 0 <= score <= 2
