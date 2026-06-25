"""Tests Fase B — calibration_metrics."""

from apps.worker.ml.calibration_metrics import (
    apply_underdog_dampening,
    brier_score_binary,
    evaluate_1x2_predictions,
    fit_poisson_elo_weights,
    log_loss_multiclass,
    propose_underdog_dampen_factor,
    blend_components,
)
from apps.worker.ml.model_combiner import ModelCombinationWeights


def test_brier_perfect_vs_random():
    perfect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    labels = [0, 1]
    assert evaluate_1x2_predictions(perfect, labels).brier_1x2 < 0.01


def test_log_loss_multiclass():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
    labels = [0, 2]
    ll = log_loss_multiclass(probs, labels)
    assert ll > 0


def test_fit_weights_on_synthetic():
    components = []
    for _ in range(20):
        components.append(
            {
                "year": 2018,
                "poisson": {"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
                "elo": {"home_win": 0.48, "draw": 0.28, "away_win": 0.24},
                "label": 0,
            }
        )
    for _ in range(12):
        components.append(
            {
                "year": 2022,
                "poisson": {"home_win": 0.20, "draw": 0.25, "away_win": 0.55},
                "elo": {"home_win": 0.24, "draw": 0.28, "away_win": 0.48},
                "label": 2,
            }
        )
    result = fit_poisson_elo_weights(components, grid_step=0.25)
    assert 0.25 <= result.weights.poisson <= 0.75
    assert result.train_report.n_samples == 20


def test_underdog_dampening_reduces_tail():
    h, d, a = apply_underdog_dampening(0.12, 0.28, 0.60, dampen_factor=0.85)
    assert h < 0.12
    assert a > 0.55  # favorito sube tras renorm
    assert abs(h + d + a - 1.0) < 0.001


def test_propose_underdog_dampen():
    assert propose_underdog_dampen_factor(10.0) == 0.82
    assert propose_underdog_dampen_factor(2.0) == 1.0


def test_blend_components():
    row = blend_components(
        {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
        ModelCombinationWeights(0.5, 0.5, 0.0),
    )
    assert abs(sum(row) - 1.0) < 0.001
    assert brier_score_binary([0.8], [1]) < brier_score_binary([0.2], [1])
