"""Tests for ML models."""

import numpy as np

from apps.worker.ml.elo import predict_match as elo_predict
from apps.worker.ml.ensemble import MatchInput, predict_match
from apps.worker.ml.evaluation import evaluate_prediction
from apps.worker.ml.gk import GoalkeeperProfile, apply_gk_adjustment
from apps.worker.ml.poisson import predict_match as poisson_predict


def test_elo_probabilities_sum_to_one():
    result = elo_predict(1600, 1400)
    total = result.home_win + result.draw + result.away_win
    assert abs(total - 1.0) < 0.01


def test_poisson_score_matrix_sums_to_one():
    result = poisson_predict(1.5, 1.1)
    assert abs(result.score_matrix.sum() - 1.0) < 0.01


def test_gk_adjustment_reduces_lambda_for_good_gk():
    good_gk = GoalkeeperProfile(name="Test", save_pct=0.78, xga_90=0.9)
    adj = apply_gk_adjustment(1.5, 1.2, away_gk=good_gk)
    assert adj.home_xga_adjusted < 1.5


def test_ensemble_produces_predictions():
    result = predict_match(MatchInput(home_elo=1550, away_elo=1450))
    assert len(result.predictions) > 0
    assert len(result.combinations) > 0
    x12 = [p for p in result.predictions if p.market_type == "1X2"]
    assert len(x12) == 3


def test_evaluation_correct_1x2():
    ev = evaluate_prediction("1X2", "home_win", 0.6, 2, 1)
    assert ev["is_correct"] is True
    assert ev["actual_outcome"] == "home_win"


def test_ensemble_weights_change_prediction():
    from apps.worker.ml.ensemble import EnsembleWeights

    base = predict_match(MatchInput(home_elo=1600, away_elo=1400))
    heavy_elo = predict_match(
        MatchInput(home_elo=1600, away_elo=1400),
        weights=EnsembleWeights(elo=0.80, poisson=0.10, gk=0.05, xgboost=0.05),
    )
    p_base = next(p for p in base.predictions if p.predicted_outcome == "home_win")
    p_elo = next(p for p in heavy_elo.predictions if p.predicted_outcome == "home_win")
    assert p_elo.probability > p_base.probability
