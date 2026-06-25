"""Tests Dixon-Coles contextual + shape calibration."""

import numpy as np

from apps.worker.ml.dixon_coles import (
    classify_match_context,
    prepare_dixon_coles_lambdas,
    resolve_rho_for_context,
)
from apps.worker.ml.poisson import (
    build_score_matrix,
    build_score_matrix_dixon_coles,
    outcome_probabilities,
    predict_match,
)
from apps.worker.ml.shape_calibration import apply_poisson_shape_calibration


def test_classify_match_context():
    assert classify_match_context(1.1, 1.0, elo_home=1510, elo_away=1505) == "close"
    assert classify_match_context(1.9, 0.7, elo_home=1700, elo_away=1450) == "mismatch"
    assert classify_match_context(1.4, 1.2, elo_home=1580, elo_away=1520) == "balanced"


def test_mismatch_no_dixon_coles():
    dc = prepare_dixon_coles_lambdas(1.9, 0.8, elo_home=1680, elo_away=1420)
    assert dc.match_context == "mismatch"
    assert dc.dixon_coles_applied is False
    assert dc.lambda_corrected_home > dc.lambda_base_home


def test_close_gets_negative_rho():
    rho = resolve_rho_for_context("close")
    assert rho < 0


def test_dixon_coles_boosts_draw_vs_independent():
    lh, la = 1.35, 1.10
    plain = outcome_probabilities(build_score_matrix(lh, la))
    dc = outcome_probabilities(build_score_matrix_dixon_coles(lh, la, rho=-0.13))
    assert dc["draw"] > plain["draw"]


def test_shape_calibration_boosts_draw_in_close():
    probs = {"home_win": 0.38, "draw": 0.28, "away_win": 0.34}
    out, meta = apply_poisson_shape_calibration(probs, "close")
    assert out["draw"] > probs["draw"]
    assert meta["shape_flags"]


def test_predict_match_exposes_lambda_base_and_corrected():
    pred = predict_match(1.4, 1.1, elo_home=1550, elo_away=1500)
    assert pred.lambda_base_home == 1.4
    assert pred.dc_meta["match_context"] in ("close", "balanced", "mismatch")
    assert pred.score_matrix.shape[0] == pred.score_matrix.shape[1]
