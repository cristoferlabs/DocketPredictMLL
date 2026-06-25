"""Tests Fase C — model_learning (Bayesian + deploy gate)."""

import json
from pathlib import Path

from apps.worker.ml.model_learning import (
    LearningState,
    apply_learning_corrections,
    bayesian_outcome_update,
    deploy_calibration_gate,
    load_learning_state,
    save_learning_state,
    update_from_wc_evaluation,
)


class _Audit:
    favorite_bias_score = 0.12


def test_bayesian_update_moves_bias_toward_outcome():
    state = LearningState()
    probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    state = bayesian_outcome_update(probs, actual_label=0, state=state, learning_rate=0.2)
    assert state.logit_bias["home_win"] > 0
    assert state.n_updates == 1


def test_apply_learning_corrections_normalized():
    state = LearningState(logit_bias={"home_win": 0.1, "draw": 0.0, "away_win": -0.05})
    h, d, a = apply_learning_corrections(0.5, 0.28, 0.22, state=state)
    assert abs(h + d + a - 1.0) < 0.001
    assert h > 0.5


def test_update_from_wc_evaluation_persists(tmp_path, monkeypatch):
    path = tmp_path / "wc_learning_state.json"
    monkeypatch.setattr(
        "apps.worker.ml.model_learning.LEARNING_STATE_PATH",
        path,
    )
    update_from_wc_evaluation(
        model_probs={"home_win": 0.6, "draw": 0.22, "away_win": 0.18},
        predicted_probability=0.6,
        predicted_outcome="Brazil",
        team_home="Brazil",
        team_away="Scotland",
        actual_label=0,
        brier_score=0.16,
        clv_vs_close=0.03,
    )
    loaded = load_learning_state()
    assert loaded.n_updates == 1
    assert loaded.rolling_clv_n == 1


def test_deploy_gate_approves_clean_audit():
    ok, reasons = deploy_calibration_gate(
        audit=_Audit(),
        live_brier=0.22,
        historical_brier=0.65,
        max_bias=0.25,
        max_live_brier=0.70,
    )
    assert ok is True
    assert reasons == []


def test_deploy_gate_blocks_high_bias():
    bad = type("A", (), {"favorite_bias_score": 0.42})()
    ok, reasons = deploy_calibration_gate(
        audit=bad,
        live_brier=0.22,
        historical_brier=0.65,
    )
    assert ok is False
    assert any("favorite_bias" in r for r in reasons)


def test_save_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setattr("apps.worker.ml.model_learning.LEARNING_STATE_PATH", path)
    s = LearningState(n_updates=5, results_since_retrain=3)
    save_learning_state(s)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["n_updates"] == 5
    assert load_learning_state().results_since_retrain == 3
