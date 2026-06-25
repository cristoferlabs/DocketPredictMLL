"""Tests joint calibration objective layer."""

import json
from pathlib import Path

from apps.worker.ml.joint_calibration import (
    JointCalibrationModel,
    JointObjectiveWeights,
    apply_joint_pricing_calibration,
    blend_toward_market,
    clv_proxy_penalty,
    fit_joint_calibration,
    joint_objective,
    load_joint_calibration_model,
    log_loss_1x2,
    market_divergence_penalty,
    save_joint_calibration_model,
)


def test_market_divergence_zero_when_equal():
    p = {"home_win": 0.45, "draw": 0.28, "away_win": 0.27}
    assert market_divergence_penalty(p, p) == 0.0


def test_blend_reduces_divergence():
    model = {"home_win": 0.50, "draw": 0.30, "away_win": 0.20}
    market = {"home_win": 0.40, "draw": 0.28, "away_win": 0.32}
    blended = blend_toward_market(model, market, 0.25)
    assert market_divergence_penalty(blended, market) < market_divergence_penalty(model, market)


def test_joint_objective_includes_all_terms():
    p = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    m = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
    w = JointObjectiveWeights(lambda_market=0.5, mu_clv=0.2)
    loss = joint_objective(p, label="home_win", market_probs=m, weights=w)
    expected = (
        log_loss_1x2(p, "home_win")
        + 0.5 * market_divergence_penalty(p, m)
        + 0.2 * clv_proxy_penalty(p, m)
    )
    assert abs(loss - expected) < 1e-9


def test_apply_joint_pricing_uses_learned_beta(tmp_path, monkeypatch):
    artifact = tmp_path / "wc_joint_calibration.json"
    model = JointCalibrationModel(market_blend_by_context={"mismatch": 0.30})
    artifact.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    monkeypatch.setattr("apps.worker.ml.joint_calibration.JOINT_ARTIFACT_PATH", artifact)
    monkeypatch.setattr(
        "apps.shared.config.get_settings",
        lambda: type("S", (), {"joint_calibration_enabled": True, "poisson_shape_use_learned": True})(),
    )
    shape = {"home_win": 0.58, "draw": 0.22, "away_win": 0.20}
    market = {"home_win": 0.48, "draw": 0.26, "away_win": 0.26}
    out, meta = apply_joint_pricing_calibration(
        shape, market, "mismatch", model=load_joint_calibration_model()
    )
    assert meta["applied"] is True
    assert meta["market_blend_beta"] == 0.30
    assert abs(out["home_win"] - shape["home_win"]) < abs(shape["home_win"] - market["home_win"])


def test_apply_skips_when_no_improvement(tmp_path, monkeypatch):
    artifact = tmp_path / "wc_joint_calibration.json"
    model = JointCalibrationModel(market_blend_by_context={"balanced": 0.25})
    artifact.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    monkeypatch.setattr("apps.worker.ml.joint_calibration.JOINT_ARTIFACT_PATH", artifact)

    p = {"home_win": 0.45, "draw": 0.28, "away_win": 0.27}
    out, meta = apply_joint_pricing_calibration(p, p, "balanced", model=model)
    assert meta["applied"] is False
    assert out == p


def test_fit_joint_on_synthetic_rows():
    rows = [
        {
            "context": "mismatch",
            "label": "home_win",
            "p_shape": {"home_win": 0.60, "draw": 0.22, "away_win": 0.18},
            "p_market": {"home_win": 0.48, "draw": 0.27, "away_win": 0.25},
        },
        {
            "context": "close",
            "label": "draw",
            "p_shape": {"home_win": 0.35, "draw": 0.32, "away_win": 0.33},
            "p_market": None,
        },
    ]
    model, metrics = fit_joint_calibration(rows)
    assert "mismatch" in model.market_blend_by_context
    assert metrics["n_total"] == 2
    path = save_joint_calibration_model(model)
    assert Path(path).exists()
