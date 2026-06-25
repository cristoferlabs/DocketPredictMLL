"""Tests shape calibration learned model."""

import json
from pathlib import Path

from apps.worker.ml.shape_calibration import (
    ShapeCalibrationModel,
    ShapeFeatures,
    apply_poisson_shape_calibration,
    fit_shape_calibration_from_archives,
    load_shape_calibration_model,
    save_shape_calibration_model,
)


def test_apply_uses_learned_draw_factor(tmp_path, monkeypatch):
  artifact = tmp_path / "wc_shape_calibration.json"
  model = ShapeCalibrationModel(
      draw_by_context={"close": 1.12, "balanced": 1.0, "mismatch": 1.0},
      peak_bins=[{"min_peak": 0.0, "max_peak": 1.0, "favorite_scale": 1.0, "n": 10}],
  )
  artifact.write_text(json.dumps(model.to_dict()), encoding="utf-8")
  monkeypatch.setattr(
      "apps.worker.ml.shape_calibration.SHAPE_ARTIFACT_PATH",
      artifact,
  )
  probs = {"home_win": 0.36, "draw": 0.30, "away_win": 0.34}
  out, meta = apply_poisson_shape_calibration(
      probs,
      "close",
      features=ShapeFeatures(lambda_total=2.2, elo_gap=40),
      model=load_shape_calibration_model(),
  )
  assert out["draw"] > probs["draw"]
  assert meta["draw_factor"] == 1.12
  assert meta["learned"] is True


def test_lambda_bin_overrides_context_base(tmp_path, monkeypatch):
  artifact = tmp_path / "wc_shape_calibration.json"
  model = ShapeCalibrationModel(
      draw_by_context={"close": 1.05},
      lambda_bins={
          "close": [
              {"lambda_min": 2.0, "lambda_max": 2.5, "draw_factor": 1.15, "n": 12},
          ]
      },
      peak_bins=[{"min_peak": 0.0, "max_peak": 1.0, "favorite_scale": 1.0, "n": 10}],
  )
  artifact.write_text(json.dumps(model.to_dict()), encoding="utf-8")
  monkeypatch.setattr("apps.worker.ml.shape_calibration.SHAPE_ARTIFACT_PATH", artifact)
  probs = {"home_win": 0.38, "draw": 0.28, "away_win": 0.34}
  _, meta = apply_poisson_shape_calibration(
      probs,
      "close",
      features=ShapeFeatures(lambda_total=2.2, elo_gap=30),
      model=load_shape_calibration_model(),
  )
  assert meta["draw_factor"] == 1.15


def test_favorite_scale_boosts_strong_favorite(tmp_path, monkeypatch):
  artifact = tmp_path / "wc_shape_calibration.json"
  model = ShapeCalibrationModel(
      draw_by_context={"mismatch": 1.0},
      peak_bins=[
          {"min_peak": 0.0, "max_peak": 0.50, "favorite_scale": 1.0, "n": 20},
          {"min_peak": 0.50, "max_peak": 1.0, "favorite_scale": 1.10, "n": 15},
      ],
  )
  artifact.write_text(json.dumps(model.to_dict()), encoding="utf-8")
  monkeypatch.setattr("apps.worker.ml.shape_calibration.SHAPE_ARTIFACT_PATH", artifact)
  probs = {"home_win": 0.58, "draw": 0.22, "away_win": 0.20}
  out, meta = apply_poisson_shape_calibration(
      probs,
      "mismatch",
      model=load_shape_calibration_model(),
  )
  assert out["home_win"] > probs["home_win"]
  assert meta["favorite_scale"] == 1.10


def test_fit_shape_from_archives_minimal():
  """Smoke: archives vacíos no rompen."""
  model, metrics = fit_shape_calibration_from_archives({})
  assert metrics.get("n_matches", 0) == 0
  path = save_shape_calibration_model(model)
  assert Path(path).exists()
