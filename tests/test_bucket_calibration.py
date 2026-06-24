"""Tests bucket calibration A+B."""

from apps.worker.ml.calibration import (
    apply_bucket_1x2,
    classify_team_win_bucket,
    calibrate_model_markets,
    merge_bucket_config,
)


def test_classify_buckets():
    assert classify_team_win_bucket(0.62) == "favorite"
    assert classify_team_win_bucket(0.45) == "medium"
    assert classify_team_win_bucket(0.28) == "underdog"


def test_apply_bucket_boosts_favorite_reduces_draw():
    h, d, a = apply_bucket_1x2(
        0.235,
        0.262,
        0.503,
        {
            "team_win": {"favorite": 1.15, "medium": 1.0, "underdog": 0.88},
            "draw": 0.90,
            "draw_dampen_threshold": 0.55,
            "draw_dampen_factor": 0.88,
            "underdog_cap_max_p": 0.30,
            "underdog_cap_factor": 0.85,
        },
    )
    assert a > h
    assert d <= 0.262
    assert abs(h + d + a - 1.0) < 1e-6


def test_calibrate_model_markets_with_buckets():
    factors = merge_bucket_config(
        None,
        {
            "team_win": {"favorite": 1.12, "medium": 1.0, "underdog": 0.86},
            "draw": 0.88,
            "draw_dampen_factor": 0.90,
            "draw_dampen_threshold": 0.55,
            "underdog_cap_max_p": 0.30,
            "underdog_cap_factor": 0.85,
        },
    )
    out = calibrate_model_markets(0.235, 0.262, 0.503, 0.5, 0.5, 0.5, 0.5, factors=factors)
    assert out["away_win"] > out["home_win"]
    assert out["draw"] <= 0.265
