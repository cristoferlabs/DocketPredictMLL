"""Tests live calibration — α por régimen, tournament, shrink, anti-overfit."""

from apps.api.services.ev_policy import ev_calibrated, ev_decimal
from apps.api.services.live_calibration import (
    LiveCalibrationContext,
    apply_live_calibration,
    apply_pre_alpha_bucket_calibration,
    apply_tournament_factors,
    apply_underdog_shrink,
    cap_alpha_for_alignment,
    compute_dynamic_alpha,
    resolve_alpha_regime,
)
from apps.api.services.worldcup_engine import ModelMarkets
from apps.shared.config import Settings


def _model(**kwargs) -> ModelMarkets:
    defaults = dict(
        home_win=0.45,
        draw=0.28,
        away_win=0.27,
        over_25=0.52,
        under_25=0.48,
        btts_yes=0.50,
        btts_no=0.50,
        lambda_home=1.4,
        lambda_away=1.1,
    )
    defaults.update(kwargs)
    return ModelMarkets(**defaults)


def test_alpha_regime_piecewise_wc():
    s = Settings()
    assert resolve_alpha_regime(5.0, s, is_wc=True) == (s.cal_alpha_regime_low, "aligned")
    assert resolve_alpha_regime(15.0, s, is_wc=True) == (s.cal_alpha_regime_medium, "moderate")
    assert resolve_alpha_regime(25.0, s, is_wc=True) == (s.cal_alpha_regime_high, "high")
    assert resolve_alpha_regime(35.0, s, is_wc=True) == (s.cal_alpha_regime_max, "extreme")


def test_dynamic_alpha_extreme_higher_than_aligned():
    s = Settings()
    low = compute_dynamic_alpha(
        LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=5.0, n_books=5),
        s,
    )
    high = compute_dynamic_alpha(
        LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=35.0, n_books=5),
        s,
    )
    assert high > low
    assert high == s.cal_alpha_regime_max


def test_dynamic_alpha_balanced_in_low_band():
    s = Settings()
    ctx = LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=8.0)
    alpha = compute_dynamic_alpha(ctx, s)
    assert alpha == s.cal_alpha_regime_low


def test_alpha_cap_when_aligned_with_market():
    s = Settings()
    capped = cap_alpha_for_alignment(0.65, divergence_cal_pp=3.0, settings=s)
    assert capped <= s.cal_alpha_aligned_cap


def test_tournament_boosts_under_draw():
    s = Settings()
    probs = {"home_win": 0.40, "draw": 0.28, "away_win": 0.32}
    totals = {"over_25": 0.55, "under_25": 0.45}
    out, out_totals, applied = apply_tournament_factors(
        probs, totals, competition="fifa_world_cup", settings=s
    )
    assert applied
    assert out["draw"] > probs["draw"]
    assert out_totals is not None
    assert out_totals["under_25"] > totals["under_25"]
    assert out_totals["over_25"] < totals["over_25"]


def test_underdog_shrink_pulls_toward_stat():
    p_stat = {"home_win": 0.12, "draw": 0.22, "away_win": 0.66}
    p_cal = {"home_win": 0.28, "draw": 0.24, "away_win": 0.48}
    out, applied = apply_underdog_shrink(
        p_stat,
        p_cal,
        gap_pp_threshold=15.0,
        stat_weight=0.85,
        mismatch_high=True,
        divergence_cal_pp=30.0,
    )
    assert applied
    assert out["home_win"] < p_cal["home_win"]
    assert out["home_win"] > p_stat["home_win"]


def test_shrink_inactive_without_mismatch():
    p_stat = {"home_win": 0.12, "draw": 0.22, "away_win": 0.66}
    p_cal = {"home_win": 0.28, "draw": 0.24, "away_win": 0.48}
    out, applied = apply_underdog_shrink(
        p_stat,
        p_cal,
        gap_pp_threshold=15.0,
        stat_weight=0.85,
        mismatch_high=False,
        divergence_cal_pp=30.0,
    )
    assert not applied
    assert out == p_cal


def test_ev_uses_calibrated_probs():
    p_stat, p_cal = 0.35, 0.28
    odds = 3.2
    assert ev_calibrated(p_cal, odds) < ev_decimal(p_stat, odds)


def test_pre_alpha_bucket_runs_before_alpha():
    model = _model(home_win=0.48, draw=0.26, away_win=0.26)
    market = {"home_win": 0.72, "draw": 0.18, "away_win": 0.10}
    ctx = LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=24.0, n_books=5)
    result = apply_live_calibration(model, market, context=ctx)
    assert result.meta.get("pre_alpha_bucket") is True
    assert result.statistical["home_win"] == 0.48
    assert result.meta["order"].startswith("stat→pre_alpha_bucket")


def test_pre_alpha_lifts_compressed_favorite_vs_market():
    s = Settings()
    probs = {"home_win": 0.52, "draw": 0.22, "away_win": 0.26}
    market = {"home_win": 0.78, "draw": 0.14, "away_win": 0.08}
    out, meta = apply_pre_alpha_bucket_calibration(probs, settings=s, market_fair=market)
    assert meta["pre_alpha_bucket"] is True
    assert out["home_win"] > probs["home_win"]


def test_statistical_probs_preserved_in_meta():
    model = _model(home_win=0.55, draw=0.25, away_win=0.20)
    market = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
    ctx = LiveCalibrationContext(
        competition="fifa_world_cup",
        max_divergence_pp=15.0,
        n_books=5,
    )
    result = apply_live_calibration(model, market, context=ctx)
    assert result.statistical["home_win"] == 0.55
    assert result.calibrated.home_win != 0.55
    assert result.meta.get("alpha_regime") in ("moderate", "high", "aligned")
    assert "statistical" in result.calibrated.blend_meta


def test_overfit_warning_when_cal_collapses_to_market():
    model = _model(home_win=0.70, draw=0.18, away_win=0.12)
    market = {"home_win": 0.42, "draw": 0.28, "away_win": 0.30}
    ctx = LiveCalibrationContext(competition="fifa_world_cup", max_divergence_pp=28.0, n_books=5)
    result = apply_live_calibration(model, market, context=ctx)
    if result.meta["divergence_cal_pp"] < result.meta["divergence_stat_pp"] * 0.4:
        assert result.meta.get("overfit_warning") is True
