"""Tests ENGINE v3 — model_combiner (Poisson + ELO + mercado calibración)."""

from apps.worker.ml.model_combiner import (
    ModelCombinationWeights,
    Probabilities1X2,
    apply_market_calibration_layer,
    combine_1x2,
    combine_poisson_elo,
)


def _approx_one(probs: Probabilities1X2, tol: float = 0.01) -> None:
    total = probs.home_win + probs.draw + probs.away_win
    assert 1.0 - tol <= total <= 1.0 + tol


def test_poisson_elo_blend_sums_to_one():
    poisson = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    elo = {"home_win": 0.48, "draw": 0.28, "away_win": 0.24}
    blended, applied = combine_poisson_elo(
        poisson, elo, weights=ModelCombinationWeights(poisson=0.5, elo=0.5)
    )
    _approx_one(blended)
    assert applied["poisson"] == 0.5
    assert applied["market"] == 0.0


def test_default_weights_50_30():
    poisson = {"home_win": 0.60, "draw": 0.22, "away_win": 0.18}
    elo = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
    result = combine_1x2(poisson, elo, weights=ModelCombinationWeights(0.5, 0.3, 0.2))
    # 0.5*0.6 + 0.3*0.4 = 0.42 home (before norm)
    assert result.decision.home_win > result.decision.away_win
    _approx_one(result.decision)
    assert result.blend_applied is False


def test_market_calibration_does_not_change_decision():
    poisson = {"home_win": 0.70, "draw": 0.18, "away_win": 0.12}
    elo = {"home_win": 0.65, "draw": 0.20, "away_win": 0.15}
    market = {"home_win": 0.50, "draw": 0.28, "away_win": 0.22}
    result = combine_1x2(
        poisson,
        elo,
        weights=ModelCombinationWeights(0.5, 0.3, 0.2),
        market_fair=market,
        calibration_layer=True,
    )
    assert result.blend_applied is True
    assert result.anchored is not None
    _approx_one(result.decision)
    _approx_one(result.anchored)
    # Decision = solo Poisson+ELO; anchored más cerca del mercado en home
    assert result.anchored.home_win < result.decision.home_win
    assert result.decision.home_win != result.anchored.home_win


def test_apply_market_calibration_layer():
    model = Probabilities1X2(0.70, 0.18, 0.12)
    market = {"home_win": 0.50, "draw": 0.28, "away_win": 0.22}
    anchored = apply_market_calibration_layer(
        model, market, weights=ModelCombinationWeights(0.5, 0.3, 0.2)
    )
    # 0.8 * 0.70 + 0.2 * 0.50 = 0.66 antes de renorm
    assert anchored.home_win < model.home_win
    _approx_one(anchored)


def test_apply_market_calibration_layer_with_explicit_alpha():
    model = Probabilities1X2(0.70, 0.18, 0.12)
    market = {"home_win": 0.50, "draw": 0.28, "away_win": 0.22}
    anchored = apply_market_calibration_layer(model, market, alpha=0.40)
    assert anchored.home_win < model.home_win
    _approx_one(anchored)


def test_compute_model_markets_uses_combiner():
    from apps.api.services.worldcup_engine import compute_model_markets

    m = compute_model_markets(1.4, 1.1, 1950, 1800)
    total = m.home_win + m.draw + m.away_win
    assert 0.99 <= total <= 1.01
    assert m.blend_meta.get("engine") == "model_combiner_v1"
    assert "poisson" in m.blend_meta
    assert "elo" in m.blend_meta
    assert m.blend_meta.get("blend_applied") is False
