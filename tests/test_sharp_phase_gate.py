"""Tests SHARP v3 — fases cold / warm / mature."""

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.api.services.bet_pipeline import BetPipelineResult, ModelLayer, MarketLayer
from apps.api.services.market_dominance import MarketDominanceResult
from apps.api.services.market_uncertainty import EvBand
from apps.api.services.odds_context import MarketContext1X2
from apps.api.services.sharp_engine import (
    _apply_sharp_phase_gate,
    resolve_sharp_phase,
)
from apps.api.services.trading_types import TradingPick
from apps.api.services.worldcup_engine import MatchAnalysis, ModelMarkets
from apps.shared.config import Settings
from apps.worker.ml.model_learning import LearningState


def _dec(soft: str = "STRONG_BET", mds_path: str = "") -> BetDecisionResult:
    return BetDecisionResult(
        action=soft,  # type: ignore[arg-type]
        soft_action=soft,  # type: ignore[arg-type]
        no_bet=False,
        blocked_reason=None,
        tree_path=["start", mds_path] if mds_path else ["start"],
        pick=TradingPick(market="1X2", selection="Home", model_prob=0.55),
        ev_band=EvBand("x", 0.08, 0.06, 0.04),
        confidence_score=72,
    )


def test_resolve_sharp_phase_thresholds():
    s = Settings(sharp_phase_cold_n=10, sharp_phase_mature_n=25)
    assert resolve_sharp_phase(5, s) == "cold"
    assert resolve_sharp_phase(15, s) == "warm"
    assert resolve_sharp_phase(30, s) == "mature"


def test_cold_strong_passes_with_mds_and_low_delta():
    s = Settings()
    cal = {"alpha": 0.62, "divergence_cal_pp": 10.0, "shrink_applied": False}
    out = _apply_sharp_phase_gate(
        _dec(),
        mds=72,
        phase="cold",
        state=LearningState(),
        settings=s,
        cal_meta=cal,
    )
    assert out.soft_action == "STRONG_BET"


def test_cold_downgrades_weak_mds():
    s = Settings()
    cal = {"alpha": 0.60, "divergence_cal_pp": 8.0, "shrink_applied": True}
    out = _apply_sharp_phase_gate(
        _dec(),
        mds=60,
        phase="cold",
        state=LearningState(),
        settings=s,
        cal_meta=cal,
    )
    assert out.soft_action == "WEAK_BET"


def test_mature_blocks_strong_on_negative_clv():
    s = Settings()
    cal = {"alpha": 0.55, "divergence_cal_pp": 8.0, "shrink_applied": True}
    state = LearningState(rolling_clv_sum=-0.5, rolling_clv_n=30)
    out = _apply_sharp_phase_gate(
        _dec(),
        mds=72,
        phase="mature",
        state=state,
        settings=s,
        cal_meta=cal,
    )
    assert out.soft_action in ("WEAK_BET", "NO_BET")
    assert "CLV" in (out.blocked_reason or "")
