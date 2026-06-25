"""Tests deploy gate — ROI backtest bloquea calibración mala."""

from apps.worker.ml.model_learning import deploy_calibration_gate


class _Audit:
    def __init__(self, score: float = 0.1):
        self.favorite_bias_score = score


def test_deploy_gate_blocks_negative_backtest_roi():
    approved, reasons = deploy_calibration_gate(
        audit=_Audit(0.1),
        live_brier=0.55,
        historical_brier=0.60,
        backtest_roi=-0.08,
        min_roi_backtest=0.0,
        backtest_roi_details={"bets": 42},
    )
    assert not approved
    assert any("backtest_roi" in r for r in reasons)


def test_deploy_gate_passes_when_roi_ok():
    approved, reasons = deploy_calibration_gate(
        audit=_Audit(0.1),
        live_brier=0.55,
        historical_brier=0.60,
        backtest_roi=0.04,
        min_roi_backtest=0.0,
    )
    assert approved
    assert not reasons


def test_deploy_gate_skips_roi_when_none():
    approved, _ = deploy_calibration_gate(
        audit=_Audit(0.1),
        live_brier=None,
        historical_brier=None,
        backtest_roi=None,
    )
    assert approved
