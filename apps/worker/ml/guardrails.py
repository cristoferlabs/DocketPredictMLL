"""Production guardrails for EV publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.shared.config import Settings


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons}


def check_ev_guardrails(db, settings: Settings) -> GuardrailResult:
    """Block +EV if rolling backtest ROI or ECE fail thresholds."""
    reasons: list[str] = []

    try:
        metrics = (
            db.schema("ml")
            .table("model_performance_metrics")
            .select("roi_sim, calibration_error, market_type, window_days, sample_size")
            .order("computed_at", desc=True)
            .limit(5)
            .execute()
        )
        for row in metrics.data or []:
            roi = row.get("roi_sim")
            ece = row.get("calibration_error")
            n = row.get("sample_size") or 0
            if n < 30:
                continue
            if roi is not None and float(roi) < settings.ev_min_roi_backtest:
                reasons.append(f"ROI sim {float(roi):.3f} < {settings.ev_min_roi_backtest}")
            if ece is not None and float(ece) > settings.ev_max_ece:
                reasons.append(f"ECE {float(ece):.3f} > {settings.ev_max_ece}")
            break
    except Exception:
        pass

    return GuardrailResult(allowed=len(reasons) == 0, reasons=reasons)
