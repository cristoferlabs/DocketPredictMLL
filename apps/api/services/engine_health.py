"""Salud del motor — alertas CLV, Brier y ROI live."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from apps.worker.ml.live_roi import LiveRoiResult, simulate_live_roi_from_db
from apps.worker.ml.model_learning import load_learning_state

HealthStatus = Literal["ok", "warning", "critical"]

ALERT_STATE_PATH = Path("artifacts/calibration/last_health_alert.json")


@dataclass
class EngineHealth:
    status: HealthStatus
    alerts: list[str] = field(default_factory=list)
    clv_rolling: float | None = None
    brier_rolling: float | None = None
    roi_sharp: LiveRoiResult | None = None
    roi_positive_ev: LiveRoiResult | None = None

    def banner_line(self) -> str | None:
        if self.status == "ok":
            return None
        icon = "🔴" if self.status == "critical" else "🟡"
        return f"{icon} Motor: {self.status.upper()} — {self.alerts[0] if self.alerts else 'revisar métricas'}"


def evaluate_engine_health(db, *, settings: Any | None = None) -> EngineHealth:
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    state = load_learning_state()
    alerts: list[str] = []
    status: HealthStatus = "ok"

    clv_min_n = int(getattr(settings, "clv_alert_min_samples", 5))
    clv_neg = float(getattr(settings, "clv_alert_negative_threshold", -0.01))
    brier_max = float(settings.model_max_live_brier_1x2)
    roi_warn = float(getattr(settings, "live_roi_alert_threshold", -0.05))

    if state.rolling_clv_n >= clv_min_n and state.rolling_clv is not None:
        if state.rolling_clv < clv_neg:
            alerts.append(
                f"CLV rolling {state.rolling_clv*100:+.2f}pp "
                f"(n={state.rolling_clv_n}) — por debajo del umbral"
            )
            status = "critical"

    if state.rolling_brier_n >= clv_min_n and state.rolling_brier is not None:
        if state.rolling_brier > brier_max:
            alerts.append(
                f"Brier rolling {state.rolling_brier:.3f} > máx {brier_max:.2f}"
            )
            if status != "critical":
                status = "warning"

    roi_sharp = simulate_live_roi_from_db(
        db,
        scope="sharp",
        min_ev_fair=settings.ev_min_edge_fair,
    )
    roi_ev = simulate_live_roi_from_db(
        db,
        scope="positive_ev",
        min_ev_fair=settings.ev_min_edge_fair,
    )

    if roi_sharp.bets >= 3 and roi_sharp.roi is not None and roi_sharp.roi < roi_warn:
        alerts.append(
            f"ROI SHARP live {roi_sharp.roi*100:+.1f}% "
            f"({roi_sharp.bets} bets, hit {roi_sharp.hit_rate*100:.0f}%)"
        )
        if status == "ok":
            status = "warning"

    if not alerts and state.rolling_clv_n >= clv_min_n and state.rolling_clv is not None:
        if state.rolling_clv < 0:
            alerts.append(
                f"CLV rolling ligeramente negativo ({state.rolling_clv*100:+.2f}pp)"
            )
            status = "warning"

    return EngineHealth(
        status=status,
        alerts=alerts,
        clv_rolling=state.rolling_clv,
        brier_rolling=state.rolling_brier,
        roi_sharp=roi_sharp,
        roi_positive_ev=roi_ev,
    )


def format_health_alerts(health: EngineHealth) -> list[str]:
    lines: list[str] = []
    if health.status == "ok":
        lines.append("✅ Salud motor: OK")
        if health.clv_rolling is not None:
            lines.append(f"  CLV rolling: {health.clv_rolling*100:+.2f}pp")
        return lines

    lines.append(f"{'🔴' if health.status == 'critical' else '🟡'} Salud motor: {health.status.upper()}")
    for a in health.alerts:
        lines.append(f"  • {a}")
    return lines


def _alert_fingerprint(health: EngineHealth) -> str:
    return "|".join(sorted(health.alerts))


def should_send_telegram_alert(health: EngineHealth, *, cooldown_hours: int = 12) -> bool:
    if health.status == "ok":
        return False
    fp = _alert_fingerprint(health)
    if not ALERT_STATE_PATH.exists():
        return True
    try:
        raw = json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        if raw.get("fingerprint") == fp:
            last = raw.get("sent_at")
            if last:
                then = datetime.fromisoformat(last.replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - then).total_seconds() / 3600
                if age_h < cooldown_hours:
                    return False
        return True
    except Exception:
        return True


def mark_alert_sent(health: EngineHealth) -> None:
    ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_PATH.write_text(
        json.dumps(
            {
                "fingerprint": _alert_fingerprint(health),
                "status": health.status,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def maybe_notify_engine_health(db, *, settings: Any | None = None) -> dict[str, Any]:
    """Envía alerta a Telegram group si salud crítica/warning (con cooldown)."""
    from apps.api.services.telegram_client import TelegramClient
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    if not getattr(settings, "engine_health_telegram_alerts", True):
        return {"sent": False, "reason": "disabled"}

    health = evaluate_engine_health(db, settings=settings)
    cooldown_hours = int(getattr(settings, "engine_health_alert_cooldown_hours", 12))
    if not should_send_telegram_alert(health, cooldown_hours=cooldown_hours):
        return {"sent": False, "reason": "cooldown", "status": health.status}

    chat_id = settings.telegram_group_id
    if not chat_id:
        return {"sent": False, "reason": "no telegram_group_id", "status": health.status}

    client = TelegramClient()
    if not client.is_configured:
        return {"sent": False, "reason": "no bot token", "status": health.status}

    lines = ["🚨 ENGINE HEALTH ALERT", "─────────────────", *format_health_alerts(health)]
    if health.roi_sharp and health.roi_sharp.bets:
        r = health.roi_sharp
        lines.append(
            f"\nROI SHARP: {r.roi*100:+.1f}% | {r.bets} bets | hit {(r.hit_rate or 0)*100:.0f}%"
        )
    lines.append("\nRevisa /stats para detalle.")

    await client.send_message(chat_id, "\n".join(lines))
    mark_alert_sent(health)
    return {"sent": True, "status": health.status, "alerts": health.alerts}
