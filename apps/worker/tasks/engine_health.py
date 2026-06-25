"""Cron — alertas de salud del motor (CLV / Brier / ROI live)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def check_engine_health(ctx: dict) -> dict:
    from apps.api.services.engine_health import (
        evaluate_engine_health,
        maybe_notify_engine_health,
    )
    from apps.shared.supabase_client import get_supabase

    db = get_supabase()
    health = evaluate_engine_health(db)
    notify = await maybe_notify_engine_health(db)
    summary = {
        "status": health.status,
        "alerts": health.alerts,
        "notify": notify,
        "clv_rolling": health.clv_rolling,
        "roi_sharp_bets": health.roi_sharp.bets if health.roi_sharp else 0,
    }
    logger.info("check_engine_health: %s", summary)
    return summary
