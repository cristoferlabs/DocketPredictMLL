"""Deduplicate Telegram updates across pollers, webhooks and n8n."""

from __future__ import annotations

import logging

import redis

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

UPDATE_TTL_SECONDS = 86_400  # 24h


def claim_update(update_id: int | None) -> bool:
    """
    Return True if this worker should process the update (first claim wins).
    Uses Redis SET NX so duplicate delivery from multiple pollers/webhooks is ignored.
    """
    if update_id is None:
        return True
    try:
        client = redis.from_url(get_settings().redis_url, decode_responses=True)
        key = f"telegram:update:{update_id}"
        return bool(client.set(key, "1", nx=True, ex=UPDATE_TTL_SECONDS))
    except Exception as exc:
        logger.warning("telegram dedup unavailable (%s); processing update anyway", exc)
        return True
