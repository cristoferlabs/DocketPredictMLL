"""Telegram long-polling bridge — recibe mensajes y los procesa con el agente local."""

import asyncio
import logging
import sys

import httpx

from apps.api.services.telegram_agent import TelegramAgentService
from apps.api.services.telegram_poll_lock import acquire_poll_lock
from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("telegram_poll")

CONFLICT_EXIT_THRESHOLD = 3
_CONFLICT_MESSAGE = (
    "Otro proceso (probablemente n8n Telegram Trigger) está usando este bot. "
    "Cierra n8n o desactiva sus workflows Telegram, o usa TELEGRAM_INGESTION_MODE=n8n "
    "y no ejecutes start-telegram.bat."
)


async def clear_webhook(token: str) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        await client.get(f"https://api.telegram.org/bot{token}/deleteWebhook")


async def poll_loop() -> None:
    acquire_poll_lock()
    settings = get_settings()
    if settings.telegram_ingestion_mode.lower() == "n8n":
        logger.error(
            "TELEGRAM_INGESTION_MODE=n8n: usa n8n (telegram_inbound) + API, no polling local. "
            "Para dev sin n8n, pon TELEGRAM_INGESTION_MODE=poll en .env"
        )
        sys.exit(1)
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN no configurado en .env")
        sys.exit(1)

    await clear_webhook(settings.telegram_bot_token)
    db = get_supabase()
    agent = TelegramAgentService(db)
    offset = 0
    base = f"https://api.telegram.org/bot{settings.telegram_bot_token}"

    logger.info("Polling activo. Escribe /hoy al bot o al grupo. Ctrl+C para salir.")

    conflict_streak = 0

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            try:
                resp = await client.get(
                    f"{base}/getUpdates",
                    params={"offset": offset, "timeout": 30, "allowed_updates": '["message","callback_query"]'},
                )
                data = resp.json()
                if not data.get("ok"):
                    if data.get("error_code") == 409:
                        conflict_streak += 1
                        logger.error(
                            "getUpdates 409 Conflict (%s/%s): %s",
                            conflict_streak,
                            CONFLICT_EXIT_THRESHOLD,
                            data.get("description"),
                        )
                        if conflict_streak >= CONFLICT_EXIT_THRESHOLD:
                            logger.error(_CONFLICT_MESSAGE)
                            sys.exit(1)
                        await asyncio.sleep(5)
                        continue
                    conflict_streak = 0
                    logger.error("getUpdates error: %s", data)
                    await asyncio.sleep(5)
                    continue

                conflict_streak = 0

                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    try:
                        result = await agent.handle_update(upd)
                        if result.get("duplicate"):
                            logger.debug("Update %s duplicado (omitido)", upd["update_id"])
                        else:
                            logger.info("Procesado update %s: %s", upd["update_id"], result)
                    except Exception as exc:
                        logger.exception("Error procesando update %s: %s", upd.get("update_id"), exc)

            except httpx.ReadTimeout:
                continue
            except Exception as exc:
                logger.exception("Poll error: %s", exc)
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(poll_loop())
