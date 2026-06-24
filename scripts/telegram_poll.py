"""Telegram long-polling bridge — recibe mensajes y los procesa con el agente local."""

import asyncio
import logging
import sys
from pathlib import Path

import httpx

from apps.api.services.telegram_agent import TelegramAgentService
from apps.api.services.telegram_poll_lock import acquire_poll_lock
from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("telegram_poll")

CONFLICT_EXIT_THRESHOLD = 3
OFFSET_PATH = Path(__file__).resolve().parents[1] / ".telegram_poll.offset"
_CONFLICT_MESSAGE = (
    "Otro proceso está haciendo getUpdates con este bot (409 Conflict).\n"
    "Desactiva TODOS los workflows n8n con Telegram Trigger, por ejemplo:\n"
    "  • telegram_inbound\n"
    "  • Bot Interactivo - Mundial 2026\n"
    "  • RSL Engine - Flow 6: Bot Telegram Comandos\n"
    "Solo puede haber UN consumidor por token.\n"
    "Si prefieres n8n: TELEGRAM_INGESTION_MODE=n8n y NO ejecutes start-telegram.bat."
)


def _load_offset() -> int:
    try:
        return int(OFFSET_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _save_offset(offset: int) -> None:
    try:
        OFFSET_PATH.write_text(str(offset), encoding="utf-8")
    except OSError as exc:
        logger.debug("no se pudo guardar offset: %s", exc)


async def clear_webhook(token: str) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        await client.get(
            f"https://api.telegram.org/bot{token}/deleteWebhook",
            params={"drop_pending_updates": True},
        )


async def probe_poll_exclusive(token: str) -> bool:
    """Return True if no other process holds getUpdates."""
    base = f"https://api.telegram.org/bot{token}"
    poll_params = {
        "offset": -1,
        "timeout": 1,
        "allowed_updates": '["message","callback_query"]',
    }
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(5):
            resp = await client.get(f"{base}/getUpdates", params=poll_params)
            data = resp.json()
            if data.get("ok"):
                return True
            if data.get("error_code") == 409:
                logger.warning("Preflight 409 (%s/5): otro poller activo", attempt + 1)
                await asyncio.sleep(4)
                continue
            logger.error("Preflight getUpdates error: %s", data)
            return False
    return False


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

    token = settings.telegram_bot_token
    await clear_webhook(token)
    if not await probe_poll_exclusive(token):
        logger.error(_CONFLICT_MESSAGE)
        sys.exit(1)

    db = get_supabase()
    agent = TelegramAgentService(db)
    offset = _load_offset()
    base = f"https://api.telegram.org/bot{token}"

    if offset:
        logger.info("Polling activo (offset=%s). Escribe /hoy al bot. Ctrl+C para salir.", offset)
    else:
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
                    _save_offset(offset)
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
