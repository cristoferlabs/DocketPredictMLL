"""Telegram webhook — Bot API updates via n8n or direct."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from apps.api.deps import get_db
from apps.api.services.telegram_agent import TelegramAgentService
from apps.shared.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class TelegramWebhookResponse(BaseModel):
    ok: bool
    chat_id: str | None = None
    message_length: int | None = None
    sent: bool | None = None
    detail: str | None = None


async def _parse_update_body(request: Request) -> dict[str, Any]:
    """Accept dict or JSON string (n8n JSON.stringify legacy)."""
    raw = await request.json()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="body must be a Telegram Update object")
    return raw


@router.post("/telegram", response_model=TelegramWebhookResponse)
async def telegram_webhook(request: Request, db=Depends(get_db)):
    """
    Recibe un Telegram Update (desde n8n Telegram Trigger o webhook directo).
    Procesa con el agente, envía respuesta al chat y retorna status.
    """
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise HTTPException(status_code=503, detail="TELEGRAM_BOT_TOKEN no configurado en .env")

    if settings.telegram_ingestion_mode.lower() == "poll":
        raise HTTPException(
            status_code=409,
            detail=(
                "TELEGRAM_INGESTION_MODE=poll: usa scripts/start-telegram.bat, "
                "no el webhook /webhooks/telegram. Desactiva telegram_inbound en n8n."
            ),
        )

    update = await _parse_update_body(request)
    agent = TelegramAgentService(db)
    try:
        result = await agent.handle_update(update)
        return TelegramWebhookResponse(**result)
    except Exception as exc:
        logger.exception("telegram webhook error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/telegram/setup")
async def telegram_setup_info():
    """Instrucciones para conectar el bot."""
    settings = get_settings()
    return {
        "bot_configured": bool(settings.telegram_bot_token),
        "group_id": settings.telegram_group_id or None,
        "webhook_path": "/webhooks/telegram",
        "n8n": "Importar n8n/workflows/telegram_inbound.json",
        "comandos": ["/hoy", "/alta", "Colombia vs Brasil", "/help"],
    }
