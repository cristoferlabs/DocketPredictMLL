"""Telegram Bot API client."""

import logging
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, token: str | None = None):
        settings = get_settings()
        self.token = token or settings.telegram_bot_token
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    async def send_message(
        self,
        chat_id: str | int,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")

        payload: dict[str, Any] = {
            "chat_id": str(chat_id),
            "text": text[:4000],
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if parse_mode:
            payload["parse_mode"] = parse_mode

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.base_url}/sendMessage", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        if not self.is_configured or not callback_query_id:
            return
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{self.base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text[:200]},
            )
