"""SportMonks API client (optional; depends on subscription)."""

import logging
from datetime import date
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sportmonks.com/v3/football"


class SportMonksClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.sportmonks_key

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {"data": []}
        p = {"api_token": self.api_key, **(params or {})}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}{path}", params=p)
            resp.raise_for_status()
            return resp.json()

    async def get_fixtures_by_date(self, match_date: date | None = None) -> list[dict]:
        d = match_date or date.today()
        data = await self._get(f"/fixtures/date/{d.isoformat()}")
        return data.get("data") or []
