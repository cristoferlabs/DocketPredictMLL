"""Football-Data.org API client (free tier — current season fixtures)."""

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Top leagues on free tier
DEFAULT_COMPETITIONS = ["PD", "PL", "SA", "BL1", "FL1", "CL", "WC"]


class FootballDataClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.football_data_key
        self.headers = {"X-Auth-Token": self.api_key, "Accept": "application/json"}

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {"matches": [], "error": "missing_key"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}{path}", headers=self.headers, params=params or {})
            if resp.status_code == 429:
                logger.warning("Football-Data rate limit hit")
                return {"matches": [], "error": "rate_limit"}
            resp.raise_for_status()
            return resp.json()

    async def get_matches_by_date(
        self,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> list[dict]:
        today = date.today()
        params: dict[str, Any] = {
            "dateFrom": (date_from or today).isoformat(),
            "dateTo": (date_to or today + timedelta(days=7)).isoformat(),
        }
        if status:
            params["status"] = status
        data = await self._get("/matches", params)
        return data.get("matches", [])

    async def get_competition_matches(
        self,
        competition_code: str,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> list[dict]:
        today = date.today()
        params: dict[str, Any] = {
            "dateFrom": (date_from or today).isoformat(),
            "dateTo": (date_to or today + timedelta(days=7)).isoformat(),
        }
        if status:
            params["status"] = status
        data = await self._get(f"/competitions/{competition_code}/matches", params)
        return data.get("matches", [])

    async def get_upcoming(
        self,
        days_ahead: int = 7,
        competitions: list[str] | None = None,
    ) -> list[dict]:
        today = date.today()
        end = today + timedelta(days=days_ahead)
        seen: set[int] = set()
        matches: list[dict] = []

        for code in competitions or DEFAULT_COMPETITIONS:
            try:
                rows = await self.get_competition_matches(code, today, end)
                for m in rows:
                    mid = m.get("id")
                    if mid and mid not in seen:
                        seen.add(mid)
                        matches.append(m)
            except Exception as exc:
                logger.warning("Football-Data competition %s: %s", code, exc)

        if not matches:
            matches = await self.get_matches_by_date(today, end)

        return matches
