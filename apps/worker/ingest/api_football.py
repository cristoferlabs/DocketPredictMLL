"""API-Football HTTP client."""

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"


class ApiFootballClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.api_football_key
        self.headers = {
            "x-apisports-key": self.api_key,
            "Accept": "application/json",
        }

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY not set; returning empty response")
            return {"response": [], "errors": {"key": "missing"}}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}/{endpoint}", headers=self.headers, params=params or {})
            resp.raise_for_status()
            return resp.json()

    async def get_leagues(self, country: str | None = None, season: int | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        data = await self._get("leagues", params)
        return data.get("response", [])

    async def get_fixtures(
        self,
        league_id: int | None = None,
        season: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        status: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        if status:
            params["status"] = status
        data = await self._get("fixtures", params)
        return data.get("response", [])

    async def get_fixture_statistics(self, fixture_id: int) -> list[dict]:
        data = await self._get("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])

    async def get_upcoming_fixtures(self, league_id: int, days_ahead: int = 7, season: int | None = None) -> list[dict]:
        today = date.today()
        return await self.get_fixtures(
            league_id=league_id,
            season=season or today.year,
            from_date=today,
            to_date=today + timedelta(days=days_ahead),
        )
