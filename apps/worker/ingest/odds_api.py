"""The Odds API client (free tier — betting odds for EV)."""

import logging
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# soccer_epl, soccer_spain_la_liga, etc.
SOCCER_SPORTS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]


class OddsApiClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.odds_api_key

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            return []
        p = {"apiKey": self.api_key, **(params or {})}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}{path}", params=p)
            resp.raise_for_status()
            return resp.json()

    async def get_soccer_odds(self, sports: list[str] | None = None) -> list[dict]:
        results: list[dict] = []
        for sport in sports or SOCCER_SPORTS:
            try:
                data = await self._get(
                    f"/sports/{sport}/odds",
                    {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
                )
                if isinstance(data, list):
                    for event in data:
                        event["_sport_key"] = sport
                        results.append(event)
            except Exception as exc:
                logger.warning("Odds API sport %s: %s", sport, exc)
        return results
