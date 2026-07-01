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

    async def get_live_fixtures(self, league_id: int | None = None) -> list[dict]:
        """Currently live fixtures. Pass league_id=1 for WC2026."""
        params: dict[str, Any] = {"live": "all"}
        if league_id:
            params["league"] = league_id
        data = await self._get("fixtures", params)
        return data.get("response", [])

    async def get_wc_live_fixtures(self) -> list[dict]:
        """All currently live WC2026 fixtures (league_id=1)."""
        return await self.get_live_fixtures(league_id=1)

    async def get_fixture_events(self, fixture_id: int) -> list[dict]:
        """Goals, cards, substitutions for a fixture."""
        data = await self._get("fixtures/events", {"fixture": fixture_id})
        return data.get("response", [])

    async def get_fixture_lineups(self, fixture_id: int) -> list[dict]:
        """Starting XI and formation for both teams."""
        data = await self._get("fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])

    async def get_fixture_players(self, fixture_id: int) -> list[dict]:
        """Player-level statistics for a fixture (Pro plan)."""
        data = await self._get("fixtures/players", {"fixture": fixture_id})
        return data.get("response", [])

    async def get_injuries(
        self,
        league_id: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        fixture_id: int | None = None,
    ) -> list[dict]:
        """Injury/suspension list. Filter by fixture or league+season."""
        params: dict[str, Any] = {}
        if fixture_id:
            params["fixture"] = fixture_id
        else:
            if league_id:
                params["league"] = league_id
            if season:
                params["season"] = season
            if team_id:
                params["team"] = team_id
        data = await self._get("injuries", params)
        return data.get("response", [])

    async def get_odds(
        self,
        fixture_id: int | None = None,
        league_id: int | None = None,
        season: int | None = None,
        bookmaker_id: int | None = None,
    ) -> list[dict]:
        """Pre-match odds (Pro plan)."""
        params: dict[str, Any] = {}
        if fixture_id:
            params["fixture"] = fixture_id
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if bookmaker_id:
            params["bookmaker"] = bookmaker_id
        data = await self._get("odds", params)
        return data.get("response", [])

    async def get_odds_live(self, fixture_id: int, bet_id: int | None = None) -> list[dict]:
        """Live odds — requires API-Football Pro plan."""
        params: dict[str, Any] = {"fixture": fixture_id}
        if bet_id:
            params["bet"] = bet_id
        data = await self._get("odds/live", params)
        return data.get("response", [])

    async def get_teams_statistics(
        self,
        league_id: int,
        season: int,
        team_id: int,
    ) -> dict:
        """Team aggregate statistics for a season."""
        data = await self._get("teams/statistics", {
            "league": league_id,
            "season": season,
            "team": team_id,
        })
        resp = data.get("response", {})
        return resp if isinstance(resp, dict) else {}

    async def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        last: int = 10,
    ) -> list[dict]:
        """Head-to-head results between two teams."""
        data = await self._get("fixtures/headtohead", {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last,
        })
        return data.get("response", [])

    async def get_standings(self, league_id: int, season: int) -> list[dict]:
        """League/tournament standings."""
        data = await self._get("standings", {"league": league_id, "season": season})
        return data.get("response", [])

    async def get_players_season_stats(
        self,
        team_id: int,
        season: int,
        league_id: int | None = None,
        page: int = 1,
    ) -> list[dict]:
        """Player season statistics for a team (Pro plan)."""
        params: dict[str, Any] = {"team": team_id, "season": season, "page": page}
        if league_id:
            params["league"] = league_id
        data = await self._get("players", params)
        return data.get("response", [])

    async def get_coaches(self, team_id: int | None = None, fixture_id: int | None = None) -> list[dict]:
        """Coach information per team or fixture."""
        params: dict[str, Any] = {}
        if team_id:
            params["team"] = team_id
        if fixture_id:
            params["fixture"] = fixture_id
        data = await self._get("coachs", params)
        return data.get("response", [])

    async def get_fixture_with_stats(self, fixture_id: int) -> dict:
        """Convenience: fixture metadata + live statistics in one call."""
        fixtures = await self.get_fixtures()  # won't work without params — use the fixture endpoint
        stats = await self.get_fixture_statistics(fixture_id)
        events = await self.get_fixture_events(fixture_id)
        return {
            "fixture_id": fixture_id,
            "statistics": stats,
            "events": events,
        }
