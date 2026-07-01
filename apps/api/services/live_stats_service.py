"""
Live Stats Service — fetches and caches live match data from API-Football.

Caches results in Redis (TTL 60s) to avoid hammering the API during
concurrent Telegram requests for the same live match.
Falls back gracefully when API is unavailable or match is not live.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from apps.worker.ingest.api_football import ApiFootballClient
from apps.worker.ml.poisson_live import (
    GameState,
    LiveStats,
    live_game_state_from_api_football,
    live_stats_from_api_football,
)

logger = logging.getLogger(__name__)

_CACHE_TTL = 60   # seconds — keep live stats fresh but avoid API spam
_WC_LEAGUE_ID = 1

# API-Football status codes that mean "match is in progress"
_LIVE_STATUS = {"1H", "HT", "2H", "ET", "BT", "INT", "LIVE", "P"}


async def _redis_get(key: str) -> str | None:
    try:
        import redis.asyncio as aioredis
        from apps.shared.config import get_settings
        r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        val = await r.get(key)
        await r.aclose()
        return val
    except Exception:
        return None


async def _redis_set(key: str, value: str, ttl: int) -> None:
    try:
        import redis.asyncio as aioredis
        from apps.shared.config import get_settings
        r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        await r.setex(key, ttl, value)
        await r.aclose()
    except Exception:
        pass


def _normalize_name(name: str) -> str:
    """Lowercase + remove common suffixes for fuzzy matching."""
    from apps.api.services.worldcup_engine import name_match as _nm
    return name.lower().strip()


def _teams_match(api_name: str, query: str) -> bool:
    """Flexible name matching between API-Football and openfootball names."""
    from apps.api.services.worldcup_engine import name_match
    return name_match(api_name, query) or name_match(query, api_name)


def _find_fixture(live_fixtures: list[dict], team_home: str, team_away: str) -> dict | None:
    """Find the fixture that matches the given team names."""
    for fx in live_fixtures:
        teams = fx.get("teams", {})
        api_home = teams.get("home", {}).get("name", "")
        api_away = teams.get("away", {}).get("name", "")
        if _teams_match(api_home, team_home) and _teams_match(api_away, team_away):
            return fx
        # Also check reversed (some sources swap home/away)
        if _teams_match(api_home, team_away) and _teams_match(api_away, team_home):
            return fx
    return None


def _count_red_cards(events: list[dict], fixture: dict) -> tuple[int, int]:
    """Count red cards (including second yellows) per team from fixture events."""
    home_id = fixture.get("teams", {}).get("home", {}).get("id")
    away_id = fixture.get("teams", {}).get("away", {}).get("id")
    home_rc = away_rc = 0
    for ev in events:
        if ev.get("type") != "Card":
            continue
        detail = ev.get("detail", "")
        if detail not in ("Red Card", "Second Yellow Card"):
            continue
        team_id = ev.get("team", {}).get("id")
        if team_id == home_id:
            home_rc += 1
        elif team_id == away_id:
            away_rc += 1
    return home_rc, away_rc


def _is_live_status(fixture: dict) -> bool:
    short = fixture.get("fixture", {}).get("status", {}).get("short", "")
    return short in _LIVE_STATUS


async def fetch_live_match_data(
    team_home: str,
    team_away: str,
) -> tuple[GameState | None, LiveStats | None]:
    """
    Fetch live game state and statistics for a match currently in progress.

    Returns (None, None) when:
    - The match is not currently live in API-Football
    - API-Football is unavailable (network error, missing key)
    - No matching fixture found for the given team names

    Results are cached in Redis for 60 seconds to handle concurrent requests.
    """
    cache_key = f"live:{_normalize_name(team_home)}:{_normalize_name(team_away)}"

    # Try Redis cache first
    cached = await _redis_get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if data.get("not_live"):
                return None, None
            gs = GameState(**data["game_state"])
            ls = LiveStats(**data["live_stats"])
            logger.debug("Live cache hit: %s vs %s", team_home, team_away)
            return gs, ls
        except Exception as exc:
            logger.debug("Live cache parse error: %s", exc)

    client = ApiFootballClient()

    # Fetch all live WC fixtures
    try:
        live_fixtures = await client.get_wc_live_fixtures()
    except Exception as exc:
        logger.warning("API-Football live fixtures error: %s", exc)
        return None, None

    fixture = _find_fixture(live_fixtures, team_home, team_away)
    if not fixture or not _is_live_status(fixture):
        # Cache the "not live" result to avoid re-querying immediately
        await _redis_set(cache_key, json.dumps({"not_live": True}), _CACHE_TTL)
        return None, None

    game_state = live_game_state_from_api_football(fixture)
    fixture_id = fixture.get("fixture", {}).get("id")
    live_stats = LiveStats()

    if fixture_id:
        # Fetch live statistics
        try:
            raw_stats = await client.get_fixture_statistics(fixture_id)
            if raw_stats:
                live_stats = live_stats_from_api_football(raw_stats)
        except Exception as exc:
            logger.warning("Live stats error fixture %s: %s", fixture_id, exc)

        # Enrich game_state with red card count from events
        try:
            events = await client.get_fixture_events(fixture_id)
            h_rc, a_rc = _count_red_cards(events, fixture)
            game_state.home_red_cards = h_rc
            game_state.away_red_cards = a_rc
        except Exception as exc:
            logger.debug("Live events error: %s", exc)

    # Persist to Redis
    try:
        cache_data = {
            "game_state": {
                "minutes_elapsed": game_state.minutes_elapsed,
                "home_goals": game_state.home_goals,
                "away_goals": game_state.away_goals,
                "home_red_cards": game_state.home_red_cards,
                "away_red_cards": game_state.away_red_cards,
                "is_extra_time": game_state.is_extra_time,
            },
            "live_stats": {
                field: getattr(live_stats, field)
                for field in live_stats.__dataclass_fields__
            },
        }
        await _redis_set(cache_key, json.dumps(cache_data), _CACHE_TTL)
    except Exception as exc:
        logger.debug("Live cache write error: %s", exc)

    logger.info(
        "Live match detected: %s vs %s | %d-%d min %d",
        team_home, team_away,
        game_state.home_goals, game_state.away_goals, game_state.minutes_elapsed,
    )
    return game_state, live_stats
