"""
Team WC Stats Service — fetches per-team historical stats from API-Football.

Aggregates shots on target, corners, and cards from all finished WC 2026
fixtures for a team. Results are cached in Redis for 6 hours to avoid
excessive API usage. Falls back gracefully when API is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass

from apps.worker.ingest.api_football import ApiFootballClient

logger = logging.getLogger(__name__)

_WC_LEAGUE_ID = 1
_WC_SEASON = 2026
_TEAM_STATS_TTL = 21600    # 6 hours
_FIXTURE_STATS_TTL = 86400  # 24 hours — finished match stats never change
_ALL_FIXTURES_TTL = 1800    # 30 min — shared fixture list cache
_STATS_ODDS_TTL = 3600      # 1 hour — pre-match odds refresh hourly
_DONE_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}

# API-Football bet IDs for stats markets
_STATS_BET_IDS = {
    45: "corners",   # Corners Over/Under
    87: "sot",       # Total ShotOnGoal
    80: "cards",     # Cards Over/Under
}


@dataclass
class StatsOdds:
    """Best available bookmaker odds for stats markets from API-Football."""
    corners_over_85: float | None = None
    corners_under_85: float | None = None
    corners_over_95: float | None = None
    corners_under_95: float | None = None
    corners_over_105: float | None = None
    corners_under_105: float | None = None
    sot_over_75: float | None = None
    sot_under_75: float | None = None
    sot_over_85: float | None = None
    sot_under_85: float | None = None
    sot_over_95: float | None = None
    sot_under_95: float | None = None
    cards_over_25: float | None = None
    cards_under_25: float | None = None
    cards_over_35: float | None = None
    cards_under_35: float | None = None
    cards_over_45: float | None = None
    cards_under_45: float | None = None


@dataclass
class TeamWCStats:
    """Per-team historical stats from finished WC 2026 matches."""
    team_id: int
    team_name: str
    matches_played: int
    avg_shots_on_target: float
    avg_shots_total: float
    avg_corners: float
    avg_yellow_cards: float
    avg_red_cards: float
    avg_xg: float | None = None
    source: str = "api_football"


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


def _parse_stat_val(statistics: list[dict], stat_type: str) -> float | None:
    for s in statistics:
        if s.get("type") == stat_type:
            val = s.get("value")
            if val is None:
                return None
            try:
                return float(str(val).rstrip("%"))
            except (TypeError, ValueError):
                return None
    return None


async def _get_fixture_team_stats(
    client: ApiFootballClient,
    fixture_id: int,
    team_id: int,
) -> dict[str, float | None]:
    """Fetch and parse one team's stats from one fixture. Cached per fixture."""
    cache_key = f"fx_stats:{fixture_id}:{team_id}"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    try:
        raw = await client.get_fixture_statistics(fixture_id)
    except Exception as exc:
        logger.debug("Fixture stats %s: %s", fixture_id, exc)
        return {}

    result: dict[str, float | None] = {}
    for block in raw:
        if block.get("team", {}).get("id") != team_id:
            continue
        stats = block.get("statistics", [])
        result = {
            "sot":          _parse_stat_val(stats, "Shots on Goal"),
            "shots":        _parse_stat_val(stats, "Total Shots"),
            "corners":      _parse_stat_val(stats, "Corner Kicks"),
            "yellow_cards": _parse_stat_val(stats, "Yellow Cards"),
            "red_cards":    _parse_stat_val(stats, "Red Cards"),
            "xg":           _parse_stat_val(stats, "expected_goals"),
        }
        break

    if result:
        await _redis_set(cache_key, json.dumps(result), _FIXTURE_STATS_TTL)
    return result


async def _get_all_wc_fixtures() -> list[dict]:
    """All WC 2026 fixtures, cached 30 min to avoid redundant API calls."""
    cache_key = "wc_all_fixtures:2026"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass
    client = ApiFootballClient()
    try:
        fixtures = await client.get_fixtures(league_id=_WC_LEAGUE_ID, season=_WC_SEASON)
        await _redis_set(cache_key, json.dumps(fixtures), _ALL_FIXTURES_TTL)
        return fixtures
    except Exception as exc:
        logger.warning("_get_all_wc_fixtures: %s", exc)
        return []


async def find_fixture_id(team_home: str, team_away: str) -> int | None:
    """Look up the API-Football fixture_id for a specific WC 2026 match."""
    cache_key = (
        f"fx_id:{team_home.lower().replace(' ', '_')}"
        f":{team_away.lower().replace(' ', '_')}"
    )
    cached = await _redis_get(cache_key)
    if cached:
        try:
            return int(cached)
        except Exception:
            pass

    from apps.api.services.worldcup_engine import name_match
    for fx in await _get_all_wc_fixtures():
        teams = fx.get("teams", {})
        h = teams.get("home", {}).get("name", "")
        a = teams.get("away", {}).get("name", "")
        fx_id = fx.get("fixture", {}).get("id")
        if not fx_id:
            continue
        if (name_match(h, team_home) and name_match(a, team_away)) or \
           (name_match(h, team_away) and name_match(a, team_home)):
            await _redis_set(cache_key, str(fx_id), _TEAM_STATS_TTL)
            return fx_id
    return None


def _parse_stats_odds_response(response: list[dict]) -> StatsOdds:
    """Parse API-Football odds response → best odds per stats market."""
    best: dict[str, float] = {}
    for event in response:
        for bk in event.get("bookmakers", []):
            for bet in bk.get("bets", []):
                market_key = _STATS_BET_IDS.get(bet.get("id"))
                if not market_key:
                    continue
                for v in bet.get("values", []):
                    val_str = v.get("value", "").lower().strip()
                    try:
                        odd = float(v.get("odd", 0))
                    except (TypeError, ValueError):
                        continue
                    if odd <= 1.0:
                        continue
                    for direction in ("over", "under"):
                        if val_str.startswith(direction):
                            try:
                                line = float(val_str[len(direction):].strip())
                                line_key = str(int(round(line * 10)))
                                attr = f"{market_key}_{direction}_{line_key}"
                                if attr in StatsOdds.__dataclass_fields__:
                                    best[attr] = max(best.get(attr, 0.0), odd)
                            except (ValueError, TypeError):
                                pass
    result = StatsOdds()
    for attr, val in best.items():
        if val > 1.0:
            setattr(result, attr, round(val, 2))
    return result


async def get_stats_market_odds(fixture_id: int) -> StatsOdds:
    """
    Fetch real bookmaker odds for Corners, SoT, and Cards from API-Football.

    Takes the best (highest) odds across all bookmakers. Cached 1 hour.
    Returns empty StatsOdds if API unavailable or no stats markets found.
    """
    cache_key = f"stats_odds:{fixture_id}"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            return StatsOdds(**{k: v for k, v in data.items() if k in StatsOdds.__dataclass_fields__})
        except Exception:
            pass

    client = ApiFootballClient()
    try:
        response = await client.get_odds(fixture_id=fixture_id)
    except Exception as exc:
        logger.warning("get_stats_market_odds fixture %s: %s", fixture_id, exc)
        return StatsOdds()

    result = _parse_stats_odds_response(response)
    await _redis_set(cache_key, json.dumps(asdict(result)), _STATS_ODDS_TTL)
    logger.info(
        "Stats odds loaded for fixture %s | CornO9.5=%s SoTO8.5=%s CardsO3.5=%s",
        fixture_id,
        result.corners_over_95, result.sot_over_85, result.cards_over_35,
    )
    return result


async def get_team_wc_stats(team_name: str) -> TeamWCStats | None:
    """
    Fetch and cache WC 2026 historical stats for a team.

    Returns None when the team has no finished WC 2026 fixtures or the API
    is unreachable. Caches team-level results in Redis for 6 hours and
    per-fixture results for 24 hours.
    """
    cache_key = f"wc_team_stats:{team_name.lower().replace(' ', '_')}:2026"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if data.get("not_found"):
                return None
            return TeamWCStats(**{k: v for k, v in data.items() if k in TeamWCStats.__dataclass_fields__})
        except Exception as exc:
            logger.debug("Team stats cache error: %s", exc)

    try:
        all_fixtures = await _get_all_wc_fixtures()
    except Exception as exc:
        logger.warning("API-Football fixtures error for team stats: %s", exc)
        return None

    from apps.api.services.worldcup_engine import name_match
    client = ApiFootballClient()

    team_id: int | None = None
    finished_ids: list[int] = []

    for fx in all_fixtures:
        teams = fx.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        fx_id = fx.get("fixture", {}).get("id")
        if not fx_id:
            continue

        home_name = home.get("name", "")
        away_name = away.get("name", "")

        matched_home = name_match(home_name, team_name) or name_match(team_name, home_name)
        matched_away = name_match(away_name, team_name) or name_match(team_name, away_name)

        if matched_home:
            if team_id is None:
                team_id = home.get("id")
            if status in _DONE_STATUSES:
                finished_ids.append(fx_id)
        elif matched_away:
            if team_id is None:
                team_id = away.get("id")
            if status in _DONE_STATUSES:
                finished_ids.append(fx_id)

    if not team_id or not finished_ids:
        logger.info("No finished WC 2026 fixtures for '%s'", team_name)
        await _redis_set(cache_key, json.dumps({"not_found": True}), 3600)
        return None

    # Fetch all fixture stats concurrently
    tasks = [_get_fixture_team_stats(client, fxid, team_id) for fxid in finished_ids]
    stats_per_match = await asyncio.gather(*tasks)

    sot_l, shots_l, corners_l, yellow_l, red_l, xg_l = [], [], [], [], [], []
    for s in stats_per_match:
        if not s:
            continue
        if s.get("sot") is not None:
            sot_l.append(s["sot"])
        if s.get("shots") is not None:
            shots_l.append(s["shots"])
        if s.get("corners") is not None:
            corners_l.append(s["corners"])
        if s.get("yellow_cards") is not None:
            yellow_l.append(s["yellow_cards"])
        if s.get("red_cards") is not None:
            red_l.append(s["red_cards"])
        if s.get("xg") is not None:
            xg_l.append(s["xg"])

    if not sot_l and not corners_l:
        logger.info("No fixture stats data returned for '%s'", team_name)
        await _redis_set(cache_key, json.dumps({"not_found": True}), 3600)
        return None

    def _avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    result = TeamWCStats(
        team_id=team_id,
        team_name=team_name,
        matches_played=len(finished_ids),
        avg_shots_on_target=_avg(sot_l),
        avg_shots_total=_avg(shots_l),
        avg_corners=_avg(corners_l),
        avg_yellow_cards=_avg(yellow_l),
        avg_red_cards=_avg(red_l),
        avg_xg=_avg(xg_l) if xg_l else None,
        source="api_football",
    )

    await _redis_set(cache_key, json.dumps(asdict(result)), _TEAM_STATS_TTL)
    logger.info(
        "WC team stats loaded: %s | SoT=%.1f, Corners=%.1f, Cards=%.1f (n=%d matches)",
        team_name,
        result.avg_shots_on_target,
        result.avg_corners,
        result.avg_yellow_cards + result.avg_red_cards,
        len(finished_ids),
    )
    return result


async def get_match_team_stats(
    team_home: str,
    team_away: str,
) -> tuple[TeamWCStats | None, TeamWCStats | None]:
    """Fetch WC stats for both teams concurrently."""
    results = await asyncio.gather(
        get_team_wc_stats(team_home),
        get_team_wc_stats(team_away),
        return_exceptions=True,
    )
    home_s = results[0] if not isinstance(results[0], Exception) else None
    away_s = results[1] if not isinstance(results[1], Exception) else None
    return home_s, away_s  # type: ignore[return-value]
