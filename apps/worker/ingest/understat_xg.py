"""
Understat xG lookup — async DB queries for team and player season stats.

Used by wc_features.py to replace estimated lambdas with real xG priors
when the team plays in one of the 6 supported leagues.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Season most recently loaded into ml.team_season_xg
DEFAULT_SEASON = "2024-25"

# Name normalization aliases: understat team name → possible canonical forms
# used by worldcup_engine.name_match (substring / lower-alpha compare)
_ALIASES: dict[str, list[str]] = {
    "Manchester City":       ["man city", "manchester city"],
    "Manchester United":     ["man utd", "manchester united"],
    "Paris Saint Germain":   ["psg", "paris saint germain", "paris sg"],
    "Zenit St. Petersburg":  ["zenit", "zenit st. petersburg"],
    "FC Krasnodar":          ["krasnodar"],
    "Bayern Munich":         ["fc bayern münchen", "fc bayern munich", "bayern munich", "fc bayern"],
    "Borussia Dortmund":     ["bvb", "borussia dortmund"],
}


def _normalize(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _team_matches(db_team: str, query_team: str) -> bool:
    """Loose name match: exact on normalized, or alias list."""
    n_db = _normalize(db_team)
    n_q = _normalize(query_team)
    if n_db == n_q or n_db in n_q or n_q in n_db:
        return True
    for canonical, aliases in _ALIASES.items():
        if n_db in [_normalize(a) for a in aliases + [canonical]]:
            if n_q in [_normalize(a) for a in aliases + [canonical]]:
                return True
    return False


async def get_team_season_xg(
    team: str,
    *,
    season: str = DEFAULT_SEASON,
    league_slug: str | None = None,
) -> dict[str, Any] | None:
    """
    Return the best matching team row from ml.team_season_xg.

    Returns dict with keys: xg_per_game, xga_per_game, xg, xga, matches,
    league_slug, season, team — or None if not found.
    """
    try:
        from apps.shared.supabase_client import get_supabase_client
        client = get_supabase_client()
    except Exception:
        return None

    try:
        q = client.schema("ml").table("team_season_xg").select(
            "team,league_slug,season,matches,xg,xga,xg_per_game,xga_per_game,xpts"
        ).eq("season", season)

        if league_slug:
            q = q.eq("league_slug", league_slug)

        resp = q.execute()
        rows = resp.data or []
    except Exception as exc:
        log.debug("understat_xg lookup failed: %s", exc)
        return None

    for row in rows:
        if _team_matches(row["team"], team):
            return row

    return None


async def get_player_season_xg(
    team: str,
    *,
    season: str = DEFAULT_SEASON,
    league_slug: str | None = None,
    min_minutes: int = 900,
) -> list[dict[str, Any]]:
    """
    Return top players by xG90 for a team (sorted desc).
    Useful for GK-module augmentation and forward xG priors.
    """
    try:
        from apps.shared.supabase_client import get_supabase_client
        client = get_supabase_client()
    except Exception:
        return []

    try:
        q = (
            client.schema("ml")
            .table("player_season_xg")
            .select("player,team,apps,minutes,goals,assists,xg,xa,xg90,xa90")
            .eq("season", season)
            .gte("minutes", min_minutes)
            .order("xg90", desc=True)
        )
        if league_slug:
            q = q.eq("league_slug", league_slug)

        resp = q.execute()
        rows = resp.data or []
    except Exception as exc:
        log.debug("player_season_xg lookup failed: %s", exc)
        return []

    return [r for r in rows if _team_matches(r["team"], team)]


async def enrich_lambda_from_understat(
    team: str,
    *,
    season: str = DEFAULT_SEASON,
) -> tuple[float | None, float | None]:
    """
    Return (xg_per_game, xga_per_game) for team from Understat.

    These map directly to:
    - xg_per_game → lambda attack prior (replaces hist_avg_gf fallback)
    - xga_per_game → rival defense strength prior (replaces WC_AVG_GOALS fallback)

    Returns (None, None) if team not found.
    """
    row = await get_team_season_xg(team, season=season)
    if not row:
        return None, None

    xg_pg = row.get("xg_per_game")
    xga_pg = row.get("xga_per_game")

    if xg_pg is None and row.get("xg") and row.get("matches"):
        xg_pg = round(float(row["xg"]) / float(row["matches"]), 3)
    if xga_pg is None and row.get("xga") and row.get("matches"):
        xga_pg = round(float(row["xga"]) / float(row["matches"]), 3)

    return (float(xg_pg) if xg_pg else None, float(xga_pg) if xga_pg else None)
