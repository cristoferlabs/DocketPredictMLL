"""Normalize Football-Data.org payloads into domain tables."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from supabase import Client

logger = logging.getLogger(__name__)

FD_STATUS = {
    "SCHEDULED": "scheduled",
    "TIMED": "scheduled",
    "IN_PLAY": "live",
    "PAUSED": "live",
    "LIVE": "live",
    "FINISHED": "finished",
    "POSTPONED": "postponed",
    "SUSPENDED": "postponed",
    "CANCELLED": "cancelled",
    "AWARDED": "finished",
}


def _checksum(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class FootballDataNormalizer:
    def __init__(self, db: Client, source_id: str):
        self.db = db
        self.source_id = source_id
        self.source_slug = "football-data"

    def store_raw(self, entity_type: str, external_id: str, payload: dict) -> None:
        self.db.schema("ops").table("raw_ingestions").upsert(
            {
                "source_id": self.source_id,
                "entity_type": entity_type,
                "external_id": external_id,
                "payload": payload,
                "checksum": _checksum(payload),
            },
            on_conflict="source_id,external_id",
        ).execute()

    def _get_or_create_league(self, competition: dict) -> str:
        ext_id = str(competition.get("id", competition.get("code", "unknown")))
        existing = (
            self.db.table("leagues")
            .select("id")
            .contains("external_ids", {self.source_slug: ext_id})
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]["id"]
        area = competition.get("area", {})
        result = self.db.table("leagues").insert(
            {
                "name": competition.get("name", "Unknown"),
                "country": area.get("name") if isinstance(area, dict) else None,
                "external_ids": {self.source_slug: ext_id, "code": competition.get("code")},
            }
        ).execute()
        return result.data[0]["id"]

    def _get_or_create_season(self, league_id: str, year: str) -> str:
        existing = (
            self.db.table("seasons")
            .select("id")
            .eq("league_id", league_id)
            .eq("year", year)
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]["id"]
        result = self.db.table("seasons").insert(
            {"league_id": league_id, "year": year, "is_active": True}
        ).execute()
        return result.data[0]["id"]

    def _get_or_create_team(self, season_id: str, team: dict) -> str:
        ext_id = str(team.get("id") or team.get("name") or "unknown")
        name = team.get("name") or team.get("shortName") or f"Team {ext_id}"
        existing = (
            self.db.table("teams")
            .select("id")
            .eq("season_id", season_id)
            .contains("external_ids", {self.source_slug: ext_id})
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]["id"]
        result = self.db.table("teams").insert(
            {
                "season_id": season_id,
                "name": name,
                "short_name": team.get("tla") or team.get("shortName"),
                "external_ids": {self.source_slug: ext_id},
            }
        ).execute()
        return result.data[0]["id"]

    def normalize_match(self, match: dict) -> str | None:
        ext_id = str(match["id"])
        self.store_raw("match", ext_id, match)

        competition = match.get("competition", {})
        league_id = self._get_or_create_league(competition)
        season_year = str(match.get("season", {}).get("startDate", match.get("utcDate", ""))[:4])
        if len(season_year) != 4:
            season_year = str(datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).year)
        season_id = self._get_or_create_season(league_id, season_year)

        home_team_id = self._get_or_create_team(season_id, match["homeTeam"])
        away_team_id = self._get_or_create_team(season_id, match["awayTeam"])

        status = FD_STATUS.get(match.get("status", "SCHEDULED"), "scheduled")

        match_row = {
            "season_id": season_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "kickoff_at": match.get("utcDate"),
            "status": status,
            "round": str(match.get("matchday", "")) if match.get("matchday") else None,
            "external_ids": {self.source_slug: ext_id},
        }

        existing = (
            self.db.table("matches")
            .select("id")
            .contains("external_ids", {self.source_slug: ext_id})
            .limit(1)
            .execute()
        )

        if existing.data:
            match_id = existing.data[0]["id"]
            self.db.table("matches").update(match_row).eq("id", match_id).execute()
        else:
            match_id = self.db.table("matches").insert(match_row).execute().data[0]["id"]

        score = match.get("score", {})
        ft = score.get("fullTime", {})
        if status == "finished" and ft.get("home") is not None:
            ht = score.get("halfTime", {})
            self.db.table("match_results").upsert(
                {
                    "match_id": match_id,
                    "home_goals": ft.get("home", 0),
                    "away_goals": ft.get("away", 0),
                    "ht_home_goals": ht.get("home"),
                    "ht_away_goals": ht.get("away"),
                    "stats_summary": score,
                },
                on_conflict="match_id",
            ).execute()

        self.db.schema("ops").table("raw_ingestions").update(
            {"processed_at": datetime.now(timezone.utc).isoformat()}
        ).eq("source_id", self.source_id).eq("external_id", ext_id).execute()

        return match_id
