"""Normalize API-Football raw payloads into domain tables."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from supabase import Client

logger = logging.getLogger(__name__)

STATUS_MAP = {
    "TBD": "scheduled",
    "NS": "scheduled",
    "1H": "live",
    "HT": "live",
    "2H": "live",
    "ET": "live",
    "P": "live",
    "FT": "finished",
    "AET": "finished",
    "PEN": "finished",
    "PST": "postponed",
    "CANC": "cancelled",
    "ABD": "cancelled",
    "AWD": "finished",
    "WO": "finished",
}


def _checksum(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _external_id(source_slug: str, entity_type: str, ext_id: str) -> str:
    return f"{source_slug}:{entity_type}:{ext_id}"


class ApiFootballNormalizer:
    def __init__(self, db: Client, source_id: str):
        self.db = db
        self.source_id = source_id
        self.source_slug = "api-football"

    def store_raw(self, entity_type: str, external_id: str, payload: dict) -> str | None:
        checksum = _checksum(payload)
        row = {
            "source_id": self.source_id,
            "entity_type": entity_type,
            "external_id": external_id,
            "payload": payload,
            "checksum": checksum,
        }
        result = self.db.schema("ops").table("raw_ingestions").upsert(
            row, on_conflict="source_id,external_id"
        ).execute()
        return result.data[0]["id"] if result.data else None

    def _get_or_create_league(self, league_data: dict) -> str:
        ext_id = str(league_data["id"])
        existing = (
            self.db.table("leagues")
            .select("id")
            .contains("external_ids", {self.source_slug: ext_id})
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]["id"]

        result = self.db.table("leagues").insert(
            {
                "name": league_data.get("name", "Unknown"),
                "country": league_data.get("country"),
                "external_ids": {self.source_slug: ext_id},
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

    def _get_or_create_team(self, season_id: str, team_data: dict) -> str:
        ext_id = str(team_data["id"])
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
                "name": team_data.get("name", "Unknown"),
                "short_name": team_data.get("code"),
                "external_ids": {self.source_slug: ext_id},
            }
        ).execute()
        return result.data[0]["id"]

    def normalize_fixture(self, fixture_payload: dict) -> str | None:
        """Upsert match from API-Football fixture response item."""
        fixture = fixture_payload.get("fixture", fixture_payload)
        league_info = fixture_payload.get("league", {})
        teams = fixture_payload.get("teams", {})
        goals = fixture_payload.get("goals", {})
        score = fixture_payload.get("score", {})

        ext_fixture_id = str(fixture["id"])
        self.store_raw("fixture", ext_fixture_id, fixture_payload)

        league_id = self._get_or_create_league(
            {"id": league_info.get("id"), "name": league_info.get("name"), "country": league_info.get("country")}
        )
        season_year = str(league_info.get("season", fixture.get("date", "")[:4]))
        season_id = self._get_or_create_season(league_id, season_year)

        home_team_id = self._get_or_create_team(season_id, teams.get("home", {}))
        away_team_id = self._get_or_create_team(season_id, teams.get("away", {}))

        status_short = fixture.get("status", {}).get("short", "NS")
        status = STATUS_MAP.get(status_short, "scheduled")

        match_row = {
            "season_id": season_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "kickoff_at": fixture.get("date"),
            "status": status,
            "round": league_info.get("round"),
            "external_ids": {self.source_slug: ext_fixture_id},
        }

        existing = (
            self.db.table("matches")
            .select("id")
            .contains("external_ids", {self.source_slug: ext_fixture_id})
            .limit(1)
            .execute()
        )

        if existing.data:
            match_id = existing.data[0]["id"]
            self.db.table("matches").update(match_row).eq("id", match_id).execute()
        else:
            result = self.db.table("matches").insert(match_row).execute()
            match_id = result.data[0]["id"]

        if status == "finished" and goals.get("home") is not None:
            self._upsert_result(match_id, goals, score)

        self.db.schema("ops").table("raw_ingestions").update(
            {"processed_at": datetime.now(timezone.utc).isoformat()}
        ).eq("source_id", self.source_id).eq("external_id", ext_fixture_id).execute()

        return match_id

    def _upsert_result(self, match_id: str, goals: dict, score: dict) -> None:
        ht = score.get("halftime", {})
        row = {
            "match_id": match_id,
            "home_goals": goals.get("home", 0),
            "away_goals": goals.get("away", 0),
            "ht_home_goals": ht.get("home"),
            "ht_away_goals": ht.get("away"),
            "stats_summary": {"fulltime": goals, "halftime": ht},
        }
        self.db.table("match_results").upsert(row, on_conflict="match_id").execute()

    def normalize_statistics(self, match_id: str, stats_payload: list[dict]) -> None:
        ext_id = f"stats-{match_id}"
        self.store_raw("fixture_statistics", ext_id, {"data": stats_payload})

        xg_home, xg_away = None, None
        possession_home, possession_away = None, None
        shots_home, shots_away = 0, 0

        for team_stats in stats_payload:
            team_side = team_stats.get("team", {}).get("name", "")
            statistics = {s["type"]: s["value"] for s in team_stats.get("statistics", [])}
            xg_val = statistics.get("expected_goals")
            if xg_val is not None:
                try:
                    xg_float = float(str(xg_val).replace("%", ""))
                except ValueError:
                    xg_float = None
            else:
                xg_float = None

            poss = statistics.get("Ball Possession")
            if poss:
                try:
                    poss_val = float(str(poss).replace("%", ""))
                except ValueError:
                    poss_val = None
            else:
                poss_val = None

            shots = statistics.get("Total Shots", 0)
            try:
                shots_int = int(shots) if shots else 0
            except (ValueError, TypeError):
                shots_int = 0

            # Heuristic: first team in response is usually home
            if xg_home is None:
                xg_home, possession_home, shots_home = xg_float, poss_val, shots_int
            else:
                xg_away, possession_away, shots_away = xg_float, poss_val, shots_int

        self.db.table("match_stats").insert(
            {
                "match_id": match_id,
                "source_id": self.source_id,
                "possession": possession_home,
                "shots": shots_home + shots_away,
                "xg": xg_home,
                "raw": {"home_xg": xg_home, "away_xg": xg_away, "payload": stats_payload},
            }
        ).execute()
