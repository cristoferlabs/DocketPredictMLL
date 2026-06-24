"""Persist and load World Cup ELO ratings by team name."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apps.api.services.worldcup_engine import calc_elo_ratings, name_match, normalize_openfootball
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives

logger = logging.getLogger(__name__)

COMPETITION = "fifa_world_cup"


def ratings_dict_from_archives(
    data_2018: dict,
    data_2022: dict,
    data_2026: dict,
) -> dict[str, float]:
    return calc_elo_ratings(data_2018, data_2022, data_2026)


async def load_wc_elo_from_db(db=None) -> dict[str, float]:
    """Latest ELO per team from ml.wc_team_elo."""
    db = db or get_supabase()
    ratings: dict[str, float] = {}
    try:
        rows = (
            db.schema("ml")
            .table("wc_team_elo")
            .select("team_name, rating")
            .eq("competition", COMPETITION)
            .order("played_at", desc=True)
            .limit(500)
            .execute()
        )
        for row in rows.data or []:
            name = row["team_name"]
            if name not in ratings:
                ratings[name] = float(row["rating"])
    except Exception as exc:
        logger.warning("load_wc_elo_from_db: %s", exc)
    return ratings


def merge_elo_sources(
    computed: dict[str, float],
    from_db: dict[str, float],
) -> dict[str, float]:
    """DB takes precedence when team exists; else computed."""
    merged = dict(computed)
    for team, rating in from_db.items():
        merged[team] = rating
    return merged


async def get_wc_elo_ratings(db=None) -> dict[str, float]:
    archives = await fetch_all_worldcup_archives()
    computed = ratings_dict_from_archives(
        archives.get(2018, {}),
        archives.get(2022, {}),
        archives.get(2026, {}),
    )
    from_db = await load_wc_elo_from_db(db)
    return merge_elo_sources(computed, from_db)


async def persist_elo_snapshot(
    ratings: dict[str, float],
    *,
    match_date: str | None = None,
    source: str = "calc_elo_ratings",
    db=None,
) -> int:
    """Insert current ratings snapshot for all teams."""
    db = db or get_supabase()
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    for team, rating in ratings.items():
        try:
            db.schema("ml").table("wc_team_elo").insert(
                {
                    "competition": COMPETITION,
                    "team_name": team,
                    "rating": rating,
                    "match_date": match_date,
                    "played_at": now,
                    "source": source,
                }
            ).execute()
            inserted += 1
        except Exception as exc:
            logger.warning("wc_team_elo insert %s: %s", team, exc)
    return inserted


def extract_finished_wc_matches(data_2026: dict) -> list[dict[str, Any]]:
    norm = normalize_openfootball(data_2026)
    finished = []
    for rnd in norm.get("rounds", []):
        for m in rnd.get("matches", []):
            if m.get("score", {}).get("ft"):
                finished.append({**m, "roundName": rnd.get("name")})
    finished.sort(key=lambda x: x.get("date", ""))
    return finished


async def update_elo_after_finished_matches(ctx: dict) -> dict:
    """Recompute ELO from archives and persist snapshot after new WC results."""
    db = get_supabase()
    job_id = None
    try:
        ins = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "update_elo", "status": "running"})
            .execute()
        )
        job_id = ins.data[0]["id"] if ins.data else None
    except Exception:
        pass

    archives = await fetch_all_worldcup_archives()
    ratings = ratings_dict_from_archives(
        archives.get(2018, {}),
        archives.get(2022, {}),
        archives.get(2026, {}),
    )
    finished = extract_finished_wc_matches(archives.get(2026, {}))
    last_date = finished[-1].get("date", "")[:10] if finished else None

    inserted = await persist_elo_snapshot(ratings, match_date=last_date, db=db)

    result = {
        "status": "completed",
        "teams": len(ratings),
        "finished_matches_2026": len(finished),
        "rows_inserted": inserted,
        "last_match_date": last_date,
    }

    if job_id:
        db.schema("ops").table("job_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": result,
            }
        ).eq("id", job_id).execute()

    logger.info("update_elo: %s", result)
    return result
