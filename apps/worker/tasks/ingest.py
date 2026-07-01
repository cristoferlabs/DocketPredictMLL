"""Multi-source fixture ingestion orchestrator."""

import logging
from datetime import date, datetime, timezone
from typing import Any

from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.api_football import ApiFootballClient
from apps.worker.ingest.football_data import FootballDataClient
from apps.worker.ingest.football_data_normalizer import FootballDataNormalizer
from apps.worker.ingest.normalizer import ApiFootballNormalizer
from apps.worker.ingest.odds_api import OddsApiClient
from apps.worker.ingest.sportmonks import SportMonksClient

logger = logging.getLogger(__name__)

DEFAULT_API_FOOTBALL_LEAGUES = [140, 39, 135, 78, 61]
WC_LEAGUE_ID = 1
WC_SEASON = 2026


async def _get_source_id(db, slug: str) -> str | None:
    result = (
        db.schema("ops").table("data_sources").select("id").eq("slug", slug).limit(1).execute()
    )
    return result.data[0]["id"] if result.data else None


async def _ingest_football_data(
    db,
    days_ahead: int,
    competition_code: str | None = None,
) -> dict[str, Any]:
    source_id = await _get_source_id(db, "football-data")
    if not source_id:
        return {"source": "football-data", "processed": 0, "errors": ["source not in DB"]}

    settings = get_settings()
    if not settings.football_data_key:
        return {"source": "football-data", "processed": 0, "errors": ["FOOTBALL_DATA_KEY not set"]}

    client = FootballDataClient()
    normalizer = FootballDataNormalizer(db, source_id)
    processed = 0
    errors: list[str] = []

    try:
        if competition_code:
            today = date.today()
            matches = await client.get_competition_matches(
                competition_code,
                today,
                today + __import__("datetime").timedelta(days=days_ahead),
            )
        else:
            matches = await client.get_upcoming(days_ahead=days_ahead)
    except Exception as exc:
        return {"source": "football-data", "processed": 0, "errors": [str(exc)]}

    for match in matches:
        try:
            if normalizer.normalize_match(match):
                processed += 1
        except Exception as exc:
            errors.append(f"match {match.get('id')}: {exc}")

    return {
        "source": "football-data",
        "processed": processed,
        "total_fetched": len(matches),
        "errors": errors,
    }


async def _ingest_api_football(
    db,
    league_external_id: int | None,
    season: int,
    days_ahead: int,
) -> dict[str, Any]:
    source_id = await _get_source_id(db, "api-football")
    if not source_id:
        return {"source": "api-football", "processed": 0, "errors": ["source not in DB"]}

    settings = get_settings()
    if not settings.api_football_key:
        return {"source": "api-football", "processed": 0, "errors": ["API_FOOTBALL_KEY not set"]}

    client = ApiFootballClient()
    normalizer = ApiFootballNormalizer(db, source_id)
    league_ids = [league_external_id] if league_external_id else DEFAULT_API_FOOTBALL_LEAGUES
    processed = 0
    errors: list[str] = []

    for lid in league_ids:
        try:
            fixtures = await client.get_upcoming_fixtures(lid, days_ahead, season)
            for fixture in fixtures:
                try:
                    match_id = normalizer.normalize_fixture(fixture)
                    if match_id:
                        processed += 1
                except Exception as exc:
                    errors.append(f"fixture {fixture.get('fixture', {}).get('id')}: {exc}")
        except Exception as exc:
            errors.append(f"league {lid}: {exc}")

    return {
        "source": "api-football",
        "processed": processed,
        "season": season,
        "leagues": league_ids,
        "errors": errors,
    }


async def _ingest_odds(db) -> dict[str, Any]:
    source_id = await _get_source_id(db, "odds-api")
    settings = get_settings()
    if not settings.odds_api_key or not source_id:
        return {"source": "odds-api", "processed": 0, "errors": ["ODDS_API_KEY not set"]}

    client = OddsApiClient()
    try:
        events = await client.get_soccer_odds()
    except Exception as exc:
        return {"source": "odds-api", "processed": 0, "errors": [str(exc)]}

    stored = 0
    for event in events:
        ext_id = event.get("id", "")
        if not ext_id:
            continue
        db.schema("ops").table("raw_ingestions").upsert(
            {
                "source_id": source_id,
                "entity_type": "odds_event",
                "external_id": ext_id,
                "payload": event,
                "checksum": str(hash(str(event))),
            },
            on_conflict="source_id,external_id",
        ).execute()
        stored += 1

    return {"source": "odds-api", "processed": stored, "total_events": len(events), "errors": []}


async def _ingest_sportmonks(db) -> dict[str, Any]:
    source_id = await _get_source_id(db, "sportmonks")
    settings = get_settings()
    if not settings.sportmonks_key or not source_id:
        return {"source": "sportmonks", "processed": 0, "skipped": True, "errors": []}

    client = SportMonksClient()
    normalizer = None  # SportMonks normalizer TBD when subscription allows
    try:
        fixtures = await client.get_fixtures_by_date()
    except Exception as exc:
        return {"source": "sportmonks", "processed": 0, "errors": [str(exc)]}

    if not fixtures:
        return {
            "source": "sportmonks",
            "processed": 0,
            "skipped": True,
            "errors": ["Sin datos (plan free puede no incluir este endpoint)"],
        }

    return {"source": "sportmonks", "processed": 0, "total_fetched": len(fixtures), "errors": []}


async def ingest_fixtures(
    ctx: dict,
    league_external_id: int | None = None,
    season: int | None = None,
    days_ahead: int = 7,
    competition_code: str | None = None,
    sources: list[str] | None = None,
) -> dict:
    """
    Ingest fixtures from free-tier sources.

    Default order:
    1. football-data — current season (gratis, recomendado)
    2. api-football — solo si season <= 2024 (límite plan free)
    3. odds-api — cuotas para EV
    4. sportmonks — si hay datos en el plan
    """
    db = get_supabase()
    job_id = None
    try:
        job_insert = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "ingest_fixtures", "status": "running"})
            .execute()
        )
        job_id = job_insert.data[0]["id"] if job_insert.data else None
    except Exception as exc:
        logger.warning("Could not create job_run: %s", exc)

    active_sources = sources or ["football-data", "api-football", "odds-api", "sportmonks"]
    results: list[dict] = []
    total_processed = 0

    if "football-data" in active_sources:
        results.append(await _ingest_football_data(db, days_ahead, competition_code or "WC"))

    # Cache openfootball archives for Mundial analytics
    if competition_code in (None, "WC"):
        try:
            from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives

            wc_source = await _get_source_id(db, "custom")
            archives = await fetch_all_worldcup_archives()
            for year, payload in archives.items():
                if payload and wc_source:
                    db.schema("ops").table("raw_ingestions").upsert(
                        {
                            "source_id": wc_source,
                            "entity_type": "worldcup_json",
                            "external_id": f"openfootball-{year}",
                            "payload": payload,
                            "checksum": str(hash(str(year))),
                        },
                        on_conflict="source_id,external_id",
                    ).execute()
            results.append({"source": "openfootball", "processed": len(archives), "errors": []})
        except Exception as exc:
            results.append({"source": "openfootball", "processed": 0, "errors": [str(exc)]})

    if "api-football" in active_sources:
        season_year = season or date.today().year
        if league_external_id or season:
            results.append(
                await _ingest_api_football(db, league_external_id, season_year, days_ahead)
            )
        # Always ingest WC2026 fixtures when running WC competition context
        if competition_code in (None, "WC"):
            wc_result = await _ingest_api_football(db, WC_LEAGUE_ID, WC_SEASON, days_ahead)
            wc_result["source"] = "api-football-wc"
            results.append(wc_result)

    if "odds-api" in active_sources:
        results.append(await _ingest_odds(db))

    if "sportmonks" in active_sources:
        results.append(await _ingest_sportmonks(db))

    total_processed = sum(r.get("processed", 0) for r in results)
    all_errors = [e for r in results for e in r.get("errors", [])]

    summary = {
        "processed": total_processed,
        "sources": results,
        "errors": all_errors,
        "days_ahead": days_ahead,
    }

    if job_id:
        db.schema("ops").table("job_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": summary,
            }
        ).eq("id", job_id).execute()

    logger.info("ingest_fixtures completed: %s", summary)
    return summary
