"""Prediction query endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.deps import get_db

router = APIRouter()


@router.get("/match/{match_id}")
async def get_match_predictions(match_id: UUID, db=Depends(get_db)):
    """Get all predictions and betting combinations for a match."""
    match = (
        db.table("matches")
        .select("*, home_team:teams!matches_home_team_id_fkey(name), away_team:teams!matches_away_team_id_fkey(name)")
        .eq("id", str(match_id))
        .limit(1)
        .execute()
    )
    if not match.data:
        raise HTTPException(status_code=404, detail="Match not found")

    predictions = (
        db.schema("ml")
        .table("predictions")
        .select("*")
        .eq("match_id", str(match_id))
        .order("created_at", desc=True)
        .execute()
    )

    combinations = (
        db.schema("ml")
        .table("betting_combinations")
        .select("*, betting_combination_legs(*)")
        .eq("match_id", str(match_id))
        .order("priority")
        .execute()
    )

    return {
        "match": match.data[0],
        "predictions": predictions.data or [],
        "combinations": combinations.data or [],
    }


@router.get("/upcoming")
async def list_upcoming_predictions(
    limit: int = Query(20, ge=1, le=100),
    db=Depends(get_db),
):
    """List predictions for upcoming scheduled matches."""
    matches = (
        db.table("matches")
        .select("id, kickoff_at, status, home_team:teams!matches_home_team_id_fkey(name), away_team:teams!matches_away_team_id_fkey(name)")
        .eq("status", "scheduled")
        .order("kickoff_at")
        .limit(limit)
        .execute()
    )

    results: list[dict[str, Any]] = []
    for m in matches.data or []:
        preds = (
            db.schema("ml")
            .table("predictions")
            .select("market_type, predicted_outcome, probability, confidence_tier")
            .eq("match_id", m["id"])
            .execute()
        )
        results.append({**m, "predictions": preds.data or []})

    return {"matches": results, "count": len(results)}


@router.get("/search")
async def search_matches(q: str = Query(..., min_length=2), db=Depends(get_db)):
    """Search teams/matches by name for WhatsApp agent context."""
    teams = (
        db.table("teams")
        .select("id, name, short_name")
        .ilike("name", f"%{q}%")
        .limit(10)
        .execute()
    )

    team_ids = [t["id"] for t in teams.data or []]
    matches: list[dict] = []
    if team_ids:
        for tid in team_ids[:5]:
            home = (
                db.table("matches")
                .select("id, kickoff_at, status, home_team:teams!matches_home_team_id_fkey(name), away_team:teams!matches_away_team_id_fkey(name)")
                .eq("home_team_id", tid)
                .in_("status", ["scheduled", "live"])
                .order("kickoff_at")
                .limit(3)
                .execute()
            )
            away = (
                db.table("matches")
                .select("id, kickoff_at, status, home_team:teams!matches_home_team_id_fkey(name), away_team:teams!matches_away_team_id_fkey(name)")
                .eq("away_team_id", tid)
                .in_("status", ["scheduled", "live"])
                .order("kickoff_at")
                .limit(3)
                .execute()
            )
            matches.extend(home.data or [])
            matches.extend(away.data or [])

    seen = set()
    unique_matches = []
    for m in matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique_matches.append(m)

    return {"teams": teams.data or [], "matches": unique_matches[:10]}
