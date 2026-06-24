"""Prediction generation tasks."""

import logging

from apps.shared.supabase_client import get_supabase
from apps.worker.ml.ensemble import MatchInput, predict_match
from apps.worker.ml.gk import GoalkeeperProfile
from apps.worker.ml.model_loader import load_active_xgboost, load_ensemble_weights
from apps.worker.ml.odds_math import fair_h2h_market

logger = logging.getLogger(__name__)


async def _get_team_elo(db, team_id: str, league_id: str) -> float:
    result = (
        db.table("team_elo_ratings")
        .select("rating")
        .eq("team_id", team_id)
        .eq("league_id", league_id)
        .order("played_at", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return float(result.data[0]["rating"])
    return 1500.0


async def _get_team_gk(db, team_id: str) -> GoalkeeperProfile | None:
    result = (
        db.table("goalkeepers")
        .select("name, save_pct, xga_90, is_starter")
        .eq("team_id", team_id)
        .eq("is_starter", True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    gk = result.data[0]
    return GoalkeeperProfile(
        name=gk["name"],
        save_pct=float(gk["save_pct"]) if gk.get("save_pct") else None,
        xga_90=float(gk["xga_90"]) if gk.get("xga_90") else None,
        is_starter=True,
    )


async def _get_match_odds(db, match_id: str) -> dict[str, float]:
    """Load fair odds from latest odds-api raw ingestion if available."""
    try:
        rows = (
            db.schema("ops")
            .table("raw_ingestions")
            .select("payload")
            .eq("source", "odds-api")
            .order("ingested_at", desc=True)
            .limit(20)
            .execute()
        )
        for row in rows.data or []:
            payload = row.get("payload") or {}
            events = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(events, list):
                continue
            for ev in events:
                if ev.get("id") == match_id or str(ev.get("id")) == str(match_id):
                    fair = fair_h2h_market(ev)
                    return {
                        "home_win": fair.get("home", {}).get("fair_odds", 2.0),
                        "draw": fair.get("draw", {}).get("fair_odds", 3.5),
                        "away_win": fair.get("away", {}).get("fair_odds", 3.5),
                        "over_25": 1.85,
                    }
    except Exception as exc:
        logger.debug("odds load: %s", exc)
    return {}


async def predict_upcoming_matches(ctx: dict) -> dict:
    """Run ensemble predictions for all scheduled matches."""
    db = get_supabase()

    model_version = (
        db.schema("ml")
        .table("model_versions")
        .select("id")
        .eq("model_type", "ensemble")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    if not model_version.data:
        return {"error": "No active ensemble model version"}

    model_version_id = model_version.data[0]["id"]
    xgb_model = load_active_xgboost(db)

    matches = (
        db.table("matches")
        .select("id, home_team_id, away_team_id, season_id, seasons(league_id)")
        .eq("status", "scheduled")
        .execute()
    )

    created = 0
    for match in matches.data or []:
        league_id = match.get("seasons", {}).get("league_id", "")
        weights = load_ensemble_weights(db, league_id) if league_id else None
        home_elo = await _get_team_elo(db, match["home_team_id"], league_id)
        away_elo = await _get_team_elo(db, match["away_team_id"], league_id)
        home_gk = await _get_team_gk(db, match["home_team_id"])
        away_gk = await _get_team_gk(db, match["away_team_id"])

        stats = (
            db.table("match_stats")
            .select("xg, raw")
            .eq("match_id", match["id"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        home_xg, away_xg = None, None
        if stats.data:
            raw = stats.data[0].get("raw", {})
            home_xg = raw.get("home_xg")
            away_xg = raw.get("away_xg")

        odds = await _get_match_odds(db, match["id"])

        result = predict_match(
            MatchInput(
                home_elo=home_elo,
                away_elo=away_elo,
                home_xg=home_xg,
                away_xg=away_xg,
                home_gk=home_gk,
                away_gk=away_gk,
            ),
            xgb_model=xgb_model,
            weights=weights,
            odds=odds or None,
        )

        training_meta = {
            "elo_probs": {
                "home_win": result.elo.home_win,
                "draw": result.elo.draw,
                "away_win": result.elo.away_win,
            },
            "poisson_probs": {
                "home_win": float(
                    sum(
                        result.poisson.score_matrix[i, j]
                        for i in range(result.poisson.score_matrix.shape[0])
                        for j in range(result.poisson.score_matrix.shape[1])
                        if i > j
                    )
                ),
                "draw": float(
                    sum(
                        result.poisson.score_matrix[i, j]
                        for i in range(result.poisson.score_matrix.shape[0])
                        for j in range(result.poisson.score_matrix.shape[1])
                        if i == j
                    )
                ),
                "away_win": float(
                    sum(
                        result.poisson.score_matrix[i, j]
                        for i in range(result.poisson.score_matrix.shape[0])
                        for j in range(result.poisson.score_matrix.shape[1])
                        if i < j
                    )
                ),
                "over_25": result.poisson.over_25,
                "btts_yes": result.poisson.btts_yes,
                "lambda_home": result.poisson.lambda_home,
                "lambda_away": result.poisson.lambda_away,
            },
            "gk_adjustment": {
                "home_factor": result.gk.home_factor,
                "away_factor": result.gk.away_factor,
            },
            "xgboost_probs": result.xgboost_probs,
            "ensemble_weights": {
                "elo": weights.elo if weights else 0.25,
                "poisson": weights.poisson if weights else 0.35,
                "gk": weights.gk if weights else 0.15,
                "xgboost": weights.xgboost if weights else 0.25,
            },
        }

        for pred in result.predictions:
            metadata = {**pred.metadata, **training_meta}
            db.schema("ml").table("predictions").insert(
                {
                    "match_id": match["id"],
                    "model_version_id": model_version_id,
                    "market_type": pred.market_type,
                    "predicted_outcome": pred.predicted_outcome,
                    "probability": pred.probability,
                    "confidence_tier": pred.confidence_tier,
                    "metadata": metadata,
                }
            ).execute()
            created += 1

        for combo in result.combinations:
            combo_insert = (
                db.schema("ml")
                .table("betting_combinations")
                .insert(
                    {
                        "match_id": match["id"],
                        "priority": combo.priority,
                        "expected_value": combo.expected_value,
                        "kelly_fraction": combo.kelly_fraction,
                        "status": "pending",
                    }
                )
                .execute()
            )
            if combo_insert.data:
                combo_id = combo_insert.data[0]["id"]
                for leg in combo.legs:
                    db.schema("ml").table("betting_combination_legs").insert(
                        {
                            "combination_id": combo_id,
                            "market_type": leg["market_type"],
                            "selection": leg["selection"],
                            "odds": leg.get("odds"),
                            "probability": leg.get("probability"),
                        }
                    ).execute()

    logger.info("predict_upcoming_matches: created %d predictions", created)
    return {"predictions_created": created, "matches_processed": len(matches.data or [])}
