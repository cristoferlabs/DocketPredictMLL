"""Persist and evaluate World Cup predictions from Telegram/API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apps.worker.ml.evaluation import evaluate_prediction

logger = logging.getLogger(__name__)

COMPETITION = "fifa_world_cup"


def save_wc_prediction(
    db,
    *,
    team_home: str,
    team_away: str,
    match_date: str | None,
    market_type: str,
    predicted_outcome: str,
    probability: float,
    expected_value_fair: float | None = None,
    edge_fair: float | None = None,
    kelly_stake: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    try:
        row = (
            db.schema("ml")
            .table("wc_predictions")
            .insert(
                {
                    "competition": COMPETITION,
                    "team_home": team_home,
                    "team_away": team_away,
                    "match_date": match_date,
                    "market_type": market_type,
                    "predicted_outcome": predicted_outcome,
                    "probability": probability,
                    "expected_value_fair": expected_value_fair,
                    "edge_fair": edge_fair,
                    "kelly_stake": kelly_stake,
                    "metadata": metadata or {},
                }
            )
            .execute()
        )
        return row.data[0]["id"] if row.data else None
    except Exception as exc:
        logger.warning("save_wc_prediction: %s", exc)
        return None


def save_odds_snapshot(
    db,
    *,
    match_key: str,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    odds_decimal: float,
    fair_odds: float | None = None,
    snapshot_type: str = "pick",
) -> None:
    try:
        db.schema("ml").table("odds_snapshots").insert(
            {
                "competition": COMPETITION,
                "match_key": match_key,
                "team_home": team_home,
                "team_away": team_away,
                "market": market,
                "selection": selection,
                "odds_decimal": odds_decimal,
                "fair_odds": fair_odds,
                "snapshot_type": snapshot_type,
            }
        ).execute()
    except Exception as exc:
        logger.warning("odds_snapshot: %s", exc)


def compute_clv(pick_odds: float, closing_odds: float) -> float:
    """CLV as % edge vs closing line (positive = beat the close)."""
    if pick_odds <= 1 or closing_odds <= 1:
        return 0.0
    pick_impl = 1.0 / pick_odds
    close_impl = 1.0 / closing_odds
    return round((close_impl - pick_impl) / close_impl, 4)


async def evaluate_wc_predictions(db, finished_matches: list[dict] | None = None) -> dict:
    """Evaluate pending wc_predictions against finished WC results."""
    pending = (
        db.schema("ml")
        .table("wc_predictions")
        .select("*")
        .is_("evaluated_at", "null")
        .limit(200)
        .execute()
    )
    if not pending.data:
        return {"evaluated": 0}

    if finished_matches is None:
        from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
        from apps.worker.tasks.update_elo import extract_finished_wc_matches

        archives = await fetch_all_worldcup_archives()
        finished_matches = extract_finished_wc_matches(archives.get(2026, {}))

    result_index: dict[tuple[str, str], dict] = {}
    for m in finished_matches:
        t1 = m.get("team1", {}).get("name", "")
        t2 = m.get("team2", {}).get("name", "")
        ft = m.get("score", {}).get("ft", [0, 0])
        key = (t1.lower(), t2.lower())
        result_index[key] = {
            "home_goals": int(ft[0]),
            "away_goals": int(ft[1]),
            "date": (m.get("date") or "")[:10],
        }
        result_index[(t2.lower(), t1.lower())] = result_index[key]

    evaluated = 0
    now = datetime.now(timezone.utc).isoformat()

    for pred in pending.data or []:
        key = (pred["team_home"].lower(), pred["team_away"].lower())
        res = result_index.get(key)
        if not res:
            continue

        eval_result = evaluate_prediction(
            market_type=pred["market_type"],
            predicted_outcome=pred["predicted_outcome"],
            probability=float(pred["probability"]),
            home_goals=res["home_goals"],
            away_goals=res["away_goals"],
        )

        db.schema("ml").table("wc_predictions").update(
            {
                "actual_outcome": eval_result["actual_outcome"],
                "is_correct": eval_result["is_correct"],
                "brier_score": eval_result["brier_score"],
                "evaluated_at": now,
            }
        ).eq("id", pred["id"]).execute()
        evaluated += 1

    return {"evaluated": evaluated}
