"""Persist parlay tickets for learning loop."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

COMPETITION = "fifa_world_cup"


def save_parlay_ticket(
    db,
    *,
    legs: list[dict[str, Any]],
    combined_prob: float,
    combined_odds: float,
    ev_parlay: float,
    combo_score: float,
    correlation_penalty: float,
    stake_pct: float,
    n_legs: int,
) -> str | None:
    """Guarda combinada en wc_predictions (metadata engine=parlay)."""
    if not db or not legs:
        return None
    parlay_id = str(uuid.uuid4())
    label = " + ".join(
        f"{l.get('team1')} vs {l.get('team2')}:{l.get('selection')}" for l in legs[:5]
    )
    try:
        row = (
            db.schema("ml")
            .table("wc_predictions")
            .insert(
                {
                    "competition": COMPETITION,
                    "team_home": legs[0].get("team1", ""),
                    "team_away": legs[0].get("team2", ""),
                    "match_date": legs[0].get("fecha"),
                    "market_type": "PARLAY",
                    "predicted_outcome": label[:200],
                    "probability": combined_prob,
                    "expected_value_fair": ev_parlay,
                    "kelly_stake": stake_pct / 100.0,
                    "metadata": {
                        "engine": "parlay",
                        "parlay_id": parlay_id,
                        "n_legs": n_legs,
                        "combined_odds": combined_odds,
                        "combo_score": combo_score,
                        "correlation_penalty": correlation_penalty,
                        "legs": legs,
                    },
                }
            )
            .execute()
        )
        return row.data[0]["id"] if row.data else parlay_id
    except Exception as exc:
        logger.warning("save_parlay_ticket: %s", exc)
        return None
