"""
label_picks — resolves outcomes for pending picks in ml.picks_log.

Goals-based markets (1X2, OU, DC, BTTS): resolved from WC archive results.
Stats markets (CORNERS, SOT, CARDS): resolved from API-Football fixture statistics.

Run after each round of matches finishes.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from supabase import Client

logger = logging.getLogger(__name__)

_GOALS_MARKETS = {"1X2", "DC", "BTTS", "OU_1.5", "OU_2.5", "OU_3.5", "OU"}
_STATS_MARKETS = {"CORNERS", "SOT", "CARDS"}


async def _fetch_wc_results() -> dict[tuple[str, str], dict]:
    """Build result index (team1, team2) → {home_goals, away_goals} from WC archive."""
    from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
    from apps.worker.tasks.update_elo import extract_finished_wc_matches

    archives = await fetch_all_worldcup_archives()
    finished = extract_finished_wc_matches(archives.get(2026, {}))
    index: dict[tuple[str, str], dict] = {}
    for m in finished:
        t1 = m.get("team1", {}).get("name", "")
        t2 = m.get("team2", {}).get("name", "")
        ft = m.get("score", {}).get("ft", [0, 0])
        entry = {"home_goals": int(ft[0]), "away_goals": int(ft[1])}
        index[(t1.lower(), t2.lower())] = entry
        index[(t2.lower(), t1.lower())] = entry
    return index


async def _fetch_stats_for_fixture(fixture_id: int) -> dict[str, float | None]:
    """
    Fetch match statistics from API-Football for a finished fixture.
    Returns totals: corners, shots_on_target, cards (yellow+red).
    """
    from apps.worker.ingest.api_football import ApiFootballClient
    client = ApiFootballClient()
    try:
        raw = await client.get_fixture_statistics(fixture_id)
    except Exception as exc:
        logger.warning("fixture_statistics %s: %s", fixture_id, exc)
        return {}

    def _val(stats: list[dict], stat_type: str) -> float:
        for s in stats:
            if s.get("type") == stat_type:
                v = s.get("value")
                try:
                    return float(str(v).rstrip("%"))
                except (TypeError, ValueError):
                    pass
        return 0.0

    totals: dict[str, float] = {"corners": 0.0, "sot": 0.0, "cards": 0.0}
    for block in raw:
        stats = block.get("statistics", [])
        totals["corners"] += _val(stats, "Corner Kicks")
        totals["sot"]     += _val(stats, "Shots on Goal")
        totals["cards"]   += _val(stats, "Yellow Cards") + _val(stats, "Red Cards")
    return totals


def _resolve_stats_outcome(selection: str, actual_value: float) -> bool | None:
    """
    Resolve over/under outcome for a stats market pick.
    Returns True (won), False (lost), None (unknown).
    """
    import re
    sel_l = selection.lower()
    m = re.search(r"(\d+\.?\d*)", sel_l)
    if not m:
        return None
    line = float(m.group(1))
    if "over" in sel_l:
        return actual_value > line
    if "under" in sel_l:
        return actual_value < line
    return None


async def label_pending_picks(db: Client) -> dict:
    """
    Main job: fetch pending picks, resolve outcomes, update ml.picks_log.
    Returns summary: {labeled_goals, labeled_stats, pending_stats_no_fixture}.
    """
    # Only look at picks older than today (matches should be finished)
    yesterday = str(date.today() - timedelta(days=1))
    try:
        pending_res = (
            db.schema("ml").table("picks_log")
            .select("id, match_key, team_home, team_away, fecha, market_type, selection, model_prob")
            .is_("outcome", "null")
            .lte("fecha", yesterday)
            .limit(500)
            .execute()
        )
    except Exception as exc:
        logger.warning("label_picks fetch: %s", exc)
        return {"error": str(exc)}

    pending = pending_res.data or []
    if not pending:
        return {"labeled_goals": 0, "labeled_stats": 0, "pending_stats_no_fixture": 0}

    # Split by market type
    goals_rows = [r for r in pending if r["market_type"] in _GOALS_MARKETS]
    stats_rows = [r for r in pending if r["market_type"] in _STATS_MARKETS]

    labeled_goals = 0
    labeled_stats = 0
    pending_stats_no_fixture = 0
    now_str = date.today().isoformat()

    # ── Resolve goals-based markets ───────────────────────────────────────────
    if goals_rows:
        result_index = await _fetch_wc_results()
        from apps.worker.ml.evaluation import evaluate_prediction

        for row in goals_rows:
            key = (row["team_home"].lower(), row["team_away"].lower())
            res = result_index.get(key)
            if not res:
                continue
            eval_r = evaluate_prediction(
                market_type=row["market_type"],
                predicted_outcome=row["selection"],
                probability=float(row["model_prob"]),
                home_goals=res["home_goals"],
                away_goals=res["away_goals"],
                team_home=row["team_home"],
                team_away=row["team_away"],
            )
            if eval_r["actual_outcome"] == "unknown":
                continue
            try:
                db.schema("ml").table("picks_log").update({
                    "outcome":    eval_r["is_correct"],
                    "labeled_at": now_str,
                }).eq("id", row["id"]).execute()
                labeled_goals += 1
            except Exception as exc:
                logger.warning("label goals row %s: %s", row["id"], exc)

    # ── Resolve stats markets via API-Football ────────────────────────────────
    if stats_rows:
        from apps.api.services.team_wc_stats_service import find_fixture_id

        # Group by match to avoid redundant API calls
        match_groups: dict[str, list[dict]] = {}
        for row in stats_rows:
            mk = row["match_key"]
            match_groups.setdefault(mk, []).append(row)

        for mk, rows in match_groups.items():
            team_home = rows[0]["team_home"]
            team_away = rows[0]["team_away"]

            fixture_id = await find_fixture_id(team_home, team_away)
            if not fixture_id:
                logger.info("No fixture_id for %s vs %s", team_home, team_away)
                pending_stats_no_fixture += len(rows)
                continue

            stats = await _fetch_stats_for_fixture(fixture_id)
            if not stats:
                pending_stats_no_fixture += len(rows)
                continue

            stat_map = {
                "CORNERS": stats.get("corners", 0.0),
                "SOT":     stats.get("sot", 0.0),
                "CARDS":   stats.get("cards", 0.0),
            }

            for row in rows:
                actual_value = stat_map.get(row["market_type"])
                if actual_value is None:
                    continue
                outcome = _resolve_stats_outcome(row["selection"], actual_value)
                if outcome is None:
                    continue
                try:
                    db.schema("ml").table("picks_log").update({
                        "outcome":      outcome,
                        "actual_value": actual_value,
                        "labeled_at":   now_str,
                    }).eq("id", row["id"]).execute()
                    labeled_stats += 1
                except Exception as exc:
                    logger.warning("label stats row %s: %s", row["id"], exc)

    logger.info(
        "label_picks: goals=%d stats=%d no_fixture=%d",
        labeled_goals, labeled_stats, pending_stats_no_fixture,
    )
    return {
        "labeled_goals":          labeled_goals,
        "labeled_stats":          labeled_stats,
        "pending_stats_no_fixture": pending_stats_no_fixture,
    }
