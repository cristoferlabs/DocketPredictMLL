"""Pre-kickoff closing line capture — cierra loop CLV."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from apps.api.services.odds_context import compute_market_context, find_wc_odds_event
from apps.api.services.worldcup_engine import find_upcoming_matches, name_match
from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.clv import record_closing_snapshot

logger = logging.getLogger(__name__)


async def capture_closing_lines(ctx: dict) -> dict:
    """
    Job pre-kickoff: captura líneas de cierre para picks con predicción activa.

    predicción → opening → pick → **closing** → resultado (evaluate_wc_predictions)
    """
    db = get_supabase()
    settings = get_settings()
    horizon_hours = 36
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=horizon_hours)

    try:
        archives = await fetch_all_worldcup_archives()
        d26 = archives.get(2026, {})
    except Exception as exc:
        logger.warning("capture_closing_lines archives: %s", exc)
        d26 = {}

    upcoming = find_upcoming_matches(d26, days_ahead=14)
    match_dates: dict[tuple[str, str], str] = {}
    for m in upcoming:
        md = (m.get("date") or "")[:10]
        if not md:
            continue
        try:
            kickoff = datetime.fromisoformat(md).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if kickoff > cutoff:
            continue
        t1 = m.get("team1", {}).get("name", "")
        t2 = m.get("team2", {}).get("name", "")
        if t1 and t2:
            match_dates[(t1.lower(), t2.lower())] = md
            match_dates[(t2.lower(), t1.lower())] = md

    pending_preds = (
        db.schema("ml")
        .table("wc_predictions")
        .select("id, team_home, team_away, market_type, predicted_outcome, metadata")
        .is_("evaluated_at", "null")
        .limit(200)
        .execute()
    )

    captured = 0
    skipped = 0
    errors: list[str] = []

    for pred in pending_preds.data or []:
        th = pred["team_home"]
        ta = pred["team_away"]
        key = (th.lower(), ta.lower())
        if key not in match_dates:
            skipped += 1
            continue

        meta = pred.get("metadata") or {}
        clv = meta.get("clv") or {}
        if clv.get("clv_stage") in ("closed", "complete"):
            skipped += 1
            continue

        try:
            odds_event = await find_wc_odds_event(th, ta, db=db)
            if not odds_event:
                skipped += 1
                continue

            market_ctx = compute_market_context(
                _dummy_model_from_pred(pred),
                th,
                ta,
                odds_event,
            )
            if not market_ctx.has_market:
                skipped += 1
                continue

            selection = pred["predicted_outcome"]
            outcome = next(
                (o for o in market_ctx.outcomes if name_match(o.selection, selection)),
                None,
            )
            if not outcome or not outcome.market_odds or outcome.market_odds <= 1:
                skipped += 1
                continue

            match_key = f"{th}|{ta}"
            clv_val = record_closing_snapshot(
                db,
                match_key=match_key,
                team_home=th,
                team_away=ta,
                market=pred.get("market_type", "1X2"),
                selection=outcome.selection,
                closing_odds=outcome.market_odds,
                prediction_id=pred["id"],
            )
            captured += 1
            logger.info(
                "CLV closing %s vs %s %s @ %.2f clv=%s",
                th,
                ta,
                outcome.selection,
                outcome.market_odds,
                clv_val,
            )
        except Exception as exc:
            errors.append(f"{th} vs {ta}: {exc}")

    summary = {
        "captured": captured,
        "skipped": skipped,
        "errors": errors[:10],
        "horizon_hours": horizon_hours,
    }
    logger.info("capture_closing_lines: %s", summary)
    return summary


def _dummy_model_from_pred(pred: dict):
    """Modelo mínimo para compute_market_context (solo necesita probs > 0)."""
    from apps.api.services.worldcup_engine import ModelMarkets

    p = float(pred.get("probability") or 0.33)
    return ModelMarkets(
        home_win=p if pred.get("predicted_outcome") == pred.get("team_home") else 0.2,
        draw=0.25,
        away_win=p if pred.get("predicted_outcome") == pred.get("team_away") else 0.2,
        over_25=0.5,
        under_25=0.5,
        btts_yes=0.5,
        btts_no=0.5,
        lambda_home=1.2,
        lambda_away=1.2,
        confidence="medium",
    )
