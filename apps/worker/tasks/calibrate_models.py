"""Weekly isotonic calibration fit from WC historical data."""

import logging
from datetime import datetime, timezone

from apps.api.services.worldcup_engine import set_calibration_factors
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.calibration import expected_calibration_error, fit_calibration_bundle, reliability_bins
from apps.worker.ml.wc_historical import actual_outcomes, extract_finished_matches, predict_match_historical

logger = logging.getLogger(__name__)

COMPETITION = "fifa_world_cup"
MIN_SAMPLES = 20


async def calibrate_models(ctx: dict) -> dict:
    """
    Fit isotonic regression on WC 2018+2022, derive scalar factors,
    persist to ml.calibration_factors and snapshots. Rollback if ECE worsens.
    """
    db = get_supabase()
    job_id = None
    try:
        ins = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "calibrate_models", "status": "running"})
            .execute()
        )
        job_id = ins.data[0]["id"] if ins.data else None
    except Exception as exc:
        logger.warning("job_run: %s", exc)

    archives = await fetch_all_worldcup_archives()
    factors, _calibrators, metrics = fit_calibration_bundle(archives, train_years=[2018, 2022])

    # Re-fit including WC 2026 when enough finished results exist
    from apps.worker.tasks.update_elo import extract_finished_wc_matches

    finished_2026 = extract_finished_wc_matches(archives.get(2026, {}))
    if len(finished_2026) >= 10:
        factors, _calibrators, metrics = fit_calibration_bundle(
            archives, train_years=[2018, 2022, 2026]
        )
        logger.info("calibrate_models: included WC 2026 results n=%s", len(finished_2026))

    if metrics["sample_size"] < MIN_SAMPLES:
        result = {"status": "skipped", "reason": f"samples {metrics['sample_size']} < {MIN_SAMPLES}"}
        _finish_job(db, job_id, result)
        return result

    # Compare average ECE before vs after on full train set
    matches = extract_finished_matches(archives, years=[2018, 2022])
    p_over, y_over = [], []
    p_over_cal, y_over_cal = [], []
    for m in matches:
        probs = predict_match_historical(m, archives)
        actual = actual_outcomes(m)
        p_over.append(probs["over_25"])
        y_over.append(actual["over_25"])
        from apps.worker.ml.calibration import calibrate_model_markets

        cal = calibrate_model_markets(
            probs["home_win"],
            probs["draw"],
            probs["away_win"],
            probs["over_25"],
            probs["under_25"],
            probs["btts_yes"],
            probs["btts_no"],
            factors=factors,
        )
        p_over_cal.append(cal["over_25"])
        y_over_cal.append(actual["over_25"])

    ece_before = expected_calibration_error(p_over, y_over)
    ece_after = expected_calibration_error(p_over_cal, y_over_cal)

    prev_ece = None
    try:
        prev = (
            db.schema("ml")
            .table("calibration_snapshots")
            .select("ece")
            .eq("competition", COMPETITION)
            .eq("market", "over_under_2.5")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if prev.data:
            prev_ece = float(prev.data[0]["ece"])
    except Exception:
        pass

    rollback = prev_ece is not None and ece_after > prev_ece * 1.10
    if rollback:
        result = {
            "status": "rollback",
            "ece_before": ece_before,
            "ece_after": ece_after,
            "prev_ece": prev_ece,
        }
        _finish_job(db, job_id, result)
        return result

    # Deactivate previous factors
    try:
        db.schema("ml").table("calibration_factors").update({"is_active": False}).eq(
            "competition", COMPETITION
        ).eq("is_active", True).execute()
    except Exception as exc:
        logger.warning("deactivate factors: %s", exc)

  # Insert new factors
    inserted = 0
    market_outcomes = {
        "1X2": factors.get("1X2", {}),
        "over_under_2.5": factors.get("over_under_2.5", {}),
        "btts": factors.get("btts", {}),
    }
    for market, outcomes in market_outcomes.items():
        for outcome, factor in outcomes.items():
            try:
                db.schema("ml").table("calibration_factors").insert(
                    {
                        "competition": COMPETITION,
                        "market": market,
                        "outcome": outcome,
                        "factor": factor,
                        "method": "isotonic",
                        "sample_size": metrics["sample_size"],
                        "ece": ece_after,
                        "is_active": True,
                    }
                ).execute()
                inserted += 1
            except Exception as exc:
                logger.warning("insert factor %s/%s: %s", market, outcome, exc)

    bins = reliability_bins(p_over_cal, y_over_cal, n_bins=5)
    try:
        db.schema("ml").table("calibration_snapshots").insert(
            {
                "competition": COMPETITION,
                "market": "over_under_2.5",
                "window_days": 0,
                "ece": ece_after,
                "brier": None,
                "hit_rate": None,
                "sample_size": metrics["sample_size"],
                "reliability": bins,
            }
        ).execute()
    except Exception as exc:
        logger.warning("calibration_snapshots: %s", exc)

    set_calibration_factors(factors)

    result = {
        "status": "calibrated",
        "factors_inserted": inserted,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "metrics": metrics,
    }
    _finish_job(db, job_id, result)
    logger.info("calibrate_models: %s", result)
    return result


def _finish_job(db, job_id: str | None, result: dict) -> None:
    if not job_id:
        return
    db.schema("ops").table("job_runs").update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metadata": result,
        }
    ).eq("id", job_id).execute()
