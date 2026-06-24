"""Self-evaluation tasks — compare predictions vs actual results."""

import logging
from datetime import datetime, timezone

from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase
from apps.worker.ml.calibration import expected_calibration_error
from apps.worker.ml.evaluation import evaluate_prediction
from apps.worker.ml.wc_predictions import evaluate_wc_predictions

logger = logging.getLogger(__name__)


async def evaluate_pending(ctx: dict) -> dict:
    """
    Find finished matches with predictions but no evaluation,
    compare outcomes, write prediction_evaluations and update metrics.
    """
    db = get_supabase()
    settings = get_settings()
    batch_size = settings.evaluation_batch_size

    job_id = None
    try:
        job_insert = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "evaluate_pending", "status": "running"})
            .execute()
        )
        job_id = job_insert.data[0]["id"] if job_insert.data else None
    except Exception as exc:
        logger.warning("Could not create job_run: %s", exc)

    finished_matches = (
        db.table("matches")
        .select("id")
        .eq("status", "finished")
        .limit(batch_size * 2)
        .execute()
    )

    evaluated_count = 0
    correct_count = 0
    errors: list[str] = []

    for match_row in finished_matches.data or []:
        match_id = match_row["id"]

        result = (
            db.table("match_results")
            .select("home_goals, away_goals")
            .eq("match_id", match_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            continue

        home_goals = int(result.data[0]["home_goals"])
        away_goals = int(result.data[0]["away_goals"])

        predictions = (
            db.schema("ml")
            .table("predictions")
            .select("id, match_id, market_type, predicted_outcome, probability, model_version_id, metadata")
            .eq("match_id", match_id)
            .execute()
        )

        for pred in predictions.data or []:
            existing_eval = (
                db.schema("ml")
                .table("prediction_evaluations")
                .select("id")
                .eq("prediction_id", pred["id"])
                .limit(1)
                .execute()
            )
            if existing_eval.data:
                continue

            try:
                eval_result = evaluate_prediction(
                    market_type=pred["market_type"],
                    predicted_outcome=pred["predicted_outcome"],
                    probability=float(pred["probability"]),
                    home_goals=home_goals,
                    away_goals=away_goals,
                )

                db.schema("ml").table("prediction_evaluations").insert(
                    {
                        "prediction_id": pred["id"],
                        "actual_outcome": eval_result["actual_outcome"],
                        "is_correct": eval_result["is_correct"],
                        "brier_score": eval_result["brier_score"],
                        "log_loss": eval_result["log_loss"],
                        "notes": eval_result.get("notes"),
                    }
                ).execute()

                evaluated_count += 1
                if eval_result["is_correct"]:
                    correct_count += 1

                await _update_model_weights(db, pred, eval_result)

            except Exception as exc:
                errors.append(f"prediction {pred['id']}: {exc}")

        if evaluated_count >= batch_size:
            break

    try:
        wc_result = await evaluate_wc_predictions(db)
        result_summary_wc = wc_result
    except Exception as exc:
        logger.warning("evaluate_wc_predictions: %s", exc)
        result_summary_wc = {"evaluated": 0, "error": str(exc)}

    hit_rate = correct_count / evaluated_count if evaluated_count > 0 else 0.0
    result_summary = {
        "evaluated": evaluated_count,
        "correct": correct_count,
        "hit_rate": round(hit_rate, 4),
        "wc_predictions": result_summary_wc,
        "errors": errors,
    }

    if job_id:
        db.schema("ops").table("job_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": result_summary,
            }
        ).eq("id", job_id).execute()

    total_evals = (
        db.schema("ml")
        .table("prediction_evaluations")
        .select("id", count="exact")
        .execute()
    )
    total_count = total_evals.count or 0

    if total_count >= settings.retrain_threshold:
        from arq import create_pool
        from arq.connections import RedisSettings

        try:
            pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            await pool.enqueue_job("train_models")
            await pool.close()
            result_summary["retrain_enqueued"] = True
        except Exception as exc:
            logger.warning("Could not enqueue retrain: %s", exc)

    # Rolling ECE snapshot for 1X2 binary (predicted outcome prob vs hit)
    try:
        recent = (
            db.schema("ml")
            .table("prediction_evaluations")
            .select("brier_score, is_correct, predictions(probability, market_type)")
            .order("evaluated_at", desc=True)
            .limit(200)
            .execute()
        )
        probs, hits = [], []
        for row in recent.data or []:
            pred = row.get("predictions") or {}
            if pred.get("market_type") != "1X2":
                continue
            probs.append(float(pred.get("probability", 0.33)))
            hits.append(1 if row.get("is_correct") else 0)
        if len(probs) >= 20:
            ece = expected_calibration_error(probs, hits)
            db.schema("ml").table("calibration_snapshots").insert(
                {
                    "competition": "production",
                    "market": "1X2",
                    "window_days": 30,
                    "ece": ece,
                    "sample_size": len(probs),
                    "reliability": [],
                }
            ).execute()
            result_summary["rolling_ece_1x2"] = ece
    except Exception as exc:
        logger.debug("rolling ece: %s", exc)

    logger.info("evaluate_pending completed: %s", result_summary)
    return result_summary


async def _update_model_weights(db, prediction: dict, eval_result: dict) -> None:
    """Adjust model weights based on evaluation feedback."""
    match = (
        db.table("matches")
        .select("season_id, seasons(league_id)")
        .eq("id", prediction.get("match_id", ""))
        .limit(1)
        .execute()
    )
    if not match.data:
        return

    league_id = match.data[0].get("seasons", {}).get("league_id")
    if not league_id:
        return

    market_type = prediction["market_type"]
    source = (prediction.get("metadata") or {}).get("source", "ensemble")

    model_type_map = {
        "elo": "elo",
        "poisson": "poisson",
        "gk": "gk",
        "ensemble": "ensemble",
        "xgboost": "xgboost",
    }
    model_type = model_type_map.get(source, "ensemble")

    existing = (
        db.schema("ml")
        .table("model_weights")
        .select("id, weight")
        .eq("league_id", league_id)
        .eq("market_type", market_type)
        .eq("model_type", model_type)
        .limit(1)
        .execute()
    )

    delta = 0.01 if eval_result["is_correct"] else -0.01
    if existing.data:
        new_weight = max(0.05, min(0.60, float(existing.data[0]["weight"]) + delta))
        db.schema("ml").table("model_weights").update(
            {"weight": new_weight, "updated_at": "now()"}
        ).eq("id", existing.data[0]["id"]).execute()
    else:
        base_weight = 0.25 + delta
        db.schema("ml").table("model_weights").insert(
            {
                "league_id": league_id,
                "market_type": market_type,
                "model_type": model_type,
                "weight": max(0.05, min(0.60, base_weight)),
            }
        ).execute()
