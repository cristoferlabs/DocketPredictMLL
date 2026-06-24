"""Model training tasks."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from apps.shared.supabase_client import get_supabase
from apps.worker.ml.calibration import expected_calibration_error
from apps.worker.ml.model_loader import ARTIFACTS_DIR
from apps.worker.ml.xgboost_model import OUTCOME_TO_LABEL, XGBoostModel

logger = logging.getLogger(__name__)


async def train_models(ctx: dict) -> dict:
    """Retrain XGBoost from evaluated predictions (walk-forward style)."""
    db = get_supabase()

    job_id = None
    try:
        job_insert = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "train_models", "status": "running"})
            .execute()
        )
        job_id = job_insert.data[0]["id"] if job_insert.data else None
    except Exception as exc:
        logger.warning("Could not create job_run: %s", exc)

    evals = (
        db.schema("ml")
        .table("prediction_evaluations")
        .select("prediction_id, actual_outcome, predictions(market_type, probability, metadata, match_id)")
        .order("evaluated_at")
        .limit(5000)
        .execute()
    )

    wc_evals = (
        db.schema("ml")
        .table("wc_predictions")
        .select("market_type, predicted_outcome, probability, actual_outcome, is_correct, metadata")
        .not_.is_("evaluated_at", "null")
        .limit(2000)
        .execute()
    )

    X_rows: list[list[float]] = []
    y_labels: list[int] = []
    holdout_probs: list[float] = []
    holdout_correct: list[int] = []

    xgb_model = XGBoostModel()

    for ev in evals.data or []:
        pred = ev.get("predictions")
        if not pred or pred.get("market_type") != "1X2":
            continue

        meta = pred.get("metadata") or {}
        elo_probs = meta.get("elo_probs", {"home_win": 0.33, "draw": 0.33, "away_win": 0.33})
        poisson_probs = meta.get("poisson_probs", {"home_win": 0.33, "draw": 0.33, "away_win": 0.33})
        gk_adj = meta.get("gk_adjustment", {"home_factor": 1.0, "away_factor": 1.0})

        features = xgb_model.build_feature_vector(elo_probs, poisson_probs, gk_adj)
        X_rows.append(features.tolist())

        actual = ev.get("actual_outcome", "")
        label = OUTCOME_TO_LABEL.get(actual, 1)
        y_labels.append(label)

        if len(X_rows) % 5 == 0:
            holdout_probs.append(float(pred.get("probability", 0.33)))
            holdout_correct.append(1 if pred.get("predicted_outcome") == actual else 0)

    for wc in wc_evals.data or []:
        if wc.get("market_type") != "1X2":
            continue
        meta = wc.get("metadata") or {}
        elo_probs = meta.get("elo_probs", {"home_win": 0.33, "draw": 0.33, "away_win": 0.33})
        poisson_probs = meta.get("poisson_probs", {"home_win": 0.33, "draw": 0.33, "away_win": 0.33})
        gk_adj = meta.get("gk_adjustment", {"home_factor": 1.0, "away_factor": 1.0})
        extra = meta.get("wc_features", {})
        features = xgb_model.build_feature_vector(elo_probs, poisson_probs, gk_adj, extra_features=extra)
        X_rows.append(features.tolist())
        actual = wc.get("actual_outcome", "")
        label = OUTCOME_TO_LABEL.get(actual, 1)
        y_labels.append(label)

    if len(X_rows) < 50:
        result = {"status": "skipped", "reason": f"insufficient samples ({len(X_rows)} < 50)"}
        if job_id:
            db.schema("ops").table("job_runs").update(
                {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(), "metadata": result}
            ).eq("id", job_id).execute()
        return result

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    metrics = xgb_model.train(X, y)
    version = datetime.now(timezone.utc).strftime("1.%Y%m%d.%H%M")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / f"xgb_{version}.json"
    xgb_model.save(artifact_path)

    ece = expected_calibration_error(holdout_probs, holdout_correct) if holdout_probs else 0.0
    metrics["holdout_ece"] = ece

    db.schema("ml").table("model_versions").update({"is_active": False}).eq(
        "model_type", "xgboost"
    ).execute()

    new_version = (
        db.schema("ml")
        .table("model_versions")
        .insert(
            {
                "model_type": "xgboost",
                "version": version,
                "artifact_path": str(artifact_path),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "is_active": True,
            }
        )
        .execute()
    )

    model_version_id = new_version.data[0]["id"] if new_version.data else None

    if model_version_id:
        hit_rate = metrics.get("train_accuracy", 0)
        db.schema("ml").table("model_performance_metrics").insert(
            {
                "model_version_id": model_version_id,
                "market_type": "1X2",
                "window_days": 90,
                "hit_rate": hit_rate,
                "sample_size": len(y_labels),
                "calibration_error": ece,
            }
        ).execute()

    result = {
        "status": "trained",
        "version": version,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "samples": len(y_labels),
    }

    if job_id:
        db.schema("ops").table("job_runs").update(
            {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(), "metadata": result}
        ).eq("id", job_id).execute()

    logger.info("train_models completed: %s", result)
    return result
