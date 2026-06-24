"""Walk-forward backtest job."""

import logging
from datetime import datetime, timezone

from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.backtest import run_holdout_backtest, run_walk_forward_backtest

logger = logging.getLogger(__name__)


async def run_backtest(ctx: dict) -> dict:
    """Run WC backtest and persist metrics to ml.model_performance_metrics."""
    db = get_supabase()
    job_id = None
    try:
        ins = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "run_backtest", "status": "running"})
            .execute()
        )
        job_id = ins.data[0]["id"] if ins.data else None
    except Exception as exc:
        logger.warning("job_run: %s", exc)

    archives = await fetch_all_worldcup_archives()

    walk = run_walk_forward_backtest(archives, train_size=40, test_size=20, years=[2018, 2022], db=db)
    holdout = run_holdout_backtest(archives, train_years=[2018], test_years=[2022], db=db)

    combined = {
        "walk_forward": walk.to_dict(),
        "holdout_2018_train_2022_test": holdout.to_dict(),
    }

    model_version = (
        db.schema("ml")
        .table("model_versions")
        .select("id")
        .eq("model_type", "ensemble")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    model_version_id = model_version.data[0]["id"] if model_version.data else None

    if model_version_id and holdout.sample_size >= 30:
        avg_ece = round((holdout.ece_over + holdout.ece_under + holdout.ece_btts) / 3, 6)
        db.schema("ml").table("model_performance_metrics").insert(
            {
                "model_version_id": model_version_id,
                "market_type": "wc_backtest",
                "window_days": 90,
                "hit_rate": holdout.hit_rate_1x2,
                "roi_sim": holdout.roi_sim,
                "calibration_error": avg_ece,
                "sample_size": holdout.sample_size,
            }
        ).execute()

    result = {
        "status": "completed",
        **combined,
        "max_drawdown": holdout.details.get("max_drawdown"),
        "roi_details": holdout.details.get("roi_details"),
    }

    if job_id:
        db.schema("ops").table("job_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": result,
            }
        ).eq("id", job_id).execute()

    logger.info("run_backtest: %s", result)
    return result
