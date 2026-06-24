"""ARQ worker entrypoint."""

import logging

from arq import cron
from arq.connections import RedisSettings

from apps.shared.config import get_settings
from apps.worker.tasks.audit_data import audit_wc_data
from apps.worker.tasks.backtest import run_backtest
from apps.worker.tasks.calibrate_models import calibrate_models
from apps.worker.tasks.clv_snapshots import capture_closing_lines
from apps.worker.tasks.evaluate import evaluate_pending
from apps.worker.tasks.ingest import ingest_fixtures
from apps.worker.tasks.predict import predict_upcoming_matches
from apps.worker.tasks.train import train_models
from apps.worker.tasks.update_elo import update_elo_after_finished_matches

logger = logging.getLogger(__name__)

_settings = get_settings()


async def startup(ctx: dict) -> None:
    settings = get_settings()
    logger.info("Worker started (env=%s)", settings.environment)
    try:
        from apps.worker.tasks.update_elo import update_elo_after_finished_matches

        result = await update_elo_after_finished_matches(ctx)
        logger.info("Startup update_elo: %s rows", result.get("rows_inserted"))
    except Exception as exc:
        logger.warning("Startup update_elo failed: %s", exc)
    try:
        from apps.worker.tasks.audit_data import audit_wc_data

        await audit_wc_data(ctx)
    except Exception as exc:
        logger.warning("Startup audit_wc_data failed: %s", exc)


async def shutdown(ctx: dict) -> None:
    logger.info("Worker shutting down")


class WorkerSettings:
    functions = [
        ingest_fixtures,
        predict_upcoming_matches,
        evaluate_pending,
        train_models,
        run_backtest,
        calibrate_models,
        update_elo_after_finished_matches,
        audit_wc_data,
        capture_closing_lines,
    ]

    cron_jobs = [
        cron(ingest_fixtures, hour=6, minute=0, run_at_startup=False),
        cron(evaluate_pending, hour={0, 6, 12, 18}, minute=30),
        cron(predict_upcoming_matches, hour=7, minute=0),
        cron(update_elo_after_finished_matches, hour={1, 13}, minute=15),
        cron(audit_wc_data, hour={8, 20}, minute=0),
        cron(run_backtest, weekday=0, hour=5, minute=0),
        cron(calibrate_models, weekday=0, hour=4, minute=0),
        cron(capture_closing_lines, hour={5, 11, 17, 23}, minute=45),
    ]

    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    job_timeout = 600
    max_jobs = _settings.arq_max_jobs
