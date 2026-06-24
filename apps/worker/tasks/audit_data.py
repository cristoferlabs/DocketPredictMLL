"""Scheduled WC data quality audit."""

import logging
from datetime import datetime, timezone

from apps.shared.supabase_client import get_supabase
from apps.worker.ml.wc_audit import audit_upcoming_matches, persist_audit_report

logger = logging.getLogger(__name__)


async def audit_wc_data(ctx: dict) -> dict:
    db = get_supabase()
    job_id = None
    try:
        ins = (
            db.schema("ops")
            .table("job_runs")
            .insert({"job_type": "audit_wc_data", "status": "running"})
            .execute()
        )
        job_id = ins.data[0]["id"] if ins.data else None
    except Exception as exc:
        logger.warning("job_run: %s", exc)

    report = await audit_upcoming_matches(db=db)
    try:
        persist_audit_report(db, report)
    except Exception as exc:
        logger.warning("persist_audit_report: %s", exc)

    result = {"status": "completed", **report.to_dict()}

    if job_id:
        db.schema("ops").table("job_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": result,
            }
        ).eq("id", job_id).execute()

    logger.info("audit_wc_data: %s", result)
    return result
