"""Job trigger endpoints for n8n cron and manual runs."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from apps.api.deps import get_arq_pool, get_db
from apps.shared.supabase_schemas import SupabaseSchemaError, safe_insert_job_run

logger = logging.getLogger(__name__)
router = APIRouter()


class JobEnqueueResponse(BaseModel):
    job: str
    enqueued: bool
    job_id: str | None = None
    detail: str | None = None


class IngestFixturesRequest(BaseModel):
    league_external_id: int | None = Field(
        None, description="API-Football league ID (140=La Liga). Solo plan free hasta 2024"
    )
    season: int | None = Field(
        None, description="Temporada API-Football (free: 2022-2024). Omitir = usa Football-Data"
    )
    days_ahead: int = Field(7, ge=1, le=30)
    competition_code: str | None = Field(
        None, description="Football-Data code: PD, PL, SA, BL1, FL1, WC, CL"
    )
    sources: list[str] | None = Field(
        None,
        description="Fuentes: football-data, api-football, odds-api, sportmonks. Omitir = todas",
    )


async def _enqueue(request: Request, job_name: str, *args: Any, **kwargs: Any) -> JobEnqueueResponse:
    pool = get_arq_pool(request)
    if not pool:
        raise HTTPException(status_code=503, detail="Worker queue unavailable (Redis not connected)")
    job = await pool.enqueue_job(job_name, *args, **kwargs)
    return JobEnqueueResponse(job=job_name, enqueued=True, job_id=job.job_id if job else None)


def _handle_supabase_error(exc: Exception) -> None:
    if isinstance(exc, SupabaseSchemaError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.exception("Supabase error in jobs endpoint")
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest-fixtures", response_model=JobEnqueueResponse)
async def ingest_fixtures(
    body: IngestFixturesRequest,
    request: Request,
    db=Depends(get_db),
):
    """Trigger fixture ingestion from API-Football (called by n8n daily cron)."""
    try:
        safe_insert_job_run(db, "ingest_fixtures", body.model_dump())
        return await _enqueue(
            request,
            "ingest_fixtures",
            league_external_id=body.league_external_id,
            season=body.season,
            days_ahead=body.days_ahead,
            competition_code=body.competition_code,
            sources=body.sources,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)


@router.post("/evaluate-pending", response_model=JobEnqueueResponse)
async def evaluate_pending(request: Request, db=Depends(get_db)):
    """Trigger self-evaluation of predictions against finished match results."""
    try:
        safe_insert_job_run(db, "evaluate_pending")
        return await _enqueue(request, "evaluate_pending")
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)


@router.post("/predict-upcoming", response_model=JobEnqueueResponse)
async def predict_upcoming(request: Request):
    """Generate predictions for upcoming matches."""
    return await _enqueue(request, "predict_upcoming_matches")


@router.post("/train", response_model=JobEnqueueResponse)
async def train_models_endpoint(request: Request):
    """Retrain XGBoost ensemble when enough evaluations exist."""
    return await _enqueue(request, "train_models")


@router.post("/run-backtest", response_model=JobEnqueueResponse)
async def run_backtest_endpoint(request: Request, db=Depends(get_db)):
    """Walk-forward backtest on WC historical data."""
    try:
        safe_insert_job_run(db, "run_backtest")
        return await _enqueue(request, "run_backtest")
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)


@router.post("/calibrate-models", response_model=JobEnqueueResponse)
async def calibrate_models_endpoint(request: Request, db=Depends(get_db)):
    """Fit isotonic calibration from WC 2018/2022 and update factors."""
    try:
        safe_insert_job_run(db, "calibrate_models")
        return await _enqueue(request, "calibrate_models")
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)


@router.post("/update-elo", response_model=JobEnqueueResponse)
async def update_elo_endpoint(request: Request, db=Depends(get_db)):
    """Recompute and persist World Cup ELO ratings after finished matches."""
    try:
        safe_insert_job_run(db, "update_elo")
        return await _enqueue(request, "update_elo_after_finished_matches")
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)


@router.post("/audit-wc-data", response_model=JobEnqueueResponse)
async def audit_wc_data_endpoint(request: Request, db=Depends(get_db)):
    """Audit data quality for upcoming WC matches."""
    try:
        safe_insert_job_run(db, "audit_wc_data")
        return await _enqueue(request, "audit_wc_data")
    except HTTPException:
        raise
    except Exception as exc:
        _handle_supabase_error(exc)
