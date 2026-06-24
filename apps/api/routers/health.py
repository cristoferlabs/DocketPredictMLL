"""Health check endpoints."""

from fastapi import APIRouter, Depends

from apps.api.deps import get_db
from apps.shared.supabase_schemas import SupabaseSchemaError, ensure_schemas_exposed

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "agente-betting-engine"}


@router.get("/health/db")
async def health_db(db=Depends(get_db)):
    try:
        result = db.table("leagues").select("id").limit(1).execute()
        return {"status": "ok", "db": "connected", "sample_count": len(result.data or [])}
    except Exception as exc:
        return {"status": "degraded", "db": "error", "detail": str(exc)}


@router.get("/health/schemas")
async def health_schemas(db=Depends(get_db)):
    """Verify ml/ops schemas are exposed in Supabase API."""
    try:
        ensure_schemas_exposed(db)
        return {"status": "ok", "schemas": ["public", "ml", "ops"]}
    except SupabaseSchemaError as exc:
        return {"status": "error", "schemas": "not_exposed", "detail": str(exc)}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}
