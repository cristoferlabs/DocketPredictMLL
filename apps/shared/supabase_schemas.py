"""Supabase schema helpers."""

from supabase import Client


class SupabaseSchemaError(RuntimeError):
    """Raised when ml/ops schemas are not exposed via PostgREST."""


def ensure_schemas_exposed(db: Client) -> None:
    """
    Verify PostgREST can reach ml and ops schemas.
    Requires Supabase Dashboard → Project Settings → API → Exposed schemas: ml, ops
    """
    for schema in ("ops", "ml"):
        try:
            db.schema(schema).table(
                "job_runs" if schema == "ops" else "model_versions"
            ).select("id").limit(1).execute()
        except Exception as exc:
            msg = str(exc)
            if "PGRST106" in msg or "Invalid schema" in msg:
                raise SupabaseSchemaError(
                    "Los schemas 'ml' y 'ops' no están expuestos en la API de Supabase. "
                    "Ve a: Project Settings → API → Exposed schemas → añade: ml, ops "
                    "(y guarda). Luego ejecuta la migración SQL si aún no lo hiciste."
                ) from exc
            raise


def safe_insert_job_run(db: Client, job_type: str, metadata: dict | None = None) -> None:
    """Insert job run log; raises SupabaseSchemaError if schemas not configured."""
    ensure_schemas_exposed(db)
    row: dict = {"job_type": job_type, "status": "pending"}
    if metadata:
        row["metadata"] = metadata
    db.schema("ops").table("job_runs").insert(row).execute()
