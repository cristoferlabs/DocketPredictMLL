"""Verify Supabase connection and schema exposure."""

from apps.shared.config import get_settings
from apps.shared.supabase_client import get_supabase


def main() -> int:
    settings = get_settings()
    ref = settings.supabase_url.split("//")[1].split(".")[0] if settings.supabase_url else "?"
    print("=== Diagnostico Supabase ===")
    print(f"Proyecto URL: {settings.supabase_url}")
    print(f"Ref: {ref}")
    print()

    db = get_supabase()
    tests = [
        ("public", "leagues"),
        ("ops", "job_runs"),
        ("ops", "data_sources"),
        ("ml", "model_versions"),
    ]
    failed = 0
    for schema, table in tests:
        try:
            db.schema(schema).table(table).select("id").limit(1).execute()
            print(f"  OK   {schema}.{table}")
        except Exception as exc:
            failed += 1
            err = str(exc)
            if "PGRST106" in err:
                print(f"  FAIL {schema}.{table} - schema NO expuesto en API")
            elif "PGRST205" in err or "does not exist" in err.lower():
                print(f"  FAIL {schema}.{table} - tabla no existe (ejecuta migraciones SQL)")
            else:
                print(f"  FAIL {schema}.{table} - {err[:150]}")

    print()
    if failed:
        print("Si ops/ml fallan con 'NO expuesto':")
        print("  Dashboard -> Integrations -> Data API -> pestana Settings")
        print("  Exposed schemas: public, graphql_public, ml, ops -> Save")
        return 1

    print("Todo OK. Puedes usar /jobs/ingest-fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
