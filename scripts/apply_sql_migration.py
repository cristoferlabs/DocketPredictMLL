#!/usr/bin/env python3
"""Apply SQL migration files via direct Postgres connection (no Supabase CLI)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

DEFAULT_MIGRATION = ROOT / "supabase/migrations/20250626100000_wc_elo_predictions.sql"


def build_database_url() -> str | None:
    direct = os.getenv("DATABASE_URL", "").strip()
    if direct:
        return direct

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    password = os.getenv("SUPABASE_DB_PASSWORD", "").strip()
    match = re.search(r"https://([^.]+)\.supabase\.co", supabase_url)
    if match and password:
        ref = match.group(1)
        return f"postgresql://postgres:{password}@db.{ref}.supabase.co:5432/postgres"
    return None


def main() -> None:
    migration = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MIGRATION
    if not migration.is_file():
        print(f"No existe: {migration}")
        sys.exit(1)

    db_url = build_database_url()
    if not db_url:
        print("Falta conexion a Postgres.")
        print()
        print("Anade a .env una de estas opciones:")
        print("  DATABASE_URL=postgresql://postgres:TU_PASSWORD@db.TU_REF.supabase.co:5432/postgres")
        print("  SUPABASE_DB_PASSWORD=TU_PASSWORD   (con SUPABASE_URL ya configurado)")
        print()
        print("La password esta en: Supabase Dashboard → Project Settings → Database")
        sys.exit(1)

    try:
        import psycopg
    except ImportError:
        print("Instala el cliente Postgres:")
        print("  pip install psycopg[binary]")
        sys.exit(1)

    sql = migration.read_text(encoding="utf-8")
    print(f"Aplicando {migration.name} ...")

    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    print("Migracion aplicada correctamente.")


if __name__ == "__main__":
    main()
