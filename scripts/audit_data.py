#!/usr/bin/env python3
"""Audit data quality for upcoming WC matches."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.shared.supabase_client import get_supabase
from apps.worker.ml.wc_audit import audit_upcoming_matches, persist_audit_report


async def main() -> None:
    db = get_supabase()
    report = await audit_upcoming_matches(db=db)
    try:
        persist_audit_report(db, report)
    except Exception as exc:
        print(f"Warning: no se pudo guardar en data_quality_log: {exc}", file=sys.stderr)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **report.to_dict(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
