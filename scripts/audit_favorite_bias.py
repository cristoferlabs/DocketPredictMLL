"""Ejecuta auditoría de sesgo favorito en partidos WC próximos."""

from __future__ import annotations

import asyncio
import sys

from apps.api.services.odds_context import compute_market_context, find_wc_odds_event
from apps.api.services.worldcup_engine import analyze_match, find_upcoming_matches
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.favorite_bias_audit import (
    _match_row,
    aggregate_favorite_bias,
    format_audit_report,
)
from apps.worker.tasks.update_elo import get_wc_elo_ratings
from apps.shared.supabase_client import get_supabase


async def run_audit(days_ahead: int = 14, limit: int = 20) -> str:
    archives = await fetch_all_worldcup_archives()
    d26 = archives.get(2026, {})
    d22 = archives.get(2022, {})
    d18 = archives.get(2018, {})
    db = get_supabase()
    elo = await get_wc_elo_ratings(db)

    upcoming = find_upcoming_matches(d26, days_ahead=days_ahead)[:limit]
    rows = []

    for match in upcoming:
        analysis = analyze_match(match, d18, d22, [], elo)
        if not analysis.model:
            continue
        m = analysis.model
        t1, t2 = analysis.team1, analysis.team2
        odds = await find_wc_odds_event(t1, t2, db=db)
        ctx = compute_market_context(m, t1, t2, odds)
        if not ctx.has_market:
            continue
        outcomes: list[tuple[str, float, float | None]] = []
        for o in ctx.outcomes:
            outcomes.append((o.selection, o.model_prob, o.market_implied))
        row = _match_row(t1, t2, outcomes)
        if row:
            rows.append(row)

    audit = aggregate_favorite_bias(rows)
    return format_audit_report(audit)


def main() -> None:
    report = asyncio.run(run_audit())
    sys.stdout.buffer.write(report.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
