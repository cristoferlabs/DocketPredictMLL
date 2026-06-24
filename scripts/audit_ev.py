#!/usr/bin/env python3
"""Audit EV: compare raw vs fair expected value for today's WC matches."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.services.odds_context import compute_ev_opportunities, find_wc_odds_event
from apps.api.services.worldcup_engine import (
    analyze_match,
    calc_elo_ratings,
    find_upcoming_matches,
    set_calibration_factors,
)
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.model_loader import load_calibration_factors_from_db
from apps.worker.tasks.update_elo import get_wc_elo_ratings


async def main() -> None:
    from apps.shared.supabase_client import get_supabase

    archives = await fetch_all_worldcup_archives()
    d26 = archives.get(2026, {})
    d22 = archives.get(2022, {})
    d18 = archives.get(2018, {})

    db = get_supabase()
    factors = load_calibration_factors_from_db(db)
    if factors:
        set_calibration_factors(factors)

    try:
        elo = await get_wc_elo_ratings(db)
    except Exception:
        elo = calc_elo_ratings(d18, d22, d26)

    upcoming = find_upcoming_matches(d26, days_ahead=7)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calibration_active": bool(factors),
        "matches_audited": 0,
        "positive_ev_raw": 0,
        "positive_ev_fair": 0,
        "false_positives_removed": 0,
        "details": [],
    }

    for match in upcoming:
        analysis = analyze_match(match, d18, d22, [], elo)
        if not analysis.model:
            continue
        odds_event = await find_wc_odds_event(analysis.team1, analysis.team2)

        all_opps = compute_ev_opportunities(
            analysis.model, analysis.team1, analysis.team2, odds_event, single_best=False
        )
        best_fair = compute_ev_opportunities(
            analysis.model, analysis.team1, analysis.team2, odds_event, single_best=True
        )

        raw_positive = sum(1 for o in all_opps if o.expected_value_raw > 0)
        fair_positive = len(best_fair)

        report["matches_audited"] += 1
        report["positive_ev_raw"] += raw_positive
        report["positive_ev_fair"] += fair_positive
        report["false_positives_removed"] += max(0, raw_positive - fair_positive)

        detail: dict = {
            "match": f"{analysis.team1} vs {analysis.team2}",
            "odds_available": bool(odds_event),
            "raw_positive_count": raw_positive,
            "fair_positive_count": fair_positive,
            "opportunities": [],
        }
        for o in all_opps:
            detail["opportunities"].append(
                {
                    "market": o.market,
                    "selection": o.selection,
                    "ev_raw_pct": round(o.expected_value_raw * 100, 2),
                    "ev_fair_pct": round(o.expected_value * 100, 2),
                    "edge_fair_pct": round(o.edge_fair * 100, 2),
                    "vig_pct": o.vig_pct,
                    "delta_pp": round((o.expected_value_raw - o.expected_value) * 100, 2),
                }
            )
        report["details"].append(detail)

    if report["matches_audited"]:
        deltas = []
        for d in report["details"]:
            for o in d["opportunities"]:
                deltas.append(o["delta_pp"])
        report["mean_ev_inflation_pp"] = round(sum(deltas) / len(deltas), 2) if deltas else 0.0
    else:
        report["mean_ev_inflation_pp"] = 0.0

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
