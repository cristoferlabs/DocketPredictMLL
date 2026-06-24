"""Shared WC data quality audit logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.odds_context import find_wc_odds_event
from apps.api.services.worldcup_engine import analyze_match, find_upcoming_matches
from apps.worker.ingest.football_data import FootballDataClient
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.data_quality import (
    DataQualityReport,
    check_match_features,
    check_odds_event,
)
from apps.worker.tasks.update_elo import get_wc_elo_ratings


@dataclass
class WcAuditReport:
    matches: int = 0
    status_ok: int = 0
    status_partial: int = 0
    status_insufficient: int = 0
    avg_completeness_pct: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "status_ok": self.status_ok,
            "status_partial": self.status_partial,
            "status_insufficient": self.status_insufficient,
            "avg_completeness_pct": self.avg_completeness_pct,
            "details": self.details,
        }


async def audit_upcoming_matches(
    *,
    days_ahead: int = 14,
    db=None,
    fd_matches: list[dict] | None = None,
) -> WcAuditReport:
    archives = await fetch_all_worldcup_archives()
    d26, d22, d18 = archives.get(2026, {}), archives.get(2022, {}), archives.get(2018, {})
    elo = await get_wc_elo_ratings(db)

    if fd_matches is None:
        fd = FootballDataClient()
        try:
            fd_matches = await fd.get_competition_matches("WC", status=None) or []
        except Exception:
            fd_matches = []

    upcoming = find_upcoming_matches(d26, days_ahead=days_ahead)
    report = WcAuditReport()
    completeness_scores: list[float] = []

    for match in upcoming:
        analysis = analyze_match(match, d18, d22, fd_matches, elo)
        if not analysis.model:
            continue

        m = analysis.model
        form_n = len(analysis.forma.get(analysis.team1, [])) + len(
            analysis.forma.get(analysis.team2, [])
        )
        hist = sum(
            analysis.historico.get(t, {}).get("wc2022", {}).get("played", 0) or 0
            for t in (analysis.team1, analysis.team2)
        )
        dq = check_match_features(
            lambda_home=m.lambda_home,
            lambda_away=m.lambda_away,
            elo_home=analysis.elo.get(analysis.team1, {}).get("rating"),
            elo_away=analysis.elo.get(analysis.team2, {}).get("rating"),
            form_matches=form_n,
            hist_played=hist,
        )
        odds_event = await find_wc_odds_event(analysis.team1, analysis.team2)
        odds_flags = check_odds_event(odds_event)

        report.matches += 1
        completeness_scores.append(dq.completeness_pct)
        if dq.status == "ok":
            report.status_ok += 1
        elif dq.status == "partial":
            report.status_partial += 1
        else:
            report.status_insufficient += 1

        report.details.append(
            {
                "match": f"{analysis.team1} vs {analysis.team2}",
                "data_quality": dq.to_dict(),
                "odds_flags": [{"level": f.level, "code": f.code} for f in odds_flags],
                "xg_sources": {
                    analysis.team1: analysis.xg.get("source_home"),
                    analysis.team2: analysis.xg.get("source_away"),
                },
            }
        )

    if completeness_scores:
        report.avg_completeness_pct = round(sum(completeness_scores) / len(completeness_scores), 1)

    return report


def persist_audit_report(db, report: WcAuditReport) -> None:
    """Write batch audit summary to ops.data_quality_log."""
    if report.matches == 0:
        status = "insufficient"
        completeness = 0.0
    elif report.status_insufficient > 0:
        status = "partial"
        completeness = report.avg_completeness_pct
    elif report.status_partial > 0:
        status = "partial"
        completeness = report.avg_completeness_pct
    else:
        status = "ok"
        completeness = report.avg_completeness_pct

    flags = []
    if report.status_partial:
        flags.append(
            {
                "level": "warning",
                "code": "partial_matches",
                "message": f"{report.status_partial} partidos con datos parciales",
            }
        )
    if report.status_insufficient:
        flags.append(
            {
                "level": "critical",
                "code": "insufficient_matches",
                "message": f"{report.status_insufficient} partidos con datos insuficientes",
            }
        )

    db.schema("ops").table("data_quality_log").insert(
        {
            "context": "wc_audit",
            "status": status,
            "completeness_pct": completeness,
            "flags": flags,
        }
    ).execute()
