"""Data quality gates for predictions and EV publishing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.worker.ml.odds_math import extract_h2h_per_bookmaker, extract_totals_per_bookmaker


@dataclass
class QualityFlag:
    level: str  # info | warning | critical
    code: str
    message: str


@dataclass
class DataQualityReport:
    status: str  # ok | partial | insufficient
    flags: list[QualityFlag] = field(default_factory=list)
    completeness_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completeness_pct": self.completeness_pct,
            "flags": [{"level": f.level, "code": f.code, "message": f.message} for f in self.flags],
        }


MIN_ODDS_BOOKS = 3
MIN_TEAM_HISTORY = 5


def check_odds_event(odds_event: dict | None, min_books: int = MIN_ODDS_BOOKS) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    if not odds_event:
        flags.append(QualityFlag("critical", "no_odds", "Sin cuotas disponibles"))
        return flags

    n_h2h = len(extract_h2h_per_bookmaker(odds_event))
    n_totals = len(extract_totals_per_bookmaker(odds_event))
    if n_h2h < min_books:
        flags.append(
            QualityFlag(
                "warning",
                "few_h2h_books",
                f"Solo {n_h2h} casas con 1X2 (mínimo recomendado: {min_books})",
            )
        )
    if n_totals < min_books:
        flags.append(
            QualityFlag(
                "info",
                "few_totals_books",
                f"Solo {n_totals} casas con O/U 2.5",
            )
        )
    return flags


def check_team_history(played: int, min_matches: int = MIN_TEAM_HISTORY) -> list[QualityFlag]:
    if played < min_matches:
        return [
            QualityFlag(
                "warning",
                "thin_history",
                f"Solo {played} partidos históricos (mínimo recomendado: {min_matches})",
            )
        ]
    return []


def check_match_features(
    *,
    lambda_home: float | None,
    lambda_away: float | None,
    elo_home: float | None,
    elo_away: float | None,
    form_matches: int = 0,
    hist_played: int = 0,
) -> DataQualityReport:
    """Aggregate quality check before publishing EV."""
    flags: list[QualityFlag] = []
    checks = 0
    passed = 0

    for name, val, lo, hi in [
        ("lambda_home", lambda_home, 0.5, 4.0),
        ("lambda_away", lambda_away, 0.5, 4.0),
        ("elo_home", elo_home, 1200, 2300),
        ("elo_away", elo_away, 1200, 2300),
    ]:
        checks += 1
        if val is None:
            flags.append(QualityFlag("critical", f"missing_{name}", f"Falta {name}"))
        elif not (lo <= val <= hi):
            flags.append(
                QualityFlag("warning", f"outlier_{name}", f"{name}={val} fuera de rango [{lo}, {hi}]")
            )
        else:
            passed += 1

    flags.extend(check_team_history(hist_played))

    if form_matches == 0:
        flags.append(QualityFlag("info", "no_recent_form", "Sin forma reciente (football-data)"))

    completeness = round(passed / checks * 100, 1) if checks else 0.0
    critical = any(f.level == "critical" for f in flags)
    warnings = any(f.level == "warning" for f in flags)

    if critical:
        status = "insufficient"
    elif warnings:
        status = "partial"
    else:
        status = "ok"

    return DataQualityReport(status=status, flags=flags, completeness_pct=completeness)


def can_publish_ev(report: DataQualityReport, odds_flags: list[QualityFlag]) -> bool:
    """Block EV only on critical data/odds issues (not thin book count)."""
    all_flags = report.flags + odds_flags
    if report.status == "insufficient":
        return False
    if any(f.level == "critical" for f in all_flags):
        return False
    return True
