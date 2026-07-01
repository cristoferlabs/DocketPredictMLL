"""
Load Understat CSV files into ml.team_season_xg and ml.player_season_xg.

Usage:
    python scripts/load_understat_csvs.py

Place CSVs in DATA/understat/ before running.
Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CSV file → (league_slug, season) mapping ──────────────────────────────────
# Files are Understat exports; numbered variants correspond to different leagues.
# season "2024-25": confirmed by EPL (Arsenal 85pts), La Liga (Barça 94pts), etc.

CHEMP_MAP = {
    "league-chemp.csv":      ("epl",        "2024-25"),
    "league-chemp (1).csv":  ("la_liga",    "2024-25"),
    "league-chemp (2).csv":  ("bundesliga", "2024-25"),
    "league-chemp (3).csv":  ("serie_a",    "2024-25"),
    "league-chemp (4).csv":  ("ligue_1",    "2024-25"),
    "league-chemp (5).csv":  ("rpl",        "2024-25"),
}

PLAYERS_MAP = {
    "league-players.csv":      ("epl",        "2024-25"),
    "league-players (1).csv":  ("la_liga",    "2024-25"),
    "league-players (2).csv":  ("bundesliga", "2024-25"),
    "league-players (3).csv":  ("serie_a",    "2024-25"),
    "league-players (4).csv":  ("ligue_1",    "2024-25"),
    "league-players (5).csv":  ("rpl",        "2024-25"),
}

DATA_DIR = Path(__file__).parent.parent / "DATA" / "understat"


def _read_csv(path: Path) -> list[dict]:
    """Read semicolon-delimited CSV with BOM stripping."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    return [
        {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
        for row in reader
    ]


def _int(v: str) -> int | None:
    try:
        return int(float(v)) if v else None
    except (ValueError, TypeError):
        return None


def _float(v: str) -> float | None:
    try:
        return float(v) if v else None
    except (ValueError, TypeError):
        return None


def load_team_rows(client, filename: str, league_slug: str, season: str) -> int:
    path = DATA_DIR / filename
    if not path.exists():
        log.warning("Missing: %s", path)
        return 0

    rows = _read_csv(path)
    records = []
    for r in rows:
        matches = _int(r.get("matches", ""))
        xg = _float(r.get("xG", ""))
        xga = _float(r.get("xGA", ""))
        records.append({
            "league_slug":   league_slug,
            "season":        season,
            "team":          r.get("team", "").strip(),
            "position":      _int(r.get("number", "")),
            "matches":       matches,
            "wins":          _int(r.get("wins", "")),
            "draws":         _int(r.get("draws", "")),
            "loses":         _int(r.get("loses", "")),
            "goals":         _int(r.get("goals", "")),
            "goals_against": _int(r.get("ga", "")),
            "points":        _int(r.get("points", "")),
            "xg":            xg,
            "xga":           xga,
            "xpts":          _float(r.get("xPTS", "")),
            "xg_per_game":   round(xg / matches, 3) if xg and matches else None,
            "xga_per_game":  round(xga / matches, 3) if xga and matches else None,
        })

    if not records:
        return 0

    resp = (
        client.table("team_season_xg")
        .upsert(records, on_conflict="league_slug,season,team")
        .execute()
    )
    n = len(resp.data) if resp.data else len(records)
    log.info("  %-20s %s  → %d teams upserted", league_slug, season, n)
    return n


def load_player_rows(client, filename: str, league_slug: str, season: str) -> int:
    path = DATA_DIR / filename
    if not path.exists():
        log.warning("Missing: %s", path)
        return 0

    rows = _read_csv(path)
    records = []
    for r in rows:
        records.append({
            "league_slug": league_slug,
            "season":      season,
            "player":      r.get("player", "").strip(),
            "team":        r.get("team", "").strip(),
            "apps":        _int(r.get("apps", "")),
            "minutes":     _int(r.get("min", "")),
            "goals":       _int(r.get("goals", "")),
            "assists":     _int(r.get("a", "")),
            "xg":          _float(r.get("xG", "")),
            "xa":          _float(r.get("xA", "")),
            "xg90":        _float(r.get("xG90", "")),
            "xa90":        _float(r.get("xA90", "")),
        })

    if not records:
        return 0

    # Batch in chunks of 500 to avoid request size limits
    total = 0
    for i in range(0, len(records), 500):
        batch = records[i : i + 500]
        resp = (
            client.table("player_season_xg")
            .upsert(batch, on_conflict="league_slug,season,player,team")
            .execute()
        )
        total += len(resp.data) if resp.data else len(batch)

    log.info("  %-20s %s  → %d players upserted", league_slug, season, total)
    return total


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        log.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)

    from supabase import create_client
    base_client = create_client(url, key)
    # schema() returns a NEW scoped client — must capture it
    client = base_client.schema("ml")

    log.info("=== Loading Understat team standings ===")
    teams_total = 0
    for filename, (slug, season) in CHEMP_MAP.items():
        teams_total += load_team_rows(client, filename, slug, season)

    log.info("=== Loading Understat player stats ===")
    players_total = 0
    for filename, (slug, season) in PLAYERS_MAP.items():
        players_total += load_player_rows(client, filename, slug, season)

    log.info("Done. Teams: %d | Players: %d", teams_total, players_total)


if __name__ == "__main__":
    main()
