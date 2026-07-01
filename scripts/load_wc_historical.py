"""
Fetch openfootball World Cup JSON (2014 / 2018 / 2022) and load into
ml.wc_match_history for ELO training and Poisson calibration.

Usage:
    python scripts/load_wc_historical.py

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
Also adds 2026 matches already played (if available in openfootball).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

WC_URLS = {
    2014: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json",
    2018: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
    2026: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json",
}

KO_KEYWORDS = ("round of", "quarter", "semi", "final", "third")


def _is_knockout(round_name: str) -> bool:
    r = (round_name or "").lower()
    return any(k in r for k in KO_KEYWORDS)


async def _fetch_json(url: str) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _parse_matches(data: dict, year: int) -> list[dict]:
    """Convert openfootball JSON (flat matches list) to DB rows."""
    raw_matches = data.get("matches", [])
    records = []

    for m in raw_matches:
        score = m.get("score", {})
        ft = score.get("ft")
        ht = score.get("ht")

        # Skip unplayed matches (no final score)
        if not ft or len(ft) < 2:
            continue

        team1 = m.get("team1", "")
        team2 = m.get("team2", "")
        if not team1 or not team2:
            continue

        round_name = m.get("round", "")
        group_name = m.get("group", None)
        date_str = (m.get("date") or "")[:10] or None

        records.append({
            "tournament_year": year,
            "match_date":      date_str,
            "round":           round_name,
            "group_name":      group_name,
            "team_home":       team1,
            "team_away":       team2,
            "score_home":      int(ft[0]),
            "score_away":      int(ft[1]),
            "ht_home":         int(ht[0]) if ht and len(ht) >= 2 else None,
            "ht_away":         int(ht[1]) if ht and len(ht) >= 2 else None,
            "venue":           m.get("ground", None),
            "is_knockout":     _is_knockout(round_name),
        })

    return records


async def load_year(client, year: int) -> int:
    url = WC_URLS[year]
    log.info("Fetching WC %d ...", year)
    try:
        data = await _fetch_json(url)
    except Exception as exc:
        log.warning("WC %d fetch failed: %s", year, exc)
        return 0

    records = _parse_matches(data, year)
    if not records:
        log.info("  WC %d: no completed matches found", year)
        return 0

    resp = (
        client.table("wc_match_history")
        .upsert(records, on_conflict="tournament_year,team_home,team_away,match_date")
        .execute()
    )
    n = len(resp.data) if resp.data else len(records)
    log.info("  WC %d: %d matches upserted", year, n)
    return n


async def main_async() -> None:
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
    client = base_client.schema("ml")

    log.info("=== Loading WC match history (openfootball) ===")
    total = 0
    for year in (2014, 2018, 2022, 2026):
        total += await load_year(client, year)

    log.info("Done. Total matches loaded: %d", total)
    log.info("")
    log.info("Summary per tournament:")
    log.info("  2014 Brasil   — 64 matches expected")
    log.info("  2018 Russia   — 64 matches expected")
    log.info("  2022 Qatar    — 64 matches expected")
    log.info("  2026 USA/MX/CA — partial (ongoing)")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
