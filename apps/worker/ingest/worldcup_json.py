"""Fetch openfootball worldcup JSON archives."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

WORLDCUP_URLS = {
    2026: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
    2018: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
}


async def fetch_worldcup_year(year: int) -> dict[str, Any]:
    url = WORLDCUP_URLS.get(year)
    if not url:
        return {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def fetch_all_worldcup_archives() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for year in WORLDCUP_URLS:
        try:
            result[year] = await fetch_worldcup_year(year)
        except Exception as exc:
            logger.warning("worldcup %s fetch failed: %s", year, exc)
            result[year] = {}
    return result
