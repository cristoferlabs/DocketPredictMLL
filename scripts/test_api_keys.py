#!/usr/bin/env python3
"""Test connectivity for all configured data API keys (no secrets printed)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.services.injury_news import fetch_injury_report
from apps.shared.config import get_settings
from apps.worker.ingest.api_football import ApiFootballClient
from apps.worker.ingest.football_data import FootballDataClient
from apps.worker.ingest.odds_api import OddsApiClient
from apps.worker.ingest.sportmonks import SportMonksClient


def _mask(key: str) -> str:
    if not key:
        return "(vacía)"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


async def test_football_data() -> dict:
    s = get_settings()
    if not s.football_data_key:
        return {"ok": False, "detail": "FOOTBALL_DATA_KEY no configurada en .env"}
    client = FootballDataClient()
    try:
        matches = await client.get_competition_matches("WC", status="SCHEDULED")
        return {"ok": True, "detail": f"WC partidos programados: {len(matches)}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


async def test_odds_api() -> dict:
    s = get_settings()
    if not s.odds_api_key:
        return {"ok": False, "detail": "ODDS_API_KEY no configurada en .env"}
    client = OddsApiClient()
    try:
        data = await client._get(
            "/sports/soccer_fifa_world_cup/odds",
            {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
        )
        n = len(data) if isinstance(data, list) else 0
        return {"ok": True, "detail": f"Eventos WC con cuotas: {n}"}
    except Exception as exc:
        msg = str(exc)
        if "quota" in msg.lower() or "Usage quota" in msg or "OUT_OF_USAGE" in msg:
            return {
                "ok": False,
                "detail": "Cuota mensual agotada (0 créditos). Nueva clave en the-odds-api.com o espera reset.",
            }
        if "401" in msg:
            return {"ok": False, "detail": "401 — clave inválida o cuota agotada. Revisa the-odds-api.com"}
        return {"ok": False, "detail": msg}


async def test_api_football() -> dict:
    s = get_settings()
    if not s.api_football_key:
        return {"ok": False, "detail": "API_FOOTBALL_KEY no configurada en .env"}
    client = ApiFootballClient()
    try:
        data = await client._get("status")
        account = (data.get("response") or {}).get("account", {})
        return {
            "ok": True,
            "detail": f"Plan {account.get('plan', '?')} | requests hoy: {account.get('requests', {}).get('current', '?')}",
        }
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


async def test_sportmonks() -> dict:
    s = get_settings()
    if not s.sportmonks_key:
        return {"ok": False, "detail": "SPORTMONKS_KEY no configurada en .env"}
    client = SportMonksClient()
    try:
        from datetime import date

        fixtures = await client.get_fixtures_by_date(date.today())
        return {"ok": True, "detail": f"Fixtures hoy: {len(fixtures)}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


async def test_injury_news() -> dict:
    s = get_settings()
    if not s.gnews_api_key and not s.newsapi_key:
        return {"ok": False, "detail": "GNEWS_API_KEY y NEWSAPI_KEY no configuradas"}
    report = await fetch_injury_report("Brazil", "Scotland")
    sources = []
    if s.gnews_api_key:
        sources.append("GNews" + (" OK" if report.gnews_ok else " fallo"))
    if s.newsapi_key:
        sources.append("NewsAPI" + (" OK" if report.newsapi_ok else " fallo"))
    return {
        "ok": report.gnews_ok or report.newsapi_ok,
        "detail": f"{'; '.join(sources)} | artículos lesión: {len(report.articles)}",
    }


async def main() -> None:
    s = get_settings()
    print("=== Test de claves API (AGENTE) ===\n")
    print("Claves detectadas en .env:")
    for name, val in [
        ("FOOTBALL_DATA_KEY", s.football_data_key),
        ("ODDS_API_KEY", s.odds_api_key),
        ("API_FOOTBALL_KEY", s.api_football_key),
        ("SPORTMONKS_KEY", s.sportmonks_key),
        ("GNEWS_API_KEY", s.gnews_api_key),
        ("NEWSAPI_KEY", s.newsapi_key),
    ]:
        status = "configurada" if val else "FALTA"
        print(f"  {name}: {status} {_mask(val) if val else ''}")

    print()
    tests = [
        ("Football-Data (forma WC)", test_football_data()),
        ("Odds API (cuotas EV)", test_odds_api()),
        ("API-Football (stats/xG)", test_api_football()),
        ("SportMonks (opcional)", test_sportmonks()),
        ("Noticias lesiones (GNews+NewsAPI)", test_injury_news()),
    ]
    for label, coro in tests:
        result = await coro
        icon = "OK" if result["ok"] else "FALLO"
        print(f"[{icon}] {label}")
        print(f"      {result['detail']}")
        print()

    print("Uso en el proyecto:")
    print("  FOOTBALL_DATA_KEY -> forma reciente + partidos WC (telegram, audit)")
    print("  ODDS_API_KEY      -> EV fair, /alta, snapshots CLV")
    print("  API_FOOTBALL_KEY  -> worker ingest ligas + stats fixture (plan free: 2022-2024)")
    print("  SPORTMONKS_KEY    -> worker ingest opcional")
    print("  GNEWS_API_KEY     -> alertas lesiones/bajas en análisis Telegram")
    print("  NEWSAPI_KEY       -> alertas lesiones/bajas en análisis Telegram")
    print()
    print("Ingesta manual: POST http://localhost:8000/jobs/ingest-fixtures")
    print('  body: {"competition_code":"WC","sources":["football-data","odds-api"]}')


if __name__ == "__main__":
    asyncio.run(main())
