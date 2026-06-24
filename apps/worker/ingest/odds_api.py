"""The Odds API client (free tier — betting odds for EV)."""

import logging
import time
from typing import Any

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# soccer_epl, soccer_spain_la_liga, etc.
SOCCER_SPORTS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]


WC_SPORT = "soccer_fifa_world_cup"

_live_odds_warned = False
_live_disabled_at: float | None = None
_LIVE_RETRY_SECONDS = 600.0  # reintentar tras nueva clave / reset mensual


class OddsApiClient:
    _live_disabled: bool | None = None

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.odds_api_key

    async def is_live_available(self) -> bool:
        """Un solo check por proceso — evita spam 401/cuota en bucles."""
        global _live_disabled_at
        if not self.api_key:
            OddsApiClient._live_disabled = True
            return False
        if OddsApiClient._live_disabled is True:
            now = time.monotonic()
            if _live_disabled_at is not None and now - _live_disabled_at < _LIVE_RETRY_SECONDS:
                return False
            status = await self.check_status()
            OddsApiClient._live_disabled = not status.get("ok", False)
            if OddsApiClient._live_disabled:
                _live_disabled_at = now
            return not OddsApiClient._live_disabled
        if OddsApiClient._live_disabled is False:
            return True
        status = await self.check_status()
        OddsApiClient._live_disabled = not status.get("ok", False)
        if OddsApiClient._live_disabled:
            _live_disabled_at = time.monotonic()
            global _live_odds_warned
            if not _live_odds_warned:
                reason = status.get("reason", "error")
                detail = status.get("detail", "")
                logger.warning(
                    "Odds API live no disponible (%s). Usando caché DB si existe. %s",
                    reason,
                    detail,
                )
                _live_odds_warned = True
        return not OddsApiClient._live_disabled

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            return []
        p = {"apiKey": self.api_key, **(params or {})}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}{path}", params=p)
            resp.raise_for_status()
            return resp.json()

    async def get_wc_odds(self) -> list[dict]:
        """World Cup odds only (1 request)."""
        if not await self.is_live_available():
            return []
        try:
            data = await self._get(
                f"/sports/{WC_SPORT}/odds",
                {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
            )
            if isinstance(data, list):
                for event in data:
                    event["_sport_key"] = WC_SPORT
                return data
        except Exception as exc:
            OddsApiClient._live_disabled = True
            global _live_odds_warned
            if not _live_odds_warned:
                logger.warning("Odds API WC: %s", exc)
                _live_odds_warned = True
        return []

    async def check_status(self) -> dict[str, Any]:
        """Quota / auth check without consuming extra credits if possible."""
        global _live_disabled_at
        if not self.api_key:
            return {"ok": False, "reason": "ODDS_API_KEY no configurada en .env"}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/sports/{WC_SPORT}/odds",
                    params={
                        "apiKey": self.api_key,
                        "regions": "eu",
                        "markets": "h2h",
                        "oddsFormat": "decimal",
                    },
                )
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if resp.status_code == 200:
                    n = len(body) if isinstance(body, list) else 0
                    OddsApiClient._live_disabled = False
                    _live_disabled_at = None
                    return {
                        "ok": True,
                        "events": n,
                        "remaining": remaining,
                        "used": used,
                    }
                code = body.get("error_code", "") if isinstance(body, dict) else ""
                if code == "OUT_OF_USAGE_CREDITS" or "quota" in str(body.get("message", "")).lower():
                    OddsApiClient._live_disabled = True
                    _live_disabled_at = time.monotonic()
                    return {
                        "ok": False,
                        "reason": "cuota_mensual_agotada",
                        "remaining": remaining or "0",
                        "used": used,
                        "detail": body.get("message", "Sin créditos"),
                    }
                if resp.status_code == 401:
                    return {"ok": False, "reason": "clave_invalida", "detail": body.get("message", "401")}
                return {"ok": False, "reason": "error", "detail": body.get("message", resp.text[:200])}
        except Exception as exc:
            return {"ok": False, "reason": "error", "detail": str(exc)}

    async def get_soccer_odds(self, sports: list[str] | None = None) -> list[dict]:
        results: list[dict] = []
        for sport in sports or SOCCER_SPORTS:
            try:
                data = await self._get(
                    f"/sports/{sport}/odds",
                    {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
                )
                if isinstance(data, list):
                    for event in data:
                        event["_sport_key"] = sport
                        results.append(event)
            except Exception as exc:
                logger.warning("Odds API sport %s: %s", sport, exc)
        return results
