"""Betting Terminal — router de estados y callbacks."""

from __future__ import annotations

import logging
from typing import Any

from supabase import Client

from apps.api.services.telegram_client import TelegramClient
from apps.api.services.parlay_engine import (
    build_parlays_from_sharp_picks,
    extract_sharp_parlay_pick,
    format_parlay_message,
)
from apps.api.services.telegram_terminal.formatters import (
    build_ranked_picks,
    format_exploration,
    format_full_analysis,
    format_match_dashboard,
    format_opportunities,
)
from apps.api.services.telegram_terminal.keyboards import (
    dashboard_keyboard,
    exploration_keyboard,
    subview_keyboard,
)
from apps.api.services.telegram_terminal.orchestrator import MatchBundle, load_match_bundle
from apps.api.services.telegram_terminal.session import (
    load_session,
    match_key,
    parse_match_key,
    save_session,
)
from apps.api.services.telegram_terminal.states import TerminalState
from apps.api.services.worldcup_engine import find_upcoming_matches, name_match, normalize_openfootball
from apps.shared.config import get_settings
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ingest.football_data import FootballDataClient
from apps.worker.tasks.update_elo import get_wc_elo_ratings
from apps.api.services.telegram_terminal.orchestrator import load_match_bundle

logger = logging.getLogger(__name__)


class BettingTerminal:
    """Capa UI Telegram — orquesta vistas sin aplicar gates de decisión."""

    def __init__(self, db: Client, telegram: TelegramClient | None = None):
        self.db = db
        self.telegram = telegram or TelegramClient()
        self._wc_cache: tuple[dict, dict, dict, list] | None = None

    async def _load_worldcup_data(self) -> tuple[dict, dict, dict, list]:
        if self._wc_cache:
            return self._wc_cache
        archives = await fetch_all_worldcup_archives()
        fd = FootballDataClient()
        try:
            fd_matches = await fd.get_competition_matches("WC", status=None)
            if not fd_matches:
                from datetime import date, timedelta

                today = date.today()
                fd_matches = await fd.get_competition_matches(
                    "WC", date_from=today - timedelta(days=60), date_to=today + timedelta(days=30)
                )
        except Exception as exc:
            logger.warning("FD WC matches: %s", exc)
            fd_matches = []
        self._wc_cache = (
            archives.get(2026, {}),
            archives.get(2022, {}),
            archives.get(2018, {}),
            fd_matches,
        )
        return self._wc_cache

    async def get_today_matches(self) -> tuple[str, dict | None, list[dict]]:
        d26, _, _, _ = await self._load_worldcup_data()
        upcoming = find_upcoming_matches(d26, days_ahead=14)
        slim = []
        for m in upcoming[:12]:
            t1 = m.get("team1", {}).get("name", "TBD")
            t2 = m.get("team2", {}).get("name", "TBD")
            slim.append(
                {
                    "team1": t1,
                    "team2": t2,
                    "fecha": (m.get("date") or "")[:10],
                    "raw": m,
                }
            )
        text = format_exploration(slim)
        markup = exploration_keyboard(slim) if slim else None
        return text, markup, slim

    def _resolve_match(self, matches_cache: list[dict], key: str) -> dict | None:
        t1, t2, fecha = parse_match_key(key)
        for m in matches_cache:
            if name_match(m["team1"], t1) and name_match(m["team2"], t2):
                return m.get("raw") or m
            if fecha and m.get("fecha") == fecha:
                if name_match(m["team1"], t1) and name_match(m["team2"], t2):
                    return m.get("raw") or m
        return None

    def _resolve_match_by_index(self, matches_cache: list[dict], idx: int) -> dict | None:
        if 0 <= idx < len(matches_cache):
            entry = matches_cache[idx]
            return entry.get("raw") or entry
        return None

    def _find_match_by_teams(self, d26: dict, t1: str, t2: str) -> dict | None:
        data = normalize_openfootball(d26)
        for rnd in data.get("rounds", []):
            for m in rnd.get("matches", []):
                a = m.get("team1", {}).get("name", "")
                b = m.get("team2", {}).get("name", "")
                if (name_match(a, t1) and name_match(b, t2)) or (
                    name_match(a, t2) and name_match(b, t1)
                ):
                    return {**m, "roundName": rnd.get("name")}
        return None

    async def _load_bundle(self, match: dict, historical_accuracy: float | None) -> MatchBundle:
        d26, d22, d18, fd = await self._load_worldcup_data()
        return await load_match_bundle(
            match,
            db=self.db,
            d18=d18,
            d22=d22,
            fd_matches=fd,
            elo_ratings=await get_wc_elo_ratings(self.db),
            historical_accuracy=historical_accuracy,
        )

    async def get_match_dashboard(
        self, match: dict, *, historical_accuracy: float | None = None
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        text = format_match_dashboard(bundle.analysis, bundle.market_ctx)
        return text, dashboard_keyboard()

    async def get_opportunities(
        self, match: dict, *, historical_accuracy: float | None = None
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        picks = build_ranked_picks(
            bundle.analysis, bundle.ev_opps, bundle.sharp, bundle.market_ctx
        )
        text = format_opportunities(bundle.analysis, picks)
        return text, subview_keyboard()

    async def get_parlays(
        self, match: dict, *, historical_accuracy: float | None = None
    ) -> tuple[str, dict]:
        """Portfolio v3 — scan multi-partido, picks SHARP únicamente."""
        d26, d22, d18, fd = await self._load_worldcup_data()
        upcoming = find_upcoming_matches(d26, days_ahead=14)
        settings = get_settings()
        sharp_picks = []

        for m in upcoming[: settings.parlay_max_matches_scan]:
            bundle = await load_match_bundle(
                m,
                db=self.db,
                d18=d18,
                d22=d22,
                fd_matches=fd,
                elo_ratings=await get_wc_elo_ratings(self.db),
                historical_accuracy=historical_accuracy,
            )
            if not bundle.sharp or not bundle.analysis.model:
                continue
            sp = extract_sharp_parlay_pick(
                bundle.analysis,
                bundle.sharp,
                bundle.market_ctx,
                bundle.ev_opps,
            )
            if sp:
                sharp_picks.append(sp)

        result = build_parlays_from_sharp_picks(sharp_picks)
        t1 = match.get("team1", {}).get("name", "")
        t2 = match.get("team2", {}).get("name", "")
        header = f"Contexto: {t1} vs {t2}\nParlays multi-partido (QUANT v3)\n"
        text = header + format_parlay_message(result)
        return text, subview_keyboard()

    async def get_full_analysis(
        self, match: dict, *, historical_accuracy: float | None = None
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        text = format_full_analysis(
            bundle.analysis,
            bundle.market_ctx,
            bundle.sharp,
            bundle.ev_opps,
        )
        return text, subview_keyboard()

    async def handle_callback(
        self,
        chat_id: str,
        callback_data: str,
        *,
        session_id: str | None,
        context: dict[str, Any],
        historical_accuracy: float | None = None,
    ) -> tuple[str, dict | None, dict[str, Any], str | None]:
        """Devuelve (text, markup, new_context, new_session_id)."""
        ctx = dict(context)
        sid = session_id

        if callback_data == "t:hoy":
            text, markup, cache = await self.get_today_matches()
            ctx["state"] = TerminalState.EXPLORATION.value
            ctx["match_key"] = None
            ctx["matches_cache"] = cache
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="today")
            return text, markup, ctx, sid

        if callback_data.startswith("t:m:"):
            try:
                idx = int(callback_data.split(":")[2])
            except (IndexError, ValueError):
                return "Partido no válido.", None, ctx, sid
            cache = ctx.get("matches_cache") or []
            if not cache:
                _, _, cache = await self.get_today_matches()
            match = self._resolve_match_by_index(cache, idx)
            if not match:
                return "Partido no encontrado. Usa /hoy.", None, ctx, sid
            t1 = match.get("team1", {}).get("name", "")
            t2 = match.get("team2", {}).get("name", "")
            fecha = (match.get("date") or "")[:10]
            ctx["match_key"] = match_key(t1, t2, fecha)
            ctx["matches_cache"] = cache
            ctx["state"] = TerminalState.MATCH_SELECTED.value
            text, markup = await self.get_match_dashboard(match, historical_accuracy=historical_accuracy)
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="match_select")
            return text, markup, ctx, sid

        key = ctx.get("match_key")
        if not key:
            text, markup, cache = await self.get_today_matches()
            ctx["matches_cache"] = cache
            return text, markup, ctx, sid

        cache = ctx.get("matches_cache") or []
        match = self._resolve_match(cache, key)
        if not match:
            d26, _, _, _ = await self._load_worldcup_data()
            t1, t2, _ = parse_match_key(key)
            match = self._find_match_by_teams(d26, t1, t2)
        if not match:
            return "Partido expirado — usa /hoy de nuevo.", None, ctx, sid

        if callback_data == "t:d":
            ctx["state"] = TerminalState.MATCH_SELECTED.value
            text, markup = await self.get_match_dashboard(match, historical_accuracy=historical_accuracy)
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="dashboard")
            return text, markup, ctx, sid

        if callback_data == "t:o":
            ctx["state"] = TerminalState.OPPORTUNITIES_VIEW.value
            text, markup = await self.get_opportunities(match, historical_accuracy=historical_accuracy)
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="opportunities")
            return text, markup, ctx, sid

        if callback_data == "t:p":
            ctx["state"] = TerminalState.PARLAY_VIEW.value
            text, markup = await self.get_parlays(match, historical_accuracy=historical_accuracy)
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="parlay")
            return text, markup, ctx, sid

        if callback_data == "t:a":
            ctx["state"] = TerminalState.ANALYSIS_VIEW.value
            text, markup = await self.get_full_analysis(match, historical_accuracy=historical_accuracy)
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="analysis")
            return text, markup, ctx, sid

        return "Acción no reconocida.", dashboard_keyboard(), ctx, sid

    async def handle_today_command(
        self, chat_id: str, *, session_id: str | None, context: dict[str, Any]
    ) -> tuple[str, dict | None, dict[str, Any], str | None]:
        text, markup, cache = await self.get_today_matches()
        ctx = dict(context)
        ctx["state"] = TerminalState.EXPLORATION.value
        ctx["match_key"] = None
        ctx["matches_cache"] = cache
        sid = save_session(
            self.db, chat_id, session_id=session_id, context=ctx, intent="today", inbound_text="/hoy"
        )
        return text, markup, ctx, sid

    async def handle_team_query(
        self,
        chat_id: str,
        t1: str,
        t2: str,
        *,
        session_id: str | None,
        context: dict[str, Any],
        historical_accuracy: float | None = None,
    ) -> tuple[str, dict | None, dict[str, Any], str | None]:
        d26, _, _, _ = await self._load_worldcup_data()
        match = self._find_match_by_teams(d26, t1, t2)
        if not match:
            return f"No encontré {t1} vs {t2}. Prueba /hoy.", None, context, session_id
        fecha = (match.get("date") or "")[:10]
        ctx = dict(context)
        ctx["match_key"] = match_key(
            match.get("team1", {}).get("name", t1),
            match.get("team2", {}).get("name", t2),
            fecha,
        )
        ctx["state"] = TerminalState.MATCH_SELECTED.value
        text, markup = await self.get_match_dashboard(match, historical_accuracy=historical_accuracy)
        sid = save_session(
            self.db,
            chat_id,
            session_id=session_id,
            context=ctx,
            intent="team_query",
            inbound_text=f"{t1} vs {t2}",
        )
        return text, markup, ctx, sid

    @staticmethod
    def is_terminal_callback(data: str) -> bool:
        return bool(data and data.startswith("t:"))

    @staticmethod
    def is_legacy_wc_callback(data: str) -> bool:
        return bool(data and data.startswith("wc:"))
