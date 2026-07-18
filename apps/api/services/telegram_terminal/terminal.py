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
from apps.api.services.live_stats_service import fetch_live_match_data
from apps.api.services.safe_combo_engine import build_live_combinations, build_safe_combinations
from apps.api.services.telegram_terminal.betting_menu import build_betting_menu
from apps.api.services.telegram_terminal.formatters import (
    build_ranked_picks,
    format_exploration,
    format_full_analysis,
    format_match_dashboard,
    format_opportunities,
    format_safe_combinations,
)
from apps.api.services.telegram_terminal.keyboards import (
    betting_menu_keyboard,
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
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        text = format_match_dashboard(
            bundle.analysis, bundle.market_ctx, bundle.ev_opps,
            home_team_stats=bundle.home_team_stats,
            away_team_stats=bundle.away_team_stats,
        )
        return text, dashboard_keyboard(match_key_str)

    async def get_opportunities(
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        picks = build_ranked_picks(
            bundle.analysis, bundle.ev_opps, bundle.sharp, bundle.market_ctx,
            home_team_stats=bundle.home_team_stats,
            away_team_stats=bundle.away_team_stats,
        )
        text = format_opportunities(bundle.analysis, picks)
        return text, subview_keyboard(match_key_str)

    async def get_safe_combinations(
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        if not bundle.analysis.model:
            return "Modelo no disponible para este partido.", subview_keyboard(match_key_str)

        # Try live data first
        live_result = await self._try_fetch_live(bundle.analysis.team1, bundle.analysis.team2, bundle.analysis.model)

        if live_result is not None:
            combos = build_live_combinations(live_result, bundle.analysis.team1, bundle.analysis.team2)
        else:
            combos = build_safe_combinations(
                bundle.analysis.model,
                bundle.analysis.team1,
                bundle.analysis.team2,
            )
        text = format_safe_combinations(bundle.analysis, combos, live_result=live_result)
        return text, subview_keyboard(match_key_str)

    async def get_parlays(
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
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
        return text, subview_keyboard(match_key_str)

    async def _try_fetch_live(self, team1: str, team2: str, model) -> "LivePoissonResult | None":
        """Fetch live game state + stats from API-Football and compute live markets. Returns None if not live."""
        try:
            from apps.worker.ml.poisson_live import compute_live_markets
            game_state, live_stats = await fetch_live_match_data(team1, team2)
            if game_state is None:
                return None
            return compute_live_markets(
                model.lambda_home,
                model.lambda_away,
                game_state,
                live_stats,
            )
        except Exception as exc:
            logger.warning("Live market computation failed: %s", exc)
            return None

    async def get_betting_menu(
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
    ) -> tuple[str, dict]:
        """Opción E — menú unificado: mercados + combinaciones + recomendación."""
        bundle = await self._load_bundle(match, historical_accuracy)

        # Detect live match and overlay live Poisson markets
        live_result = None
        if bundle.analysis.model:
            live_result = await self._try_fetch_live(
                bundle.analysis.team1, bundle.analysis.team2, bundle.analysis.model
            )

        text = build_betting_menu(
            analysis=bundle.analysis,
            ev_opps=bundle.ev_opps,
            sharp=bundle.sharp,
            odds_event=bundle.odds_event,
            live_result=live_result,
            home_team_stats=bundle.home_team_stats,
            away_team_stats=bundle.away_team_stats,
            stats_odds=bundle.stats_odds,
            db=self.db,
        )
        return text, betting_menu_keyboard(match_key_str)

    async def get_full_analysis(
        self, match: dict, *, historical_accuracy: float | None = None, match_key_str: str = ""
    ) -> tuple[str, dict]:
        bundle = await self._load_bundle(match, historical_accuracy)
        text = format_full_analysis(
            bundle.analysis,
            bundle.market_ctx,
            bundle.sharp,
            bundle.ev_opps,
        )
        return text, subview_keyboard(match_key_str)

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
            mk_str = match_key(t1, t2, fecha)
            ctx["match_key"] = mk_str
            ctx["matches_cache"] = cache
            ctx["state"] = TerminalState.MATCH_SELECTED.value
            text, markup = await self.get_match_dashboard(
                match, historical_accuracy=historical_accuracy, match_key_str=mk_str
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="match_select")
            return text, markup, ctx, sid

        # For subview callbacks: extract embedded match_key if present.
        # New format: "t:v:team1|team2|fecha" — the match is self-contained in the button.
        # Old format (without embedded key): fall back to session match_key.
        _cb_parts = callback_data.split(":", 2)
        _embedded_key = (
            _cb_parts[2]
            if len(_cb_parts) == 3 and _cb_parts[1] not in ("m", "hoy")
            else None
        )
        # Normalise so the if-branches below compare plain view codes
        base_cb = f"t:{_cb_parts[1]}" if len(_cb_parts) >= 2 else callback_data

        key = _embedded_key or ctx.get("match_key")
        if _embedded_key and _embedded_key != ctx.get("match_key"):
            # Sync session to the match embedded in the button click
            ctx["match_key"] = _embedded_key

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

        if base_cb == "t:d":
            ctx["state"] = TerminalState.MATCH_SELECTED.value
            text, markup = await self.get_match_dashboard(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="dashboard")
            return text, markup, ctx, sid

        if base_cb == "t:o":
            ctx["state"] = TerminalState.OPPORTUNITIES_VIEW.value
            text, markup = await self.get_opportunities(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="opportunities")
            return text, markup, ctx, sid

        if base_cb == "t:e":
            ctx["state"] = TerminalState.OPPORTUNITIES_VIEW.value
            text, markup = await self.get_betting_menu(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="betting_menu")
            return text, markup, ctx, sid

        if base_cb == "t:c":
            ctx["state"] = TerminalState.PARLAY_VIEW.value
            text, markup = await self.get_safe_combinations(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="safe_combos")
            return text, markup, ctx, sid

        if base_cb == "t:p":
            ctx["state"] = TerminalState.PARLAY_VIEW.value
            text, markup = await self.get_parlays(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
            sid = save_session(self.db, chat_id, session_id=sid, context=ctx, intent="parlay")
            return text, markup, ctx, sid

        if base_cb == "t:a":
            ctx["state"] = TerminalState.ANALYSIS_VIEW.value
            text, markup = await self.get_full_analysis(
                match, historical_accuracy=historical_accuracy, match_key_str=key
            )
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
        mk_str = match_key(
            match.get("team1", {}).get("name", t1),
            match.get("team2", {}).get("name", t2),
            fecha,
        )
        ctx["match_key"] = mk_str
        ctx["state"] = TerminalState.MATCH_SELECTED.value
        text, markup = await self.get_match_dashboard(
            match, historical_accuracy=historical_accuracy, match_key_str=mk_str
        )
        sid = save_session(
            self.db,
            chat_id,
            session_id=session_id,
            context=ctx,
            intent="team_query",
            inbound_text=f"{t1} vs {t2}",
        )
        return text, markup, ctx, sid

    async def handle_live_command(
        self,
        chat_id: str,
        *,
        session_id: str | None,
        context: dict[str, Any],
        historical_accuracy: float | None = None,
    ) -> tuple[str, dict | None, dict[str, Any], str | None]:
        """Show all live WC matches with score, time, and best live combo."""
        from apps.worker.ingest.api_football import ApiFootballClient
        from apps.worker.ml.poisson_live import (
            compute_live_markets,
            live_game_state_from_api_football,
            live_stats_from_api_football,
        )

        _LIVE_CODES = {"1H", "HT", "2H", "ET", "BT", "INT", "LIVE", "P"}
        _STATUS_LABEL = {
            "1H": "1ª Parte", "HT": "Descanso",
            "2H": "2ª Parte", "ET": "Prórroga",
            "BT": "Desc. Prórroga", "P": "Penaltis",
        }

        client = ApiFootballClient()
        try:
            live_fixtures = await client.get_wc_live_fixtures()
        except Exception as exc:
            logger.warning("Live fixtures fetch error: %s", exc)
            return (
                "⚠️ No se pudo conectar con la API en vivo. Intenta más tarde.",
                None, context, session_id,
            )

        active = [
            fx for fx in live_fixtures
            if fx.get("fixture", {}).get("status", {}).get("short", "") in _LIVE_CODES
        ]
        if not active:
            return (
                "⚪ No hay partidos WC en vivo ahora mismo.\n\nUsa /hoy para ver próximos partidos.",
                None, context, session_id,
            )

        d26, _, _, _ = await self._load_worldcup_data()
        ctx = dict(context)
        cache_entries: list[dict] = []
        lines = ["🔴 PARTIDOS EN VIVO — WC 2026\n"]

        for fx in active[:4]:
            fx_info = fx.get("fixture", {})
            teams = fx.get("teams", {})
            goals = fx.get("goals", {})
            status_info = fx_info.get("status", {})

            api_home = teams.get("home", {}).get("name", "?")
            api_away = teams.get("away", {}).get("name", "?")
            g_h = goals.get("home") or 0
            g_a = goals.get("away") or 0
            elapsed = status_info.get("elapsed") or 0
            status_label = _STATUS_LABEL.get(status_info.get("short", ""), "En vivo")

            lines.append(f"⚽ {api_home} {g_h}–{g_a} {api_away}")
            lines.append(f"   {status_label} | {elapsed}'")

            match = self._find_match_by_teams(d26, api_home, api_away)
            if match:
                t1_name = match.get("team1", {}).get("name", api_home) if isinstance(match.get("team1"), dict) else api_home
                t2_name = match.get("team2", {}).get("name", api_away) if isinstance(match.get("team2"), dict) else api_away
                cache_entries.append({
                    "team1": t1_name,
                    "team2": t2_name,
                    "fecha": (match.get("date") or "")[:10],
                    "raw": match,
                })
                try:
                    bundle = await self._load_bundle(match, historical_accuracy)
                    if bundle.analysis.model:
                        gs = live_game_state_from_api_football(fx)
                        fx_id = fx_info.get("id")
                        try:
                            raw_stats = await client.get_fixture_statistics(fx_id) if fx_id else []
                            ls = live_stats_from_api_football(raw_stats) if raw_stats else None
                        except Exception:
                            ls = None
                        live_result = compute_live_markets(
                            bundle.analysis.model.lambda_home,
                            bundle.analysis.model.lambda_away,
                            gs, ls,
                        )
                        combos = build_live_combinations(
                            live_result, bundle.analysis.team1, bundle.analysis.team2
                        )
                        if combos:
                            best = combos[0]
                            stars = (
                                "⭐⭐⭐" if best.decision == "STRONG_BET"
                                else "⭐⭐" if best.decision == "MODERATE_BET"
                                else "⭐"
                            )
                            lines.append(
                                f"   🔝 {best.leg1_label} + {best.leg2_label} {stars}"
                                f" — {best.combo_prob * 100:.0f}%"
                            )
                except Exception as exc:
                    logger.debug("Live analysis error %s vs %s: %s", api_home, api_away, exc)

            lines.append("")

        if cache_entries:
            lines.append("Selecciona un partido para análisis completo:")
            markup = exploration_keyboard(cache_entries)
            ctx["matches_cache"] = cache_entries
        else:
            lines.append("Escribe [equipo1] vs [equipo2] para análisis completo.")
            markup = None

        ctx["state"] = TerminalState.EXPLORATION.value
        ctx["match_key"] = None
        sid = save_session(self.db, chat_id, session_id=session_id, context=ctx, intent="live")
        return "\n".join(lines), markup, ctx, sid

    @staticmethod
    def is_terminal_callback(data: str) -> bool:
        return bool(data and data.startswith("t:"))

    @staticmethod
    def is_legacy_wc_callback(data: str) -> bool:
        return bool(data and data.startswith("wc:"))
