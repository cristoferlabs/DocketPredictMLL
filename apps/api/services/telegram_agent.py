"""Telegram agent — Mundial 2026 with model-first predictions and separate odds EV."""

import hashlib
import json
import logging
import re
from typing import Any

from supabase import Client

from apps.api.services.injury_news import fetch_injury_report
from apps.api.services.live_calibration import calibrate_analysis_model
from apps.api.services.llm import get_llm_client
from apps.api.services.odds_context import (
    compute_ev_opportunities,
    compute_market_context,
    find_wc_odds_event,
    odds_unavailable_reason,
)
from apps.api.services.telegram_client import TelegramClient
from apps.api.services.telegram_terminal import BettingTerminal
from apps.api.services.telegram_terminal.session import load_session
from apps.api.services.worldcup_engine import (
    analyze_match,
    find_upcoming_matches,
    get_calibration_factors,
    name_match,
    set_calibration_factors,
)
from apps.shared.config import get_settings
from apps.worker.ingest.football_data import FootballDataClient
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.clv import record_pick_snapshot, record_wc_market_snapshots
from apps.worker.ml.data_quality import can_publish_ev, check_match_features, check_odds_event
from apps.worker.ml.ev_anomaly import evaluate_pick, log_anomaly_to_db
from apps.worker.ml.guardrails import check_ev_guardrails
from apps.worker.ml.model_loader import load_calibration_factors_from_db
from apps.worker.ml.wc_audit import audit_upcoming_matches
from apps.api.services.engine_health import evaluate_engine_health
from apps.api.services.stats_report import build_pro_stats_report, build_roi_report
from apps.worker.ml.wc_predictions import save_wc_prediction
from apps.api.services.sharp_engine import SharpBetResult, run_sharp_engine
from apps.api.services.parlay_engine import (
    ParlayBuildResult,
    build_parlays_from_sharp_picks,
    extract_sharp_parlay_pick,
    format_parlay_message,
)
from apps.api.services.parlay_tracking import save_parlay_ticket
from apps.api.services.engine_constants import ENGINE_VERSION_TAG
from apps.api.services.trading_card import (
    build_trading_card,
    build_trading_card_from_dict,
    format_trading_message,
)
from apps.worker.tasks.update_elo import get_wc_elo_ratings

logger = logging.getLogger(__name__)

TELEGRAM_SYSTEM = """Eres el analista del Mundial 2026. Respondes en español, texto plano y emojis.
REGLAS CRITICAS:
- Las probabilidades del MODELO (Poisson+ELO) son la fuente de verdad para predicciones.
- Las cuotas de casas de apuestas son SOLO contexto de mercado para calcular valor esperado (EV).
- NUNCA ajustes las probabilidades del modelo para alinearlas con las cuotas.
- Si modelo y mercado difieren, explica la divergencia; el modelo manda, el mercado informa precio.
- Formato: secciones con líneas ─────, sin markdown (sin ** ni |).
- Mercados: 1X2, Over/Under 1.5, Over/Under 2.5, Over/Under 3.5, BTTS, Doble Oportunidad (1X, X2, 12).
- Incluye disclaimer probabilístico al final."""


class TelegramAgentService:
    def __init__(self, db: Client):
        self.db = db
        self.llm = get_llm_client()
        self.telegram = TelegramClient()
        self.terminal = BettingTerminal(db, self.telegram)
        self.settings = get_settings()
        self._wc_cache: dict | None = None

    async def _load_worldcup_data(self) -> tuple[dict, dict, dict, list]:
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

        return archives.get(2026, {}), archives.get(2022, {}), archives.get(2018, {}), fd_matches

    def _detect_command(self, text: str) -> str:
        t = (text or "").strip().lower()
        if t in ("/start", "/help", "/ayuda"):
            return "help"
        if t.startswith("/alta"):
            return "alta"
        if t.startswith("/combinada") or t.startswith("/parlay") or t.startswith("/combo"):
            return "combinada"
        if t.startswith("/stats"):
            return "stats"
        if t.startswith("/roi"):
            return "roi"
        if t.startswith("/hoy") or t.startswith("/partidos"):
            return "today"
        if t.startswith("/live") or t.startswith("/vivo") or t.startswith("/ahora"):
            return "live"
        if any(w in t for w in ("predic", "pronost", "analiza", "vs")):
            return "analyze"
        return "general"

    async def handle_update(self, update: dict[str, Any]) -> dict[str, Any]:
        """Process Telegram Update and return/send response."""
        from apps.api.services.telegram_dedup import claim_update

        update_id = update.get("update_id")
        if not claim_update(update_id):
            logger.info("duplicate telegram update %s skipped", update_id)
            return {"ok": True, "duplicate": True, "update_id": update_id}

        chat_id, text, callback_id, callback_data = self._parse_update(update)
        if not chat_id:
            return {"ok": False, "error": "no chat_id"}

        session_id, session_ctx = load_session(self.db, chat_id)
        hist_acc = self._load_historical_accuracy()

        if callback_id:
            await self.telegram.answer_callback_query(callback_id)

        markup: dict | None = None

        if callback_data and BettingTerminal.is_terminal_callback(callback_data):
            msg, markup, _, session_id = await self.terminal.handle_callback(
                chat_id,
                callback_data,
                session_id=session_id,
                context=session_ctx,
                historical_accuracy=hist_acc,
            )
            await self.telegram.send_message(chat_id, msg, reply_markup=markup)
            return {"ok": True, "chat_id": chat_id, "sent": True, "terminal": True}

        if callback_data and BettingTerminal.is_legacy_wc_callback(callback_data):
            teams = callback_data.replace("wc:", "").split("|")
            if len(teams) >= 2:
                msg, markup, _, session_id = await self.terminal.handle_team_query(
                    chat_id,
                    teams[0],
                    teams[1],
                    session_id=session_id,
                    context=session_ctx,
                    historical_accuracy=hist_acc,
                )
                await self.telegram.send_message(chat_id, msg, reply_markup=markup)
                return {"ok": True, "chat_id": chat_id, "sent": True, "terminal": True}

        command = self._detect_command(text)
        await self._save_session(chat_id, text, command)

        if command == "help":
            msg = self._help_text()
        elif command == "stats":
            msg = await self._stats_message()
        elif command == "roi":
            msg = await self._roi_message()
        elif command == "today":
            msg, markup, _, session_id = await self.terminal.handle_today_command(
                chat_id, session_id=session_id, context=session_ctx
            )
            await self.telegram.send_message(chat_id, msg, reply_markup=markup)
            return {"ok": True, "chat_id": chat_id, "sent": True, "terminal": True}
        elif command == "live":
            msg, markup, _, session_id = await self.terminal.handle_live_command(
                chat_id, session_id=session_id, context=session_ctx,
                historical_accuracy=hist_acc,
            )
            await self.telegram.send_message(chat_id, msg, reply_markup=markup)
            return {"ok": True, "chat_id": chat_id, "sent": True, "terminal": True}
        elif command == "combinada":
            msg = await self._combinada_message()
        elif command == "alta":
            msg = await self._alta_message()
        elif command == "analyze":
            t1, t2 = self._extract_teams(text)
            if t1 and t2:
                msg, markup, _, session_id = await self.terminal.handle_team_query(
                    chat_id,
                    t1,
                    t2,
                    session_id=session_id,
                    context=session_ctx,
                    historical_accuracy=hist_acc,
                )
                await self.telegram.send_message(chat_id, msg, reply_markup=markup)
                return {"ok": True, "chat_id": chat_id, "sent": True, "terminal": True}
            msg = await self._analyze_message(text, alta_only=False)
        elif command == "general":
            msg = self._help_text()
        else:
            msg = self._help_text()

        await self.telegram.send_message(chat_id, msg, reply_markup=markup)
        return {"ok": True, "chat_id": chat_id, "message_length": len(msg)}

    def _parse_update(self, update: dict) -> tuple[str, str, str, str]:
        if update.get("callback_query"):
            cq = update["callback_query"]
            return (
                str(cq.get("message", {}).get("chat", {}).get("id", "")),
                "",
                cq.get("id", ""),
                cq.get("data", ""),
            )
        msg = update.get("message", {})
        return (
            str(msg.get("chat", {}).get("id", "")),
            msg.get("text", ""),
            "",
            "",
        )

    def _help_text(self) -> str:
        return (
            "🖥️ Betting Terminal v2 — Mundial 2026\n\n"
            "Comandos:\n"
            "/hoy — Explorar próximos partidos\n"
            "/live — Ver partidos WC en vivo + mejor combo\n"
            "/alta — Scan SHARP multi-partido\n"
            "/combinada — Parlays multi-partido\n"
            "/stats — Brier, CLV, ROI, tiers SHARP\n"
            "/roi — ROI live + backtest (compacto)\n"
            "Colombia vs Brasil — Dashboard del partido\n\n"
            "Flujo: /hoy → partido → opciones / combinadas / análisis\n"
            "El terminal muestra datos — tú decides.\n\n"
            "⚠️ Predicciones probabilísticas, no garantías."
        )

    async def _today_matches_menu(self) -> tuple[str, dict | None]:
        text, markup, _ = await self.terminal.get_today_matches()
        return text, markup

    def _persist_sharp_pick(self, analysis, sharp: SharpBetResult) -> None:
        """Log SHARP aprobado/WATCH con tier para hit-rate por tier en /stats."""
        if not sharp or not sharp.decision.pick:
            return
        if not sharp.sharp_allowed and sharp.decision.soft_action != "WATCH":
            return
        dec = sharp.decision
        pick = dec.pick
        m = analysis.model
        if not m:
            return
        pred_id = save_wc_prediction(
            self.db,
            team_home=analysis.team1,
            team_away=analysis.team2,
            match_date=(analysis.fecha or "")[:10] or None,
            market_type=pick.market or "1X2",
            predicted_outcome=pick.selection,
            probability=pick.model_prob,
            expected_value_fair=pick.ev_fair,
            edge_fair=pick.edge_fair,
            kelly_stake=pick.kelly_stake,
            metadata={
                "source": "sharp_scan",
                "sharp_tier": sharp.portfolio_tier,
                "sharp_allowed": sharp.sharp_allowed,
                "soft_action": dec.soft_action,
                "mds": sharp.mds,
                "rank_score": sharp.rank_score,
                "sharp_phase": (m.blend_meta or {}).get("calibration", {}).get("sharp_phase"),
                "alpha_regime": (m.blend_meta or {}).get("calibration", {}).get("alpha_regime"),
                "alpha": (m.blend_meta or {}).get("calibration", {}).get("alpha"),
                "prob_statistical": (m.blend_meta or {}).get("statistical"),
                "prob_calibrated": {
                    "home_win": m.home_win,
                    "draw": m.draw,
                    "away_win": m.away_win,
                },
                "shrink_applied": (m.blend_meta or {}).get("calibration", {}).get("shrink_applied"),
                "model_1x2": {
                    "home_win": m.home_win,
                    "draw": m.draw,
                    "away_win": m.away_win,
                },
                "blend_meta": m.blend_meta,
            },
        )
        if pred_id and pick.raw_odds and pick.raw_odds > 1:
            record_pick_snapshot(
                self.db,
                team_home=analysis.team1,
                team_away=analysis.team2,
                market=pick.market or "1X2",
                selection=pick.selection,
                odds_decimal=pick.raw_odds,
                fair_odds=pick.fair_odds,
                prediction_id=pred_id,
            )

    def _save_model_prediction(self, analysis) -> None:
        """Persist model 1X2 pick for learning loop (every analysis)."""
        m = analysis.model
        if not m:
            return
        candidates = [
            (analysis.team1, m.home_win),
            ("Empate", m.draw),
            (analysis.team2, m.away_win),
        ]
        outcome, prob = max(candidates, key=lambda x: x[1])
        save_wc_prediction(
            self.db,
            team_home=analysis.team1,
            team_away=analysis.team2,
            match_date=(analysis.fecha or "")[:10] or None,
            market_type="1X2",
            predicted_outcome=outcome,
            probability=prob,
            metadata={
                "source": "telegram_analyze",
                "xg_sources": {
                    analysis.team1: analysis.xg.get("source_home"),
                    analysis.team2: analysis.xg.get("source_away"),
                },
                "wc_features": {
                    "lambda_home": m.lambda_home,
                    "lambda_away": m.lambda_away,
                    "xg_home": analysis.xg.get(analysis.team1, 0),
                    "xg_away": analysis.xg.get(analysis.team2, 0),
                },
                "elo_probs": {
                    "home_win": m.home_win,
                    "draw": m.draw,
                    "away_win": m.away_win,
                },
                "model_1x2": {
                    "home_win": m.home_win,
                    "draw": m.draw,
                    "away_win": m.away_win,
                },
                "blend_meta": m.blend_meta,
            },
        )

    async def _run_sharp_for_match(
        self,
        match: dict,
        *,
        d18: dict,
        d22: dict,
        fd_matches: list,
        elo_ratings: dict,
    ) -> tuple[Any, SharpBetResult, str]:
        """Analiza un partido y devuelve (analysis, sharp, quality_note)."""
        analysis = analyze_match(match, d18, d22, fd_matches, elo_ratings)
        quality_note = ""
        ev_opps: list = []
        market_ctx = None
        dq_completeness = 100.0
        hist_played = 20

        if not analysis.model:
            return analysis, None, "sin modelo"  # type: ignore[return-value]

        m = analysis.model
        odds_event = await find_wc_odds_event(analysis.team1, analysis.team2, db=self.db)
        hist_played = (
            (analysis.historico.get(analysis.team1, {}).get("wc2022", {}).get("played", 0) or 0)
            + (analysis.historico.get(analysis.team2, {}).get("wc2022", {}).get("played", 0) or 0)
        )
        form_n = len(analysis.forma.get(analysis.team1, [])) + len(
            analysis.forma.get(analysis.team2, [])
        )
        dq = check_match_features(
            lambda_home=m.lambda_home,
            lambda_away=m.lambda_away,
            elo_home=analysis.elo.get(analysis.team1, {}).get("rating"),
            elo_away=analysis.elo.get(analysis.team2, {}).get("rating"),
            form_matches=form_n,
            hist_played=hist_played,
        )
        odds_flags = check_odds_event(odds_event, min_books=self.settings.min_odds_books)
        dq_label = (
            f"OK ({dq.completeness_pct:.0f}%)"
            if dq.status == "ok"
            else f"{dq.status.upper()} ({dq.completeness_pct:.0f}%)"
        )
        quality_note = f"Datos {dq_label}"
        dq_completeness = dq.completeness_pct

        calibrate_analysis_model(
            analysis,
            odds_event,
            data_quality_pct=dq_completeness,
            hist_played=hist_played,
        )
        m = analysis.model
        market_ctx = compute_market_context(m, analysis.team1, analysis.team2, odds_event)

        if can_publish_ev(dq, odds_flags) and odds_event and market_ctx.has_market:
            guard = check_ev_guardrails(self.db, self.settings)
            if guard.allowed:
                ev_opps = compute_ev_opportunities(
                    analysis.model, analysis.team1, analysis.team2, odds_event
                )
                ev_opps = [
                    o for o in ev_opps if o.edge_fair >= self.settings.ev_min_edge_fair
                ][: self.settings.ev_max_daily_picks]

        injury = await fetch_injury_report(analysis.team1, analysis.team2)
        sharp = run_sharp_engine(
            analysis,
            ev_opps,
            market_ctx=market_ctx,
            injury_report=injury,
            data_quality_pct=dq_completeness,
            hist_played=hist_played,
            historical_accuracy=self._load_historical_accuracy(),
            settings=self.settings,
        )
        return analysis, sharp, quality_note

    def _format_sharp_scan_line(self, analysis, sharp: SharpBetResult) -> str:
        dec = sharp.decision
        pick = dec.pick
        if not pick:
            return ""
        label = pick.selection
        if pick.selection == analysis.team1:
            label = f"{analysis.team1} gana"
        elif pick.selection == analysis.team2:
            label = f"{analysis.team2} gana"
        trust_s = ""
        if dec.trust:
            trust_s = (
                f" | arb {dec.trust.trust_side} "
                f"({dec.trust.model_confidence:.0%}/{dec.trust.market_confidence:.0%})"
            )
        return (
            f"⚽ {analysis.team1} vs {analysis.team2}\n"
            f"   🎯 {label} | tier {sharp.portfolio_tier} | "
            f"EV {sharp.ev_final*100:+.1f}% | MDS {sharp.mds}{trust_s}\n"
            f"   💵 Stake {dec.stake_pct:g}% | {dec.classification}"
        )

    async def _alta_message(self) -> str:
        """Escanea partidos próximos y lista singles SHARP aprobados."""
        d26, d22, d18, fd_matches = await self._load_worldcup_data()
        elo_ratings = await get_wc_elo_ratings(self.db)
        factors = load_calibration_factors_from_db(self.db)
        if factors:
            set_calibration_factors(factors)

        upcoming = find_upcoming_matches(d26, days_ahead=14)
        if not upcoming:
            return "No hay partidos próximos para escanear SHARP."

        scan_limit = self.settings.parlay_max_matches_scan
        approved: list[tuple[Any, SharpBetResult]] = []
        watch: list[tuple[Any, SharpBetResult]] = []

        for match in upcoming[:scan_limit]:
            analysis, sharp, _ = await self._run_sharp_for_match(
                match, d18=d18, d22=d22, fd_matches=fd_matches, elo_ratings=elo_ratings
            )
            if sharp is None:
                continue
            if sharp.sharp_allowed:
                approved.append((analysis, sharp))
                self._persist_sharp_pick(analysis, sharp)
            elif sharp.decision.soft_action == "WATCH":
                watch.append((analysis, sharp))

        approved.sort(key=lambda x: (-x[1].mds, -x[1].ev_final))

        health = evaluate_engine_health(self.db, settings=self.settings)
        lines = [
            f"🟢 {ENGINE_VERSION_TAG}",
            "💎 SHARP ENGINE — singles aprobados",
            f"Escaneados: {min(len(upcoming), scan_limit)} partidos",
            "",
        ]
        banner = health.banner_line()
        if banner:
            lines.insert(1, banner)
            lines.insert(2, "")

        if approved:
            lines.append(f"✅ Aprobados ({len(approved)})")
            for analysis, sharp in approved[:5]:
                lines.append(self._format_sharp_scan_line(analysis, sharp))
                lines.append("")
        else:
            lines.append("❌ Ningún single SHARP aprobado hoy.")
            lines.append(
                f"   (EV≥{self.settings.ev_min_edge_fair:.0%}, "
                f"MDS≥{self.settings.sharp_min_mds}, "
                f"conf≥{self.settings.sharp_min_confidence:.0%})"
            )
            lines.append("")

        if watch:
            lines.append(f"👁️ Vigilar ({len(watch)}) — stake 0%")
            for analysis, sharp in watch[:3]:
                dec = sharp.decision
                pick = dec.pick
                sel = pick.selection if pick else "—"
                lines.append(f"• {analysis.team1} vs {analysis.team2} → {sel} (MDS {sharp.mds})")
            lines.append("")

        lines.append("─────────────────")
        lines.append("Detalle completo: escribe «Czech Republic vs Mexico»")
        lines.append("🔵 Combinadas: /combinada o /parlay")
        lines.append("")
        lines.append("⚠️ Predicciones probabilísticas, no garantías.")
        return "\n".join(lines)

    async def _analyze_message(self, text: str, alta_only: bool) -> str:
        d26, d22, d18, fd_matches = await self._load_worldcup_data()
        elo_ratings = await get_wc_elo_ratings(self.db)

        t1, t2 = self._extract_teams(text)
        match = self._find_match(d26, t1, t2) if t1 and t2 else None

        if not match:
            upcoming = find_upcoming_matches(d26, days_ahead=14)
            if upcoming:
                match = upcoming[0]
            else:
                return "No encontré ese partido. Prueba /hoy o escribe: Colombia vs Brasil"

        factors = load_calibration_factors_from_db(self.db)
        if factors:
            set_calibration_factors(factors)

        analysis = analyze_match(match, d18, d22, fd_matches, elo_ratings)
        self._save_model_prediction(analysis)
        odds_event = await find_wc_odds_event(analysis.team1, analysis.team2, db=self.db)
        ev_opps: list = []
        quality_note = ""
        market_ctx = None
        dq_completeness = 100.0
        hist_played = 20

        if analysis.model:
            m = analysis.model
            hist_played = (
                (analysis.historico.get(analysis.team1, {}).get("wc2022", {}).get("played", 0) or 0)
                + (analysis.historico.get(analysis.team2, {}).get("wc2022", {}).get("played", 0) or 0)
            )
            form_n = len(analysis.forma.get(analysis.team1, [])) + len(
                analysis.forma.get(analysis.team2, [])
            )
            dq = check_match_features(
                lambda_home=m.lambda_home,
                lambda_away=m.lambda_away,
                elo_home=analysis.elo.get(analysis.team1, {}).get("rating"),
                elo_away=analysis.elo.get(analysis.team2, {}).get("rating"),
                form_matches=form_n,
                hist_played=hist_played,
            )
            odds_flags = check_odds_event(odds_event, min_books=self.settings.min_odds_books)
            dq_label = f"OK ({dq.completeness_pct:.0f}%)" if dq.status == "ok" else f"{dq.status.upper()} ({dq.completeness_pct:.0f}%)"
            quality_note = f"Datos {dq_label}"
            dq_completeness = dq.completeness_pct

            calibrate_analysis_model(
                analysis,
                odds_event,
                data_quality_pct=dq_completeness,
                hist_played=hist_played,
            )
            m = analysis.model
            market_ctx = compute_market_context(m, analysis.team1, analysis.team2, odds_event)
            if market_ctx.has_market:
                saved = record_wc_market_snapshots(
                    self.db,
                    team_home=analysis.team1,
                    team_away=analysis.team2,
                    market_ctx=market_ctx,
                )
                if saved:
                    logger.debug("odds_snapshots market: %s filas", saved)

            if can_publish_ev(dq, odds_flags):
                guard = check_ev_guardrails(self.db, self.settings)
                if guard.allowed:
                    ev_opps = compute_ev_opportunities(
                        analysis.model, analysis.team1, analysis.team2, odds_event
                    )
                    filtered = []
                    for o in ev_opps:
                        if o.edge_fair < self.settings.ev_min_edge_fair:
                            continue
                        stake = evaluate_pick(
                            model_prob=o.model_prob,
                            fair_odds=o.fair_odds,
                            edge_fair=o.edge_fair,
                            ev_fair=o.expected_value,
                            fair_implied=1.0 / o.fair_odds if o.fair_odds > 1 else 0,
                        )
                        if not stake.allowed:
                            log_anomaly_to_db(
                                self.db,
                                f"{analysis.team1} vs {analysis.team2}",
                                stake.flags,
                                {"market": o.market, "selection": o.selection},
                            )
                            continue
                        o.metadata["kelly_stake"] = stake.stake_units
                        o.metadata["edge_fair"] = o.edge_fair
                        filtered.append(o)
                    ev_opps = filtered[: self.settings.ev_max_daily_picks]
                else:
                    quality_note += f"\nEV bloqueado: guardrail ({', '.join(guard.reasons)})"

            if not odds_event or not market_ctx.has_market:
                block = await odds_unavailable_reason(self.db)
                quality_note += f"\nEV bloqueado: {block}"
            elif not can_publish_ev(dq, odds_flags):
                quality_note += "\nEV bloqueado por calidad insuficiente"
            elif not ev_opps:
                quality_note += "\nEV bloqueado: sin valor positivo"

        if not analysis.model:
            return f"⚽ {analysis.team1} vs {analysis.team2}\n\nSin modelo disponible."

        for o in ev_opps:
            kelly = (o.metadata or {}).get("kelly_stake", 0)
            pred_id = save_wc_prediction(
                self.db,
                team_home=analysis.team1,
                team_away=analysis.team2,
                match_date=(analysis.fecha or "")[:10] or None,
                market_type=o.market,
                predicted_outcome=o.selection,
                probability=o.model_prob,
                expected_value_fair=o.expected_value,
                edge_fair=o.edge_fair,
                kelly_stake=kelly,
                metadata={
                    "priority": o.priority,
                    "fair_odds": o.fair_odds,
                    "model_1x2": {
                        "home_win": analysis.model.home_win,
                        "draw": analysis.model.draw,
                        "away_win": analysis.model.away_win,
                    },
                    "blend_meta": analysis.model.blend_meta,
                    "wc_features": {
                        "lambda_home": analysis.model.lambda_home,
                        "lambda_away": analysis.model.lambda_away,
                        "xg_home": analysis.xg.get(analysis.team1, 0),
                        "xg_away": analysis.xg.get(analysis.team2, 0),
                    },
                },
            )
            if o.raw_odds:
                record_pick_snapshot(
                    self.db,
                    team_home=analysis.team1,
                    team_away=analysis.team2,
                    market=o.market,
                    selection=o.selection,
                    odds_decimal=o.raw_odds,
                    fair_odds=o.fair_odds,
                    prediction_id=pred_id,
                )

        card = build_trading_card(
            analysis,
            ev_opps,
            odds_available=bool(odds_event),
            market_ctx=market_ctx,
            injury_report=await fetch_injury_report(analysis.team1, analysis.team2),
            data_quality_pct=dq_completeness,
            hist_played=hist_played,
            historical_accuracy=self._load_historical_accuracy(),
        )
        if card.market_divergence_flag and card.no_bet:
            quality_note += (
                f"\nEV bloqueado: filtro mercado (capa extrema, "
                f"Δ {card.max_divergence*100:.0f}%)"
            )
            if card.diagnosis:
                quality_note += f"\nDiagnóstico: {card.diagnosis.label}"
            if card.dominance:
                quality_note += (
                    f"\nDominance: {card.dominance.classification} "
                    f"(model={card.dominance.model_reliability:.2f}, "
                    f"market={card.dominance.market_reliability:.2f})"
                )
            if card.decision and card.decision.tree_path:
                quality_note += f"\nÁrbol: {' → '.join(card.decision.tree_path[-4:])}"
        return format_trading_message(
            card,
            quality_note=quality_note,
            alta_header=alta_only,
        )

    def _load_historical_accuracy(self) -> float | None:
        """Precisión histórica del modelo (1X2) desde métricas en DB."""
        try:
            rows = (
                self.db.schema("ml")
                .table("model_performance_metrics")
                .select("accuracy, sample_size")
                .eq("market_type", "1X2")
                .order("computed_at", desc=True)
                .limit(1)
                .execute()
            )
            row = (rows.data or [None])[0]
            if row and (row.get("sample_size") or 0) >= 20:
                acc = row.get("accuracy")
                if acc is not None:
                    return float(acc)
        except Exception:
            pass
        return None

    async def _combinada_message(self) -> str:
        """Escanea partidos — PARLAY v3 solo desde picks SHARP validados."""
        d26, d22, d18, fd_matches = await self._load_worldcup_data()
        elo_ratings = await get_wc_elo_ratings(self.db)
        factors = load_calibration_factors_from_db(self.db)
        if factors:
            set_calibration_factors(factors)

        upcoming = find_upcoming_matches(d26, days_ahead=14)
        if not upcoming:
            return "No hay partidos próximos para armar combinadas."

        scan_limit = self.settings.parlay_max_matches_scan
        sharp_picks = []

        for match in upcoming[:scan_limit]:
            analysis, sharp, _ = await self._run_sharp_for_match(
                match, d18=d18, d22=d22, fd_matches=fd_matches, elo_ratings=elo_ratings
            )
            if sharp is None or not analysis.model:
                continue
            odds_event = await find_wc_odds_event(analysis.team1, analysis.team2, db=self.db)
            market_ctx = compute_market_context(
                analysis.model, analysis.team1, analysis.team2, odds_event
            )
            ev_opps = []
            if odds_event and market_ctx.has_market:
                ev_opps = compute_ev_opportunities(
                    analysis.model, analysis.team1, analysis.team2, odds_event, single_best=False
                )
            sp = extract_sharp_parlay_pick(analysis, sharp, market_ctx, ev_opps)
            if sp:
                sharp_picks.append(sp)

        result = build_parlays_from_sharp_picks(sharp_picks)
        if result.tickets:
            t = result.tickets[0]
            save_parlay_ticket(
                self.db,
                legs=[
                    {
                        "team1": l.team1,
                        "team2": l.team2,
                        "fecha": l.fecha,
                        "selection": l.selection,
                        "model_prob": l.model_prob,
                        "market_prob": l.market_prob,
                        "odds": l.odds,
                        "pick_score": l.pick_score,
                    }
                    for l in t.legs
                ],
                combined_prob=t.combined_prob,
                combined_odds=t.combined_odds,
                ev_parlay=t.ev_parlay,
                combo_score=t.combo_score,
                correlation_penalty=t.correlation_penalty,
                stake_pct=t.stake_pct,
                n_legs=t.n_legs,
            )
        return format_parlay_message(result)

    def _extract_teams(self, text: str) -> tuple[str, str]:
        t = text.strip()
        for sep in (" vs ", " v ", " - "):
            if sep in t.lower():
                parts = re.split(re.escape(sep), t, flags=re.I, maxsplit=1)
                if len(parts) == 2:
                    return parts[0].strip().title(), parts[1].strip().title()
        return "", ""

    def _find_match(self, data_2026: dict, t1: str, t2: str) -> dict | None:
        from apps.api.services.worldcup_engine import normalize_openfootball

        d26 = normalize_openfootball(data_2026)
        for rnd in d26.get("rounds", []):
            for m in rnd.get("matches", []):
                a = m.get("team1", {}).get("name", "")
                b = m.get("team2", {}).get("name", "")
                if (name_match(a, t1) and name_match(b, t2)) or (
                    name_match(a, t2) and name_match(b, t1)
                ):
                    return {**m, "roundName": rnd.get("name")}
        return None

    def _analysis_to_dict(self, a, ev_opps, odds_event) -> dict:
        m = a.model
        return {
            "partido": f"{a.team1} vs {a.team2}",
            "fecha": a.fecha,
            "ronda": a.ronda,
            "grupo": a.grupo,
            "modelo": {
                "1x2": {
                    a.team1: m.home_win if m else None,
                    "empate": m.draw if m else None,
                    a.team2: m.away_win if m else None,
                },
                "over_25": m.over_25 if m else None,
                "under_25": m.under_25 if m else None,
                "btts_si": m.btts_yes if m else None,
                "lambda_home": m.lambda_home if m else None,
                "lambda_away": m.lambda_away if m else None,
                "confianza": m.confidence if m else None,
            },
            "elo": a.elo,
            "xg": a.xg,
            "forma": a.forma,
            "historico": a.historico,
            "local_visitante": a.local_visitante,
            "mercado_casas": {
                "disponible": bool(odds_event),
                "nota": "Solo para EV; NO modifica el modelo",
            },
            "oportunidades_ev": [
                {
                    "mercado": o.market,
                    "seleccion": o.selection,
                    "prob_modelo": o.model_prob,
                    "cuota_fair": o.fair_odds,
                    "cuota_bruta": o.raw_odds,
                    "vig_pct": o.vig_pct,
                    "ev_fair": o.expected_value,
                    "ev_bruto": o.expected_value_raw,
                    "prioridad": o.priority,
                }
                for o in ev_opps
            ],
        }

    async def _format_with_llm(self, user_text: str, data: dict) -> str:
        """Siempre formato trading (sin LLM) para decisiones de apuesta consistentes."""
        card = build_trading_card_from_dict(data)
        return format_trading_message(card)

    def _template_analysis(self, data: dict) -> str:
        card = build_trading_card_from_dict(data)
        return format_trading_message(card)

    async def _stats_message(self) -> str:
        """Dashboard cuant: Brier, CLV, ROI, tiers SHARP."""
        try:
            return build_pro_stats_report(self.db, settings=self.settings)
        except Exception as exc:
            logger.warning("stats_report: %s", exc)
            return (
                "📊 Stats — error generando reporte.\n"
                f"Detalle: {exc}\n"
                "Revisa logs o ejecuta: python scripts/run_learning_cycle.py"
            )

    async def _roi_message(self) -> str:
        """ROI live SHARP/+EV y backtest histórico."""
        try:
            return build_roi_report(self.db, settings=self.settings)
        except Exception as exc:
            logger.warning("roi_report: %s", exc)
            return f"💰 ROI — error generando reporte.\nDetalle: {exc}"

    async def _save_session(self, chat_id: str, text: str, intent: str) -> None:
        chat_hash = hashlib.sha256(str(chat_id).encode()).hexdigest()
        try:
            existing = (
                self.db.table("telegram_sessions")
                .select("id")
                .eq("chat_hash", chat_hash)
                .limit(1)
                .execute()
            )
            if existing.data:
                sid = existing.data[0]["id"]
                self.db.table("telegram_sessions").update(
                    {"last_intent": intent, "context": {"last_text": text}}
                ).eq("id", sid).execute()
            else:
                ins = (
                    self.db.table("telegram_sessions")
                    .insert({"chat_hash": chat_hash, "last_intent": intent, "context": {"chat_id": chat_id}})
                    .execute()
                )
                sid = ins.data[0]["id"] if ins.data else None
            if sid:
                self.db.table("telegram_messages").insert(
                    {"session_id": sid, "direction": "inbound", "content": text}
                ).execute()
        except Exception as exc:
            logger.warning("telegram session save: %s", exc)
