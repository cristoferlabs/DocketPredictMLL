"""Telegram agent — Mundial 2026 with model-first predictions and separate odds EV."""

import hashlib
import json
import logging
import re
from typing import Any

from supabase import Client

from apps.api.services.llm import get_llm_client
from apps.api.services.odds_context import compute_ev_opportunities, find_wc_odds_event
from apps.api.services.telegram_client import TelegramClient
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
from apps.worker.ml.clv import record_pick_snapshot
from apps.worker.ml.data_quality import can_publish_ev, check_match_features, check_odds_event
from apps.worker.ml.ev_anomaly import evaluate_pick, log_anomaly_to_db
from apps.worker.ml.guardrails import check_ev_guardrails
from apps.worker.ml.model_loader import load_calibration_factors_from_db
from apps.worker.ml.wc_audit import audit_upcoming_matches
from apps.worker.ml.wc_predictions import save_wc_prediction
from apps.api.services.trading_card import build_trading_card, format_trading_message

logger = logging.getLogger(__name__)

TELEGRAM_SYSTEM = """Eres el analista del Mundial 2026. Respondes en español, texto plano y emojis.
REGLAS CRITICAS:
- Las probabilidades del MODELO (Poisson+ELO) son la fuente de verdad para predicciones.
- Las cuotas de casas de apuestas son SOLO contexto de mercado para calcular valor esperado (EV).
- NUNCA ajustes las probabilidades del modelo para alinearlas con las cuotas.
- Si modelo y mercado difieren, explica la divergencia; el modelo manda, el mercado informa precio.
- Formato: secciones con líneas ─────, sin markdown (sin ** ni |).
- Mercados: 1X2, Over/Under 2.5, BTTS.
- Incluye disclaimer probabilístico al final."""


class TelegramAgentService:
    def __init__(self, db: Client):
        self.db = db
        self.llm = get_llm_client()
        self.telegram = TelegramClient()
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
        if t.startswith("/stats"):
            return "stats"
        if t.startswith("/hoy") or t.startswith("/partidos"):
            return "today"
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

        if callback_id:
            await self.telegram.answer_callback_query(callback_id)

        if callback_data and callback_data.startswith("wc:"):
            teams = callback_data.replace("wc:", "").split("|")
            if len(teams) >= 2:
                text = f"analiza {teams[0]} vs {teams[1]}"

        command = self._detect_command(text)
        await self._save_session(chat_id, text, command)

        if command == "help":
            msg = self._help_text()
        elif command == "stats":
            msg = await self._stats_message()
        elif command == "today":
            msg, markup = await self._today_matches_menu()
            await self.telegram.send_message(chat_id, msg, reply_markup=markup)
            return {"ok": True, "chat_id": chat_id, "sent": True}
        elif command in ("analyze", "alta", "general"):
            msg = await self._analyze_message(text, command == "alta")
        else:
            msg = self._help_text()

        await self.telegram.send_message(chat_id, msg)
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
            "⚽ Agente Mundial 2026\n\n"
            "Comandos:\n"
            "/hoy — Partidos próximos\n"
            "/alta — Oportunidades con EV fair positivo (modelo > mercado sin vig)\n"
            "/stats — Métricas de calibración, calidad de datos y guardrails\n"
            "Colombia vs Brasil — Análisis con pick, EV, semáforo y stake\n\n"
            "El modelo usa Poisson + ELO + datos WC 2018/2022/2026.\n"
            "Las cuotas de casas NO modifican el modelo; solo calculan EV.\n\n"
            "⚠️ Predicciones probabilísticas, no garantías."
        )

    async def _today_matches_menu(self) -> tuple[str, dict | None]:
        d26, d22, d18, _ = await self._load_worldcup_data()
        upcoming = find_upcoming_matches(d26, days_ahead=7)
        if not upcoming:
            return "No hay partidos del Mundial en los próximos 7 días.", None

        rows = []
        for m in upcoming[:8]:
            t1 = m.get("team1", {}).get("name", "TBD")
            t2 = m.get("team2", {}).get("name", "TBD")
            fecha = (m.get("date") or "")[:10]
            rows.append(
                [{"text": f"{t1} vs {t2}", "callback_data": f"wc:{t1}|{t2}|{fecha}"}]
            )

        return (
            f"📅 Partidos Mundial próximos ({len(upcoming)} encontrados)\nToca un partido:",
            {"inline_keyboard": rows},
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
            },
        )

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
        odds_event = await find_wc_odds_event(analysis.team1, analysis.team2)
        ev_opps: list = []
        quality_note = ""

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
            quality_note = f"datos: {dq.status} ({dq.completeness_pct}%)"

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
                    quality_note += f" | guardrail: {', '.join(guard.reasons)}"
            else:
                quality_note += " | EV bloqueado por calidad insuficiente"

        if not analysis.model:
            return f"⚽ {analysis.team1} vs {analysis.team2}\n\nSin modelo disponible."

        for o in ev_opps:
            kelly = (o.metadata or {}).get("kelly_stake", 0)
            save_wc_prediction(
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
                )

        card = build_trading_card(
            analysis, ev_opps, odds_available=bool(odds_event)
        )
        return format_trading_message(
            card,
            quality_note=quality_note,
            alta_header=alta_only,
        )

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
        payload = json.dumps({"consulta": user_text, "datos": data}, ensure_ascii=False)
        if self.settings.llm_provider != "template":
            reply = await self.llm.chat(TELEGRAM_SYSTEM, payload)
            if reply:
                return reply
        return self._template_analysis(data)

    def _template_analysis(self, data: dict) -> str:
        m = data.get("modelo", {})
        x12 = m.get("1x2", {})
        lines = [
            f"⚽ {data.get('partido')}",
            f"📅 {data.get('fecha')} | {data.get('ronda')}",
            "─────────────────",
            "📐 MODELO (Poisson + ELO)",
        ]
        for k, v in x12.items():
            if v is not None:
                lines.append(f"  {k}: {float(v)*100:.1f}%")
        if m.get("over_25"):
            lines.append(f"  Over 2.5: {float(m['over_25'])*100:.1f}%")
        if m.get("btts_si"):
            lines.append(f"  BTTS Sí: {float(m['btts_si'])*100:.1f}%")
        evs = data.get("oportunidades_ev", [])
        if evs:
            lines.append("─────────────────")
            lines.append("💎 Valor vs mercado (EV fair)")
            for o in evs[:3]:
                vig = o.get("vig_pct", 0)
                lines.append(
                    f"  {o['mercado']} {o['seleccion']}: "
                    f"EV fair {o['ev_fair']*100:+.1f}% "
                    f"(bruto {o.get('ev_bruto', 0)*100:+.1f}%, vig {vig:.1f}%) "
                    f"[{o['prioridad']}]"
                )
        lines.append("\n⚠️ Predicciones probabilísticas, no garantías.")
        return "\n".join(lines)

    async def _stats_message(self) -> str:
        """Model calibration and data quality summary."""
        lines = ["📊 Estadísticas del modelo", "─────────────────"]

        factors = get_calibration_factors()
        active_markets = [k for k, v in factors.items() if any(x != 1.0 for x in v.values())]
        if active_markets:
            lines.append(f"Calibración activa: {', '.join(active_markets)}")
        else:
            lines.append("Calibración: identidad (pendiente fit histórico)")

        try:
            snap = (
                self.db.schema("ml")
                .table("calibration_snapshots")
                .select("ece, brier, hit_rate, sample_size, market, created_at")
                .order("created_at", desc=True)
                .limit(3)
                .execute()
            )
            if snap.data:
                lines.append("\nÚltimas métricas (Supabase):")
                for row in snap.data:
                    lines.append(
                        f"  {row.get('market')}: ECE {float(row.get('ece', 0)):.3f} | "
                        f"Brier {row.get('brier') or '—'} | n={row.get('sample_size', 0)}"
                    )
        except Exception as exc:
            logger.debug("calibration_snapshots: %s", exc)
            lines.append("\nMétricas Supabase: no disponibles (ejecuta migración)")

        lines.append("\nGuardrails activos:")
        lines.append(f"  EV min edge fair: {self.settings.ev_min_edge_fair*100:.1f}%")
        lines.append(f"  EV max edge fair: {self.settings.ev_max_edge_fair*100:.1f}%")
        lines.append(f"  Kelly fracción: {self.settings.kelly_fraction}")
        lines.append(f"  ECE máx: {self.settings.ev_max_ece}")
        lines.append(f"  ROI backtest mín: {self.settings.ev_min_roi_backtest}")
        lines.append(f"  Casas mínimas (aviso): {self.settings.min_odds_books}")

        try:
            dq_log = (
                self.db.schema("ops")
                .table("data_quality_log")
                .select("status, completeness_pct, flags, created_at")
                .eq("context", "wc_audit")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if dq_log.data:
                row = dq_log.data[0]
                pct = row.get("completeness_pct")
                pct_str = f"{float(pct):.1f}" if pct is not None else "—"
                lines.append(f"\nAuditoría datos WC: {row.get('status')} ({pct_str}% completitud)")
                flags = row.get("flags") or []
                for f in flags[:3]:
                    lines.append(f"  • {f.get('message', f.get('code', ''))}")
            else:
                live = await audit_upcoming_matches(db=self.db, days_ahead=14)
                lines.append(
                    f"\nAuditoría datos WC (en vivo): "
                    f"{live.status_ok} ok / {live.status_partial} parcial / "
                    f"{live.status_insufficient} insuficiente "
                    f"({live.avg_completeness_pct}% avg)"
                )
        except Exception as exc:
            logger.debug("data quality stats: %s", exc)

        try:
            elo_n = (
                self.db.schema("ml")
                .table("wc_team_elo")
                .select("id", count="exact")
                .execute()
            )
            pred_n = (
                self.db.schema("ml")
                .table("wc_predictions")
                .select("id", count="exact")
                .execute()
            )
            lines.append(
                f"\nBase aprendizaje: {elo_n.count or 0} filas ELO | "
                f"{pred_n.count or 0} predicciones WC"
            )
        except Exception:
            pass

        lines.append("\nScripts: audit_ev.py | audit_data.py | run_update_elo.py")
        return "\n".join(lines)

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
