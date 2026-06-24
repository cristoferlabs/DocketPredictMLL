"""LLM agent service for WhatsApp conversational layer."""

import json
import logging
from typing import Any

from supabase import Client

from apps.api.services.llm import get_llm_client
from apps.shared.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un agente de predicciones de fútbol basado en modelos estadísticos
(Poisson, ELO, métricas de portero y XGBoost). Respondes en español, de forma concisa.

Reglas:
- Basa tus respuestas SOLO en los datos estructurados que recibes (predicciones, probabilidades, combinaciones).
- Indica siempre el nivel de confianza (alta/media/baja) y la prioridad de cada combinación.
- No inventes estadísticas ni resultados.
- Si no hay datos suficientes, dilo claramente y sugiere consultar más cerca del partido.
- Incluye disclaimer: las predicciones son probabilísticas, no garantías."""


class AgentService:
    def __init__(self, db: Client):
        self.db = db
        self.llm = get_llm_client()
        self.settings = get_settings()

    async def handle_message(
        self, text: str, context: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], str]:
        intent = self._detect_intent(text)
        structured = await self._fetch_context(text, intent)

        user_payload = json.dumps(
            {"user_message": text, "intent": intent, "data": structured},
            ensure_ascii=False,
        )

        message: str | None = None
        if self.settings.llm_provider != "template":
            message = await self.llm.chat(SYSTEM_PROMPT, user_payload)

        if not message:
            message = self._fallback_response(structured, intent)

        return message, structured.get("combinations", []), intent

    def _detect_intent(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ("predic", "pronost", "apuesta", "combo", "cuota")):
            return "predict"
        if any(w in lower for w in ("resultado", "ayer", "ganó", "perdió")):
            return "results"
        if any(w in lower for w in ("hola", "ayuda", "help", "qué puedes")):
            return "help"
        return "general"

    async def _fetch_context(self, text: str, intent: str) -> dict[str, Any]:
        words = [w for w in text.split() if len(w) > 2]
        search_term = " ".join(words[:3]) if words else text[:30]

        teams = (
            self.db.table("teams")
            .select("id, name")
            .ilike("name", f"%{search_term}%")
            .limit(5)
            .execute()
        )

        matches_data: list[dict] = []
        for team in teams.data or []:
            for col in ("home_team_id", "away_team_id"):
                rows = (
                    self.db.table("matches")
                    .select(
                        "id, kickoff_at, status, "
                        "home_team:teams!matches_home_team_id_fkey(name), "
                        "away_team:teams!matches_away_team_id_fkey(name)"
                    )
                    .eq(col, team["id"])
                    .in_("status", ["scheduled", "live"])
                    .order("kickoff_at")
                    .limit(3)
                    .execute()
                )
                matches_data.extend(rows.data or [])

        seen: set[str] = set()
        unique_matches = []
        for m in matches_data:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique_matches.append(m)

        predictions: list[dict] = []
        combinations: list[dict] = []
        for m in unique_matches[:3]:
            preds = (
                self.db.schema("ml")
                .table("predictions")
                .select("market_type, predicted_outcome, probability, confidence_tier")
                .eq("match_id", m["id"])
                .execute()
            )
            predictions.extend([{**p, "match_id": m["id"]} for p in preds.data or []])

            combos = (
                self.db.schema("ml")
                .table("betting_combinations")
                .select("priority, expected_value, kelly_fraction, betting_combination_legs(*)")
                .eq("match_id", m["id"])
                .execute()
            )
            combinations.extend(combos.data or [])

        return {
            "matches": unique_matches,
            "predictions": predictions,
            "combinations": combinations,
            "intent": intent,
        }

    def _fallback_response(self, structured: dict[str, Any], intent: str) -> str:
        if intent == "help":
            return (
                "Soy el agente de predicciones de fútbol. Puedes preguntarme por predicciones "
                "de equipos o partidos (ej: 'predicción Real Madrid vs Barcelona'). "
                "Uso modelos Poisson, ELO, GK y XGBoost. Las predicciones son probabilísticas."
            )
        matches = structured.get("matches", [])
        predictions = structured.get("predictions", [])
        if not matches:
            return (
                "No encontré partidos próximos para tu consulta. "
                "Intenta con el nombre exacto del equipo o liga."
            )
        lines = ["Predicciones disponibles:\n"]
        for m in matches[:2]:
            home = m.get("home_team", {}).get("name", "?")
            away = m.get("away_team", {}).get("name", "?")
            lines.append(f"{home} vs {away}")
            match_preds = [p for p in predictions if p.get("match_id") == m["id"]]
            for p in match_preds[:5]:
                prob = float(p["probability"]) * 100
                lines.append(
                    f"  - {p['market_type']}: {p['predicted_outcome']} "
                    f"({prob:.1f}%, confianza {p['confidence_tier']})"
                )
        lines.append("\nPredicciones probabilísticas, no garantías.")
        return "\n".join(lines)
