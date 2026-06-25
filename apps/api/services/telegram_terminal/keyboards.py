"""Teclados inline del betting terminal."""

from __future__ import annotations

from apps.api.services.engine_constants import ENGINE_VERSION_TAG


def exploration_keyboard(matches: list[dict]) -> dict:
    rows = []
    for i, m in enumerate(matches[:12]):
        t1 = m.get("team1", "TBD")
        t2 = m.get("team2", "TBD")
        rows.append([{"text": f"{i + 1}. {t1} vs {t2}", "callback_data": f"t:m:{i}"}])
    return {"inline_keyboard": rows}


def dashboard_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "🎯 Ver opciones", "callback_data": "t:o"},
                {"text": "💎 Combinadas", "callback_data": "t:p"},
            ],
            [
                {"text": "🔬 Análisis técnico", "callback_data": "t:a"},
                {"text": "📅 Volver a /hoy", "callback_data": "t:hoy"},
            ],
        ]
    }


def subview_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "◀ Dashboard", "callback_data": "t:d"},
                {"text": "📅 /hoy", "callback_data": "t:hoy"},
            ],
            [
                {"text": "🎯 Opciones", "callback_data": "t:o"},
                {"text": "💎 Combinadas", "callback_data": "t:p"},
                {"text": "🔬 Análisis", "callback_data": "t:a"},
            ],
        ]
    }


def terminal_header() -> str:
    return f"🖥️ BETTING TERMINAL v2 — {ENGINE_VERSION_TAG}"
