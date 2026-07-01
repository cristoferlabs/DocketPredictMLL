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


def _cb(view: str, mk: str) -> str:
    """Build callback data: embed match_key so each button is self-contained."""
    return f"t:{view}:{mk}" if mk else f"t:{view}"


def dashboard_keyboard(match_key: str = "") -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "📊 Apuestas", "callback_data": _cb("e", match_key)},
                {"text": "🎯 Opciones EV", "callback_data": _cb("o", match_key)},
            ],
            [
                {"text": "🔄 Combos", "callback_data": _cb("c", match_key)},
                {"text": "🔬 Análisis", "callback_data": _cb("a", match_key)},
            ],
            [
                {"text": "💎 Parlays", "callback_data": _cb("p", match_key)},
                {"text": "📅 /hoy", "callback_data": "t:hoy"},
            ],
        ]
    }


def subview_keyboard(match_key: str = "") -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "◀ Dashboard", "callback_data": _cb("d", match_key)},
                {"text": "📅 /hoy", "callback_data": "t:hoy"},
            ],
            [
                {"text": "📊 Apuestas", "callback_data": _cb("e", match_key)},
                {"text": "🔄 Combos", "callback_data": _cb("c", match_key)},
                {"text": "🎯 Opciones", "callback_data": _cb("o", match_key)},
            ],
        ]
    }


def betting_menu_keyboard(match_key: str = "") -> dict:
    """Keyboard shown after the full betting menu (t:e view)."""
    return {
        "inline_keyboard": [
            [
                {"text": "🔄 Ver combos detalle", "callback_data": _cb("c", match_key)},
                {"text": "🎯 Opciones EV", "callback_data": _cb("o", match_key)},
            ],
            [
                {"text": "◀ Dashboard", "callback_data": _cb("d", match_key)},
                {"text": "📅 /hoy", "callback_data": "t:hoy"},
            ],
        ]
    }


def terminal_header() -> str:
    return f"🖥️ BETTING TERMINAL v2 — {ENGINE_VERSION_TAG}"
