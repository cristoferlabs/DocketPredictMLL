"""Sesión Telegram — estado de navegación del terminal."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from supabase import Client

from apps.api.services.telegram_terminal.states import TerminalState

logger = logging.getLogger(__name__)


def _chat_hash(chat_id: str) -> str:
    return hashlib.sha256(str(chat_id).encode()).hexdigest()


def default_context() -> dict[str, Any]:
    return {
        "state": TerminalState.EXPLORATION.value,
        "match_key": None,
        "matches_cache": [],
    }


def load_session(db: Client, chat_id: str) -> tuple[str | None, dict[str, Any]]:
    """Devuelve (session_id, context)."""
    ctx = default_context()
    try:
        row = (
            db.table("telegram_sessions")
            .select("id, context")
            .eq("chat_hash", _chat_hash(chat_id))
            .limit(1)
            .execute()
        )
        if row.data:
            sid = row.data[0]["id"]
            stored = row.data[0].get("context") or {}
            if isinstance(stored, dict):
                ctx.update({k: v for k, v in stored.items() if k != "chat_id"})
            return sid, ctx
    except Exception as exc:
        logger.warning("telegram session load: %s", exc)
    return None, ctx


def save_session(
    db: Client,
    chat_id: str,
    *,
    session_id: str | None,
    context: dict[str, Any],
    intent: str = "terminal",
    inbound_text: str = "",
) -> str | None:
    payload = {**context, "chat_id": chat_id}
    try:
        if session_id:
            db.table("telegram_sessions").update(
                {"last_intent": intent, "context": payload}
            ).eq("id", session_id).execute()
        else:
            ins = (
                db.table("telegram_sessions")
                .insert(
                    {
                        "chat_hash": _chat_hash(chat_id),
                        "last_intent": intent,
                        "context": payload,
                    }
                )
                .execute()
            )
            session_id = ins.data[0]["id"] if ins.data else None
        if session_id and inbound_text:
            db.table("telegram_messages").insert(
                {"session_id": session_id, "direction": "inbound", "content": inbound_text}
            ).execute()
        return session_id
    except Exception as exc:
        logger.warning("telegram session save: %s", exc)
        return session_id


def match_key(team1: str, team2: str, fecha: str = "") -> str:
    return f"{team1}|{team2}|{fecha[:10]}"


def parse_match_key(key: str) -> tuple[str, str, str]:
    parts = (key or "").split("|", 2)
    if len(parts) >= 2:
        return parts[0], parts[1], parts[2] if len(parts) > 2 else ""
    return "", "", ""
