"""WhatsApp webhook handlers (Meta Cloud API via n8n)."""

import hashlib
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from apps.api.deps import get_arq_pool, get_db
from apps.api.services.agent import AgentService
from apps.shared.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class WhatsAppInboundPayload(BaseModel):
    """Normalized payload from n8n after Meta webhook processing."""

    phone: str = Field(..., description="Sender phone number (E.164)")
    message_id: str | None = None
    text: str = Field(..., description="User message text")
    timestamp: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class WhatsAppResponse(BaseModel):
    phone: str
    message: str
    combinations: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str | None = None


@router.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
):
    """Meta webhook verification (can be called directly or via n8n)."""
    settings = get_settings()
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return int(hub_challenge) if hub_challenge.isdigit() else hub_challenge
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp", response_model=WhatsAppResponse)
async def whatsapp_inbound(
    payload: WhatsAppInboundPayload,
    request: Request,
    db=Depends(get_db),
):
    """Receive normalized WhatsApp message from n8n and return agent response."""
    phone_hash = hashlib.sha256(payload.phone.encode()).hexdigest()

    session_result = (
        db.table("whatsapp_sessions")
        .select("id, context")
        .eq("phone_hash", phone_hash)
        .limit(1)
        .execute()
    )

    if session_result.data:
        session = session_result.data[0]
        session_id = session["id"]
        context = session.get("context") or {}
    else:
        insert_result = (
            db.table("whatsapp_sessions")
            .insert({"phone_hash": phone_hash, "context": {}})
            .execute()
        )
        session = insert_result.data[0]
        session_id = session["id"]
        context = {}

    db.table("whatsapp_messages").insert(
        {
            "session_id": session_id,
            "direction": "inbound",
            "content": payload.text,
            "api_message_id": payload.message_id,
        }
    ).execute()

    agent = AgentService(db)
    response_text, combinations, intent = await agent.handle_message(payload.text, context)

    db.table("whatsapp_messages").insert(
        {
            "session_id": session_id,
            "direction": "outbound",
            "content": response_text,
        }
    ).execute()

    db.table("whatsapp_sessions").update(
        {"last_intent": intent, "context": {**context, "last_query": payload.text}}
    ).eq("id", session_id).execute()

    arq_pool = get_arq_pool(request)
    if arq_pool and intent == "predict":
        await arq_pool.enqueue_job("predict_upcoming_matches")

    return WhatsAppResponse(
        phone=payload.phone,
        message=response_text,
        combinations=combinations,
        session_id=session_id,
    )
