"""LLM client — Ollama (gratis, local) con fallback a plantillas."""

import logging
from typing import Protocol

import httpx

from apps.shared.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def chat(self, system: str, user: str) -> str | None: ...


class OllamaClient:
    """Cliente para Ollama (https://ollama.com) — sin costo, corre en local o en tu servidor."""

    def __init__(self, base_url: str, model: str, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def chat(self, system: str, user: str) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 800},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                return content or None
        except httpx.ConnectError:
            logger.warning(
                "Ollama no disponible en %s. ¿Está corriendo? Ej: ollama serve",
                self.base_url,
            )
            return None
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return None

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


class TemplateLLMClient:
    """Sin LLM externo — siempre devuelve None para usar fallback del agente."""

    async def chat(self, system: str, user: str) -> str | None:
        return None


def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.llm_timeout,
        )
    return TemplateLLMClient()
