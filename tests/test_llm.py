"""Tests for LLM client."""

import pytest

from apps.api.services.llm import OllamaClient, TemplateLLMClient, get_llm_client
from apps.shared.config import get_settings


@pytest.mark.asyncio
async def test_template_llm_returns_none():
    client = TemplateLLMClient()
    assert await client.chat("system", "user") is None


@pytest.mark.asyncio
async def test_ollama_unavailable_returns_none():
    client = OllamaClient("http://127.0.0.1:59999", "llama3.2", timeout=1.0)
    assert await client.chat("system", "user") is None


def test_get_llm_client_ollama_by_default():
    get_settings.cache_clear()
    client = get_llm_client()
    assert isinstance(client, OllamaClient)
