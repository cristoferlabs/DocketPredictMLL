"""FastAPI dependencies."""

from fastapi import Request

from apps.shared.supabase_client import get_supabase


def get_db():
    return get_supabase()


def get_arq_pool(request: Request):
    return request.app.state.arq_pool
