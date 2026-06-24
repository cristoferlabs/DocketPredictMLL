#!/usr/bin/env bash
# Railway start command for API service
exec uvicorn apps.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
