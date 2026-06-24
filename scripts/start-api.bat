@echo off
cd /d "%~dp0.."
python -m uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
