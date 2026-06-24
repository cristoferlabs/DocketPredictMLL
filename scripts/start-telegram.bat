@echo off
cd /d "%~dp0.."
echo ============================================
echo  Telegram polling (modo poll — dev local)
echo  No ejecutar junto con n8n Telegram Trigger
echo  en el mismo bot. Usa TELEGRAM_INGESTION_MODE.
echo ============================================
findstr /B /C:"TELEGRAM_INGESTION_MODE=n8n" .env >nul 2>&1
if %ERRORLEVEL%==0 (
  echo.
  echo ERROR: TELEGRAM_INGESTION_MODE=n8n en .env
  echo Usa n8n con telegram_inbound + API. No ejecutes este script.
  echo Para polling local: TELEGRAM_INGESTION_MODE=poll en .env
  echo.
  exit /b 1
)
python scripts/telegram_poll.py
