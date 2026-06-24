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

  echo.

  echo Opcion A - n8n ^(recomendado si ya usas n8n^):

  echo   1. Deja TELEGRAM_INGESTION_MODE=n8n

  echo   2. NO ejecutes este script

  echo   3. Activa workflow telegram_inbound en n8n + API en :8000

  echo.

  echo Opcion B - polling local ^(sin n8n^):

  echo   1. En .env cambia a: TELEGRAM_INGESTION_MODE=poll

  echo   2. Desactiva Telegram Trigger en n8n

  echo   3. Vuelve a ejecutar start-telegram.bat

  echo.

  exit /b 1

)

python scripts/telegram_poll.py

