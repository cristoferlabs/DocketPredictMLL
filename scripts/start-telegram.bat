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

  echo   2. Desactiva TODOS los workflows con Telegram Trigger en n8n:

  echo      - telegram_inbound

  echo      - Bot Interactivo - Mundial 2026

  echo      - RSL Engine - Flow 6: Bot Telegram Comandos

  echo   3. Vuelve a ejecutar start-telegram.bat

  echo.

  exit /b 1

)

REM Cierra instancias zombie de polling local (causan 409 entre ellas)
for /f "tokens=2 delims=="" skip=1" %%P in ('wmic process where "CommandLine like '%%telegram_poll%%'" get ProcessId /format:list 2^>nul ^| find "="') do (
  if not "%%P"=="" taskkill /PID %%P /F >nul 2>&1
)
if exist .telegram_poll.lock del /f .telegram_poll.lock >nul 2>&1

python scripts/telegram_poll.py

