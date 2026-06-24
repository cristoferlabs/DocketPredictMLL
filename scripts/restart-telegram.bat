@echo off
cd /d "%~dp0.."

echo ============================================
echo  Reiniciar Telegram polling (ENGINE v2)
echo  Desactiva en n8n: "Bot Interactivo - Mundial 2026"
echo ============================================

for /f "tokens=2 delims=="" skip=1" %%P in ('wmic process where "CommandLine like '%%telegram_poll%%'" get ProcessId /format:list 2^>nul ^| find "="') do (
  if not "%%P"=="" taskkill /PID %%P /F >nul 2>&1
)
if exist .telegram_poll.lock del /f .telegram_poll.lock >nul 2>&1

echo.
echo Iniciando polling local (TELEGRAM_INGESTION_MODE=poll)...
start "telegram-poll" cmd /k scripts\start-telegram.bat
