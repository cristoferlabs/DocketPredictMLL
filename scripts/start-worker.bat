@echo off
cd /d "%~dp0.."
echo Comprobando Redis en localhost:6379...
python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',6379)); s.close(); print('Redis OK')" 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Redis no esta corriendo en localhost:6379
  echo.
  echo Opciones en Windows:
  echo   1. Docker:  docker compose up -d redis
  echo   2. Memurai: https://www.memurai.com/ ^(gratis para desarrollo^)
  echo   3. WSL:     sudo apt install redis-server ^&^& redis-server
  echo.
  exit /b 1
)
python -m arq apps.worker.main.WorkerSettings
