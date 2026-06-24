@echo off
cd /d "%~dp0.."

where docker >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] Docker no esta instalado o no esta en el PATH.
  echo.
  echo 1. Instala Docker Desktop: https://www.docker.com/products/docker-desktop/
  echo    O con winget: winget install Docker.DockerDesktop
  echo 2. Abre Docker Desktop y espera a que diga "Engine running"
  echo 3. Cierra y abre CMD de nuevo, luego ejecuta este script otra vez.
  echo.
  exit /b 1
)

echo Iniciando Redis en Docker (puerto 6379)...
docker compose up -d redis
if errorlevel 1 (
  echo.
  echo No se pudo levantar Redis. Asegurate de que Docker Desktop este abierto.
  exit /b 1
)

echo.
echo Redis listo en redis://localhost:6379
echo.
echo Siguiente paso: scripts\start-worker.bat
docker compose ps
