@echo off
cd /d "%~dp0.."
docker compose stop redis
echo Redis detenido.
