@echo off
cd /d "%~dp0.."
echo ============================================
echo  Tunel publico para API local (n8n Cloud)
echo  n8n Cloud NO puede usar 127.0.0.1:8000
echo ============================================
echo.
echo Requisitos:
echo   1. API corriendo: scripts\start-api.bat
echo   2. cloudflared: winget install Cloudflare.cloudflared
echo.
echo Copia la URL https://....trycloudflare.com que aparezca abajo
echo y pegala en n8n -^> workflow telegram_inbound -^> nodo "Config y Normalize":
echo   const API_BASE_URL = 'ESA_URL_AQUI';
echo (Tu plan no tiene Variables; edita el nodo Code, no Settings)
echo Luego re-publica telegram_inbound.
echo ============================================
echo.
set "CF=cloudflared"
where cloudflared >nul 2>&1
if errorlevel 1 (
  set "CF=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
)
if not exist "%CF%" (
  echo ERROR: cloudflared no encontrado.
  exit /b 1
)
"%CF%" tunnel --url http://127.0.0.1:8000
