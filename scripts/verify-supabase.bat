@echo off
cd /d "%~dp0.."
python scripts\verify_supabase.py
exit /b %ERRORLEVEL%
