@echo off
setlocal
cd /d "%~dp0\.."

echo.
echo === Migracion WC: wc_team_elo, odds_snapshots, wc_predictions ===
echo.

where npx >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo [Opcion A] Supabase CLI via npx ^(requiere login + link^)
  echo   npx supabase login
  echo   npx supabase link --project-ref jwpdukjouvaygefeanfs
  echo   npx supabase db push
  echo.
)

echo [Opcion B] Script Python ^(recomendado^)
echo   1. Pon en .env: SUPABASE_DB_PASSWORD=tu_password_de_database
echo      ^(Dashboard - Project Settings - Database - Database password^)
echo   2. pip install psycopg[binary]
echo   3. python scripts/apply_sql_migration.py
echo.

echo [Opcion C] SQL Editor manual
echo   1. Abre: https://supabase.com/dashboard/project/jwpdukjouvaygefeanfs/sql/new
echo   2. Pega el contenido de:
echo      supabase\migrations\20250626100000_wc_elo_predictions.sql
echo   3. Run
echo.

set /p RUN=Ejecutar script Python ahora? (s/n): 
if /i "%RUN%"=="s" (
  python scripts/apply_sql_migration.py
  if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Si fallo, usa la Opcion C ^(SQL Editor^).
  )
)

endlocal
