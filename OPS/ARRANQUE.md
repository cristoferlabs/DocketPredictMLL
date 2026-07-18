# ARRANQUE DEL AGENTE — GUÍA DE OPERACIONES
**Proyecto:** DocketPredictMLL / MundialPredict2  
**Stack:** FastAPI · ARQ Worker · Redis · Supabase · Telegram · n8n

---

## ⚠️ IMPORTANTE — ORDEN DE ARRANQUE

> **Siempre en este orden. Si saltás un paso, el siguiente falla.**

```
1. Redis        →  2. Worker ARQ   →  3. API FastAPI   →  4. Telegram
```

---

## PASO 1 — REDIS (requerido por el Worker)

Redis es el broker de tareas del Worker ARQ. Sin él, el worker no arranca.

```bat
scripts\start-redis.bat
```

**Requiere Docker Desktop abierto y corriendo.**

Verificar que levantó:
```powershell
docker compose ps
# debe mostrar redis   running
```

Alternativa sin Docker (WSL):
```bash
sudo apt install redis-server && redis-server
```

---

## PASO 2 — WORKER ARQ (motor de tareas automáticas)

El worker ejecuta todos los cron jobs: ELO, calibración, ingestión, evaluación.

```bat
scripts\start-worker.bat
```

O directamente:
```powershell
python -m arq apps.worker.main.WorkerSettings
```

**Qué hace este proceso:**
- Actualiza ELO después de partidos terminados
- Ingesta fixtures de API-Football
- Corre calibración isotónica periódica
- Evalúa predicciones de partidos finalizados
- Ejecuta el ciclo de aprendizaje bayesiano

**Cron jobs que corren automáticamente:**

| Job | Frecuencia | Qué hace |
|-----|-----------|----------|
| `update_elo` | Diario | Recalcula ELO con resultados nuevos |
| `ingest_fixtures` | Periódico | Trae fixtures de API-Football |
| `evaluate_pending` | Periódico | Evalúa predicciones con resultado ya conocido |
| `calibrate_models` | Periódico | Ajusta factores de calibración isotónica |
| `audit_wc_data` | Diario | Verifica integridad de datos WC |

---

## PASO 3 — API FASTAPI (servidor HTTP)

La API es el punto de entrada del bot de Telegram y de n8n.

```bat
scripts\start-api.bat
```

O directamente:
```powershell
python -m uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

Verificar que levantó:
```powershell
curl http://localhost:8000/health
# debe responder {"status": "ok"}
```

---

## PASO 4 — TELEGRAM

Hay **dos modos**. Solo usar uno a la vez. Ver `.env` → `TELEGRAM_INGESTION_MODE`.

---

### MODO A — n8n (recomendado en producción)

**Cuándo usarlo:** Si tenés n8n Cloud activo con el workflow `telegram_inbound`.

```
TELEGRAM_INGESTION_MODE=n8n   ← en .env
```

1. La API debe estar corriendo en el puerto 8000
2. Exponer la API con túnel público:

```bat
scripts\start-api-tunnel.bat
```

Esto abre un túnel Cloudflare. Copiar la URL `https://xxxx.trycloudflare.com` y pegarla en n8n → workflow `telegram_inbound` → nodo "Config y Normalize":
```javascript
const API_BASE_URL = 'https://xxxx.trycloudflare.com';
```

3. Republicar el workflow en n8n.

---

### MODO B — Polling local (desarrollo / sin n8n)

**Cuándo usarlo:** Para pruebas locales sin n8n.

```
TELEGRAM_INGESTION_MODE=poll   ← en .env
```

```bat
scripts\start-telegram.bat
```

Para reiniciar el polling (mata instancias zombie primero):
```bat
scripts\restart-telegram.bat
```

> **Nunca correr MODO A y MODO B al mismo tiempo** — generan conflicto 409 en Telegram.

---

## VERIFICACIÓN RÁPIDA DEL SISTEMA

```bat
scripts\verify-supabase.bat
```

O individual:
```powershell
# Conexión Supabase
python scripts\verify_supabase.py

# API Keys (Odds API, Football-Data, Telegram, etc.)
python scripts\test_api_keys.py

# Estado Odds API (cuota restante)
python scripts\check_odds_api.py
```

---

---

# COMANDOS DE CARGA MANUAL

> Estos comandos **no corren solos**. Se ejecutan a mano cuando se necesitan.

---

## DATOS HISTÓRICOS — Cargar una vez o al inicio de torneo

### Cargar datos Understat (xG real por liga 2024-25)
```powershell
python scripts\load_understat_csvs.py
```
**Qué carga:** 112 equipos + 3,245 jugadores con xG/90 real en `ml.team_season_xg` y `ml.player_season_xg`.  
**Cuándo:** Al inicio, o cuando se agregan CSVs nuevos en `DATA/understat/`.

### Cargar historial WC (openfootball 2014/2018/2022/2026)
```powershell
python scripts\load_wc_historical.py
```
**Qué carga:** 248 partidos históricos en `ml.wc_match_history`.  
**Cuándo:** Al inicio del torneo, y después de cada ronda para capturar partidos jugados.

---

## ELO — Actualizar ratings

```powershell
python scripts\run_update_elo.py
```
**Qué hace:** Recalcula ELO de todos los equipos WC con los partidos jugados y persiste snapshot en `ml.wc_team_elo`.  
**Cuándo:** Después de cada jornada/ronda. El Worker lo hace automático, pero este comando fuerza la actualización.

---

## CALIBRACIÓN — Ajustar el modelo

### Calibración isotónica completa (desde archivos WC)
```powershell
python scripts\run_calibrate.py
```
**Qué hace:** Fitea calibración isotónica con datos WC 2018+2022, actualiza factores en `ml.calibration_factors`.  
**Cuándo:** Al inicio del torneo, y cada 20+ partidos nuevos evaluados.

### Bucket calibration (pre-α, forma del modelo)
```powershell
python scripts\fit_bucket_calibration.py
```
**Cuándo:** Cuando el Brier Score supera 0.25 o los factores de calibración cambian mucho entre runs.

### Shape calibration (ajuste pre-mercado)
```powershell
python scripts\fit_shape_calibration.py
```

### Joint calibration (outcome + mercado + CLV combinado)
```powershell
python scripts\fit_joint_calibration.py
```
**Nota:** Desactivado por defecto (`joint_calibration_enabled=False` en `.env`). Activar solo con 50+ partidos evaluados.

### Fit Dixon-Coles ρ (corrección scores bajos)
```powershell
python scripts\fit_dixon_coles_rho.py
```

### Fit pesos del modelo (Poisson/ELO blend óptimo)
```powershell
python scripts\fit_model_weights.py
```

---

## APRENDIZAJE — Ciclo de entrenamiento

```powershell
python scripts\run_learning_cycle.py
```
**Qué hace:** Evalúa predicciones pendientes → actualiza bias logit bayesiano → dispara retrain si hay suficientes resultados.  
**Cuándo:** Después de cada jornada con partidos finalizados. El Worker lo corre automático, pero este comando lo fuerza.

---

## BACKTEST — Validar el modelo

```powershell
python scripts\run_backtest.py
```
**Qué hace:** Walk-forward backtest con WC 2018 (train) → 2022 (test). Muestra ROI simulado, hit rate, Brier.  
**Cuándo:** Antes de activar el agente en una nueva competición, o después de cambios grandes al modelo.

---

## AUDITORÍAS — Diagnóstico del motor

### Auditoría EV (valor esperado real vs inflado)
```powershell
python scripts\audit_ev.py
```
**Qué hace:** Compara EV raw vs EV fair para los partidos próximos. Detecta falsos positivos.

### Auditoría sesgo favorito
```powershell
python scripts\audit_favorite_bias.py
```
**Qué hace:** Detecta si el modelo comprime favoritos, infla empates o sobreestima underdogs.

### Auditoría datos WC
```powershell
python scripts\audit_data.py
```

---

## BASE DE DATOS — Migraciones

### Ver qué migraciones están pendientes
```powershell
npx supabase db push --dry-run
```

### Aplicar migraciones pendientes
```powershell
npx supabase db push
```

### Verificar estado de migraciones aplicadas
```powershell
# En Supabase SQL Editor:
SELECT version, name FROM supabase_migrations.schema_migrations ORDER BY version;
```

---

---

# REFERENCIA RÁPIDA — UN VISTAZO

```
ARRANQUE DIARIO
───────────────
1. Docker Desktop  →  abierto y corriendo
2. scripts\start-redis.bat
3. scripts\start-worker.bat     (terminal nueva)
4. scripts\start-api.bat        (terminal nueva)
5. scripts\start-api-tunnel.bat (terminal nueva, si usas n8n)
   O
   scripts\start-telegram.bat   (si usas polling local)

DESPUÉS DE CADA RONDA WC
─────────────────────────
python scripts\load_wc_historical.py    ← captura partidos nuevos
python scripts\run_update_elo.py        ← actualiza ELO
python scripts\run_learning_cycle.py    ← ciclo de aprendizaje
python scripts\run_calibrate.py         ← recalibra (cada 20+ partidos)

INICIO DE TORNEO / UNA SOLA VEZ
─────────────────────────────────
npx supabase db push                    ← migraciones
python scripts\load_understat_csvs.py   ← xG por liga
python scripts\load_wc_historical.py    ← historial WC
python scripts\run_update_elo.py        ← ELO inicial
python scripts\run_calibrate.py         ← calibración inicial
python scripts\run_backtest.bat         ← validar modelo
```

---

# PUERTOS Y SERVICIOS

| Servicio | Puerto | URL |
|---------|--------|-----|
| API FastAPI | 8000 | http://localhost:8000 |
| Redis | 6379 | redis://localhost:6379 |
| Supabase | remoto | jpdvuafhfvdadxagqnoj.supabase.co |
| n8n Cloud | externo | ver configuración n8n |

---

# MODOS TELEGRAM — RESUMEN

| Variable `.env` | Script a usar | Cuándo |
|----------------|---------------|--------|
| `TELEGRAM_INGESTION_MODE=n8n` | `start-api-tunnel.bat` | Producción con n8n Cloud |
| `TELEGRAM_INGESTION_MODE=poll` | `start-telegram.bat` | Desarrollo local sin n8n |

> Cambiar el modo requiere reiniciar la API.
