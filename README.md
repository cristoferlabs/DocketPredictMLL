# DocketPredictMLL

Motor de predicciones probabilísticas de fútbol con ensemble **Poisson + ELO + GK + XGBoost**, calibración automática, backtesting con guardrails de EV y agente conversacional por Telegram.

Repositorio: [github.com/cristoferlabs/DocketPredictMLL](https://github.com/cristoferlabs/DocketPredictMLL)

## Características

- Ensemble multi-modelo con pesos adaptativos por liga y mercado
- Ingesta de fixtures, cuotas y resultados desde múltiples fuentes deportivas
- Autoevaluación continua y reentrenamiento de XGBoost bajo demanda
- Guardrails de valor esperado (EV), Kelly fraccional y detección de anomalías
- Bot de Telegram con deduplicación de updates y modo n8n o polling local
- API REST (FastAPI) y worker asíncrono (ARQ) desacoplados

## Arquitectura

| Componente | Rol |
|------------|-----|
| **Supabase** | Postgres: dominio (`public`), ML (`ml`), operaciones (`ops`) |
| **FastAPI** | API REST, webhooks Telegram/WhatsApp, disparo de jobs |
| **ARQ Worker** | Ingesta, predicción, evaluación, calibración y retrain |
| **Redis** | Cola de tareas ARQ y deduplicación de updates de Telegram |
| **n8n** | Puente Telegram/WhatsApp hacia la API; cron de sincronización diaria |
| **Ollama** | Capa LLM local para explicaciones del agente (opcional) |

```
Telegram / WhatsApp
        |
       n8n  ----webhook---->  FastAPI  ----enqueue---->  Redis  ---->  ARQ Worker
                                    |                              |
                                    v                              v
                               Supabase  <-------------------  modelos ML
```

## Requisitos

- Python 3.12+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Redis local)
- Cuenta [Supabase](https://supabase.com) con CLI instalada
- [Ollama](https://ollama.com) (opcional, para respuestas con LLM local)
- [n8n](https://n8n.io) (recomendado para Telegram en producción)

## Configuración

### 1. Clonar e instalar

```bash
git clone https://github.com/cristoferlabs/DocketPredictMLL.git
cd DocketPredictMLL
pip install -e ".[dev]"
```

### 2. Variables de entorno

```bash
cp .env.example .env
```

Edita `.env` con tus credenciales. **Nunca subas `.env` al repositorio**; solo `.env.example` con placeholders.

Variables principales:

| Variable | Descripción |
|----------|-------------|
| `SUPABASE_URL` | URL del proyecto Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (solo backend) |
| `REDIS_URL` | Conexión Redis (`redis://localhost:6379` en local) |
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_INGESTION_MODE` | `n8n` (producción) o `poll` (desarrollo local) |
| `LLM_PROVIDER` | `ollama` o `template` (sin LLM) |
| `API_FOOTBALL_KEY`, `FOOTBALL_DATA_KEY`, etc. | Claves de fuentes de datos |

### 3. Base de datos (Supabase)

```bash
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
```

En el Dashboard de Supabase, expón los schemas `ml` y `ops` además de `public`:

**Project Settings → API → Exposed schemas** → añadir `ml`, `ops`.

Verifica con `GET /health/schemas` tras levantar la API.

### 4. Redis local (Windows)

```cmd
scripts\start-redis.bat
```

En `.env`: `REDIS_URL=redis://localhost:6379`

### 5. Ollama (opcional)

```bash
ollama pull llama3.2
```

En `.env`: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2`

## Ejecución local

```cmd
REM Terminal 1 — API
scripts\start-api.bat

REM Terminal 2 — Worker (requiere Redis)
scripts\start-worker.bat
```

API disponible en `http://127.0.0.1:8000`. Documentación interactiva en `/docs`.

## Telegram

Telegram permite un solo consumidor de `getUpdates` por token. No mezcles polling local y n8n con el mismo bot.

| Modo | Configuración | Ejecutar | Desactivar |
|------|---------------|----------|------------|
| **n8n** (recomendado) | `TELEGRAM_INGESTION_MODE=n8n` | API + Redis + workflow `telegram_inbound` | `start-telegram.bat` |
| **poll** (dev) | `TELEGRAM_INGESTION_MODE=poll` | `start-telegram.bat` | Workflows n8n con Telegram Trigger |

Pasos para modo n8n:

1. Importar y activar `n8n/workflows/telegram_inbound.json`
2. Configurar `API_BASE_URL` en el nodo **Config y Normalize** con tu URL pública HTTPS (Railway o tunnel con `scripts\start-api-tunnel.bat`)
3. No uses `127.0.0.1` en n8n Cloud (bloqueado por SSRF)

## Workflows n8n

Importar desde `n8n/workflows/`:

| Archivo | Función |
|---------|---------|
| `telegram_inbound.json` | Telegram Trigger → API |
| `whatsapp_inbound.json` | Meta webhook → API → respuesta |
| `whatsapp_outbound.json` | Envío manual de mensajes |
| `daily_sync_trigger.json` | Cron diario: ingesta + predicción + evaluación |
| `Bot Interactivo - Mundial 2026.json` | Agente conversacional Mundial 2026 |
| `Agente IA - Análisis Mundial 2026.json` | Análisis automatizado |

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/health/schemas` | Verificación de schemas Supabase |
| POST | `/webhooks/telegram` | Update de Telegram desde n8n |
| POST | `/webhooks/whatsapp` | Mensaje normalizado desde n8n |
| POST | `/jobs/ingest-fixtures` | Disparar ingesta de fixtures |
| POST | `/jobs/evaluate-pending` | Evaluar predicciones vs resultados |
| POST | `/jobs/predict-upcoming` | Generar predicciones |
| GET | `/predictions/upcoming` | Listar predicciones próximas |

## Loop de auto-mejora

1. El worker predice partidos `scheduled` y escribe en `ml.predictions`
2. Al finalizar el partido se registran resultados en `match_results`
3. El job `evaluate_pending` compara y escribe `prediction_evaluations`
4. Se ajustan `model_weights` por liga y mercado
5. Tras acumular suficientes evaluaciones (`RETRAIN_THRESHOLD`), se reentrena XGBoost

## Estructura del proyecto

```
apps/
  api/          FastAPI, routers, servicios del agente
  worker/       ARQ worker, tareas ML e ingesta
  shared/       Configuración y cliente Supabase
supabase/
  migrations/   Esquema SQL versionado
n8n/workflows/  Automatizaciones de mensajería
scripts/        Utilidades y arranque local (Windows)
tests/          Suite pytest
```

## Despliegue (Railway)

Crear tres servicios según `railway.json`:

1. **api** — `uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT`
2. **worker** — `arq apps.worker.main.WorkerSettings`
3. **redis** — Plugin Redis de Railway; copiar `REDIS_URL` a api y worker

## Tests

```bash
pytest
```

## Disclaimer

Las predicciones son probabilísticas con fines analíticos. Verifica la regulación local sobre apuestas y juego responsable.
