"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import health, jobs, predictions, telegram, webhooks
from apps.shared.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    except Exception:
        app.state.arq_pool = None
    yield
    if app.state.arq_pool:
        await app.state.arq_pool.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Agente Betting Engine",
        description="Self-improving probabilistic betting engine for football",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:4173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.is_production:
        import os
        from pathlib import Path

        frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
        if frontend_dist.exists():
            from fastapi.staticfiles import StaticFiles
            app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    app.include_router(health.router)
    app.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
    app.include_router(telegram.router, prefix="/webhooks", tags=["telegram"])
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])

    @app.get("/")
    async def root():
        return {
            "service": "agente-betting-engine",
            "environment": settings.environment,
            "docs": "/docs",
        }

    return app


app = create_app()
