"""Shared configuration for API and worker."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    supabase_url: str = ""
    supabase_service_role_key: str = ""

  # LLM agente — Ollama (gratis, local). provider: ollama | template
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_timeout: float = 90.0

    whatsapp_verify_token: str = ""
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""

    redis_url: str = "redis://localhost:6379"

    api_football_key: str = ""
    football_data_key: str = ""
    sportmonks_key: str = ""
    odds_api_key: str = ""
    gnews_api_key: str = ""
    newsapi_key: str = ""
    openweather_key: str = ""

    telegram_bot_token: str = ""
    telegram_group_id: str = ""
    telegram_webhook_secret: str = ""
    # n8n = n8n Telegram Trigger → API webhook | poll = scripts/start-telegram.bat
    telegram_ingestion_mode: str = "n8n"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    log_level: str = "INFO"

    arq_max_jobs: int = 10
    evaluation_batch_size: int = 100
    retrain_threshold: int = 200

    # EV guardrails
    ev_min_roi_backtest: float = 0.0
    ev_max_ece: float = 0.15
    ev_min_edge_fair: float = 0.03
    ev_max_daily_picks: int = 5
    ev_max_edge_fair: float = 0.18
    ev_max_fair: float = 0.15
    ev_max_model_market_divergence: float = 0.20
    # Deprecated — blend vive en worldcup_engine.compute_model_markets (blend_poisson)
    market_blend_model_weight: float = 0.6
    market_shrink_threshold: float = 0.25
    kelly_fraction: float = 0.25
    min_odds_books: int = 3

    # Parlay / combinadas (OUTPUT B — portfolio mode)
    parlay_min_win_prob: float = 0.55
    parlay_min_ev: float = 0.01
    parlay_min_legs: int = 3
    parlay_max_legs: int = 5
    parlay_max_pool_legs: int = 8
    parlay_max_matches_scan: int = 15
    parlay_market_min_prob: float = 0.60

    # SHARP single bet gate — métrica compuesta (incluye MDS+trust); evita doble gate
    sharp_min_composite: int = 68
    sharp_min_mds: int = 70  # telemetría / display
    sharp_min_confidence: float = 0.70  # telemetría / display
    sharp_max_stake_pct: float = 2.0

    # WATCH — stake exploratorio (optimistic EV fair, no raw)
    watch_exploratory_ev_threshold: float = 0.05
    watch_exploratory_stake_pct: float = 0.25
    watch_micro_ev_threshold: float = 0.12
    watch_micro_stake_pct: float = 0.15

    # Parlay stake
    parlay_base_stake_pct: float = 0.35
    parlay_max_stake_pct: float = 1.0

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
