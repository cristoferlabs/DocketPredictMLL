"""Shared configuration for API and worker."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

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
    ev_min_roi_backtest: float = 0.02
    ev_max_ece: float = 0.15
    ev_min_edge_fair: float = 0.03
    ev_max_daily_picks: int = 5
    ev_max_edge_fair: float = 0.18
    ev_max_fair: float = 0.15
    ev_max_model_market_divergence: float = 0.20
    # ENGINE v3 — combinación Poisson / ELO / mercado (mercado solo calibración)
    model_weight_poisson: float = 0.5
    model_weight_elo: float = 0.3
    model_calibration_market_weight: float = 0.2
    # Deprecated — usar model_weight_poisson / model_weight_elo
    market_blend_model_weight: float = 0.6
    market_shrink_threshold: float = 0.25
    kelly_fraction: float = 0.25
    min_odds_books: int = 3

    # Parlay / combinadas (OUTPUT B — quant portfolio v3)
    parlay_min_win_prob: float = 0.55
    parlay_min_ev: float = 0.05
    parlay_sharp_min_ev: float = 0.03
    parlay_min_confidence: float = 55.0
    parlay_min_mds: float = 55.0
    parlay_max_correlation_score: float = 0.65
    parlay_max_pairwise_correlation: float = 0.80
    parlay_min_legs: int = 2
    parlay_max_legs: int = 5
    parlay_max_pool_legs: int = 8
    parlay_max_matches_scan: int = 15
    parlay_market_min_prob: float = 0.60

    # SHARP single bet gate — métrica compuesta (incluye MDS+trust); evita doble gate
    sharp_mode: str = "portfolio"  # portfolio | gate
    sharp_min_composite: int = 68
    sharp_portfolio_min_composite: int = 58
    sharp_watch_composite: int = 52
    sharp_portfolio_top_pct: float = 0.30
    sharp_portfolio_top_k: int = 15
    sharp_min_mds: int = 70  # telemetría / display
    sharp_min_confidence: float = 0.70  # telemetría / display
    sharp_max_stake_pct: float = 2.0          # allocator hard cap (% bankroll)
    max_stake_display_pct: float = 5.0        # display/recommendation cap (% bankroll)

    # WATCH — stake exploratorio (optimistic EV fair, no raw)
    watch_exploratory_ev_threshold: float = 0.05
    watch_exploratory_stake_pct: float = 0.25
    watch_micro_ev_threshold: float = 0.12
    watch_micro_stake_pct: float = 0.15

    # Parlay stake
    parlay_base_stake_pct: float = 0.35
    parlay_max_stake_pct: float = 1.0

    # ENGINE v3 Fase C — learning loop
    model_retrain_min_wc_results: int = 8
    model_max_live_brier_1x2: float = 0.70
    model_max_favorite_bias: float = 0.25
    model_learning_rate: float = 0.08
    model_logit_bias_cap: float = 0.35

    # Salud motor — CLV / ROI live
    clv_alert_min_samples: int = 5
    clv_alert_negative_threshold: float = -0.01
    live_roi_alert_threshold: float = -0.05
    engine_health_telegram_alerts: bool = True
    engine_health_alert_cooldown_hours: int = 12

    # Live calibration — regime-based α (piecewise)
    live_calibration_enabled: bool = True
    cal_alpha_regime_t1: float = 10.0
    cal_alpha_regime_t2: float = 20.0
    cal_alpha_regime_t3: float = 30.0
    cal_alpha_regime_low: float = 0.30
    cal_alpha_regime_medium: float = 0.52
    cal_alpha_regime_high: float = 0.70
    cal_alpha_regime_max: float = 0.78
    cal_alpha_normal_low: float = 0.28
    cal_alpha_normal_medium: float = 0.34
    cal_alpha_normal_high: float = 0.38
    cal_alpha_normal_max: float = 0.40
    cal_alpha_aligned_cap: float = 0.30
    cal_alpha_overfit_cap: float = 0.60
    cal_alpha_knockout_bump: float = 0.03
    cal_alpha_thin_books_penalty: float = 0.05
    cal_alpha_low_mus_bump: float = 0.03
    # Legacy aliases (deprecated — regime reemplaza boost lineal)
    cal_alpha_wc_min: float = 0.30
    cal_alpha_wc_max: float = 0.78
    cal_alpha_normal_min: float = 0.28
    cal_alpha_normal_max: float = 0.40
    cal_alpha_mismatch_boost: float = 0.70
    cal_alpha_favorite_divergence_pp: float = 20.0
    cal_alpha_favorite_boost: float = 0.10
    cal_alpha_absolute_max: float = 0.78
    tournament_under_boost: float = 0.08
    tournament_over_penalty: float = 0.08
    tournament_draw_boost: float = 0.08
    underdog_shrink_gap_pp: float = 25.0
    underdog_shrink_stat_weight: float = 0.85
    underdog_shrink_mismatch_pp: float = 20.0
    overfit_warning_ratio: float = 0.40

    # Pre-α bucket calibration (P_stat structural fix antes de régimen α)
    pre_alpha_bucket_enabled: bool = True
    pre_alpha_favorite_strong_min: float = 0.62
    pre_alpha_favorite_strong_factor: float = 1.06
    pre_alpha_market_compression_pp: float = 10.0
    pre_alpha_market_compression_gain: float = 0.35
    pre_alpha_underdog_inflation_pp: float = 8.0
    pre_alpha_underdog_dampen: float = 0.88

    # EV clamp estructural por régimen α (decisión SHARP)
    ev_cap_regime_aligned: float = 0.10
    ev_cap_regime_moderate: float = 0.12
    ev_cap_regime_high: float = 0.15
    ev_cap_regime_extreme: float = 0.18
    ev_cap_regime_default: float = 0.15
    ev_regime_clamp_enabled: bool = True

    # Poisson P1 — Dixon-Coles contextual + λ estructural
    poisson_dixon_coles_enabled: bool = True
    poisson_dc_close_total_lambda: float = 2.50
    poisson_dc_close_elo_diff: float = 80.0
    poisson_dc_mismatch_elo_diff: float = 150.0
    poisson_dc_mismatch_lambda_ratio: float = 2.0
    poisson_elo_tight_max_diff: float = 80.0
    poisson_ratio_tight_threshold: float = 1.75
    poisson_mismatch_favorite_lift: float = 1.05
    poisson_lambda_extreme_correction: bool = True
    poisson_shape_use_learned: bool = True

    # Joint calibration — outcome + mercado + CLV proxy (post-shape, pre-CAL)
    joint_calibration_enabled: bool = False
    joint_lambda_market: float = 0.35
    joint_mu_clv: float = 0.15

    # SHARP v3 — fases cold / warm / mature
    sharp_phase_cold_n: int = 10
    sharp_phase_mature_n: int = 25
    sharp_cold_strong_mds: int = 70
    sharp_cold_weak_mds_min: int = 55
    sharp_max_divergence_pp: float = 20.0
    sharp_mature_max_divergence_pp: float = 15.0
    sharp_require_shrink_active: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
