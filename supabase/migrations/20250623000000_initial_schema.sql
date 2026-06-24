-- Self-Improving Probabilistic Betting Engine — initial schema
-- Schemas: public (domain + whatsapp), ml (predictions), ops (ingestion + jobs)

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "pg_cron" WITH SCHEMA extensions;

-- Schemas
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS ops;

-- Revoke public API access to internal schemas
REVOKE ALL ON SCHEMA ml FROM PUBLIC;
REVOKE ALL ON SCHEMA ops FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA ml FROM anon, authenticated;
REVOKE ALL ON ALL TABLES IN SCHEMA ops FROM anon, authenticated;

-- =============================================================================
-- PUBLIC: Sports domain
-- =============================================================================

CREATE TABLE public.leagues (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    country     TEXT,
    external_ids JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.seasons (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    league_id   UUID NOT NULL REFERENCES public.leagues(id) ON DELETE CASCADE,
    year        TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league_id, year)
);

CREATE TABLE public.teams (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    season_id   UUID NOT NULL REFERENCES public.seasons(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    short_name  TEXT,
    external_ids JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.goalkeepers (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     UUID NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    save_pct    NUMERIC(5, 4),
    xga_90      NUMERIC(6, 3),
    form_window JSONB NOT NULL DEFAULT '{}',
    is_starter  BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE public.match_status AS ENUM (
    'scheduled', 'live', 'finished', 'postponed', 'cancelled'
);

CREATE TABLE public.matches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    season_id       UUID NOT NULL REFERENCES public.seasons(id) ON DELETE CASCADE,
    home_team_id    UUID NOT NULL REFERENCES public.teams(id),
    away_team_id    UUID NOT NULL REFERENCES public.teams(id),
    kickoff_at      TIMESTAMPTZ NOT NULL,
    status          public.match_status NOT NULL DEFAULT 'scheduled',
    round           TEXT,
    external_ids    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT matches_different_teams CHECK (home_team_id <> away_team_id)
);

CREATE TABLE public.match_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id        UUID NOT NULL UNIQUE REFERENCES public.matches(id) ON DELETE CASCADE,
    home_goals      SMALLINT NOT NULL DEFAULT 0,
    away_goals      SMALLINT NOT NULL DEFAULT 0,
    ht_home_goals   SMALLINT,
    ht_away_goals   SMALLINT,
    stats_summary   JSONB NOT NULL DEFAULT '{}',
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.match_stats (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id    UUID NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    source_id   UUID,
    possession  NUMERIC(5, 2),
    shots       SMALLINT,
    shots_on_target SMALLINT,
    xg          NUMERIC(6, 3),
    corners     SMALLINT,
    cards       SMALLINT,
    raw         JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.team_elo_ratings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     UUID NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
    league_id   UUID NOT NULL REFERENCES public.leagues(id) ON DELETE CASCADE,
    rating      NUMERIC(8, 2) NOT NULL DEFAULT 1500,
    played_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    match_id    UUID REFERENCES public.matches(id) ON DELETE SET NULL
);

CREATE INDEX idx_matches_kickoff_status ON public.matches (kickoff_at, status);
CREATE INDEX idx_matches_season ON public.matches (season_id);
CREATE INDEX idx_team_elo_team_played ON public.team_elo_ratings (team_id, played_at DESC);
CREATE INDEX idx_match_stats_match ON public.match_stats (match_id);

-- =============================================================================
-- PUBLIC: WhatsApp / agent sessions
-- =============================================================================

CREATE TABLE public.whatsapp_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_hash  TEXT NOT NULL UNIQUE,
    last_intent TEXT,
    context     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE public.message_direction AS ENUM ('inbound', 'outbound');

CREATE TABLE public.whatsapp_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES public.whatsapp_sessions(id) ON DELETE CASCADE,
    direction       public.message_direction NOT NULL,
    content         TEXT NOT NULL,
    api_message_id  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_whatsapp_messages_session ON public.whatsapp_messages (session_id, created_at DESC);

ALTER TABLE public.whatsapp_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.whatsapp_messages ENABLE ROW LEVEL SECURITY;

-- Service role only for whatsapp tables (backend writes via service_role)
CREATE POLICY whatsapp_sessions_service ON public.whatsapp_sessions
    FOR ALL USING (false);

CREATE POLICY whatsapp_messages_service ON public.whatsapp_messages
    FOR ALL USING (false);

-- =============================================================================
-- OPS: Data ingestion and job tracking
-- =============================================================================

CREATE TABLE ops.data_sources (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    api_base_url TEXT,
    priority    SMALLINT NOT NULL DEFAULT 5,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    config      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ops.raw_ingestions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID NOT NULL REFERENCES ops.data_sources(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    payload     JSONB NOT NULL,
    checksum    TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ,
    UNIQUE (source_id, external_id)
);

CREATE TYPE ops.job_status AS ENUM ('pending', 'running', 'completed', 'failed');

CREATE TABLE ops.job_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type    TEXT NOT NULL,
    status      ops.job_status NOT NULL DEFAULT 'pending',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error       TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_raw_ingestions_unprocessed ON ops.raw_ingestions (ingested_at)
    WHERE processed_at IS NULL;
CREATE INDEX idx_job_runs_type_status ON ops.job_runs (job_type, status, started_at DESC);

-- FK from match_stats to data_sources (deferred after ops tables exist)
ALTER TABLE public.match_stats
    ADD CONSTRAINT match_stats_source_id_fkey
    FOREIGN KEY (source_id) REFERENCES ops.data_sources(id) ON DELETE SET NULL;

-- Seed data sources
INSERT INTO ops.data_sources (name, slug, api_base_url, priority) VALUES
    ('API-Football', 'api-football', 'https://v3.football.api-sports.io', 1),
    ('Football-Data.org', 'football-data', 'https://api.football-data.org/v4', 2),
    ('SportMonks', 'sportmonks', 'https://api.sportmonks.com/v3', 3),
    ('Understat', 'understat', 'https://understat.com', 4),
    ('FBref', 'fbref', 'https://fbref.com', 5),
    ('Odds API', 'odds-api', 'https://api.the-odds-api.com/v4', 6),
    ('OpenWeather', 'openweather', 'https://api.openweathermap.org/data/2.5', 7),
    ('Transfermarkt', 'transfermarkt', 'https://www.transfermarkt.com', 8),
    ('Custom/Manual', 'custom', NULL, 9);

-- =============================================================================
-- ML: Features, models, predictions, betting
-- =============================================================================

CREATE TABLE ml.match_features (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id        UUID NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    feature_version TEXT NOT NULL DEFAULT 'v1',
    vector          JSONB NOT NULL DEFAULT '{}',
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (match_id, feature_version)
);

CREATE TYPE ml.model_type AS ENUM ('poisson', 'elo', 'gk', 'xgboost', 'ensemble');

CREATE TABLE ml.model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type      ml.model_type NOT NULL,
    version         TEXT NOT NULL,
    artifact_path   TEXT,
    trained_at      TIMESTAMPTZ,
    metrics         JSONB NOT NULL DEFAULT '{}',
    is_active       BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_type, version)
);

CREATE TABLE ml.model_weights (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    league_id   UUID NOT NULL REFERENCES public.leagues(id) ON DELETE CASCADE,
    market_type TEXT NOT NULL,
    model_type  ml.model_type NOT NULL,
    weight      NUMERIC(6, 4) NOT NULL DEFAULT 0.25,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league_id, market_type, model_type)
);

CREATE TYPE ml.confidence_tier AS ENUM ('high', 'medium', 'low');

CREATE TABLE ml.predictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id            UUID NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    model_version_id    UUID NOT NULL REFERENCES ml.model_versions(id),
    market_type         TEXT NOT NULL,
    predicted_outcome   TEXT NOT NULL,
    probability         NUMERIC(8, 6) NOT NULL,
    confidence_tier     ml.confidence_tier NOT NULL DEFAULT 'medium',
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ml.prediction_evaluations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id   UUID NOT NULL UNIQUE REFERENCES ml.predictions(id) ON DELETE CASCADE,
    actual_outcome  TEXT NOT NULL,
    is_correct      BOOLEAN NOT NULL,
    brier_score     NUMERIC(10, 6),
    log_loss        NUMERIC(10, 6),
    notes           TEXT,
    evaluated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE ml.betting_priority AS ENUM ('high', 'medium', 'low');
CREATE TYPE ml.betting_status AS ENUM ('pending', 'won', 'lost', 'void', 'partial');

CREATE TABLE ml.betting_combinations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id        UUID REFERENCES public.matches(id) ON DELETE CASCADE,
    parlay_id       UUID,
    priority        ml.betting_priority NOT NULL DEFAULT 'medium',
    expected_value  NUMERIC(8, 4),
    kelly_fraction  NUMERIC(8, 4),
    status          ml.betting_status NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ml.betting_combination_legs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    combination_id  UUID NOT NULL REFERENCES ml.betting_combinations(id) ON DELETE CASCADE,
    market_type     TEXT NOT NULL,
    selection       TEXT NOT NULL,
    odds            NUMERIC(8, 3),
    probability     NUMERIC(8, 6)
);

CREATE TABLE ml.model_performance_metrics (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id    UUID NOT NULL REFERENCES ml.model_versions(id) ON DELETE CASCADE,
    league_id           UUID REFERENCES public.leagues(id) ON DELETE SET NULL,
    market_type         TEXT NOT NULL,
    window_days         INTEGER NOT NULL DEFAULT 30,
    hit_rate            NUMERIC(6, 4),
    roi_sim             NUMERIC(8, 4),
    calibration_error   NUMERIC(8, 6),
    sample_size         INTEGER NOT NULL DEFAULT 0,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_predictions_match_market ON ml.predictions (match_id, market_type);
CREATE INDEX idx_predictions_created ON ml.predictions (created_at DESC);
CREATE INDEX idx_prediction_evaluations_evaluated ON ml.prediction_evaluations (evaluated_at DESC);

-- View: predictions pending evaluation (finished matches without eval)
CREATE OR REPLACE VIEW ml.predictions_pending_evaluation AS
SELECT
    p.id AS prediction_id,
    p.match_id,
    p.market_type,
    p.predicted_outcome,
    p.probability,
    m.status AS match_status,
    mr.home_goals,
    mr.away_goals
FROM ml.predictions p
JOIN public.matches m ON m.id = p.match_id
JOIN public.match_results mr ON mr.match_id = m.id
LEFT JOIN ml.prediction_evaluations pe ON pe.prediction_id = p.id
WHERE m.status = 'finished'
  AND pe.id IS NULL;

-- Seed initial model versions (placeholders)
INSERT INTO ml.model_versions (model_type, version, is_active, metrics) VALUES
    ('elo', '1.0.0', true, '{"k_factor": 32, "home_advantage": 100}'),
    ('poisson', '1.0.0', true, '{"max_goals": 10}'),
    ('gk', '1.0.0', true, '{"adjustment_cap": 0.15}'),
    ('xgboost', '1.0.0', false, '{}'),
    ('ensemble', '1.0.0', true, '{"weights": {"elo": 0.25, "poisson": 0.35, "gk": 0.15, "xgboost": 0.25}}');

-- Updated_at trigger helper
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER leagues_updated_at BEFORE UPDATE ON public.leagues
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER teams_updated_at BEFORE UPDATE ON public.teams
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER goalkeepers_updated_at BEFORE UPDATE ON public.goalkeepers
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER matches_updated_at BEFORE UPDATE ON public.matches
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER whatsapp_sessions_updated_at BEFORE UPDATE ON public.whatsapp_sessions
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Grant service_role full access to ml and ops
GRANT USAGE ON SCHEMA ml TO service_role;
GRANT USAGE ON SCHEMA ops TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA ops TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ops TO service_role;

ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT ALL ON TABLES TO service_role;
