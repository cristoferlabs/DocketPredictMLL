-- Calibration factors and snapshots for honest probability reporting

CREATE TABLE IF NOT EXISTS ml.calibration_factors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition     TEXT NOT NULL DEFAULT 'fifa_world_cup',
    market          TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    factor          NUMERIC(8, 6) NOT NULL DEFAULT 1.0,
    method          TEXT NOT NULL DEFAULT 'isotonic',
    sample_size     INTEGER NOT NULL DEFAULT 0,
    ece             NUMERIC(8, 6),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    fitted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (competition, market, outcome)
);

CREATE TABLE IF NOT EXISTS ml.calibration_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition     TEXT NOT NULL DEFAULT 'fifa_world_cup',
    market          TEXT NOT NULL,
    window_days     INTEGER NOT NULL DEFAULT 30,
    ece             NUMERIC(8, 6) NOT NULL,
    brier           NUMERIC(8, 6),
    hit_rate        NUMERIC(8, 6),
    sample_size     INTEGER NOT NULL DEFAULT 0,
    reliability     JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ops.data_quality_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context         TEXT NOT NULL,
    status          TEXT NOT NULL,
    completeness_pct NUMERIC(5, 2),
    flags           JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE ON ml.calibration_factors TO service_role;
GRANT SELECT, INSERT ON ml.calibration_snapshots TO service_role;
GRANT SELECT, INSERT ON ops.data_quality_log TO service_role;
