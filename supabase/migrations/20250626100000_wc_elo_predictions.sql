-- WC ELO by team name, odds snapshots for CLV, WC prediction log

-- ELO ratings keyed by team name for World Cup (no teams table FK required)
CREATE TABLE IF NOT EXISTS ml.wc_team_elo (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition     TEXT NOT NULL DEFAULT 'fifa_world_cup',
    team_name       TEXT NOT NULL,
    rating          NUMERIC(8, 2) NOT NULL DEFAULT 1500,
    match_date      DATE,
    opponent        TEXT,
    played_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL DEFAULT 'calc_elo_ratings'
);

CREATE INDEX IF NOT EXISTS idx_wc_team_elo_name_played
    ON ml.wc_team_elo (competition, team_name, played_at DESC);

-- Odds snapshots for CLV (capture at pick time vs pre-kickoff)
CREATE TABLE IF NOT EXISTS ml.odds_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition     TEXT NOT NULL DEFAULT 'fifa_world_cup',
    match_key       TEXT NOT NULL,
    team_home       TEXT NOT NULL,
    team_away       TEXT NOT NULL,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    odds_decimal    NUMERIC(8, 4) NOT NULL,
    fair_odds       NUMERIC(8, 4),
    snapshot_type   TEXT NOT NULL DEFAULT 'pick',
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_match
    ON ml.odds_snapshots (match_key, captured_at DESC);

-- WC predictions from Telegram / API (learning loop)
CREATE TABLE IF NOT EXISTS ml.wc_predictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition         TEXT NOT NULL DEFAULT 'fifa_world_cup',
    team_home           TEXT NOT NULL,
    team_away           TEXT NOT NULL,
    match_date          DATE,
    market_type         TEXT NOT NULL,
    predicted_outcome   TEXT NOT NULL,
    probability         NUMERIC(8, 6) NOT NULL,
    expected_value_fair NUMERIC(8, 4),
    edge_fair           NUMERIC(8, 4),
    kelly_stake         NUMERIC(8, 4),
    actual_outcome      TEXT,
    is_correct          BOOLEAN,
    brier_score         NUMERIC(10, 6),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evaluated_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_wc_predictions_match
    ON ml.wc_predictions (team_home, team_away, match_date);

GRANT SELECT, INSERT, UPDATE ON ml.wc_team_elo TO service_role;
GRANT SELECT, INSERT ON ml.odds_snapshots TO service_role;
GRANT SELECT, INSERT, UPDATE ON ml.wc_predictions TO service_role;
