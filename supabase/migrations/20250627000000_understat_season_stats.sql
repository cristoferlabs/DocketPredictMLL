-- Understat season xG/xGA + openfootball WC match history
-- Feeds: Poisson lambda priors, GK adjustment, ELO training corpus

-- ============================================================
-- ml.team_season_xg  — standings with xG/xGA (from Understat)
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.team_season_xg (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    league_slug  TEXT NOT NULL,   -- epl | la_liga | bundesliga | serie_a | ligue_1 | rpl
    season       TEXT NOT NULL,   -- e.g. "2024-25"
    team         TEXT NOT NULL,
    position     SMALLINT,
    matches      SMALLINT,
    wins         SMALLINT,
    draws        SMALLINT,
    loses        SMALLINT,
    goals        SMALLINT,
    goals_against SMALLINT,
    points       SMALLINT,
    xg           NUMERIC(8, 2),   -- total expected goals scored
    xga          NUMERIC(8, 2),   -- total expected goals conceded
    xpts         NUMERIC(8, 2),   -- expected points
    xg_per_game  NUMERIC(6, 3),   -- derived: xg / matches
    xga_per_game NUMERIC(6, 3),   -- derived: xga / matches
    source       TEXT NOT NULL DEFAULT 'understat',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league_slug, season, team)
);

CREATE INDEX IF NOT EXISTS idx_team_season_xg_lookup
    ON ml.team_season_xg (league_slug, season, team);

-- ============================================================
-- ml.player_season_xg  — player stats with xG/xA per 90
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.player_season_xg (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    league_slug  TEXT NOT NULL,
    season       TEXT NOT NULL,
    player       TEXT NOT NULL,
    team         TEXT NOT NULL,
    apps         SMALLINT,
    minutes      INTEGER,
    goals        SMALLINT,
    assists      SMALLINT,
    xg           NUMERIC(7, 3),
    xa           NUMERIC(7, 3),
    xg90         NUMERIC(6, 3),   -- xG per 90 minutes
    xa90         NUMERIC(6, 3),   -- xA per 90 minutes
    source       TEXT NOT NULL DEFAULT 'understat',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league_slug, season, player, team)
);

CREATE INDEX IF NOT EXISTS idx_player_season_xg_team
    ON ml.player_season_xg (league_slug, season, team);

-- ============================================================
-- ml.wc_match_history  — openfootball WC 2014/2018/2022 results
-- Primary training corpus for ELO + Poisson calibration
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.wc_match_history (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tournament_year  SMALLINT NOT NULL,   -- 2014 | 2018 | 2022
    match_date       DATE,
    round            TEXT,                -- "Matchday 1", "Round of 16", etc.
    group_name       TEXT,                -- "Group A", null for KO rounds
    team_home        TEXT NOT NULL,
    team_away        TEXT NOT NULL,
    score_home       SMALLINT,
    score_away       SMALLINT,
    ht_home          SMALLINT,
    ht_away          SMALLINT,
    venue            TEXT,
    is_knockout      BOOLEAN NOT NULL DEFAULT false,
    source           TEXT NOT NULL DEFAULT 'openfootball',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tournament_year, team_home, team_away, match_date)
);

CREATE INDEX IF NOT EXISTS idx_wc_match_history_year
    ON ml.wc_match_history (tournament_year, match_date);

CREATE INDEX IF NOT EXISTS idx_wc_match_history_teams
    ON ml.wc_match_history (team_home, team_away);

-- ============================================================
-- Grants
-- ============================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON ml.team_season_xg TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ml.player_season_xg TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ml.wc_match_history TO service_role;

ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT ALL ON TABLES TO service_role;
