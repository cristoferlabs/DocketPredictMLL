-- Calibration tracking log: records every recommended pick per market type
-- with enough data to compute hit-rate vs model probability per market.
-- Separate from wc_predictions (which tracks CLV/kelly/portfolio).

CREATE TABLE IF NOT EXISTS ml.picks_log (
  id            BIGSERIAL PRIMARY KEY,
  match_key     TEXT        NOT NULL,
  team_home     TEXT        NOT NULL,
  team_away     TEXT        NOT NULL,
  fecha         DATE,

  -- Market classification
  market_type   TEXT        NOT NULL,  -- '1X2','OU_2.5','OU_1.5','OU_3.5','DC','BTTS','CORNERS','SOT','CARDS'
  selection     TEXT        NOT NULL,  -- 'Belgium', 'Over 9.5', 'Corners Over 9.5', etc.

  -- Model snapshot at recommendation time
  model_prob    NUMERIC(6,4) NOT NULL,
  market_odds   NUMERIC(7,3),          -- real bookmaker odds (NULL if none available)
  ev_pct        NUMERIC(7,2),          -- EV% at recommendation time

  -- Outcome (set by label_picks job after match)
  outcome       BOOLEAN,               -- NULL = pending, TRUE = won, FALSE = lost
  actual_value  NUMERIC(5,1),          -- for STATS markets: actual corners/sot/cards count
  labeled_at    TIMESTAMPTZ,

  logged_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for calibration queries
CREATE INDEX IF NOT EXISTS picks_log_market_type_idx   ON ml.picks_log (market_type);
CREATE INDEX IF NOT EXISTS picks_log_fecha_idx         ON ml.picks_log (fecha);
CREATE INDEX IF NOT EXISTS picks_log_match_key_idx     ON ml.picks_log (match_key);
CREATE INDEX IF NOT EXISTS picks_log_pending_idx       ON ml.picks_log (fecha, outcome) WHERE outcome IS NULL;
