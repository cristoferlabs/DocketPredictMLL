-- Ensure service_role can access ml/ops via PostgREST (after exposing schemas in API settings)

GRANT USAGE ON SCHEMA ml TO service_role, postgres;
GRANT USAGE ON SCHEMA ops TO service_role, postgres;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA ops TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ops TO service_role;

ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT ALL ON TABLES TO service_role;

-- View used by evaluation worker
GRANT SELECT ON ml.predictions_pending_evaluation TO service_role;
