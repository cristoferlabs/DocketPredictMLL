"""ARQ task — ciclo de aprendizaje WC (retrain pesos tras N resultados)."""

from __future__ import annotations

import logging

from apps.worker.ml.model_learning import maybe_retrain_wc_weights

logger = logging.getLogger(__name__)


async def wc_learning_cycle(ctx: dict) -> dict:
    """Auto-retrain Poisson/ELO cuando results_since_retrain >= umbral."""
    from apps.shared.supabase_client import get_supabase

    db = get_supabase()
    result = await maybe_retrain_wc_weights(db)
    logger.info("wc_learning_cycle: %s", result)
    return result
