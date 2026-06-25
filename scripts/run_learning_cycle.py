"""
Fase C — ciclo manual de aprendizaje (eval + retrain + estado).

Uso:
  python scripts/run_learning_cycle.py
"""

from __future__ import annotations

import asyncio
import sys

from apps.shared.supabase_client import get_supabase
from apps.worker.ml.model_learning import load_learning_state, maybe_retrain_wc_weights
from apps.worker.ml.wc_predictions import evaluate_wc_predictions


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    db = get_supabase()
    print("=== FASE C: Learning cycle ===\n")

    eval_result = await evaluate_wc_predictions(db)
    print(f"Evaluadas: {eval_result.get('evaluated', 0)}")
    print(f"Retrain: {eval_result.get('retrain', {})}")

    retrain = await maybe_retrain_wc_weights(db)
    print(f"\nRetrain directo: {retrain}")

    state = load_learning_state()
    print(f"\nEstado aprendizaje:")
    print(f"  updates: {state.n_updates}")
    print(f"  rolling Brier: {state.rolling_brier}")
    print(f"  rolling CLV: {state.rolling_clv}")
    print(f"  since retrain: {state.results_since_retrain}")
    print(f"  bias logit: {state.logit_bias}")


if __name__ == "__main__":
    asyncio.run(main())
