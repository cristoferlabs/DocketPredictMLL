"""
Fase B — Fit pesos Poisson/ELO minimizando Brier en histórico WC.

Uso:
  python scripts/fit_model_weights.py
"""

from __future__ import annotations

import asyncio
import sys

from apps.shared.config import get_settings
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.calibration_metrics import (
    blend_components,
    collect_1x2_components,
    evaluate_1x2_predictions,
    fit_poisson_elo_weights,
    save_fitted_model_weights,
)
from apps.worker.ml.model_combiner import ModelCombinationWeights


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== FASE B: Fit pesos Poisson/ELO (Brier + LogLoss) ===\n")
    archives = await fetch_all_worldcup_archives()
    components = collect_1x2_components(archives, years=[2018, 2022])
    print(f"Partidos históricos: {len(components)}")

    settings = get_settings()
    baseline = ModelCombinationWeights(
        poisson=settings.model_weight_poisson,
        elo=settings.model_weight_elo,
        market=settings.model_calibration_market_weight,
    )
    base_probs = [blend_components(c["poisson"], c["elo"], baseline) for c in components]
    base_labels = [c["label"] for c in components]
    base_report = evaluate_1x2_predictions(base_probs, base_labels)
    print(
        f"Baseline ({baseline.poisson:.0%}/{baseline.elo:.0%}): "
        f"Brier={base_report.brier_1x2:.4f} | LogLoss={base_report.log_loss_1x2:.4f} | "
        f"dog infl={base_report.underdog_inflation_pp:+.1f}pp"
    )

    result = fit_poisson_elo_weights(
        components,
        market_weight=settings.model_calibration_market_weight,
        train_years=[2018],
        test_years=[2022],
    )
    w = result.weights
    print(
        f"\nÓptimo train 2018: Poisson {w.poisson:.0%} + ELO {w.elo:.0%}"
    )
    tr = result.train_report
    print(
        f"  Train: Brier={tr.brier_1x2:.4f} | LogLoss={tr.log_loss_1x2:.4f} | "
        f"ECE={tr.ece_max_prob:.4f} | hit={tr.hit_rate_1x2:.1%}"
    )
    if result.test_report:
        te = result.test_report
        print(
            f"  Test 2022: Brier={te.brier_1x2:.4f} | LogLoss={te.log_loss_1x2:.4f} | "
            f"dog infl={te.underdog_inflation_pp:+.1f}pp"
        )

    if result.underdog_dampen_factor < 1.0:
        print(f"\nUnderdog dampen: factor {result.underdog_dampen_factor:.2f}")

    path = save_fitted_model_weights(result)
    print(f"\n✓ Guardado: {path}")
    print("Reinicia API/bot para cargar pesos fitted.")


if __name__ == "__main__":
    asyncio.run(main())
