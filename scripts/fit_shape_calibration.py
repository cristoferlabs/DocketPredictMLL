"""
Fit shape calibration (draw + tail) desde WC histórico 2018+2022.

Uso:
  python scripts/fit_shape_calibration.py
"""

from __future__ import annotations

import asyncio
import sys

from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.shape_calibration import (
    _collect_shape_rows,
    evaluate_shape_model,
    fit_shape_calibration_from_archives,
    save_shape_calibration_model,
)


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    archives = await fetch_all_worldcup_archives()
    model, metrics = fit_shape_calibration_from_archives(archives, train_years=[2018, 2022])
    rows = _collect_shape_rows(archives, train_years=[2018, 2022])
    eval_metrics = evaluate_shape_model(rows, model)

    print("=== Fit Shape Calibration (learned) ===\n")
    print(f"Partidos: {metrics.get('n_matches', 0)}\n")

    print("Draw factor por contexto:")
    for ctx, m in (metrics.get("by_context") or {}).items():
        print(
            f"  {ctx}: n={m['n']} draw_rate={m['draw_rate']:.3f} "
            f"p_draw={m['mean_p_draw']:.3f} → factor={m['draw_factor']}"
        )

    print("\nDraw factor × λ (tertiles):")
    for ctx, bins in (metrics.get("lambda_bins") or {}).items():
        print(f"  {ctx}: {bins}")

    print("\nFavorite scale por peak:")
    for m in metrics.get("peak_bins") or []:
        print(
            f"  {m['bin']}: n={m['n']} fav_rate={m['fav_win_rate']:.3f} "
            f"p_fav={m['mean_p_fav']:.3f} → scale={m['favorite_scale']}"
        )

    print(f"\nLog-loss: before={eval_metrics.get('log_loss_before')} "
          f"after={eval_metrics.get('log_loss_after')} "
          f"Δ={eval_metrics.get('delta')}")

    path = save_shape_calibration_model(model)
    print(f"\nGuardado: {path}")


if __name__ == "__main__":
    asyncio.run(main())
