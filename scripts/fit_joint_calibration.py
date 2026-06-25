"""
Fit joint calibration — Loss = log-loss + λ·market_div + μ·clv_proxy.

Uso:
  python scripts/fit_joint_calibration.py
"""

from __future__ import annotations

import asyncio
import sys

from apps.api.services.odds_context import load_wc_odds_events
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.joint_calibration import (
    build_joint_training_rows,
    fit_joint_calibration,
    save_joint_calibration_model,
)


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    db = get_supabase()
    odds_events, api_status = await load_wc_odds_events(db)
    archives = await fetch_all_worldcup_archives()

    if api_status.get("ok"):
        print(f"Odds API OK — {len(odds_events)} eventos WC")
    else:
        print(f"⚠️  Odds desde caché — {len(odds_events)} eventos ({api_status.get('reason', '?')})")

    rows = build_joint_training_rows(archives, odds_events, train_years=[2018, 2022])
    model, metrics = fit_joint_calibration(rows)

    print("\n=== Fit Joint Calibration ===\n")
    print(f"Filas total: {metrics.get('n_total')} "
          f"(outcome={metrics.get('n_outcome_only')}, market={metrics.get('n_market')})")
    print(f"λ_market={model.weights.lambda_market}  μ_clv={model.weights.mu_clv}\n")

    print("β blend por contexto:")
    for ctx, m in (metrics.get("by_context") or {}).items():
        print(f"  {ctx}: β={m['beta']}  loss={m['joint_loss']}  n={m['n']}")

    if metrics.get("n_market"):
        print(
            f"\nDivergencia mercado (MSE): "
            f"{metrics.get('market_divergence_mean_before')} → {metrics.get('market_divergence_mean_after')}"
        )
        print(
            f"Log-loss (filas mercado): "
            f"{metrics.get('log_loss_market_rows_before')} → {metrics.get('log_loss_market_rows_after')}"
        )

    path = save_joint_calibration_model(model)
    print(f"\nGuardado: {path}")


if __name__ == "__main__":
    asyncio.run(main())
