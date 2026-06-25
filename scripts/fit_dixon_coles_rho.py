"""
Fit ρ Dixon-Coles por contexto (close / balanced) sobre WC histórico.

Uso:
  python scripts/fit_dixon_coles_rho.py
"""

from __future__ import annotations

import asyncio
import sys

from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.dixon_coles import fit_rho_by_context_from_archives, save_fitted_rho_by_context


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    archives = await fetch_all_worldcup_archives()
    fitted, metrics = fit_rho_by_context_from_archives(archives, train_years=[2018, 2022])

    print("=== Fit Dixon-Coles ρ por contexto ===\n")
    for ctx, m in metrics.items():
        print(f"  {ctx}: n={m.get('n')} ρ={m.get('rho')} {m.get('note', '')}")

    path = save_fitted_rho_by_context(fitted, metrics=metrics)
    print(f"\nGuardado: {path}")
    print(f"ρ fitted: {fitted}")


if __name__ == "__main__":
    asyncio.run(main())
