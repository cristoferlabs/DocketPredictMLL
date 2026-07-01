"""
Calibración completa — Dixon-Coles ρ + Joint Calibration
Uso:  python run_calibration.py
"""
import asyncio
import sys
import os

# Garantizar encoding UTF-8 en Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.dixon_coles import (
    fit_rho_by_context_from_archives,
    save_fitted_rho_by_context,
    load_fitted_rho_by_context,
)
from apps.worker.ml.joint_calibration import (
    build_joint_training_rows,
    fit_joint_calibration,
    save_joint_calibration_model,
)


async def main() -> None:
    # ── 1. Carga de archives ──────────────────────────────────────────
    print("=" * 60)
    print("CARGANDO WC 2018 + 2022...")
    print("=" * 60)
    archives = await fetch_all_worldcup_archives()
    loaded = {y: bool(d) for y, d in archives.items()}
    print(f"  Archivos: {loaded}\n")

    # ── 2. Dixon-Coles ρ por contexto ────────────────────────────────
    print("=" * 60)
    print("PASO 1 — DIXON-COLES rho (calibracion por contexto)")
    print("=" * 60)
    print("  Fiteando rho sobre partidos WC2018+2022...")

    rho_fitted, rho_metrics = fit_rho_by_context_from_archives(
        archives,
        train_years=[2018, 2022],
    )

    print("\n  Resultado por contexto:")
    for ctx, val in rho_fitted.items():
        m = rho_metrics.get(ctx, {})
        n = m.get("n", "?")
        note = m.get("note", "")
        ll = m.get("log_likelihood")
        ll_s = f"  log_likelihood={ll:.2f}" if ll is not None else f"  {note}"
        print(f"    {ctx:12s}: rho={val:+.4f}  n={n}{ll_s}")

    path = save_fitted_rho_by_context(rho_fitted, metrics=rho_metrics)
    print(f"\n  Guardado en: {path}")
    print("  OK - Dixon-Coles calibrado\n")

    # ── 3. Joint Calibration ─────────────────────────────────────────
    print("=" * 60)
    print("PASO 2 — JOINT CALIBRATION (blend Poisson+ELO hacia mercado)")
    print("=" * 60)
    print("  Construyendo filas de entrenamiento (solo WC historico, sin odds de mercado)...")

    rows = build_joint_training_rows(
        archives,
        odds_events=[],          # sin odds live — solo data historica WC
        train_years=[2018, 2022],
    )

    n_rows = len(rows)
    if n_rows < 5:
        print(f"  AVISO: solo {n_rows} filas — resultado puede ser poco fiable")
    else:
        print(f"  Filas construidas: {n_rows}")

    print("  Fiteando joint calibration...")
    model, metrics = fit_joint_calibration(rows)

    print(f"\n  Resultado:")
    print(f"    Total filas:   {metrics.get('n_total', '?')}")
    print(f"    Con mercado:   {metrics.get('n_market', 0)}")
    print(f"    Solo outcome:  {metrics.get('n_outcome_only', '?')}")
    print(f"    lambda_market: {getattr(model.weights, 'lambda_market', '?')}")
    print(f"    mu_clv:        {getattr(model.weights, 'mu_clv', '?')}")

    print("\n  Beta blend por contexto:")
    for ctx, m in (metrics.get("by_context") or {}).items():
        print(f"    {ctx:12s}: beta={m.get('beta')}  loss={m.get('joint_loss')}  n={m.get('n')}")

    path_j = save_joint_calibration_model(model)
    print(f"\n  Guardado en: {path_j}")
    print("  OK - Joint calibration listo\n")

    # ── Resumen ───────────────────────────────────────────────────────
    print("=" * 60)
    print("CALIBRACION COMPLETADA")
    print("=" * 60)
    print(f"\n  Siguiente paso:")
    print(f"    python apps/worker/ml/audit_experimental.py")
    print(f"\n  Que comparar vs baseline anterior:")
    print(f"    Brier 1X2:    0.6441  -> esperar <= 0.643x")
    print(f"    Sesgo U2.5:  +8.6pp  -> esperar <= +6-7pp")
    print(f"    %U>60:       31.2%   -> esperar <= 28%")


if __name__ == "__main__":
    asyncio.run(main())
