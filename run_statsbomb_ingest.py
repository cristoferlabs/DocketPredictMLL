"""
StatsBomb Ingest + Validación de impacto en λ
Uso:  python run_statsbomb_ingest.py

Pasos:
  1. Descarga eventos WC2018/2022 de StatsBomb GitHub raw
  2. Extrae xG, shots, posesión por equipo por partido
  3. Computa perfiles xG de equipo
  4. Compara λ_actual (goles) vs λ_nuevo (xG) en los 128 partidos históricos
  5. Mide si el cambio de fuente reduce el sesgo Under 2.5
"""
import asyncio
import json
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apps.worker.ingest.statsbomb_ingest import (
    build_team_xg_database,
    save_team_xg_database,
    get_team_xg_before_date,
    compute_team_xg_profile,
    find_wc_seasons,
)
from apps.worker.ml.xg_estimator import (
    estimate_lambda_from_xg_profile,
    WC_AVG_XG_PER_GAME,
)


# ── 1. Descarga de datos StatsBomb ────────────────────────────────────────────
print("=" * 65)
print("PASO 1 — Descargando datos StatsBomb WC (GitHub raw)")
print("=" * 65)

seasons = find_wc_seasons()
if not seasons:
    print("  ERROR: no se pudo conectar a GitHub. Verifica conexion a internet.")
    sys.exit(1)

available = [(s["season_name"], s["season_id"]) for s in seasons]
print(f"  Temporadas WC disponibles: {available}")

# Identificar qué años están disponibles
available_years = []
for s in seasons:
    name = s.get("season_name", "")
    for year in [2018, 2019, 2022, 2023]:
        if str(year) in name:
            available_years.append(year)

print(f"  Años WC con eventos: {available_years}")
print(f"\n  Descargando eventos (puede tomar 2-5 min con cache)...")

team_xg_db = build_team_xg_database(years=available_years or [2018, 2022], verbose=True)

if not team_xg_db:
    print("\n  ERROR: no se descargaron datos. Posibles causas:")
    print("    - Sin conexion a internet")
    print("    - StatsBomb no tiene eventos para estos años en open-data")
    print("    - Rate limiting de GitHub")
    sys.exit(1)

path = save_team_xg_database(team_xg_db)
print(f"\n  Database guardado: {path}")
print(f"  Equipos con datos xG: {len(team_xg_db)}")

# Mostrar ejemplo de un equipo
sample_team = next(iter(team_xg_db))
sample_records = team_xg_db[sample_team]
print(f"\n  Ejemplo ({sample_team}, {len(sample_records)} partidos):")
for r in sample_records[:3]:
    print(f"    {r['date']} vs {r['opponent']}: xG={r['xg_for']:.2f} (tiros={r['shots_for']}, pos={r['possession_pct']:.0%})")

# ── 2. Validación: λ_viejo vs λ_nuevo ────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2 — Comparativa lambda: goles vs xG (128 partidos WC)")
print("=" * 65)

async def validate():
    from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
    from apps.worker.ml.wc_historical import (
        extract_finished_matches,
        _match_feature_bundle,
        actual_outcomes,
    )
    from apps.api.services.worldcup_engine import compute_model_markets

    archives = await fetch_all_worldcup_archives()
    matches = extract_finished_matches(archives, years=[2018, 2022])

    lam_old, lam_new = [], []
    u25_old, u25_new = [], []
    hits_old, hits_new = 0, 0
    n_with_xg = 0
    n_fallback = 0

    for m in matches:
        bundle = _match_feature_bundle(m, archives)
        if not bundle:
            continue

        lambdas = bundle["lambdas"]
        elo = bundle["elo"]

        # λ actual (goles históricos)
        lh_old = lambdas.lambda_home
        la_old = lambdas.lambda_away

        # λ nuevo (xG StatsBomb, con fallback a goles si no hay data)
        xg_home = get_team_xg_before_date(team_xg_db, m.team1, m.date)
        xg_away = get_team_xg_before_date(team_xg_db, m.team2, m.date)
        profile_h = compute_team_xg_profile(xg_home)
        profile_a = compute_team_xg_profile(xg_away)

        est_h = estimate_lambda_from_xg_profile(profile_h, fallback_goals=lambdas.xg_home)
        est_a = estimate_lambda_from_xg_profile(profile_a, fallback_goals=lambdas.xg_away)
        lh_new = max(0.5, min(4.0, est_h.lambda_value))
        la_new = max(0.5, min(4.0, est_a.lambda_value))

        if est_h.source == "statsbomb_wc" or est_a.source == "statsbomb_wc":
            n_with_xg += 1
        else:
            n_fallback += 1

        # Predicciones con λ antiguo
        raw_old = compute_model_markets(
            lh_old, la_old,
            elo.get(m.team1, 1500), elo.get(m.team2, 1500),
            calibrate=False, apply_joint_calibration=False,
        )

        # Predicciones con λ nuevo
        raw_new = compute_model_markets(
            lh_new, la_new,
            elo.get(m.team1, 1500), elo.get(m.team2, 1500),
            calibrate=False, apply_joint_calibration=False,
        )

        actual = actual_outcomes(m)

        lam_old.append(lh_old + la_old)
        lam_new.append(lh_new + la_new)
        u25_old.append(raw_old.under_25)
        u25_new.append(raw_new.under_25)

        best_old = max(["home_win", "draw", "away_win"],
                       key=lambda k: getattr(raw_old, k.replace("_", "_")))
        best_new = max(["home_win", "draw", "away_win"],
                       key=lambda k: getattr(raw_new, k.replace("_", "_")))

        if actual.get(best_old) == 1:
            hits_old += 1
        if actual.get(best_new) == 1:
            hits_new += 1

    n = len(lam_old)
    if n == 0:
        print("  ERROR: ningún partido procesado")
        return

    act_u25 = 53.9  # tasa real Under 2.5 en WC2018+2022

    avg_lam_old = sum(lam_old) / n
    avg_lam_new = sum(lam_new) / n
    avg_u25_old = sum(u25_old) / n * 100
    avg_u25_new = sum(u25_new) / n * 100
    sesgo_old = sum(1 for u in u25_old if u > 0.5) / n * 100 - act_u25
    sesgo_new = sum(1 for u in u25_new if u > 0.5) / n * 100 - act_u25
    pct_u60_old = sum(1 for u in u25_old if u > 0.6) / n * 100
    pct_u60_new = sum(1 for u in u25_new if u > 0.6) / n * 100

    print(f"\n  Partidos: {n} | Con xG StatsBomb: {n_with_xg} | Fallback: {n_fallback}")
    print(f"\n  {'Metrica':30s} {'Goles (actual)':>16} {'xG StatsBomb':>14} {'Delta':>8}")
    print(f"  {'-'*30} {'-'*16} {'-'*14} {'-'*8}")
    print(f"  {'lambda_total media':30s} {avg_lam_old:>16.3f} {avg_lam_new:>14.3f} {avg_lam_new-avg_lam_old:>+8.3f}")
    print(f"  {'P(Under2.5) media':30s} {avg_u25_old:>15.1f}% {avg_u25_new:>13.1f}% {avg_u25_new-avg_u25_old:>+7.1f}pp")
    print(f"  {'Sesgo Under 2.5':30s} {sesgo_old:>+15.1f}pp {sesgo_new:>+13.1f}pp {sesgo_new-sesgo_old:>+7.1f}pp")
    print(f"  {'% predicciones U>60%':30s} {pct_u60_old:>15.1f}% {pct_u60_new:>13.1f}% {pct_u60_new-pct_u60_old:>+7.1f}pp")
    print(f"  {'Hit rate 1X2':30s} {hits_old/n*100:>15.1f}% {hits_new/n*100:>13.1f}% {(hits_new-hits_old)/n*100:>+7.1f}pp")

    print(f"\n  DIAGNOSTICO:")
    sesgo_delta = sesgo_new - sesgo_old
    if sesgo_delta < -2.0:
        print(f"  [+] Sesgo Under reducido {sesgo_delta:+.1f}pp -> xG mejora el modelo")
    elif sesgo_delta > 1.0:
        print(f"  [-] Sesgo empeorado {sesgo_delta:+.1f}pp -> revisar calibracion de xG")
    else:
        print(f"  [~] Sin cambio significativo en sesgo — puede ser que pocos partidos")
        print(f"      tengan xG disponible (n_with_xg={n_with_xg} de {n})")

asyncio.run(validate())

print(f"\n{'='*65}")
print("FIN")
print(f"{'='*65}")
