#!/usr/bin/env python3
"""
Validación experimental — Auditoría motor WC2026
Ejecutar desde la raíz del repo:  python apps/worker/ml/audit_experimental.py

Corre el motor partido por partido sobre WC 2018+2022 (walk-forward, leak-free).
Computa distribución real de lambdas + 3 ablaciones aisladas.
NO modifica ningún archivo de producción (todos los cambios son monkey-patches
en memoria que se restauran al final de cada ablación).
"""
import sys
import os
import math
import asyncio

# Resolver raíz del repo independientemente del directorio de trabajo
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np

# ─── helpers de métricas ──────────────────────────────────────────────────────

def brier_multiclass(probs_list, labels):
    total = 0.0
    for probs, label in zip(probs_list, labels):
        for i, pi in enumerate(probs):
            yi = 1.0 if i == label else 0.0
            total += (pi - yi) ** 2
    return total / len(probs_list) if probs_list else 0.0


def log_loss_multiclass(probs_list, labels, eps=1e-12):
    total = 0.0
    for probs, label in zip(probs_list, labels):
        total -= math.log(max(probs[label], eps))
    return total / len(probs_list) if probs_list else 0.0


def hit_rate(probs_list, labels):
    correct = sum(1 for p, y in zip(probs_list, labels) if p.index(max(p)) == y)
    return correct / len(probs_list) if probs_list else 0.0


def brier_binary(probs, actuals):
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs) if probs else 0.0


def roi_flat(probs_1x2, labels):
    """Bet 1 unit on most probable outcome when p > 0.36. Cuota = 1/p_modelo."""
    profit = 0.0
    bets = 0
    for probs, label in zip(probs_1x2, labels):
        best_idx = probs.index(max(probs))
        p_best = probs[best_idx]
        if p_best <= 0.36:
            continue
        bets += 1
        fair_odd = 1.0 / p_best
        profit += (fair_odd - 1.0) if best_idx == label else -1.0
    roi = profit / bets if bets > 0 else 0.0
    yield_ = profit / len(probs_1x2) if probs_1x2 else 0.0
    return roi, yield_, bets


def histogram_ascii(arr, bins=10, width=35):
    a = np.array(arr)
    counts, edges = np.histogram(a, bins=bins)
    max_c = max(counts) or 1
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / max_c * width)
        lines.append(f"  [{lo:4.2f}-{hi:4.2f}] {bar:<35} {c:>3}")
    return "\n".join(lines)


def pct_stats(arr):
    a = np.array(arr)
    return {
        "n":      len(a),
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(np.percentile(a, 25)),
        "p50":    float(np.percentile(a, 50)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
    }


# ─── carga de datos ───────────────────────────────────────────────────────────

print("=" * 70)
print("DESCARGANDO WC 2018 + 2022 (openfootball)...")
print("=" * 70)


async def _fetch():
    from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
    return await fetch_all_worldcup_archives()


archives = asyncio.run(_fetch())
loaded = {y: bool(d) for y, d in archives.items()}
print(f"  Archivos cargados: {loaded}")

from apps.worker.ml.wc_historical import (
    extract_finished_matches,
    actual_outcomes,
    _match_feature_bundle,
    archives_before_date,
)
from apps.api.services.worldcup_engine import compute_model_markets

matches = extract_finished_matches(archives, years=[2018, 2022])
print(f"  Partidos historicos encontrados: {len(matches)}")


# ─── función de evaluación ────────────────────────────────────────────────────

def run_evaluation(matches, archives, label="BASELINE"):
    """
    Corre el motor con las funciones tal como estén en el módulo en ese momento.
    El monkey-patching externo controla las ablaciones.
    """
    lam_home_list  = []
    lam_away_list  = []
    lam_total_list = []
    u25_prob_list  = []

    probs_1x2_list = []
    labels_list    = []
    p_over25_list  = []
    y_over25_list  = []
    p_btts_list    = []
    y_btts_list    = []
    y_u25_list     = []

    skipped = 0
    for m in matches:
        bundle = _match_feature_bundle(m, archives)
        if bundle is None:
            skipped += 1
            continue

        lam = bundle["lambdas"]
        elo = bundle["elo"]

        try:
            raw = compute_model_markets(
                lam.lambda_home,
                lam.lambda_away,
                elo.get(m.team1, 1500),
                elo.get(m.team2, 1500),
                calibrate=False,
                apply_joint_calibration=False,
            )
        except Exception as exc:
            print(f"    compute_model_markets error ({m.team1} vs {m.team2}): {exc}")
            skipped += 1
            continue

        actual = actual_outcomes(m)

        lam_home_list.append(lam.lambda_home)
        lam_away_list.append(lam.lambda_away)
        lam_total_list.append(lam.lambda_home + lam.lambda_away)
        u25_prob_list.append(raw.under_25)

        probs_1x2_list.append([raw.home_win, raw.draw, raw.away_win])
        labels_list.append(actual["label_1x2"])

        p_over25_list.append(raw.over_25)
        y_over25_list.append(float(actual["over_25"]))
        p_btts_list.append(raw.btts_yes)
        y_btts_list.append(float(actual["btts_yes"]))
        y_u25_list.append(float(actual["under_25"]))

    n = len(lam_total_list)
    if n == 0:
        print(f"  [{label}] ERROR: 0 partidos procesados (skipped={skipped})")
        return None
    print(f"  [{label}] Procesados: {n}  |  Skipped: {skipped}")

    bs         = brier_multiclass(probs_1x2_list, labels_list)
    ll         = log_loss_multiclass(probs_1x2_list, labels_list)
    hr         = hit_rate(probs_1x2_list, labels_list)
    roi, yield_, n_bets = roi_flat(probs_1x2_list, labels_list)
    bs_o       = brier_binary(p_over25_list, y_over25_list)
    bs_b       = brier_binary(p_btts_list, y_btts_list)

    act_u25    = sum(y_u25_list) / n * 100
    pct_lt25   = sum(1 for x in lam_total_list if x < 2.5) / n * 100
    pct_u60    = sum(1 for x in u25_prob_list  if x > 0.60) / n * 100
    pct_rec_u  = sum(1 for x in u25_prob_list  if x > 0.50) / n * 100

    return {
        "label":     label,
        "n":         n,
        "lam_home":  lam_home_list,
        "lam_away":  lam_away_list,
        "lam_total": lam_total_list,
        "u25_probs": u25_prob_list,
        "probs_1x2": probs_1x2_list,
        "labels":    labels_list,
        "bs":        bs,
        "ll":        ll,
        "hr":        hr,
        "roi":       roi,
        "yield":     yield_,
        "n_bets":    n_bets,
        "bs_over":   bs_o,
        "bs_btts":   bs_b,
        "act_u25":   act_u25,
        "pct_lt25":  pct_lt25,
        "pct_u60":   pct_u60,
        "pct_rec_u": pct_rec_u,
    }


# ══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 70)
print("BASELINE  (motor sin cambios, incluye Game State Model activo)")
print("-" * 70)
baseline = run_evaluation(matches, archives, "BASELINE")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 70)
print("ABLACION 1  -- sin elo_adj en rival_defense_strength")
print("-" * 70)

import apps.worker.ml.wc_features as wf_mod

_orig_rival = wf_mod.rival_defense_strength
_ORIG_WCA   = wf_mod.WC_AVG_GOALS


def _rival_no_elo(team: str, rounds: list, elo_ratings: dict) -> float:
    """Sin elo_adj: solo ratio historico de goles recibidos."""
    stats = wf_mod.team_historical_stats(rounds, team)
    if stats["played"] == 0:
        return 1.0
    ga_rate = stats["avg_ga"] if stats["avg_ga"] > 0 else wf_mod.WC_AVG_GOALS
    base = ga_rate / wf_mod.WC_AVG_GOALS
    return round(max(0.7, min(1.4, base)), 3)


wf_mod.rival_defense_strength = _rival_no_elo
ablation1 = run_evaluation(matches, archives, "ABL1 sin elo_adj")
wf_mod.rival_defense_strength = _orig_rival

# ══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 70)
print("ABLACION 2  -- WC_AVG_GOALS = 1.33 (era 1.28)")
print("-" * 70)

wf_mod.WC_AVG_GOALS = 1.33
ablation2 = run_evaluation(matches, archives, "ABL2 avg=1.33")
wf_mod.WC_AVG_GOALS = _ORIG_WCA

# ══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 70)
print("ABLACION 3  -- _dampen_low_scoring_favorite eliminada (impacto nulo)")
print("  Funcion removida definitivamente tras Ablation 3 original (n=128)")
print("-" * 70)

# Funcion ya eliminada de worldcup_engine.py — ablacion original confirmo
# Brier identico al baseline. Usamos baseline como proxy.
import apps.api.services.worldcup_engine as we_mod
ablation3 = run_evaluation(matches, archives, "ABL3 sin dampen (proxy=baseline)")


# ══════════════════════════════════════════════════════════════════════════════
# REPORTE FINAL
# ══════════════════════════════════════════════════════════════════════════════

results = [r for r in [baseline, ablation1, ablation2, ablation3] if r]

print()
print()
print("#" * 70)
print("   REPORTE EXPERIMENTAL -- AUDITORIA MOTOR WC2026")
print("#" * 70)

# ─── SECCION 1: Distribucion lambda ──────────────────────────────────────────

print()
print("=== SECCION 1: DISTRIBUCION REAL DE lambda (BASELINE) ===")
print()


def show_dist(arr, title):
    s = pct_stats(arr)
    print(f"  {title}")
    print(f"    n={s['n']}  mean={s['mean']:.3f}  median={s['median']:.3f}  std={s['std']:.3f}")
    print(f"    p10={s['p10']:.3f}  p25={s['p25']:.3f}  p75={s['p75']:.3f}  p90={s['p90']:.3f}")
    print(f"    min={s['min']:.3f}  max={s['max']:.3f}")
    print()
    print(histogram_ascii(arr, bins=10))
    print()


if baseline:
    show_dist(baseline["lam_home"],  "lambda_home")
    show_dist(baseline["lam_away"],  "lambda_away")
    show_dist(baseline["lam_total"], "lambda_total")
    lt_m = np.mean(baseline["lam_total"])
    print(f"  WC real 2018+2022: lambda_total_real ~ 2.66  (suma de las dos lambdas por partido)")
    print(f"  Motor baseline:    lambda_total_media = {lt_m:.3f}  -->  DELTA = {lt_m - 2.66:+.3f}")

# ─── SECCION 2-4: Porcentajes ─────────────────────────────────────────────

print()
print("=== SECCIONES 2-4: SESGOS DE MERCADO (BASELINE) ===")
print()
if baseline:
    b = baseline
    print(f"  Partidos evaluados:                   {b['n']}")
    print()
    print(f"  2) % partidos con lambda_total < 2.5: {b['pct_lt25']:.1f}%")
    print(f"     (WC real: ~35%% de partidos terminan con <=2 goles)")
    print()
    print(f"  3) % partidos con P(Under 2.5) > 60%: {b['pct_u60']:.1f}%")
    print(f"     (esperado calibrado: ~15-25%%)")
    print()
    print(f"  4) Motor recomienda Under 2.5:         {b['pct_rec_u']:.1f}%%  [P_under > 50%%]")
    print(f"     Tasa REAL Under 2.5 en datos:       {b['act_u25']:.1f}%%")
    print(f"     SESGO NETO:                         {b['pct_rec_u'] - b['act_u25']:+.1f}pp")

# ─── SECCION 5-8: Tabla comparativa ──────────────────────────────────────

print()
print("=== SECCIONES 5-8: COMPARATIVA METRICAS ===")
print()

H1 = f"  {'Experimento':<26} {'Brier 1X2':>10} {'LogLoss':>8} {'HitRate':>9} {'ROIsim':>8} {'Yield':>7} {'Nbets':>6}"
print(H1)
print("  " + "-" * 76)
for r in results:
    print(
        f"  {r['label']:<26} {r['bs']:>10.4f} {r['ll']:>8.4f} "
        f"{r['hr']*100:>8.1f}% {r['roi']*100:>7.1f}% {r['yield']*100:>6.2f}% {r['n_bets']:>6}"
    )
print()
print("  Referencia random (1/3 cada clase): Brier=0.6667  LogLoss=1.0986")

print()
H2 = f"  {'Experimento':<26} {'lhome_u':>8} {'laway_u':>8} {'ltot_u':>7} {'%<2.5':>7} {'%U>60':>7} {'%recU':>7} {'sesgo':>7}"
print(H2)
print("  " + "-" * 76)
for r in results:
    lhm = np.mean(r["lam_home"])
    lam = np.mean(r["lam_away"])
    ltm = np.mean(r["lam_total"])
    ses = r["pct_rec_u"] - r["act_u25"]
    print(
        f"  {r['label']:<26} {lhm:>8.3f} {lam:>8.3f} {ltm:>7.3f} "
        f"{r['pct_lt25']:>6.1f}% {r['pct_u60']:>6.1f}% {r['pct_rec_u']:>6.1f}% {ses:>+6.1f}pp"
    )
if baseline:
    print(f"\n  WC real: lhome~1.33 | laway~1.33 | ltotal~2.66 | U2.5_real={baseline['act_u25']:.1f}%")

# ─── Ranking mejoras ──────────────────────────────────────────────────────

if baseline:
    print()
    print("=== IMPACTO INDIVIDUAL DE CADA CAMBIO vs BASELINE ===")
    print()
    bb     = baseline["bs"]
    bll    = baseline["ll"]
    bhr    = baseline["hr"]
    bsesgo = baseline["pct_rec_u"] - baseline["act_u25"]

    print(f"  {'Ablacion':<26} {'dBrier':>8} {'dLogLoss':>9} {'dHitRate':>10} {'dSesgoU25':>11}")
    print("  " + "-" * 68)
    for r in [ablation1, ablation2, ablation3]:
        if r is None:
            continue
        db   = r["bs"] - bb
        dll  = r["ll"] - bll
        dhr  = (r["hr"] - bhr) * 100
        dses = (r["pct_rec_u"] - r["act_u25"]) - bsesgo
        b_s  = "mejor" if db  < -0.001 else ("PEOR" if db  > 0.001 else "=")
        l_s  = "mejor" if dll < -0.001 else ("PEOR" if dll > 0.001 else "=")
        h_s  = "mejor" if dhr >  0.5   else ("PEOR" if dhr < -0.5  else "=")
        u_s  = "mejor" if dses < -0.5  else ("PEOR" if dses > 0.5   else "=")
        print(
            f"  {r['label']:<26} {db:>+7.4f} {b_s:<6}  {dll:>+8.4f} {l_s:<6} "
            f"{dhr:>+9.1f}pp {h_s:<6} {dses:>+8.1f}pp {u_s}"
        )
    print()
    print("  Brier/LogLoss: negativo = MEJOR  |  HitRate: positivo = MEJOR")
    print("  dSesgoU25: negativo = MENOS sesgo de Under 2.5 (correccion del hallazgo)")

    ablaciones_ok = [r for r in [ablation1, ablation2, ablation3] if r]
    if ablaciones_ok:
        print()
        print("=== CONCLUSION ===")
        print()
        mejor_bs  = min(ablaciones_ok, key=lambda x: x["bs"])
        mejor_hr  = max(ablaciones_ok, key=lambda x: x["hr"])
        menor_ses = min(ablaciones_ok, key=lambda x: abs(x["pct_rec_u"] - x["act_u25"]))
        print(f"  Menor Brier:       {mejor_bs['label']}  ({mejor_bs['bs']:.4f} vs baseline {bb:.4f})")
        print(f"  Mayor Hit Rate:    {mejor_hr['label']}  ({mejor_hr['hr']*100:.1f}% vs baseline {bhr*100:.1f}%)")
        print(f"  Menor sesgo U2.5:  {menor_ses['label']}  ({menor_ses['pct_rec_u']-menor_ses['act_u25']:.1f}pp vs {bsesgo:.1f}pp baseline)")
        print()
        print("  NOTA: ROI simulado usa cuotas justas del modelo (1/p_modelo), no odds")
        print("  reales de mercado. Para CLV real: activar rolling_clv en wc_learning_state.")

print()
print("#" * 70)
print("  FIN DEL REPORTE")
print("#" * 70)
