"""
Fit bucket calibration (paso A+B) y tune vs mercado (paso C).

Uso:
  python scripts/fit_bucket_calibration.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from apps.api.services.live_calibration import calibrate_analysis_model
from apps.api.services.odds_context import (
    compute_market_context,
    find_wc_odds_event,
    load_wc_odds_events,
)
from apps.api.services.worldcup_engine import analyze_match, find_upcoming_matches, set_calibration_factors
from apps.shared.supabase_client import get_supabase
from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives
from apps.worker.ml.calibration import (
    fit_calibration_bundle,
    load_fitted_calibration_factors,
    propose_market_tune_candidates,
    save_fitted_calibration_factors,
    seed_buckets_for_market_bias,
)
from apps.worker.ml.favorite_bias_audit import (
    FavoriteBiasAudit,
    _match_row,
    aggregate_favorite_bias,
    format_audit_report,
)
from apps.worker.tasks.update_elo import get_wc_elo_ratings


class _MarketAuditSession:
    """Una carga de cuotas + auditoría repetible (sin N requests API)."""

    def __init__(self) -> None:
        self._odds_events: list[dict] = []
        self._api_status: dict = {}
        self._loaded = False
        self._warned_api = False
        self._archives: dict | None = None
        self._elo: dict | None = None

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        db = get_supabase()
        self._odds_events, self._api_status = await load_wc_odds_events(db)
        self._archives = await fetch_all_worldcup_archives()
        self._elo = await get_wc_elo_ratings(db)
        self._loaded = True
        if not self._api_status.get("ok") and not self._warned_api:
            reason = self._api_status.get("reason", "error")
            detail = self._api_status.get("detail", "")
            print(
                f"⚠️  Odds API no live ({reason}): {detail}\n"
                f"    Paso C usa caché — {len(self._odds_events)} eventos WC"
            )
            self._warned_api = True
        elif self._api_status.get("ok"):
            print(f"Odds API OK — {len(self._odds_events)} eventos WC (1 request, reutilizado)")

    async def run(self, limit: int = 24, *, use_live_calibration: bool = False) -> FavoriteBiasAudit:
        await self.ensure_loaded()
        assert self._archives is not None and self._elo is not None
        d26 = self._archives.get(2026, {})
        d22 = self._archives.get(2022, {})
        d18 = self._archives.get(2018, {})
        db = get_supabase()
        rows = []
        for match in find_upcoming_matches(d26, days_ahead=14)[:limit]:
            t1 = match.get("team1", {})
            t2 = match.get("team2", {})
            if isinstance(t1, dict):
                t1 = t1.get("name", "")
            if isinstance(t2, dict):
                t2 = t2.get("name", "")
            odds = await find_wc_odds_event(
                t1,
                t2,
                db=db,
                events_cache=self._odds_events,
            )
            analysis = analyze_match(match, d18, d22, [], self._elo, odds_event=odds)
            if not analysis.model:
                continue
            if use_live_calibration:
                calibrate_analysis_model(analysis, odds)
            ctx = compute_market_context(
                analysis.model, analysis.team1, analysis.team2, odds
            )
            if not ctx.has_market:
                continue
            outcomes = [(o.selection, o.model_prob, o.market_implied) for o in ctx.outcomes]
            row = _match_row(analysis.team1, analysis.team2, outcomes)
            if row:
                rows.append(row)
        audit = aggregate_favorite_bias(rows)
        if not rows:
            print("⚠️  Sin partidos con cuotas para auditar — paso C no puede tune vs mercado.")
        return audit


_SESSION = _MarketAuditSession()


async def run_market_audit(limit: int = 24, *, use_live_calibration: bool = False):
    return await _SESSION.run(limit=limit, use_live_calibration=use_live_calibration)


def _print_audit_summary(label: str, audit: FavoriteBiasAudit) -> None:
    print(f"\n=== {label} ===")
    print(f"favorite_bias_score: {audit.favorite_bias_score:+.3f}")
    if audit.n_matches:
        print(
            f"  compresión fav: {audit.favorite_compression_avg*100:+.1f}pp | "
            f"inflación empate: {audit.draw_inflation_avg*100:+.1f}pp | "
            f"inflación dog: {audit.underdog_inflation_avg*100:+.1f}pp"
        )
        print(f"  partidos: {audit.n_matches} | outcomes: {audit.n_outcomes}")


async def compare_live_calibration_audit(limit: int = 24) -> tuple[FavoriteBiasAudit, FavoriteBiasAudit]:
    """Compara P_stat vs P_cal sobre mismos partidos/cuotas."""
    stat = await run_market_audit(limit=limit, use_live_calibration=False)
    cal = await run_market_audit(limit=limit, use_live_calibration=True)
    _print_audit_summary("AUDIT P_statistical (modelo base)", stat)
    _print_audit_summary("AUDIT P_calibrated (live_calibration)", cal)
    if stat.n_matches and cal.n_matches:
        print("\n── Delta patch (cal − stat) ──")
        print(f"  favorite_bias_score: {cal.favorite_bias_score - stat.favorite_bias_score:+.3f}")
        print(
            f"  underdog inflation: "
            f"{(cal.underdog_inflation_avg - stat.underdog_inflation_avg)*100:+.1f}pp"
        )
        targets = (
            ("favorite_bias_score", 0.12, cal.favorite_bias_score),
            ("underdog_inflation_pp", 5.0, cal.underdog_inflation_avg * 100),
        )
        print("\n── Objetivos post-patch ──")
        for name, target, value in targets:
            ok = abs(value) < target if name != "underdog_inflation_pp" else abs(value) < target
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name}: {value:+.3f} (target < {target})")
    return stat, cal


def _apply_factors(factors: dict) -> None:
    set_calibration_factors(factors)


async def tune_step_c(factors: dict, target: float = 0.25, max_iter: int = 40):
    import copy

    best_factors = copy.deepcopy(factors)
    seeded_buckets = seed_buckets_for_market_bias(
        best_factors.get("1X2_buckets", {}),
        await run_market_audit(use_live_calibration=True),
    )
    best_factors["1X2_buckets"] = seeded_buckets
    _apply_factors(best_factors)

    best_audit = await run_market_audit(use_live_calibration=True)
    best_obj = abs(best_audit.favorite_bias_score)
    step_scale = 1.0
    iterations = 0
    stagnation = 0
    print(
        f"  seed: score {best_audit.favorite_bias_score:+.3f} | "
        f"favorite={seeded_buckets.get('team_win', {}).get('favorite')} | "
        f"max_lift={seeded_buckets.get('compressed_favorite_max_lift')}"
    )

    for i in range(max_iter):
        if abs(best_audit.favorite_bias_score) <= target:
            break

        candidates = propose_market_tune_candidates(
            best_factors, best_audit, step_scale=step_scale
        )
        accepted = False
        for candidate in candidates:
            _apply_factors(candidate)
            trial_audit = await run_market_audit(use_live_calibration=True)
            trial_obj = abs(trial_audit.favorite_bias_score)
            if trial_obj < best_obj - 1e-5:
                best_factors = candidate
                best_audit = trial_audit
                best_obj = trial_obj
                iterations = i + 1
                stagnation = 0
                step_scale = min(1.5, step_scale * 1.06)
                accepted = True
                bk = best_factors.get("1X2_buckets", {})
                tw = bk.get("team_win", {})
                print(
                    f"  iter {iterations}: score {best_audit.favorite_bias_score:+.3f} "
                    f"(fav {best_audit.favorite_compression_avg*100:+.1f}pp | "
                    f"draw {best_audit.draw_inflation_avg*100:+.1f}pp | "
                    f"dog {best_audit.underdog_inflation_avg*100:+.1f}pp) "
                    f"[fav_f={tw.get('favorite')} lift={bk.get('compressed_favorite_max_lift')}]"
                )
                break

        if accepted:
            continue
        stagnation += 1
        step_scale *= 0.55
        _apply_factors(best_factors)
        if stagnation >= 8:
            break

    _apply_factors(best_factors)
    return best_factors, best_audit, iterations


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit bucket calibration WC")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Solo comparar P_stat vs P_cal (sin fit ni tune)",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.audit_only:
        fitted = load_fitted_calibration_factors()
        if fitted:
            set_calibration_factors(fitted)
        _, cal = await compare_live_calibration_audit()
        report_path = Path("artifacts/calibration/wc_bucket_audit_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_audit_report(cal), encoding="utf-8")
        print(f"\nReporte P_cal: {report_path}")
        return

    print("=== PASO A: Fit isotonic por bucket (WC 2018+2022) ===\n")
    archives = await fetch_all_worldcup_archives()
    factors, _cals, metrics = fit_calibration_bundle(archives, train_years=[2018, 2022])
    bucket_m = metrics.get("bucket_calibration", {})
    print("Bucket ECE:", bucket_m.get("ece_by_bucket", {}))
    print("Team win factors (isotonic):", bucket_m.get("team_win_factors", {}))
    print("1X2_buckets:", factors.get("1X2_buckets", {}))

    set_calibration_factors(factors)
    await compare_live_calibration_audit()
    audit_before = await run_market_audit(use_live_calibration=True)
    print(f"\n=== Tras paso A+B (P_cal, antes tune C) ===")
    print(f"favorite_bias_score: {audit_before.favorite_bias_score:+.3f}")
    if audit_before.n_matches:
        print(
            f"  compresión fav: {audit_before.favorite_compression_avg*100:+.1f}pp | "
            f"inflación empate: {audit_before.draw_inflation_avg*100:+.1f}pp | "
            f"inflación dog: {audit_before.underdog_inflation_avg*100:+.1f}pp"
        )

    print("\n=== PASO C: Tune vs mercado live (target |score| < 0.25) ===\n")
    if audit_before.n_matches == 0:
        print("Omitido — sin cuotas de mercado. Renueva ODDS_API_KEY o ejecuta ingest de cuotas WC.")
        tuned = factors
        audit_after = audit_before
        iters = 0
    else:
        tuned, audit_after, iters = await tune_step_c(factors, target=0.25, max_iter=30)

    set_calibration_factors(tuned)

    from apps.shared.supabase_client import get_supabase
    from apps.worker.ml.backtest import evaluate_calibration_holdout_roi
    from apps.worker.ml.calibration_metrics import load_fitted_model_weights
    from apps.worker.ml.model_learning import (
        deploy_calibration_gate,
        evaluate_live_brier_from_db,
        load_learning_state,
    )

    db = get_supabase()
    live_brier = evaluate_live_brier_from_db(db)
    weights_art = load_fitted_model_weights() or {}
    hist_brier = float((weights_art.get("test") or {}).get("brier_1x2", 0.0)) or None
    bt_roi, bt_details = evaluate_calibration_holdout_roi(archives, tuned, db=db)
    approved, block_reasons = deploy_calibration_gate(
        audit=audit_after,
        live_brier=live_brier,
        historical_brier=hist_brier,
        backtest_roi=bt_roi,
        backtest_roi_details=bt_details,
    )
    learning = load_learning_state()

    print(f"\n=== GATE DEPLOY (Fase C) ===")
    print(f"Holdout ROI (2022): {bt_roi if bt_roi is not None else 'n/d'}")
    if bt_details.get("bets"):
        print(f"  bets={bt_details['bets']} hit={bt_details.get('hit_rate')}")
    print(f"Live Brier (DB): {live_brier if live_brier is not None else 'n/d'}")
    print(f"Hist Brier test: {hist_brier if hist_brier else 'n/d'}")
    print(f"Rolling Brier: {learning.rolling_brier}")
    print(f"Deploy: {'✓ APROBADO' if approved else '✗ BLOQUEADO'}")
    if block_reasons:
        for r in block_reasons:
            print(f"  — {r}")

    if not approved:
        tuned = dict(tuned)
        tuned["_deploy_reasons"] = block_reasons
        path = save_fitted_calibration_factors(tuned, approved=False)
        set_calibration_factors(load_fitted_calibration_factors())
        print(f"\nCalibración candidata bloqueada: {path}")
        print("Runtime sigue con artifact aprobado previo.")
    else:
        path = save_fitted_calibration_factors(tuned, approved=True)
        print(f"\nCalibración activa para runtime.")

    print(f"\nIteraciones: {iters}")
    print(f"Score final: {audit_after.favorite_bias_score:+.3f}")
    print(f"Buckets finales:\n  {tuned.get('1X2_buckets')}")
    print(f"\nGuardado: {path}")
    report_path = path.parent / "wc_bucket_audit_report.txt"
    report_path.write_text(format_audit_report(audit_after), encoding="utf-8")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
