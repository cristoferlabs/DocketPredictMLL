"""
Fit bucket calibration (paso A+B) y tune vs mercado (paso C).

Uso:
  python scripts/fit_bucket_calibration.py
"""

from __future__ import annotations

import asyncio
import sys

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

    async def run(self, limit: int = 24) -> FavoriteBiasAudit:
        await self.ensure_loaded()
        assert self._archives is not None and self._elo is not None
        d26 = self._archives.get(2026, {})
        d22 = self._archives.get(2022, {})
        d18 = self._archives.get(2018, {})
        db = get_supabase()
        rows = []
        for match in find_upcoming_matches(d26, days_ahead=14)[:limit]:
            analysis = analyze_match(match, d18, d22, [], self._elo)
            if not analysis.model:
                continue
            odds = await find_wc_odds_event(
                analysis.team1,
                analysis.team2,
                db=db,
                events_cache=self._odds_events,
            )
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


async def run_market_audit(limit: int = 24):
    return await _SESSION.run(limit=limit)


def _apply_factors(factors: dict) -> None:
    set_calibration_factors(factors)


async def tune_step_c(factors: dict, target: float = 0.25, max_iter: int = 40):
    import copy

    best_factors = copy.deepcopy(factors)
    seeded_buckets = seed_buckets_for_market_bias(
        best_factors.get("1X2_buckets", {}),
        await run_market_audit(),
    )
    best_factors["1X2_buckets"] = seeded_buckets
    _apply_factors(best_factors)

    best_audit = await run_market_audit()
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
            trial_audit = await run_market_audit()
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
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("=== PASO A: Fit isotonic por bucket (WC 2018+2022) ===\n")
    archives = await fetch_all_worldcup_archives()
    factors, _cals, metrics = fit_calibration_bundle(archives, train_years=[2018, 2022])
    bucket_m = metrics.get("bucket_calibration", {})
    print("Bucket ECE:", bucket_m.get("ece_by_bucket", {}))
    print("Team win factors (isotonic):", bucket_m.get("team_win_factors", {}))
    print("1X2_buckets:", factors.get("1X2_buckets", {}))

    set_calibration_factors(factors)
    audit_before = await run_market_audit()
    print(f"\n=== Tras paso A+B (antes tune C) ===")
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
    path = save_fitted_calibration_factors(tuned)

    print(f"\nIteraciones: {iters}")
    print(f"Score final: {audit_after.favorite_bias_score:+.3f}")
    print(f"Buckets finales:\n  {tuned.get('1X2_buckets')}")
    print(f"\nGuardado: {path}")
    report_path = path.parent / "wc_bucket_audit_report.txt"
    report_path.write_text(format_audit_report(audit_after), encoding="utf-8")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
