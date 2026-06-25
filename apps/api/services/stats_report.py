"""Reporte cuantitativo — Brier, CLV, ROI, tiers SHARP (Telegram /stats)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.services.engine_health import evaluate_engine_health, format_health_alerts
from apps.worker.ml.calibration_metrics import load_fitted_model_weights
from apps.worker.ml.live_roi import simulate_live_roi_from_db
from apps.worker.ml.model_learning import evaluate_live_brier_from_db, load_learning_state


@dataclass
class WcPredictionStats:
    total: int = 0
    pending: int = 0
    evaluated: int = 0
    hit_rate: float | None = None
    avg_brier: float | None = None
    avg_clv: float | None = None
    clv_n: int = 0
    tier_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


def aggregate_wc_predictions(db) -> WcPredictionStats:
    """Agrega métricas live desde ml.wc_predictions."""
    stats = WcPredictionStats()
    try:
        rows = (
            db.schema("ml")
            .table("wc_predictions")
            .select(
                "id, is_correct, brier_score, evaluated_at, metadata, "
                "expected_value_fair, soft_action"
            )
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
        data = rows.data or []
        stats.total = len(data)
        evaluated_rows = [r for r in data if r.get("evaluated_at")]
        stats.evaluated = len(evaluated_rows)
        stats.pending = stats.total - stats.evaluated

        if evaluated_rows:
            correct = sum(1 for r in evaluated_rows if r.get("is_correct"))
            stats.hit_rate = round(correct / len(evaluated_rows), 4)
            briers = [
                float(r["brier_score"])
                for r in evaluated_rows
                if r.get("brier_score") is not None
            ]
            if briers:
                stats.avg_brier = round(sum(briers) / len(briers), 4)

        clv_vals: list[float] = []
        tier_buckets: dict[str, list[dict]] = {}

        for r in data:
            meta = r.get("metadata") or {}
            clv_block = meta.get("clv") or {}
            clv = clv_block.get("clv_vs_close")
            if clv is not None:
                try:
                    clv_vals.append(float(clv))
                except (TypeError, ValueError):
                    pass

            tier = meta.get("sharp_tier")
            if tier:
                tier_buckets.setdefault(str(tier), []).append(r)

        if clv_vals:
            stats.clv_n = len(clv_vals)
            stats.avg_clv = round(sum(clv_vals) / len(clv_vals), 4)

        for tier, items in tier_buckets.items():
            ev = [x for x in items if x.get("evaluated_at")]
            hits = sum(1 for x in ev if x.get("is_correct"))
            stats.tier_stats[tier] = {
                "n": len(items),
                "evaluated": len(ev),
                "hit_rate": round(hits / len(ev), 4) if ev else None,
            }
    except Exception:
        pass
    return stats


def fetch_backtest_metrics(db) -> dict[str, Any] | None:
    try:
        row = (
            db.schema("ml")
            .table("model_performance_metrics")
            .select("hit_rate, roi_sim, calibration_error, sample_size, created_at")
            .eq("market_type", "wc_backtest")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if row.data:
            return row.data[0]
    except Exception:
        pass
    return None


def count_odds_snapshots(db) -> dict[str, int]:
    counts = {"opening": 0, "pick": 0, "closing": 0, "market": 0}
    try:
        for stype in counts:
            r = (
                db.schema("ml")
                .table("odds_snapshots")
                .select("id", count="exact")
                .eq("snapshot_type", stype)
                .execute()
            )
            counts[stype] = int(r.count or 0)
    except Exception:
        pass
    return counts


def _pct(v: float | None, *, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_pp_clv(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.2f}pp vs cierre"


def build_roi_report(db, *, settings: Any | None = None) -> str:
    """Mensaje compacto Telegram /roi — ROI live + backtest."""
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    health = evaluate_engine_health(db, settings=settings)

    lines = [
        "💰 ROI — Motor WC 2026",
        "─────────────────────────────",
        f"Salud motor: {health.status.upper()}",
    ]
    roi_alerts = [a for a in health.alerts if "ROI" in a]
    if roi_alerts:
        lines.append(f"⚠️ {roi_alerts[0]}")

    lines.append("\n📡 Live (flat 1u, picks evaluados)")
    for scope, label in (
        ("sharp", "SHARP"),
        ("positive_ev", "+EV fair"),
        ("all_evaluated", "Todos evaluados"),
    ):
        r = simulate_live_roi_from_db(
            db, scope=scope, min_ev_fair=settings.ev_min_edge_fair
        )
        if r.bets > 0 and r.roi is not None:
            lines.append(
                f"  {label}: {r.roi*100:+.1f}% | {r.bets} bets | "
                f"hit {_pct(r.hit_rate)} | DD {r.max_drawdown or 0:.2f}u"
            )
            if r.skipped_no_odds:
                lines.append(f"    ({r.skipped_no_odds} sin cuota pick)")
        else:
            lines.append(f"  {label}: — (sin muestra)")

    wc = aggregate_wc_predictions(db)
    lines.append(f"\n📋 Muestra: {wc.evaluated} evaluadas / {wc.pending} pendientes")
    if wc.clv_n > 0:
        lines.append(f"  CLV medio ({wc.clv_n}): {_fmt_pp_clv(wc.avg_clv)}")

    bt = fetch_backtest_metrics(db)
    lines.append("\n📈 Backtest histórico (último job)")
    if bt:
        roi = bt.get("roi_sim")
        lines.append(
            f"  ROI sim: {_pct(float(roi)) if roi is not None else '—'} | "
            f"hit {_pct(float(bt['hit_rate'])) if bt.get('hit_rate') is not None else '—'}"
        )
        lines.append(f"  ECE {float(bt.get('calibration_error', 0)):.3f} | n={bt.get('sample_size', 0)}")
    else:
        lines.append("  Sin backtest en DB — job semanal o run_backtest")

    lines.append(f"\n🛡️ Umbral ROI backtest deploy: ≥ {settings.ev_min_roi_backtest:.0%}")
    lines.append("⚠️ ROI live necesita resultados reales (/alta + partidos jugados).")
    return "\n".join(lines)


def build_pro_stats_report(db, *, settings: Any | None = None) -> str:
    """Mensaje Telegram para /stats — dashboard cuant."""
    from apps.shared.config import get_settings

    settings = settings or get_settings()
    health = evaluate_engine_health(db, settings=settings)
    lines = [
        "📊 STATS PRO — Motor WC 2026",
        "─────────────────────────────",
    ]
    lines.extend(format_health_alerts(health))
    if health.status != "ok":
        lines.append("")

    weights = load_fitted_model_weights() or {}
    w_p = weights.get("poisson", settings.model_weight_poisson)
    w_e = weights.get("elo", settings.model_weight_elo)
    lines.append("⚙️ Blend activo")
    lines.append(f"  Poisson {float(w_p)*100:.0f}% · ELO {float(w_e)*100:.0f}%")
    dampen = weights.get("underdog_dampen_factor")
    if dampen is not None and float(dampen) < 1.0:
        lines.append(f"  Underdog dampen: {float(dampen):.2f}")

    train = weights.get("train") or {}
    test = weights.get("test") or {}
    if train.get("brier_1x2") is not None:
        lines.append("\n📐 Fit histórico (2018 train / 2022 test)")
        lines.append(
            f"  Brier train {float(train['brier_1x2']):.3f} | "
            f"test {float(test.get('brier_1x2', 0)):.3f}"
        )
        if train.get("hit_rate_1x2") is not None:
            lines.append(
                f"  Hit rate train {_pct(float(train['hit_rate_1x2']))} | "
                f"test {_pct(float(test.get('hit_rate_1x2', 0)))}"
            )
        infl = train.get("underdog_inflation_pp")
        if infl is not None:
            lines.append(f"  Sesgo underdog (train): {float(infl):+.1f}pp")

    state = load_learning_state()
    lines.append("\n🔄 Learning loop (live)")
    lines.append(f"  Updates: {state.n_updates} | desde retrain: {state.results_since_retrain}")
    lines.append(f"  Brier rolling: {state.rolling_brier or '—'}")
    lines.append(f"  CLV rolling: {_fmt_pp_clv(state.rolling_clv)}")
    if state.last_retrain_at:
        lines.append(f"  Último retrain: {state.last_retrain_at[:10]}")

    live_brier = evaluate_live_brier_from_db(db)
    wc = aggregate_wc_predictions(db)
    lines.append("\n🎯 Predicciones WC (Telegram)")
    lines.append(f"  Total {wc.total} | pendientes {wc.pending} | evaluadas {wc.evaluated}")
    if wc.hit_rate is not None:
        lines.append(f"  Hit rate live: {_pct(wc.hit_rate)}")
    if wc.avg_brier is not None:
        lines.append(f"  Brier live (DB): {wc.avg_brier:.3f}")
    elif live_brier is not None:
        lines.append(f"  Brier live (DB): {live_brier:.3f}")
    if wc.clv_n > 0:
        lines.append(f"  CLV medio ({wc.clv_n} picks): {_fmt_pp_clv(wc.avg_clv)}")

    roi_sharp = simulate_live_roi_from_db(
        db, scope="sharp", min_ev_fair=settings.ev_min_edge_fair
    )
    roi_ev = simulate_live_roi_from_db(
        db, scope="positive_ev", min_ev_fair=settings.ev_min_edge_fair
    )
    lines.append("\n💰 ROI simulado live (flat 1u)")
    if roi_sharp.bets > 0 and roi_sharp.roi is not None:
        lines.append(
            f"  SHARP: {roi_sharp.roi*100:+.1f}% | {roi_sharp.bets} bets | "
            f"hit {_pct(roi_sharp.hit_rate)} | DD {roi_sharp.max_drawdown or 0:.2f}u"
        )
    else:
        lines.append("  SHARP: — (usa /alta y espera resultados)")
    if roi_ev.bets > 0 and roi_ev.roi is not None and roi_ev.bets != roi_sharp.bets:
        lines.append(
            f"  +EV: {roi_ev.roi*100:+.1f}% | {roi_ev.bets} bets | "
            f"hit {_pct(roi_ev.hit_rate)}"
        )

    if wc.tier_stats:
        lines.append("\n🏷️ SHARP tiers (persistidos)")
        for tier in sorted(wc.tier_stats):
            t = wc.tier_stats[tier]
            hr = _pct(t["hit_rate"]) if t.get("hit_rate") is not None else "—"
            lines.append(
                f"  Tier {tier}: n={t['n']} eval={t['evaluated']} hit={hr}"
            )
    else:
        lines.append("\n🏷️ SHARP tiers: sin histórico aún (usa /alta)")

    bt = fetch_backtest_metrics(db)
    if bt:
        lines.append("\n📈 Backtest WC (último job)")
        roi = bt.get("roi_sim")
        lines.append(
            f"  ROI sim flat: {_pct(float(roi)) if roi is not None else '—'} | "
            f"hit {_pct(float(bt['hit_rate'])) if bt.get('hit_rate') is not None else '—'}"
        )
        lines.append(f"  ECE {float(bt.get('calibration_error', 0)):.3f} | n={bt.get('sample_size', 0)}")

    snaps = count_odds_snapshots(db)
    if any(snaps.values()):
        lines.append("\n📸 Snapshots odds (CLV chain)")
        lines.append(
            f"  opening {snaps['opening']} | pick {snaps['pick']} | "
            f"closing {snaps['closing']} | market {snaps['market']}"
        )

    lines.append("\n🛡️ Guardrails")
    lines.append(f"  EV min fair: {settings.ev_min_edge_fair*100:.1f}% | max: {settings.ev_max_edge_fair*100:.1f}%")
    lines.append(f"  ROI backtest mín: {settings.ev_min_roi_backtest}")
    lines.append(f"  Brier live máx: {settings.model_max_live_brier_1x2}")

    lines.append("\nScripts: run_learning_cycle.py | run_backtest.py")
    lines.append("⚠️ Brier↓ y CLV+ sostenido = motor sano.")
    return "\n".join(lines)
