"""
Per-market calibration tracker.

Logs every market recommendation (with real bookmaker odds) to ml.picks_log.
After matches finish, label_picks.py resolves outcomes and this module
provides the calibration report query.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import Client
    from apps.api.services.telegram_terminal.betting_menu import MarketRow
    from apps.api.services.worldcup_engine import MatchAnalysis

logger = logging.getLogger(__name__)

# Minimum number of labeled picks before reporting calibration for a market
_MIN_SAMPLE = 15

_MARKET_TYPE_MAP = {
    "1X2":   "1X2",
    "DC":    "DC",
    "BTTS":  "BTTS",
    "OU":    None,      # resolved per-row from label
    "STATS": None,      # resolved per-row from label
}


def _classify_market(row: "MarketRow") -> str:
    """Normalize MarketRow → canonical market_type key for picks_log."""
    mt = row.market_type
    label_l = row.label.lower()
    if mt == "1X2":
        return "1X2"
    if mt == "DC":
        return "DC"
    if mt == "BTTS":
        return "BTTS"
    if mt == "OU":
        # Extract line from label: "Over 2.5" → "OU_2.5"
        for line in ("1.5", "2.5", "3.5"):
            if line in label_l:
                return f"OU_{line}"
        return "OU"
    if mt == "STATS":
        if "corner" in label_l:
            return "CORNERS"
        if "sot" in label_l or "tiro" in label_l.replace("á", "a"):
            return "SOT"
        if "tarjeta" in label_l or "card" in label_l:
            return "CARDS"
        return "STATS"
    return mt


def log_market_rows(
    db: "Client",
    analysis: "MatchAnalysis",
    market_rows: list["MarketRow"],
    *,
    min_ev_pct: float = -10.0,
) -> int:
    """
    Log picks from the betting menu to ml.picks_log.
    Only rows with real bookmaker odds (has_market=True) are logged.
    Silently ignores errors — never raises.
    Returns count of rows inserted.
    """
    fecha_str = (analysis.fecha or "")[:10] or None
    match_key = f"{analysis.team1}|{analysis.team2}|{fecha_str or ''}"
    rows_to_insert = []

    for row in market_rows:
        if not row.has_market:
            continue
        ev = row.ev_pct or 0.0
        if ev < min_ev_pct:
            continue
        market_type = _classify_market(row)
        rows_to_insert.append({
            "match_key":   match_key,
            "team_home":   analysis.team1,
            "team_away":   analysis.team2,
            "fecha":       fecha_str,
            "market_type": market_type,
            "selection":   row.label,
            "model_prob":  round(row.model_prob, 4),
            "market_odds": round(row.market_odds, 3) if row.market_odds else None,
            "ev_pct":      round(ev, 2),
        })

    if not rows_to_insert:
        return 0

    # Avoid duplicates: skip if this match_key + market_type + selection already logged today
    try:
        existing = (
            db.schema("ml").table("picks_log")
            .select("selection, market_type")
            .eq("match_key", match_key)
            .execute()
        )
        already = {
            (r["market_type"], r["selection"])
            for r in (existing.data or [])
        }
        rows_to_insert = [
            r for r in rows_to_insert
            if (r["market_type"], r["selection"]) not in already
        ]
    except Exception as exc:
        logger.debug("picks_log dedup check: %s", exc)

    if not rows_to_insert:
        return 0

    try:
        db.schema("ml").table("picks_log").insert(rows_to_insert).execute()
        logger.info(
            "picks_log: logged %d rows for %s vs %s",
            len(rows_to_insert), analysis.team1, analysis.team2,
        )
        return len(rows_to_insert)
    except Exception as exc:
        logger.warning("picks_log insert: %s", exc)
        return 0


def get_calibration_report(
    db: "Client",
    *,
    days: int = 90,
    min_sample: int = _MIN_SAMPLE,
) -> list[dict]:
    """
    Query calibration stats per market type from ml.picks_log.
    Returns list of dicts with: market_type, picks, wins, avg_model_pct, real_hit_pct, gap_pp, flag.
    Ordered by picks desc.
    Only includes markets with at least min_sample labeled picks.
    """
    try:
        cutoff = str(date.today().replace(day=max(1, date.today().day - days)))
        rows = (
            db.schema("ml").table("picks_log")
            .select("market_type, model_prob, outcome")
            .gte("fecha", cutoff)
            .not_.is_("outcome", "null")
            .limit(5000)
            .execute()
        )
    except Exception as exc:
        logger.warning("calibration_report query: %s", exc)
        return []

    # Aggregate in Python (avoids needing RPC / SQL GROUP BY)
    from collections import defaultdict
    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "prob_sum": 0.0})
    for r in (rows.data or []):
        mt = r["market_type"]
        buckets[mt]["n"] += 1
        if r["outcome"]:
            buckets[mt]["wins"] += 1
        buckets[mt]["prob_sum"] += float(r["model_prob"] or 0)

    results = []
    for mt, b in buckets.items():
        n = b["n"]
        if n < min_sample:
            continue
        avg_model_pct = round(b["prob_sum"] / n * 100, 1)
        real_hit_pct = round(b["wins"] / n * 100, 1)
        gap_pp = round(avg_model_pct - real_hit_pct, 1)
        if abs(gap_pp) <= 5.0:
            flag = "✅"
        elif gap_pp > 5.0:
            flag = "⚠ Sobreestima"
        else:
            flag = "⚠ Subestima"
        results.append({
            "market_type":   mt,
            "picks":         n,
            "wins":          b["wins"],
            "avg_model_pct": avg_model_pct,
            "real_hit_pct":  real_hit_pct,
            "gap_pp":        gap_pp,
            "flag":          flag,
        })

    results.sort(key=lambda r: -r["picks"])
    return results


def format_calibration_report(rows: list[dict], *, days: int = 90) -> str:
    """Format calibration report as Telegram-ready text table."""
    if not rows:
        return (
            f"📊 CALIBRACIÓN ({days}d)\n\n"
            f"Sin datos suficientes aún (min {_MIN_SAMPLE} picks por mercado con resultado).\n\n"
            "Los picks se registran automáticamente cuando usas el menú de apuestas.\n"
            "Después de los partidos, el sistema etiqueta los resultados."
        )

    lines = [
        f"📊 CALIBRACIÓN DEL MODELO — últimos {days}d",
        "",
        f"{'Mercado':<14} {'Picks':>5}  {'Modelo':>6}  {'Real':>6}  {'Δ':>5}",
        "─" * 44,
    ]
    for r in rows:
        gap_s = f"{r['gap_pp']:+.1f}pp"
        lines.append(
            f"{r['market_type']:<14} {r['picks']:>5}  "
            f"{r['avg_model_pct']:>5.1f}%  {r['real_hit_pct']:>5.1f}%  "
            f"{gap_s:>6}  {r['flag']}"
        )

    # Summary
    overest = [r for r in rows if r["gap_pp"] > 5]
    underest = [r for r in rows if r["gap_pp"] < -5]
    total_picks = sum(r["picks"] for r in rows)
    total_wins = sum(r["wins"] for r in rows)
    overall_hit = round(total_wins / total_picks * 100, 1) if total_picks else 0.0

    lines += [
        "─" * 44,
        f"  Total: {total_picks} picks | Hit rate global: {overall_hit:.1f}%",
        "",
    ]
    if overest:
        lines.append(
            "⚠ Sobreestima: " + ", ".join(r["market_type"] for r in overest)
        )
        lines.append("  → El modelo asigna más prob de la que se cumple. Ajusta el threshold.")
    if underest:
        lines.append(
            "⚠ Subestima: " + ", ".join(r["market_type"] for r in underest)
        )
        lines.append("  → El modelo es conservador. EV real podría ser mayor.")
    if not overest and not underest:
        lines.append("✅ Todos los mercados bien calibrados (Δ ≤ 5pp)")

    lines += [
        "",
        "ℹ️ Δ = Modelo% − Real% | ⚠ = gap >5pp",
        "Picks registrados automáticamente desde el menú E.",
    ]
    return "\n".join(lines)
