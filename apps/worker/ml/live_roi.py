"""ROI simulado flat-stake sobre predicciones WC reales (Telegram / SHARP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RoiScope = Literal["sharp", "positive_ev", "all_evaluated"]


@dataclass
class LiveRoiResult:
    scope: RoiScope
    roi: float | None = None
    max_drawdown: float | None = None
    bets: int = 0
    wins: int = 0
    hit_rate: float | None = None
    pnl_total: float = 0.0
    staked: float = 0.0
    skipped_no_odds: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def _max_drawdown(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for p in pnl_series:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return round(max_dd, 4)


def _pick_odds(db, pred: dict) -> float | None:
    meta = pred.get("metadata") or {}
    clv = meta.get("clv") or {}
    raw = clv.get("pick_odds")
    if raw and float(raw) > 1:
        return float(raw)

    try:
        match_key = f"{pred['team_home']}|{pred['team_away']}"
        row = (
            db.schema("ml")
            .table("odds_snapshots")
            .select("odds_decimal")
            .eq("match_key", match_key)
            .eq("market", pred.get("market_type", "1X2"))
            .eq("selection", pred.get("predicted_outcome"))
            .eq("snapshot_type", "pick")
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        if row.data and row.data[0].get("odds_decimal"):
            o = float(row.data[0]["odds_decimal"])
            return o if o > 1 else None
    except Exception:
        pass
    return None


def _matches_scope(pred: dict, scope: RoiScope, *, min_ev_fair: float) -> bool:
    meta = pred.get("metadata") or {}
    if scope == "sharp":
        return bool(meta.get("sharp_allowed")) or meta.get("source") == "sharp_scan"
    if scope == "positive_ev":
        ev = pred.get("expected_value_fair")
        if ev is not None and float(ev) >= min_ev_fair:
            return True
        return bool(meta.get("sharp_allowed")) or meta.get("source") == "sharp_scan"
    return True


def simulate_live_roi_from_db(
    db,
    *,
    scope: RoiScope = "sharp",
    min_ev_fair: float = 0.03,
    flat_stake: float = 1.0,
    limit: int = 300,
) -> LiveRoiResult:
    """
    Flat stake en picks evaluados con cuota pick (CLV chain o snapshot).

    scope:
      sharp — solo picks SHARP /alta
      positive_ev — EV fair ≥ umbral o SHARP
      all_evaluated — todos los evaluados con cuota
    """
    result = LiveRoiResult(scope=scope)
    try:
        rows = (
            db.schema("ml")
            .table("wc_predictions")
            .select(
                "team_home, team_away, market_type, predicted_outcome, "
                "is_correct, expected_value_fair, metadata, evaluated_at"
            )
            .not_.is_("evaluated_at", "null")
            .order("evaluated_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        result.details = {"error": str(exc)}
        return result

    pnl: list[float] = []
    for pred in rows.data or []:
        if not _matches_scope(pred, scope, min_ev_fair=min_ev_fair):
            continue
        odds = _pick_odds(db, pred)
        if not odds or odds <= 1:
            result.skipped_no_odds += 1
            continue

        won = bool(pred.get("is_correct"))
        stake = flat_stake
        profit = stake * (odds - 1) if won else -stake
        pnl.append(profit)
        result.bets += 1
        if won:
            result.wins += 1

    if result.bets == 0:
        result.details = {
            "skipped_no_odds": result.skipped_no_odds,
            "reason": "sin apuestas simulables",
        }
        return result

    result.pnl_total = round(sum(pnl), 4)
    result.staked = round(result.bets * flat_stake, 4)
    result.roi = round(result.pnl_total / result.staked, 4)
    result.max_drawdown = _max_drawdown(pnl)
    result.hit_rate = round(result.wins / result.bets, 4)
    result.details = {
        "skipped_no_odds": result.skipped_no_odds,
        "flat_stake": flat_stake,
    }
    return result
