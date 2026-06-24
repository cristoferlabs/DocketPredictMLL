"""Closing Line Value — loop completo predicción → apertura → cierre → resultado."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apps.worker.ml.wc_predictions import compute_clv, save_odds_snapshot

logger = logging.getLogger(__name__)

COMPETITION = "fifa_world_cup"


def _odds_table(db):
    return db.schema("ml").table("odds_snapshots")


def _has_snapshot(
    db,
    *,
    match_key: str,
    market: str,
    selection: str,
    snapshot_type: str,
) -> bool:
    try:
        row = (
            _odds_table(db)
            .select("id")
            .eq("match_key", match_key)
            .eq("market", market)
            .eq("selection", selection)
            .eq("snapshot_type", snapshot_type)
            .limit(1)
            .execute()
        )
        return bool(row.data)
    except Exception:
        return False


def _latest_odds(
    db,
    *,
    match_key: str,
    market: str,
    selection: str,
    snapshot_type: str,
) -> float | None:
    try:
        row = (
            _odds_table(db)
            .select("odds_decimal")
            .eq("match_key", match_key)
            .eq("market", market)
            .eq("selection", selection)
            .eq("snapshot_type", snapshot_type)
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        if row.data:
            return float(row.data[0]["odds_decimal"])
    except Exception as exc:
        logger.warning("latest_odds %s: %s", snapshot_type, exc)
    return None


def record_opening_snapshot(
    db,
    *,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    odds_decimal: float,
    fair_odds: float | None = None,
) -> bool:
    """Primera línea vista = apertura (solo una por match/mercado/selección)."""
    if odds_decimal <= 1:
        return False
    match_key = f"{team_home}|{team_away}"
    if _has_snapshot(
        db,
        match_key=match_key,
        market=market,
        selection=selection,
        snapshot_type="opening",
    ):
        return False
    save_odds_snapshot(
        db,
        match_key=match_key,
        team_home=team_home,
        team_away=team_away,
        market=market,
        selection=selection,
        odds_decimal=odds_decimal,
        fair_odds=fair_odds,
        snapshot_type="opening",
    )
    return True


def record_wc_market_snapshots(
    db,
    *,
    team_home: str,
    team_away: str,
    market_ctx,
) -> int:
    """Persist 1X2 lines: opening (1ª vez) + market (cada análisis)."""
    if not market_ctx or not getattr(market_ctx, "has_market", False):
        return 0
    match_key = f"{team_home}|{team_away}"
    saved = 0
    for o in market_ctx.outcomes:
        raw = o.market_odds
        if not raw or raw <= 1:
            continue
        record_opening_snapshot(
            db,
            team_home=team_home,
            team_away=team_away,
            market="1X2",
            selection=o.selection,
            odds_decimal=raw,
            fair_odds=o.fair_odds,
        )
        save_odds_snapshot(
            db,
            match_key=match_key,
            team_home=team_home,
            team_away=team_away,
            market="1X2",
            selection=o.selection,
            odds_decimal=raw,
            fair_odds=o.fair_odds,
            snapshot_type="market",
        )
        saved += 1
    return saved


def record_pick_snapshot(
    db,
    *,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    odds_decimal: float,
    fair_odds: float | None = None,
    prediction_id: str | None = None,
) -> str:
    match_key = f"{team_home}|{team_away}"
    save_odds_snapshot(
        db,
        match_key=match_key,
        team_home=team_home,
        team_away=team_away,
        market=market,
        selection=selection,
        odds_decimal=odds_decimal,
        fair_odds=fair_odds,
        snapshot_type="pick",
    )
    if prediction_id:
        _attach_clv_metadata(
            db,
            prediction_id,
            {
                "pick_odds": odds_decimal,
                "pick_fair_odds": fair_odds,
                "match_key": match_key,
                "clv_stage": "pick",
            },
        )
    return match_key


def record_closing_snapshot(
    db,
    *,
    match_key: str,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    closing_odds: float,
    prediction_id: str | None = None,
) -> float | None:
    """Línea de cierre pre-kickoff; devuelve CLV vs último pick."""
    save_odds_snapshot(
        db,
        match_key=match_key,
        team_home=team_home,
        team_away=team_away,
        market=market,
        selection=selection,
        odds_decimal=closing_odds,
        snapshot_type="closing",
    )
    pick_odds = _latest_odds(
        db,
        match_key=match_key,
        market=market,
        selection=selection,
        snapshot_type="pick",
    )
    clv = compute_clv(pick_odds, closing_odds) if pick_odds else None
    if prediction_id and clv is not None:
        opening = _latest_odds(
            db,
            match_key=match_key,
            market=market,
            selection=selection,
            snapshot_type="opening",
        )
        _attach_clv_metadata(
            db,
            prediction_id,
            {
                "opening_odds": opening,
                "closing_odds": closing_odds,
                "pick_odds": pick_odds,
                "clv_vs_close": clv,
                "clv_stage": "closed",
            },
        )
    return clv


def build_clv_chain(
    db,
    *,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    prediction_id: str | None = None,
) -> dict[str, Any]:
    """
    Cadena CLV: predicción → apertura → cierre (último market si no hay closing).

    Usado al evaluar resultado para cerrar el loop.
    """
    match_key = f"{team_home}|{team_away}"
    opening = _latest_odds(
        db, match_key=match_key, market=market, selection=selection, snapshot_type="opening"
    )
    pick = _latest_odds(
        db, match_key=match_key, market=market, selection=selection, snapshot_type="pick"
    )
    closing = _latest_odds(
        db, match_key=match_key, market=market, selection=selection, snapshot_type="closing"
    )
    if closing is None:
        closing = _latest_odds(
            db, match_key=match_key, market=market, selection=selection, snapshot_type="market"
        )
    clv = compute_clv(pick, closing) if pick and closing else None
    chain = {
        "match_key": match_key,
        "opening_odds": opening,
        "pick_odds": pick,
        "closing_odds": closing,
        "clv_vs_close": clv,
        "clv_stage": "result_pending" if clv is not None else "incomplete",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    if prediction_id:
        _attach_clv_metadata(db, prediction_id, chain)
    return chain


def finalize_clv_on_result(
    db,
    *,
    prediction_id: str,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    is_correct: bool | None,
    actual_outcome: str | None,
) -> dict[str, Any]:
    """Cierra loop CLV al evaluar resultado."""
    chain = build_clv_chain(
        db,
        team_home=team_home,
        team_away=team_away,
        market=market,
        selection=selection,
        prediction_id=None,
    )
    chain["is_correct"] = is_correct
    chain["actual_outcome"] = actual_outcome
    chain["clv_stage"] = "complete"
    chain["finalized_at"] = datetime.now(timezone.utc).isoformat()
    _attach_clv_metadata(db, prediction_id, chain)
    return chain


def _attach_clv_metadata(db, prediction_id: str, patch: dict[str, Any]) -> None:
    try:
        row = (
            db.schema("ml")
            .table("wc_predictions")
            .select("metadata")
            .eq("id", prediction_id)
            .limit(1)
            .execute()
        )
        meta = (row.data[0].get("metadata") if row.data else None) or {}
        clv = dict(meta.get("clv") or {})
        clv.update(patch)
        meta["clv"] = clv
        db.schema("ml").table("wc_predictions").update({"metadata": meta}).eq(
            "id", prediction_id
        ).execute()
    except Exception as exc:
        logger.warning("attach_clv_metadata: %s", exc)


def fatigue_multiplier(form: list[dict], *, short_rest_days: int = 3) -> float:
    """
    Penalize λ when last match was within short_rest_days (calendar congestion).
    Optional Phase 3 signal — lightweight heuristic.
    """
    if not form:
        return 1.0
    last_date = (form[0].get("fecha") or "")[:10]
    if not last_date:
        return 1.0
    try:
        from datetime import date

        d = date.fromisoformat(last_date)
        days_ago = (date.today() - d).days
        if days_ago <= short_rest_days:
            return 0.96
        if days_ago <= 5:
            return 0.98
    except ValueError:
        pass
    return 1.0
