"""Closing Line Value helpers."""

from __future__ import annotations

from apps.worker.ml.wc_predictions import compute_clv, save_odds_snapshot


def record_pick_snapshot(
    db,
    *,
    team_home: str,
    team_away: str,
    market: str,
    selection: str,
    odds_decimal: float,
    fair_odds: float | None = None,
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
) -> float | None:
    """Store pre-kickoff closing odds; return CLV vs last pick snapshot if found."""
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
    try:
        picks = (
            db.schema("ml")
            .table("odds_snapshots")
            .select("odds_decimal")
            .eq("match_key", match_key)
            .eq("market", market)
            .eq("selection", selection)
            .eq("snapshot_type", "pick")
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        if picks.data:
            pick_odds = float(picks.data[0]["odds_decimal"])
            return compute_clv(pick_odds, closing_odds)
    except Exception:
        pass
    return None


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
