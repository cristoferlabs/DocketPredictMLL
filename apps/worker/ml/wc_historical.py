"""World Cup historical match extraction and leak-free feature building."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

from apps.api.services.worldcup_engine import (
    calc_elo_ratings,
    compute_model_markets,
    home_factor,
    normalize_openfootball,
    team_historical_stats,
)
from apps.worker.ml.wc_features import compute_match_lambdas


@dataclass
class HistoricalMatch:
    date: datetime
    team1: str
    team2: str
    home_goals: int
    away_goals: int
    competition: str
    round_name: str = ""


def extract_finished_matches(
    archives: dict[int, dict[str, Any]],
    years: list[int] | None = None,
) -> list[HistoricalMatch]:
    """All finished WC matches sorted by kickoff date."""
    years = years or sorted(archives.keys())
    rows: list[HistoricalMatch] = []
    for year in years:
        data = archives.get(year, {})
        if not data:
            continue
        norm = normalize_openfootball(data)
        comp = f"wc_{year}"
        for rnd in norm.get("rounds", []):
            for m in rnd.get("matches", []):
                ft = m.get("score", {}).get("ft")
                if not ft or not m.get("date"):
                    continue
                t1 = m.get("team1", {}).get("name", "")
                t2 = m.get("team2", {}).get("name", "")
                if not t1 or not t2:
                    continue
                try:
                    d = datetime.fromisoformat(m["date"].replace("Z", "+00:00").split("T")[0])
                except ValueError:
                    continue
                rows.append(
                    HistoricalMatch(
                        date=d,
                        team1=t1,
                        team2=t2,
                        home_goals=int(ft[0]),
                        away_goals=int(ft[1]),
                        competition=comp,
                        round_name=rnd.get("name", ""),
                    )
                )
    rows.sort(key=lambda r: r.date)
    return rows


def archives_before_date(
    archives: dict[int, dict],
    cutoff: datetime,
) -> tuple[dict, dict, dict]:
    """Build 2018/2022/2026-style archives using only matches strictly before cutoff."""
    d18: dict = {"rounds": []}
    d22: dict = {"rounds": []}
    d26: dict = {"rounds": []}

    def _bucket(year: int) -> dict:
        if year <= 2018:
            return d18
        if year <= 2022:
            return d22
        return d26

    for year, data in archives.items():
        norm = normalize_openfootball(data)
        round_map: dict[str, dict] = {}
        for rnd in norm.get("rounds", []):
            for m in rnd.get("matches", []):
                if not m.get("date") or not m.get("score", {}).get("ft"):
                    continue
                try:
                    d = datetime.fromisoformat(m["date"].replace("Z", "+00:00").split("T")[0])
                except ValueError:
                    continue
                if d >= cutoff:
                    continue
                rname = rnd.get("name", "Unknown")
                round_map.setdefault(rname, {"name": rname, "matches": []})
                round_map[rname]["matches"].append(m)
        target = _bucket(year)
        target["rounds"].extend(round_map.values())

    return d18, d22, d26


def predict_match_historical(
    match: HistoricalMatch,
    archives: dict[int, dict],
    *,
    calibrate: bool = False,
) -> dict[str, float]:
    """
    Model prediction using only data available before match.date (no leakage).
    Default calibrate=False for honest isotonic/backtest fitting.
    """
    from apps.worker.ml.calibration import calibrate_model_markets

    d18, d22, d26 = archives_before_date(archives, match.date)
    elo = calc_elo_ratings(d18, d22, d26)

    d18n = normalize_openfootball(d18)
    d22n = normalize_openfootball(d22)
    stats1 = team_historical_stats(d22n.get("rounds", []), match.team1)
    stats2 = team_historical_stats(d22n.get("rounds", []), match.team2)
    stats1_18 = team_historical_stats(d18n.get("rounds", []), match.team1)
    stats2_18 = team_historical_stats(d18n.get("rounds", []), match.team2)
    hist1 = stats1["avg_gf"] * 0.6 + stats1_18["avg_gf"] * 0.4 if stats1["played"] else stats1_18["avg_gf"]
    hist2 = stats2["avg_gf"] * 0.6 + stats2_18["avg_gf"] * 0.4 if stats2["played"] else stats2_18["avg_gf"]

    lambdas = compute_match_lambdas(
        match.team1,
        match.team2,
        [],
        [],
        hist1,
        hist2,
        elo,
        d18n.get("rounds", []),
        d22n.get("rounds", []),
        [],
    )
    raw = compute_model_markets(
        lambdas.lambda_home,
        lambdas.lambda_away,
        elo.get(match.team1, 1500),
        elo.get(match.team2, 1500),
        calibrate=False,
    )
    probs = {
        "home_win": raw.home_win,
        "draw": raw.draw,
        "away_win": raw.away_win,
        "over_25": raw.over_25,
        "under_25": raw.under_25,
        "btts_yes": raw.btts_yes,
        "btts_no": raw.btts_no,
    }
    if calibrate:
        from apps.api.services.worldcup_engine import get_calibration_factors

        return calibrate_model_markets(
            probs["home_win"],
            probs["draw"],
            probs["away_win"],
            probs["over_25"],
            probs["under_25"],
            probs["btts_yes"],
            probs["btts_no"],
            factors=get_calibration_factors(),
        )
    return probs


def actual_outcomes(match: HistoricalMatch) -> dict[str, Any]:
    g1, g2 = match.home_goals, match.away_goals
    total = g1 + g2
    return {
        "home_win": 1 if g1 > g2 else 0,
        "draw": 1 if g1 == g2 else 0,
        "away_win": 1 if g1 < g2 else 0,
        "over_25": 1 if total > 2.5 else 0,
        "under_25": 1 if total <= 2.5 else 0,
        "btts_yes": 1 if g1 > 0 and g2 > 0 else 0,
        "btts_no": 1 if g1 == 0 or g2 == 0 else 0,
        "label_1x2": 0 if g1 > g2 else 1 if g1 == g2 else 2,
    }


def iter_walk_forward_windows(
    matches: list[HistoricalMatch],
    train_size: int,
    test_size: int,
) -> Iterator[tuple[list[HistoricalMatch], list[HistoricalMatch]]]:
    """Rolling window by match count (WC has ~64 games per edition)."""
    if train_size + test_size > len(matches):
        yield matches[: max(1, len(matches) // 2)], matches[max(1, len(matches) // 2) :]
        return
    start = 0
    while start + train_size + test_size <= len(matches):
        train = matches[start : start + train_size]
        test = matches[start + train_size : start + train_size + test_size]
        yield train, test
        start += test_size
