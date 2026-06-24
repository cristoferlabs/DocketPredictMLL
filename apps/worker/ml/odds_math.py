"""Fair odds, devig and EV math — shared by API and worker."""

from __future__ import annotations

import statistics
from typing import Sequence


def implied_probability(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def overround(odds_dict: dict[str, float]) -> float:
    """Bookmaker margin as fraction (e.g. 0.05 = 5%)."""
    probs = [implied_probability(o) for o in odds_dict.values() if o > 1.0]
    if not probs:
        return 0.0
    return max(0.0, sum(probs) - 1.0)


def devig_multiclass(odds_dict: dict[str, float]) -> dict[str, float]:
    """
    Multiplicative devig for N outcomes from the same bookmaker.
    Returns fair probabilities summing to 1.
    """
    raw: dict[str, float] = {}
    for key, odds in odds_dict.items():
        if odds > 1.0:
            raw[key] = implied_probability(odds)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in raw.items()}


def devig_two_way(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Devig a two-way market (e.g. O/U 2.5)."""
    fair = devig_multiclass({"over": over_odds, "under": under_odds})
    return fair.get("over", 0.0), fair.get("under", 0.0)


def fair_odds(fair_prob: float) -> float:
    if fair_prob <= 0:
        return 0.0
    return round(1.0 / fair_prob, 4)


def expected_value_fair(model_prob: float, fair_odds_value: float) -> float:
    """EV against vig-free fair price."""
    if fair_odds_value <= 1.0:
        return 0.0
    return round(model_prob * fair_odds_value - 1.0, 4)


def expected_value_raw(model_prob: float, book_odds: float) -> float:
    """EV against raw book odds (often inflated)."""
    if book_odds <= 1.0:
        return 0.0
    return round(model_prob * book_odds - 1.0, 4)


def expected_value_executable(
    model_prob: float,
    book_odds: float,
    *,
    vig_removed: bool = True,
    fair_prob: float | None = None,
) -> float:
    """
    EV with explicit vig handling.
    If vig_removed=True, uses fair_prob to derive fair_odds; else raw book odds.
    """
    if vig_removed:
        fp = fair_prob if fair_prob is not None else implied_probability(book_odds)
        return expected_value_fair(model_prob, fair_odds(fp))
    return expected_value_raw(model_prob, book_odds)


def _median(values: Sequence[float]) -> float:
    clean = [v for v in values if v > 0]
    if not clean:
        return 0.0
    return statistics.median(clean)


def aggregate_fair_probs(
    per_book_fair_probs: list[dict[str, float]],
) -> dict[str, float]:
    """Average fair probabilities across bookmakers."""
    if not per_book_fair_probs:
        return {}
    keys = set()
    for d in per_book_fair_probs:
        keys.update(d.keys())
    result: dict[str, float] = {}
    for key in keys:
        vals = [d[key] for d in per_book_fair_probs if key in d and d[key] > 0]
        if vals:
            result[key] = round(sum(vals) / len(vals), 6)
    return result


def aggregate_median_fair_odds(
    per_book_fair_probs: list[dict[str, float]],
) -> dict[str, float]:
    """Median fair decimal odds per outcome across bookmakers."""
    if not per_book_fair_probs:
        return {}
    keys = set()
    for d in per_book_fair_probs:
        keys.update(d.keys())
    result: dict[str, float] = {}
    for key in keys:
        odds_list = [fair_odds(d[key]) for d in per_book_fair_probs if key in d and d[key] > 0]
        med = _median(odds_list)
        if med > 1.0:
            result[key] = round(med, 4)
    return result


def extract_h2h_per_bookmaker(event: dict) -> list[dict[str, float]]:
    """Raw h2h odds per bookmaker (home/draw/away)."""
    home_team = (event.get("home_team") or "").lower()
    away_team = (event.get("away_team") or "").lower()
    books: list[dict[str, float]] = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            odds: dict[str, float] = {}
            for out in market.get("outcomes", []):
                name = (out.get("name") or "").lower()
                price = float(out.get("price", 0))
                if price <= 1.0:
                    continue
                if name == home_team:
                    odds["home"] = price
                elif name == away_team:
                    odds["away"] = price
                else:
                    odds["draw"] = price
            if len(odds) >= 2:
                books.append(odds)
    return books


def extract_totals_per_bookmaker(event: dict, point: float = 2.5) -> list[dict[str, float]]:
    """Raw totals odds per bookmaker (over/under)."""
    books: list[dict[str, float]] = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "totals":
                continue
            odds: dict[str, float] = {}
            for out in market.get("outcomes", []):
                if float(out.get("point", 0)) != point:
                    continue
                key = "over" if (out.get("name") or "").lower() == "over" else "under"
                price = float(out.get("price", 0))
                if price > 1.0:
                    odds[key] = price
            if len(odds) == 2:
                books.append(odds)
    return books


def fair_h2h_market(event: dict) -> dict[str, dict[str, float]]:
    """
    Returns per-outcome: fair_prob, fair_odds, median_raw_odds, vig_pct.
  """
    per_book = extract_h2h_per_bookmaker(event)
    if not per_book:
        return {}

    fair_probs_list = [devig_multiclass(b) for b in per_book if len(b) >= 2]
    fair_probs = aggregate_fair_probs(fair_probs_list)
    fair_odds_map = aggregate_median_fair_odds(fair_probs_list)

    median_raw: dict[str, float] = {}
    keys = set()
    for b in per_book:
        keys.update(b.keys())
    for key in keys:
        median_raw[key] = round(_median([b[key] for b in per_book if key in b]), 4)

    avg_vig = round(
        sum(overround(b) for b in per_book) / len(per_book) * 100,
        2,
    ) if per_book else 0.0

    result: dict[str, dict[str, float]] = {}
    for key in fair_probs:
        result[key] = {
            "fair_prob": fair_probs[key],
            "fair_odds": fair_odds_map.get(key, fair_odds(fair_probs[key])),
            "raw_odds": median_raw.get(key, 0.0),
            "vig_pct": avg_vig,
        }
    return result


def fair_totals_market(event: dict, point: float = 2.5) -> dict[str, dict[str, float]]:
    per_book = extract_totals_per_bookmaker(event, point)
    if not per_book:
        return {}

    fair_probs_list = [devig_multiclass(b) for b in per_book]
    fair_probs = aggregate_fair_probs(fair_probs_list)
    fair_odds_map = aggregate_median_fair_odds(fair_probs_list)

    median_raw: dict[str, float] = {}
    for key in ("over", "under"):
        median_raw[key] = round(_median([b[key] for b in per_book if key in b]), 4)

    avg_vig = round(
        sum(overround(b) for b in per_book) / len(per_book) * 100,
        2,
    ) if per_book else 0.0

    result: dict[str, dict[str, float]] = {}
    for key in fair_probs:
        result[key] = {
            "fair_prob": fair_probs[key],
            "fair_odds": fair_odds_map.get(key, fair_odds(fair_probs[key])),
            "raw_odds": median_raw.get(key, 0.0),
            "vig_pct": avg_vig,
        }
    return result
