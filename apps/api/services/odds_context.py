"""Odds context for EV — bookmaker data NEVER modifies model probabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from apps.api.services.worldcup_engine import ModelMarkets, name_match
from apps.worker.ingest.odds_api import OddsApiClient
from apps.worker.ml.odds_math import (
    expected_value_fair,
    expected_value_raw,
    fair_h2h_market,
    fair_totals_market,
)

logger = logging.getLogger(__name__)

MIN_ODDS_BOOKS = 1


@dataclass
class EvOpportunity:
    market: str
    selection: str
    model_prob: float
    book_odds: float
    implied_prob: float
    expected_value: float
    edge_pct: float
    priority: str
    raw_odds: float = 0.0
    fair_odds: float = 0.0
    vig_pct: float = 0.0
    edge_fair: float = 0.0
    expected_value_raw: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class OutcomeEdge:
    """Model vs mercado fair para un desenlace 1X2."""

    selection: str
    model_prob: float
    fair_odds: float
    raw_odds: float | None
    fair_implied: float
    edge_pct: float


@dataclass
class MarketContext1X2:
    has_market: bool
    outcomes: list[OutcomeEdge]
    n_books: int = 0


def expected_value(model_prob: float, odds: float) -> float:
    """Backward-compatible alias — returns fair-style EV on given odds."""
    return expected_value_fair(model_prob, odds)


def _priority_fair(ev_fair: float, edge_fair: float) -> str:
    if ev_fair >= 0.04 and edge_fair >= 0.04:
        return "high"
    if ev_fair >= 0.02 and edge_fair >= 0.02:
        return "medium"
    return "low"


def _match_odds_event(ev: dict, team1: str, team2: str) -> bool:
    home = ev.get("home_team", "")
    away = ev.get("away_team", "")
    return (name_match(home, team1) and name_match(away, team2)) or (
        name_match(home, team2) and name_match(away, team1)
    )


def _find_wc_odds_in_db(db, team1: str, team2: str) -> dict | None:
    """Fallback: últimas cuotas ingeridas en ops.raw_ingestions."""
    try:
        rows = (
            db.schema("ops")
            .table("raw_ingestions")
            .select("payload")
            .eq("entity_type", "odds_event")
            .order("ingested_at", desc=True)
            .limit(300)
            .execute()
        )
        for row in rows.data or []:
            ev = row.get("payload") or {}
            if isinstance(ev, dict) and _match_odds_event(ev, team1, team2):
                return ev
    except Exception as exc:
        logger.debug("odds cache db: %s", exc)
    return None


async def find_wc_odds_event(team1: str, team2: str, db=None) -> dict | None:
    client = OddsApiClient()
    try:
        events = await client.get_soccer_odds(sports=["soccer_fifa_world_cup"])
        for ev in events:
            if _match_odds_event(ev, team1, team2):
                return ev
    except Exception as exc:
        logger.warning("Odds API live: %s", exc)

    if db is not None:
        cached = _find_wc_odds_in_db(db, team1, team2)
        if cached:
            logger.info("Cuotas WC desde caché DB: %s vs %s", team1, team2)
            return cached
    return None


def compute_market_context(
    model: ModelMarkets,
    team1: str,
    team2: str,
    odds_event: dict | None,
) -> MarketContext1X2:
    """Cuotas fair y edge por desenlace (siempre, aunque no haya +EV)."""
    if odds_event and _count_bookmakers(odds_event) >= 1:
        h2h_fair = fair_h2h_market(odds_event)
        outcomes: list[OutcomeEdge] = []
        mapping = [
            (team1, model.home_win, "home"),
            ("Empate", model.draw, "draw"),
            (team2, model.away_win, "away"),
        ]
        for selection, model_prob, key in mapping:
            if key not in h2h_fair or model_prob <= 0:
                continue
            fm = h2h_fair[key]
            fair_p = fm["fair_prob"]
            outcomes.append(
                OutcomeEdge(
                    selection=selection,
                    model_prob=model_prob,
                    fair_odds=fm["fair_odds"],
                    raw_odds=fm.get("raw_odds"),
                    fair_implied=fair_p,
                    edge_pct=round((model_prob - fair_p) * 100, 1),
                )
            )
        if outcomes:
            return MarketContext1X2(
                has_market=True,
                outcomes=outcomes,
                n_books=_count_bookmakers(odds_event),
            )

    outcomes = []
    for selection, prob in [
        (team1, model.home_win),
        ("Empate", model.draw),
        (team2, model.away_win),
    ]:
        if prob <= 0:
            continue
        outcomes.append(
            OutcomeEdge(
                selection=selection,
                model_prob=prob,
                fair_odds=round(1 / prob, 2),
                raw_odds=None,
                fair_implied=prob,
                edge_pct=0.0,
            )
        )
    return MarketContext1X2(has_market=False, outcomes=outcomes)


def _count_bookmakers(event: dict) -> int:
    return len(event.get("bookmakers", []))


def compute_ev_opportunities(
    model: ModelMarkets,
    team1: str,
    team2: str,
    odds_event: dict | None,
    *,
    single_best: bool = True,
) -> list[EvOpportunity]:
    """
    Compara modelo vs mercado fair (devig por casa + mediana).
    El modelo NO se ajusta a las cuotas.
    Por defecto devuelve una sola oportunidad (mayor fair edge).
    """
    if not odds_event:
        return []

    if _count_bookmakers(odds_event) < MIN_ODDS_BOOKS:
        logger.warning("Insufficient bookmakers for fair odds: %s", _count_bookmakers(odds_event))
        return []

    h2h_fair = fair_h2h_market(odds_event)
    totals_fair = fair_totals_market(odds_event, 2.5)

    candidates: list[EvOpportunity] = []

    mapping = [
        ("1X2", team1, model.home_win, "home", h2h_fair),
        ("1X2", "Empate", model.draw, "draw", h2h_fair),
        ("1X2", team2, model.away_win, "away", h2h_fair),
        ("Over/Under 2.5", "Over", model.over_25, "over", totals_fair),
        ("Over/Under 2.5", "Under", model.under_25, "under", totals_fair),
    ]

    for market, selection, model_prob, key, fair_market in mapping:
        if model_prob <= 0 or key not in fair_market:
            continue
        fm = fair_market[key]
        fair_p = fm["fair_prob"]
        fair_o = fm["fair_odds"]
        raw_o = fm["raw_odds"]
        vig = fm["vig_pct"]

        edge_fair = model_prob - fair_p
        ev_fair = expected_value_fair(model_prob, fair_o)
        ev_raw = expected_value_raw(model_prob, raw_o) if raw_o > 1.0 else 0.0

        if ev_fair > 0 and edge_fair > 0:
            candidates.append(
                EvOpportunity(
                    market=market,
                    selection=selection,
                    model_prob=model_prob,
                    book_odds=fair_o,
                    implied_prob=round(fair_p, 4),
                    expected_value=ev_fair,
                    edge_pct=round(edge_fair * 100, 1),
                    priority=_priority_fair(ev_fair, edge_fair),
                    raw_odds=raw_o,
                    fair_odds=fair_o,
                    vig_pct=vig,
                    edge_fair=round(edge_fair, 4),
                    expected_value_raw=ev_raw,
                )
            )

    candidates.sort(key=lambda x: x.edge_fair, reverse=True)

    if single_best and candidates:
        return [candidates[0]]
    return candidates


# Legacy helpers — delegate to odds_math for tests/imports
def extract_h2h_odds(event: dict) -> dict[str, float]:
    """Median raw odds per outcome (deprecated: use fair_h2h_market)."""
    fair = fair_h2h_market(event)
    return {k: v["raw_odds"] for k, v in fair.items() if v.get("raw_odds")}


def extract_totals_odds(event: dict, point: float = 2.5) -> dict[str, float]:
    fair = fair_totals_market(event, point)
    return {k: v["raw_odds"] for k, v in fair.items() if v.get("raw_odds")}
