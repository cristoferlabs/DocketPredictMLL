"""Odds context for EV — bookmaker data NEVER modifies model probabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from apps.api.services.worldcup_engine import ModelMarkets, name_match
from apps.api.services.ev_policy import ev_for_decision, regime_ev_cap
from apps.worker.ingest.odds_api import OddsApiClient, WC_SPORT
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
    """Modelo vs mercado — fair EV para decisión, raw EV informativo."""

    selection: str
    model_prob: float
    model_fair_odds: float  # 1 / prob modelo — cuota justa teórica
    market_odds: float | None  # mediana casas brutas (Odds API)
    edge_pct: float  # EV fair × 100 — usado en decisión (alias ev_fair_pct)
    market_implied: float | None = None  # 1 / cuota mercado (sin devig)
    divergence: float | None = None  # |model_prob - market_implied|
    fair_odds: float | None = None  # cuota fair devig
    fair_implied: float | None = None
    ev_fair_pct: float = 0.0
    ev_raw_pct: float = 0.0
    edge_fair_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.ev_fair_pct == 0.0 and self.edge_pct != 0.0:
            self.ev_fair_pct = self.edge_pct
        if self.edge_fair_pct == 0.0 and self.ev_fair_pct != 0.0:
            self.edge_fair_pct = self.ev_fair_pct


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
    for ev in _all_wc_odds_from_db(db):
        if _match_odds_event(ev, team1, team2):
            return ev
    return None


def _all_wc_odds_from_db(db) -> list[dict]:
    """Todos los eventos WC únicos en caché (más reciente por partido)."""
    try:
        rows = (
            db.schema("ops")
            .table("raw_ingestions")
            .select("payload")
            .eq("entity_type", "odds_event")
            .order("ingested_at", desc=True)
            .limit(500)
            .execute()
        )
    except Exception as exc:
        logger.debug("odds cache db list: %s", exc)
        return []

    seen: set[str] = set()
    events: list[dict] = []
    for row in rows.data or []:
        ev = row.get("payload") or {}
        if not isinstance(ev, dict):
            continue
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        if not home or not away:
            continue
        key = f"{home.lower()}|{away.lower()}"
        if key in seen:
            continue
        seen.add(key)
        events.append(ev)
    return events


def find_wc_odds_in_events(
    events: list[dict],
    team1: str,
    team2: str,
) -> dict | None:
    for ev in events:
        if _match_odds_event(ev, team1, team2):
            return ev
    return None


async def load_wc_odds_events(db=None) -> tuple[list[dict], dict]:
    """
    Carga cuotas WC una sola vez (live o caché).

    Returns (events, api_status).
    """
    client = OddsApiClient()
    status = await client.check_status()
    events: list[dict] = []
    if status.get("ok"):
        try:
            data = await client._get(
                f"/sports/{WC_SPORT}/odds",
                {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
            )
            if isinstance(data, list):
                for event in data:
                    event["_sport_key"] = WC_SPORT
                events = data
        except Exception as exc:
            logger.warning("Odds API WC fetch: %s", exc)
    if not events and db is not None:
        events = _all_wc_odds_from_db(db)
    return events, status


async def find_wc_odds_event(
    team1: str,
    team2: str,
    db=None,
    *,
    events_cache: list[dict] | None = None,
) -> dict | None:
    if events_cache is not None:
        hit = find_wc_odds_in_events(events_cache, team1, team2)
        if hit:
            return hit
        if db is not None:
            return _find_wc_odds_in_db(db, team1, team2)
        return None

    client = OddsApiClient()
    status = await client.check_status()
    if status.get("ok"):
        try:
            data = await client._get(
                f"/sports/{WC_SPORT}/odds",
                {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
            )
            if isinstance(data, list):
                for ev in data:
                    ev["_sport_key"] = WC_SPORT
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


async def odds_unavailable_reason(db=None) -> str:
    """Mensaje claro para Telegram cuando no hay cuotas de mercado."""
    status = await OddsApiClient().check_status()
    reason = status.get("reason", "")
    if reason == "cuota_mensual_agotada":
        rem = status.get("remaining", "0")
        return f"cuota Odds API agotada ({rem} créditos restantes). Nueva clave en the-odds-api.com o espera reset mensual"
    if reason == "clave_invalida":
        return "ODDS_API_KEY inválida en .env"
    if not status.get("ok") and status.get("detail"):
        return f"Odds API: {status['detail'][:80]}"

    if db is not None:
        try:
            rows = (
                db.schema("ops")
                .table("raw_ingestions")
                .select("payload")
                .eq("entity_type", "odds_event")
                .order("ingested_at", desc=True)
                .limit(50)
                .execute()
            )
            wc_cached = sum(
                1
                for r in rows.data or []
                if (r.get("payload") or {}).get("_sport_key") == WC_SPORT
                or (r.get("payload") or {}).get("sport_key") == WC_SPORT
            )
            if wc_cached == 0 and (rows.data or []):
                return "caché sin partidos WC (solo ligas europeas). Activa cuota y ejecuta ingest"
        except Exception:
            pass
    return "sin cuotas WC para este partido (API o caché vacía)"


def compute_market_context(
    model: ModelMarkets,
    team1: str,
    team2: str,
    odds_event: dict | None,
) -> MarketContext1X2:
    """
    Cuotas fair del modelo, línea de mercado (Odds API) y EV dual.

    ev_fair / edge_fair → decisión.  ev_raw → solo informativo.
    """
    mapping = [
        (team1, model.home_win, "home"),
        ("Empate", model.draw, "draw"),
        (team2, model.away_win, "away"),
    ]

    if odds_event and _count_bookmakers(odds_event) >= 1:
        h2h_fair = fair_h2h_market(odds_event)
        outcomes: list[OutcomeEdge] = []
        for selection, model_prob, key in mapping:
            if model_prob <= 0:
                continue
            model_fair = round(1 / model_prob, 2)
            fm = h2h_fair.get(key, {})
            market_o = fm.get("raw_odds") if fm else None
            fair_o = fm.get("fair_odds") if fm else None
            fair_p = fm.get("fair_prob") if fm else None
            ev_fair = (
                round(expected_value_fair(model_prob, fair_o) * 100, 1)
                if fair_o and fair_o > 1
                else 0.0
            )
            ev_raw = (
                round(expected_value_raw(model_prob, market_o) * 100, 1)
                if market_o and market_o > 1
                else 0.0
            )
            edge_pp = round((model_prob - fair_p) * 100, 1) if fair_p else 0.0
            impl = market_implied_prob(market_o) if market_o and market_o > 1 else None
            div = outcome_divergence(model_prob, market_o) if impl is not None else None
            outcomes.append(
                OutcomeEdge(
                    selection=selection,
                    model_prob=model_prob,
                    model_fair_odds=model_fair,
                    market_odds=market_o if market_o and market_o > 1 else None,
                    edge_pct=ev_fair,
                    market_implied=round(impl, 4) if impl is not None else None,
                    divergence=round(div, 4) if div is not None else None,
                    fair_odds=fair_o if fair_o and fair_o > 1 else None,
                    fair_implied=round(fair_p, 4) if fair_p else None,
                    ev_fair_pct=ev_fair,
                    ev_raw_pct=ev_raw,
                    edge_fair_pct=edge_pp,
                )
            )
        if outcomes and any(o.market_odds for o in outcomes):
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
                model_fair_odds=round(1 / prob, 2),
                market_odds=None,
                edge_pct=0.0,
            )
        )
    return MarketContext1X2(has_market=False, outcomes=outcomes)


def market_implied_prob(odds: float | None) -> float | None:
    if odds and odds > 1:
        return 1.0 / odds
    return None


def outcome_divergence(model_prob: float, market_odds: float | None) -> float | None:
    impl = market_implied_prob(market_odds)
    if impl is None:
        return None
    return abs(model_prob - impl)


def max_market_divergence(market_ctx: MarketContext1X2 | None) -> float:
    if not market_ctx or not market_ctx.has_market:
        return 0.0
    return max((o.divergence or 0.0 for o in market_ctx.outcomes), default=0.0)


def check_market_outcome_allowed(
    outcome: OutcomeEdge,
    *,
    max_divergence: float,
    max_ev: float,
) -> tuple[bool, list[str]]:
    """Bloquea picks con desacople modelo-mercado o EV outlier."""
    flags: list[str] = []
    if outcome.divergence is not None and outcome.divergence > max_divergence:
        flags.append(f"divergence>{max_divergence:.0%}")
    ev = outcome.ev_fair_pct / 100.0
    if ev > max_ev:
        flags.append(f"ev_fair>{max_ev:.0%}")
    if outcome.ev_fair_pct <= 0:
        flags.append("ev_fair<=0")
    return len(flags) == 0, flags


def best_bettable_market_ev(
    market_ctx: MarketContext1X2 | None,
    *,
    max_divergence: float,
    max_ev: float,
) -> float:
    """Mayor EV entre desenlaces que pasan guardrails de mercado."""
    if not market_ctx or not market_ctx.has_market:
        return 0.0
    allowed: list[float] = []
    for o in market_ctx.outcomes:
        ok, _ = check_market_outcome_allowed(
            o, max_divergence=max_divergence, max_ev=max_ev
        )
        if ok:
            allowed.append(o.ev_fair_pct / 100.0)
    return max(allowed, default=0.0)


def best_market_ev(market_ctx: MarketContext1X2 | None) -> float:
    """Mayor EV fair (decimal) entre desenlaces 1X2 — para decisión."""
    if not market_ctx or not market_ctx.has_market:
        return 0.0
    return max((o.ev_fair_pct / 100.0 for o in market_ctx.outcomes), default=0.0)


def best_market_ev_raw(market_ctx: MarketContext1X2 | None) -> float:
    """Mayor EV raw (informativo) entre desenlaces 1X2."""
    if not market_ctx or not market_ctx.has_market:
        return 0.0
    return max((o.ev_raw_pct / 100.0 for o in market_ctx.outcomes), default=0.0)


def _count_bookmakers(event: dict) -> int:
    return len(event.get("bookmakers", []))


def _alpha_regime_from_model(model: ModelMarkets) -> str | None:
    cal = (model.blend_meta or {}).get("calibration") or {}
    return cal.get("alpha_regime")


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
    alpha_regime = _alpha_regime_from_model(model)

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
        ev_raw_calc = expected_value_fair(model_prob, fair_o)
        ev_fair = ev_for_decision(ev_fair=ev_raw_calc, alpha_regime=alpha_regime)
        ev_raw = expected_value_raw(model_prob, raw_o) if raw_o > 1.0 else 0.0
        ev_capped = ev_raw_calc > ev_fair

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
                    metadata={
                        "prob_source": "calibrated",
                        "alpha_regime": alpha_regime,
                        "ev_raw_calc": ev_raw_calc,
                        "ev_regime_capped": ev_capped,
                        "ev_cap": regime_ev_cap(alpha_regime),
                    },
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
