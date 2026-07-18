"""Orquestación — carga datos de motores sin alterar lógica."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from supabase import Client

from apps.api.services.injury_news import InjuryReport, fetch_injury_report
from apps.api.services.live_calibration import calibrate_analysis_model
from apps.api.services.team_wc_stats_service import (
    StatsOdds,
    TeamWCStats,
    find_fixture_id,
    get_match_team_stats,
    get_stats_market_odds,
)
from apps.api.services.odds_context import (
    EvOpportunity,
    MarketContext1X2,
    compute_ev_opportunities,
    compute_market_context,
    find_wc_odds_event,
)
from apps.api.services.parlay_engine import (
    ParlayBuildResult,
    build_parlays_from_sharp_picks,
    evaluate_parlay_leg,
    extract_sharp_parlay_pick,
)
from apps.api.services.sharp_engine import SharpBetResult, run_sharp_engine
from apps.api.services.worldcup_engine import MatchAnalysis, analyze_match, set_calibration_factors
from apps.worker.ml.model_loader import load_calibration_factors_from_db


@dataclass
class MatchBundle:
    analysis: MatchAnalysis
    market_ctx: MarketContext1X2 | None = None
    ev_opps: list[EvOpportunity] = field(default_factory=list)
    sharp: SharpBetResult | None = None
    parlay_leg: Any = None
    parlay_result: ParlayBuildResult | None = None
    injury: InjuryReport | None = None
    odds_event: dict | None = None
    home_team_stats: TeamWCStats | None = None
    away_team_stats: TeamWCStats | None = None
    api_football_fixture_id: int | None = None
    stats_odds: StatsOdds | None = None


async def load_match_bundle(
    match: dict,
    *,
    db: Client,
    d18: dict,
    d22: dict,
    fd_matches: list,
    elo_ratings: dict,
    historical_accuracy: float | None = None,
) -> MatchBundle:
    """Ejecuta motores una vez — UI solo consume el bundle."""
    factors = load_calibration_factors_from_db(db)
    if factors:
        set_calibration_factors(factors)

    from apps.worker.ml.wc_features import build_match_features

    feat = build_match_features(match, d18, d22, fd_matches, elo_ratings)
    odds_event = await find_wc_odds_event(feat["team1"], feat["team2"], db=db)
    analysis = analyze_match(match, d18, d22, fd_matches, elo_ratings, odds_event=odds_event)
    if not analysis.model:
        return MatchBundle(analysis=analysis)

    hist_played = (
        (analysis.historico.get(analysis.team1, {}).get("wc2022", {}).get("played", 0) or 0)
        + (analysis.historico.get(analysis.team2, {}).get("wc2022", {}).get("played", 0) or 0)
    )
    calibrate_analysis_model(
        analysis,
        odds_event,
        data_quality_pct=100.0,
        hist_played=hist_played,
    )
    market_ctx = compute_market_context(
        analysis.model, analysis.team1, analysis.team2, odds_event
    )
    ev_opps: list[EvOpportunity] = []
    if odds_event and market_ctx.has_market:
        ev_opps = compute_ev_opportunities(
            analysis.model,
            analysis.team1,
            analysis.team2,
            odds_event,
            single_best=False,
        )

    (injury, (home_team_stats, away_team_stats), api_football_fixture_id) = await asyncio.gather(
        fetch_injury_report(analysis.team1, analysis.team2),
        get_match_team_stats(analysis.team1, analysis.team2),
        find_fixture_id(analysis.team1, analysis.team2),
    )
    stats_odds: StatsOdds | None = None
    if api_football_fixture_id:
        try:
            stats_odds = await get_stats_market_odds(api_football_fixture_id)
        except Exception as exc:
            logger.warning("get_stats_market_odds: %s", exc)

    sharp = run_sharp_engine(
        analysis,
        ev_opps,
        market_ctx=market_ctx,
        injury_report=injury,
        data_quality_pct=100.0,
        hist_played=hist_played,
        historical_accuracy=historical_accuracy,
    )
    parlay_leg = evaluate_parlay_leg(
        analysis,
        market_ctx,
        sharp.pipeline.market.dominance if sharp else None,
        injury_report=injury,
        sharp=sharp,
        ev_opps=ev_opps,
    )
    sharp_pick = extract_sharp_parlay_pick(analysis, sharp, market_ctx, ev_opps)
    parlay_result = build_parlays_from_sharp_picks([sharp_pick] if sharp_pick else [])

    return MatchBundle(
        analysis=analysis,
        market_ctx=market_ctx,
        ev_opps=ev_opps,
        sharp=sharp,
        parlay_leg=parlay_leg,
        parlay_result=parlay_result,
        injury=injury,
        odds_event=odds_event,
        home_team_stats=home_team_stats,
        away_team_stats=away_team_stats,
        api_football_fixture_id=api_football_fixture_id,
        stats_odds=stats_odds,
    )
