"""World Cup feature engineering — xG, weighted form, rival strength, lambdas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api.services.worldcup_engine import (
    HOST_BOOST,
    get_team_form_fd,
    home_factor,
    name_match,
    normalize_openfootball,
    team_historical_stats,
)
from apps.worker.ml.clv import fatigue_multiplier

WC_AVG_GOALS = 1.28
FORM_DECAY = 0.85
FORM_LIMIT = 5
DEFAULT_ELO = 1500.0


@dataclass
class TeamAttackProfile:
    xg: float
    form_gf_weighted: float
    hist_avg_gf: float
    source: str  # xg_api | weighted_form | historical


@dataclass
class MatchLambdas:
    lambda_home: float
    lambda_away: float
    xg_home: float
    xg_away: float
    profile_home: TeamAttackProfile
    profile_away: TeamAttackProfile


def _rival_elo_weight(opponent_elo: float, base_elo: float = DEFAULT_ELO) -> float:
    """Stronger opponent → higher weight on that form match."""
    return max(0.75, min(1.35, opponent_elo / base_elo))


def _extract_xg_from_fd_match(fd_match: dict, team: str, is_home: bool) -> float | None:
    """Try to read xG from football-data match payload if present."""
    stats = fd_match.get("statistics") or fd_match.get("stats") or []
    if isinstance(stats, list):
        for block in stats:
            side = (block.get("location") or block.get("side") or "").lower()
            if (is_home and side == "home") or (not is_home and side == "away"):
                for item in block.get("statistics", block.get("stats", [])):
                    if str(item.get("type", "")).lower() in ("expected_goals", "xg"):
                        val = item.get("value")
                        if val is not None:
                            try:
                                return float(str(val).replace("%", ""))
                            except ValueError:
                                pass
    raw_xg = fd_match.get("xg") or fd_match.get("expectedGoals")
    if raw_xg is not None:
        try:
            return float(raw_xg)
        except (TypeError, ValueError):
            pass
    return None


def weighted_form_xg(
    form: list[dict],
    hist_avg_gf: float,
    elo_ratings: dict[str, float],
    team: str,
) -> tuple[float, float]:
    """
    Exponential decay on last N matches + opponent strength weighting.
    Returns (weighted_gf_per_match, total_weight).
    """
    hist = hist_avg_gf if hist_avg_gf > 0 else WC_AVG_GOALS
    if not form:
        return hist, 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for i, m in enumerate(form[:FORM_LIMIT]):
        time_w = FORM_DECAY ** i
        gf = int((m.get("marcador", "0-0").split("-")[0]) or 0)
        rival = m.get("rival", "")
        rival_elo = DEFAULT_ELO
        for name, elo in elo_ratings.items():
            if name_match(name, rival):
                rival_elo = elo
                break
        opp_w = _rival_elo_weight(rival_elo)
        w = time_w * opp_w
        weighted_sum += gf * w
        weight_total += w

    if weight_total <= 0:
        return hist, 0.0
    return round(weighted_sum / weight_total, 3), round(weight_total, 3)


def estimate_team_xg(
    form: list[dict],
    hist_avg_gf: float,
    elo_ratings: dict[str, float],
    team: str,
    fd_matches: list[dict] | None = None,
    understat_xg_per_game: float | None = None,
    statsbomb_xg_per_game: float | None = None,
) -> TeamAttackProfile:
    """
    Priority: FD match xG → Understat season xG → weighted form → historical GF.
    StatsBomb xG se aplica como corrector híbrido (55/45) sobre cualquier fuente base.

    understat_xg_per_game: pre-fetched from ml.team_season_xg (caller resolves async).
    statsbomb_xg_per_game: xG/partido de StatsBomb WC (walk-forward safe, caller lo resuelve).
    """
    fd_matches = fd_matches or []
    xg_samples: list[float] = []
    for m in form[:FORM_LIMIT]:
        is_home = name_match(m.get("home_team", ""), team) if m.get("home_team") else None
        for fd in fd_matches:
            home_n = fd.get("homeTeam", {}).get("name", "")
            away_n = fd.get("awayTeam", {}).get("name", "")
            if not (name_match(home_n, team) or name_match(away_n, team)):
                continue
            if (m.get("fecha") or "")[:10] and (fd.get("utcDate") or "")[:10] != (m.get("fecha") or "")[:10]:
                continue
            is_h = name_match(home_n, team)
            xg_val = _extract_xg_from_fd_match(fd, team, is_h)
            if xg_val is not None and xg_val >= 0:
                xg_samples.append(xg_val)

    form_gf, _ = weighted_form_xg(form, hist_avg_gf, elo_ratings, team)

    # 1. Real match-level xG from Football-Data API
    if xg_samples:
        xg_real = sum(xg_samples) / len(xg_samples)
        blended = round(xg_real * 0.7 + form_gf * 0.3, 2)
        return TeamAttackProfile(
            xg=blended,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="xg_api",
        )

    # 2. Understat season xG/game — richer prior than raw goals average
    if understat_xg_per_game is not None and understat_xg_per_game > 0:
        # Blend: 60% Understat season xG + 40% weighted recent form
        base = form_gf if form else (hist_avg_gf or WC_AVG_GOALS)
        blended = round(understat_xg_per_game * 0.60 + base * 0.40, 2)
        return TeamAttackProfile(
            xg=blended,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="understat_season",
        )

    # 3. Weighted form from recent results
    if form:
        blended = round(form_gf * 0.65 + (hist_avg_gf or WC_AVG_GOALS) * 0.35, 2)
        base_profile = TeamAttackProfile(
            xg=blended,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="weighted_form",
        )
    else:
        base_profile = TeamAttackProfile(
            xg=round(hist_avg_gf or WC_AVG_GOALS, 2),
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="historical",
        )

    # Corrector híbrido StatsBomb: 55% señal de goles (outcome) + 45% xG (proceso)
    # Solo aplica cuando hay >= 2 partidos WC con xG (señal confiable)
    if statsbomb_xg_per_game is not None and statsbomb_xg_per_game > 0:
        hybrid_xg = round(0.55 * base_profile.xg + 0.45 * statsbomb_xg_per_game, 2)
        return TeamAttackProfile(
            xg=hybrid_xg,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source=f"statsbomb_hybrid({base_profile.source})",
        )

    return base_profile


def rival_defense_strength(team: str, rounds: list, elo_ratings: dict[str, float]) -> float:
    """Goals conceded relative to WC average, adjusted by ELO (strong team concedes less)."""
    stats = team_historical_stats(rounds, team)
    if stats["played"] == 0:
        elo = elo_ratings.get(team, DEFAULT_ELO)
        return max(0.85, min(1.15, WC_AVG_GOALS / (WC_AVG_GOALS * (elo / DEFAULT_ELO))))

    ga_rate = stats["avg_ga"] if stats["avg_ga"] > 0 else WC_AVG_GOALS
    base = ga_rate / WC_AVG_GOALS
    elo = elo_ratings.get(team, DEFAULT_ELO)
    elo_adj = DEFAULT_ELO / max(elo, 1200)
    return round(max(0.7, min(1.4, base * elo_adj)), 3)


_SB_DB: dict | None = None


def _get_statsbomb_db() -> dict:
    global _SB_DB
    if _SB_DB is None:
        try:
            from apps.worker.ingest.statsbomb_ingest import load_team_xg_database
            _SB_DB = load_team_xg_database()
        except Exception:
            _SB_DB = {}
    return _SB_DB


def _statsbomb_xg_for_team(
    team: str,
    match_date,  # datetime | None
    *,
    min_matches: int = 2,
) -> float | None:
    """Devuelve xG/partido StatsBomb antes de match_date. None si datos insuficientes."""
    if match_date is None:
        return None
    try:
        from apps.worker.ingest.statsbomb_ingest import (
            get_team_xg_before_date,
            compute_team_xg_profile,
        )
        db = _get_statsbomb_db()
        records = get_team_xg_before_date(db, team, match_date)
        if len(records) < min_matches:
            return None
        profile = compute_team_xg_profile(records)
        if not profile:
            return None
        from apps.worker.ml.xg_estimator import WC_SCALER
        return round(profile["xg_per_game"] * WC_SCALER, 4)
    except Exception:
        return None


def compute_match_lambdas(
    team_home: str,
    team_away: str,
    form_home: list[dict],
    form_away: list[dict],
    hist_gf_home: float,
    hist_gf_away: float,
    elo_ratings: dict[str, float],
    rounds_18: list,
    rounds_22: list,
    fd_matches: list[dict] | None = None,
    understat_xg_home: float | None = None,
    understat_xg_away: float | None = None,
    match_date=None,  # datetime | None — habilita lookup de StatsBomb walk-forward
) -> MatchLambdas:
    """
    λ_home = xG_home * rival_def_away * home_boost
    λ_away = xG_away * rival_def_home

    understat_xg_home/away: season xG/game from ml.team_season_xg (optional enrichment).
    match_date: si se provee, activa corrector híbrido StatsBomb (55/45 goals/xG).
    """
    # Lookup StatsBomb xG (walk-forward safe si match_date está disponible)
    sb_xg_home = _statsbomb_xg_for_team(team_home, match_date)
    sb_xg_away = _statsbomb_xg_for_team(team_away, match_date)

    profile_h = estimate_team_xg(
        form_home, hist_gf_home, elo_ratings, team_home, fd_matches,
        understat_xg_per_game=understat_xg_home,
        statsbomb_xg_per_game=sb_xg_home,
    )
    profile_a = estimate_team_xg(
        form_away, hist_gf_away, elo_ratings, team_away, fd_matches,
        understat_xg_per_game=understat_xg_away,
        statsbomb_xg_per_game=sb_xg_away,
    )

    combined_rounds = (rounds_22 or []) + (rounds_18 or [])
    def_away = rival_defense_strength(team_away, combined_rounds, elo_ratings)
    def_home = rival_defense_strength(team_home, combined_rounds, elo_ratings)

    hf = home_factor(team_home)
    fat_h = fatigue_multiplier(form_home)
    fat_a = fatigue_multiplier(form_away)
    lam_home = profile_h.xg * def_away * (1 + hf["boost"]) * fat_h
    lam_away = profile_a.xg * def_home * fat_a

    # Ajuste de estado de juego: integra sobre dinámicas esperadas de liderazgo
    # Equipo que lidera gestiona (λ ↓) / equipo que persigue presiona (λ ↑)
    # Corrige sesgo de -29pp en victorias visitante confirmado en auditoría
    from apps.worker.ml.game_state_model import adjust_lambdas_for_flow
    _gs = adjust_lambdas_for_flow(lam_home, lam_away)
    lam_home = _gs.lam_home_adj
    lam_away = _gs.lam_away_adj

    lam_home = round(max(0.5, min(4.0, lam_home)), 2)
    lam_away = round(max(0.5, min(4.0, lam_away)), 2)

    return MatchLambdas(
        lambda_home=lam_home,
        lambda_away=lam_away,
        xg_home=profile_h.xg,
        xg_away=profile_a.xg,
        profile_home=profile_h,
        profile_away=profile_a,
    )


def build_match_features(
    match: dict,
    data_2018: dict,
    data_2022: dict,
    fd_matches: list[dict],
    elo_ratings: dict[str, float],
    understat_xg: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full feature dict for analyze_match and prediction logging.

    understat_xg: optional pre-fetched dict keyed by team name → xg_per_game,
    populated by the async caller via enrich_lambda_from_understat().
    """
    t1 = match.get("team1", {}).get("name", "TBD")
    t2 = match.get("team2", {}).get("name", "TBD")
    d18 = normalize_openfootball(data_2018)
    d22 = normalize_openfootball(data_2022)

    stats1_22 = team_historical_stats(d22.get("rounds", []), t1)
    stats2_22 = team_historical_stats(d22.get("rounds", []), t2)
    stats1_18 = team_historical_stats(d18.get("rounds", []), t1)
    stats2_18 = team_historical_stats(d18.get("rounds", []), t2)

    form1 = get_team_form_fd(fd_matches, t1)
    form2 = get_team_form_fd(fd_matches, t2)

    hist1 = stats1_22["avg_gf"] * 0.6 + stats1_18["avg_gf"] * 0.4 if stats1_22["played"] else stats1_18["avg_gf"]
    hist2 = stats2_22["avg_gf"] * 0.6 + stats2_18["avg_gf"] * 0.4 if stats2_22["played"] else stats2_18["avg_gf"]

    us = understat_xg or {}
    lambdas = compute_match_lambdas(
        t1,
        t2,
        form1,
        form2,
        hist1,
        hist2,
        elo_ratings,
        d18.get("rounds", []),
        d22.get("rounds", []),
        fd_matches,
        understat_xg_home=us.get(t1),
        understat_xg_away=us.get(t2),
    )

    return {
        "team1": t1,
        "team2": t2,
        "form": {t1: form1, t2: form2},
        "historico": {
            t1: {"wc2022": stats1_22, "wc2018": stats1_18},
            t2: {"wc2022": stats2_22, "wc2018": stats2_18},
        },
        "xg_profiles": {
            t1: {
                "xg": lambdas.profile_home.xg,
                "source": lambdas.profile_home.source,
                "form_gf_weighted": lambdas.profile_home.form_gf_weighted,
                "understat_xg_per_game": us.get(t1),
            },
            t2: {
                "xg": lambdas.profile_away.xg,
                "source": lambdas.profile_away.source,
                "form_gf_weighted": lambdas.profile_away.form_gf_weighted,
                "understat_xg_per_game": us.get(t2),
            },
        },
        "lambdas": lambdas,
        "home_factor": {t1: home_factor(t1), t2: home_factor(t2)},
    }
