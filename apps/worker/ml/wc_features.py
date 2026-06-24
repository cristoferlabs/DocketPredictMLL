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
) -> TeamAttackProfile:
    """
    Prefer real xG from recent FD matches; else weighted form; else historical GF.
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

    if xg_samples:
        xg_real = sum(xg_samples) / len(xg_samples)
        blended = round(xg_real * 0.7 + form_gf * 0.3, 2)
        return TeamAttackProfile(
            xg=blended,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="xg_api",
        )

    if form:
        blended = round(form_gf * 0.65 + (hist_avg_gf or WC_AVG_GOALS) * 0.35, 2)
        return TeamAttackProfile(
            xg=blended,
            form_gf_weighted=form_gf,
            hist_avg_gf=hist_avg_gf,
            source="weighted_form",
        )

    return TeamAttackProfile(
        xg=round(hist_avg_gf or WC_AVG_GOALS, 2),
        form_gf_weighted=form_gf,
        hist_avg_gf=hist_avg_gf,
        source="historical",
    )


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
) -> MatchLambdas:
    """
    λ_home = xG_home * rival_def_away * home_boost
    λ_away = xG_away * rival_def_home
    """
    profile_h = estimate_team_xg(form_home, hist_gf_home, elo_ratings, team_home, fd_matches)
    profile_a = estimate_team_xg(form_away, hist_gf_away, elo_ratings, team_away, fd_matches)

    combined_rounds = (rounds_22 or []) + (rounds_18 or [])
    def_away = rival_defense_strength(team_away, combined_rounds, elo_ratings)
    def_home = rival_defense_strength(team_home, combined_rounds, elo_ratings)

    hf = home_factor(team_home)
    fat_h = fatigue_multiplier(form_home)
    fat_a = fatigue_multiplier(form_away)
    lam_home = profile_h.xg * def_away * (1 + hf["boost"]) * fat_h
    lam_away = profile_a.xg * def_home * fat_a

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
) -> dict[str, Any]:
    """Full feature dict for analyze_match and prediction logging."""
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
            },
            t2: {
                "xg": lambdas.profile_away.xg,
                "source": lambdas.profile_away.source,
                "form_gf_weighted": lambdas.profile_away.form_gf_weighted,
            },
        },
        "lambdas": lambdas,
        "home_factor": {t1: home_factor(t1), t2: home_factor(t2)},
    }
