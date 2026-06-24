"""World Cup analytics engine — ported from n8n workflows (model-first, odds separate)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from apps.worker.ml.calibration import DEFAULT_CALIBRATION_FACTORS, calibrate_model_markets
from apps.worker.ml.elo import EloConfig, predict_match as elo_predict
from apps.worker.ml.poisson import outcome_probabilities, predict_match as poisson_predict

_calibration_factors: dict[str, Any] | None = None


def set_calibration_factors(factors: dict[str, Any] | None) -> None:
    """Override calibration factors (e.g. loaded from Supabase)."""
    global _calibration_factors
    _calibration_factors = factors


def get_calibration_factors() -> dict[str, Any]:
    if _calibration_factors:
        return _calibration_factors
    from apps.worker.ml.calibration import load_fitted_calibration_factors

    return load_fitted_calibration_factors()

HOST_BOOST = {
    "united states": 0.15,
    "usa": 0.15,
    "mexico": 0.08,
    "canada": 0.08,
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def name_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    na, nb = _norm(a), _norm(b)
    return na == nb or na in nb or nb in na


def normalize_openfootball(data: dict) -> dict:
    if data.get("rounds"):
        return data
    matches = data.get("matches", [])
    round_map: dict[str, dict] = {}
    for m in matches:
        rname = m.get("round", "Unknown")
        round_map.setdefault(rname, {"name": rname, "matches": []})
        t1 = m.get("team1")
        t2 = m.get("team2")
        if isinstance(t1, str):
            m = {**m, "team1": {"name": t1}}
        if isinstance(t2, str):
            m = {**m, "team2": {"name": t2}}
        round_map[rname]["matches"].append(m)
    return {"name": data.get("name"), "rounds": list(round_map.values())}


def calc_elo_ratings(
    data_2018: dict, data_2022: dict, data_2026: dict
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    def get_elo(team: str) -> float:
        if team not in ratings:
            ratings[team] = 1500.0
        return ratings[team]

    def process(rounds: list, base_k: float) -> None:
        all_matches: list[dict] = []
        for rnd in rounds or []:
            for m in rnd.get("matches", []):
                if m.get("score", {}).get("ft") and m.get("date"):
                    all_matches.append({**m, "roundName": rnd.get("name", "")})
        all_matches.sort(key=lambda x: x["date"])
        for m in all_matches:
            t1 = m.get("team1", {}).get("name")
            t2 = m.get("team2", {}).get("name")
            if not t1 or not t2:
                continue
            r1, r2 = get_elo(t1), get_elo(t2)
            g1, g2 = m["score"]["ft"]
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            s1 = 1.0 if g1 > g2 else 0.5 if g1 == g2 else 0.0
            gd = abs(g1 - g2)
            gd_mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else min(1.75 + (gd - 3) * 0.25, 2.5)
            rn = (m.get("roundName") or "").lower()
            is_ko = any(x in rn for x in ("round of", "quarter", "semi", "final"))
            k = base_k * (1.5 if is_ko else 1.0)
            ratings[t1] = round(r1 + k * gd_mult * (s1 - e1))
            ratings[t2] = round(r2 + k * gd_mult * ((1 - s1) - (1 - e1)))

    d18 = normalize_openfootball(data_2018)
    d22 = normalize_openfootball(data_2022)
    d26 = normalize_openfootball(data_2026)
    process(d18.get("rounds", []), 30)
    process(d22.get("rounds", []), 32)
    process(d26.get("rounds", []), 40)
    return ratings


def team_historical_stats(rounds: list, team: str) -> dict[str, Any]:
    played = wins = draws = losses = gf = ga = 0
    for rnd in rounds or []:
        for m in rnd.get("matches", []):
            if not m.get("score", {}).get("ft"):
                continue
            t1 = m.get("team1", {}).get("name")
            t2 = m.get("team2", {}).get("name")
            if team not in (t1, t2):
                continue
            played += 1
            gh, gay = m["score"]["ft"]
            if t1 == team:
                gf += gh
                ga += gay
                if gh > gay:
                    wins += 1
                elif gh == gay:
                    draws += 1
                else:
                    losses += 1
            else:
                gf += gay
                ga += gh
                if gay > gh:
                    wins += 1
                elif gh == gay:
                    draws += 1
                else:
                    losses += 1
    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "avg_gf": round(gf / played, 2) if played else 0.0,
        "avg_ga": round(ga / played, 2) if played else 0.0,
    }


def get_team_form_fd(fd_matches: list[dict], team: str, limit: int = 5) -> list[dict]:
    rows = [
        m
        for m in fd_matches
        if name_match(m.get("homeTeam", {}).get("name", ""), team)
        or name_match(m.get("homeTeam", {}).get("shortName", ""), team)
        or name_match(m.get("awayTeam", {}).get("name", ""), team)
        or name_match(m.get("awayTeam", {}).get("shortName", ""), team)
    ]
    rows = [m for m in rows if m.get("score", {}).get("fullTime", {}).get("home") is not None]
    rows.sort(key=lambda m: m.get("utcDate", ""), reverse=True)
    form = []
    for m in rows[:limit]:
        is_home = name_match(m["homeTeam"].get("name", ""), team) or name_match(
            m["homeTeam"].get("shortName", ""), team
        )
        s = m["score"]["fullTime"]
        tf = s["home"] if is_home else s["away"]
        og = s["away"] if is_home else s["home"]
        opp = (
            m["awayTeam"].get("shortName") or m["awayTeam"].get("name")
            if is_home
            else m["homeTeam"].get("shortName") or m["homeTeam"].get("name")
        )
        res = "V" if tf > og else "E" if tf == og else "D"
        form.append({"fecha": (m.get("utcDate") or "")[:10], "rival": opp, "resultado": res, "marcador": f"{tf}-{og}"})
    return form


def home_factor(team: str) -> dict[str, Any]:
    for host, boost in HOST_BOOST.items():
        if name_match(team, host):
            label = "Local (sede)" if boost >= 0.12 else "Co-sede"
            return {"label": label, "boost": boost}
    return {"label": "Visitante", "boost": 0.0}


@dataclass
class ModelMarkets:
    """Probabilidades del MODELO — independientes de la casa de apuestas."""

    home_win: float
    draw: float
    away_win: float
    over_25: float
    under_25: float
    btts_yes: float
    btts_no: float
    lambda_home: float
    lambda_away: float
    confidence: str = "medium"


@dataclass
class MatchAnalysis:
    team1: str
    team2: str
    fecha: str
    ronda: str
    grupo: str
    estadio: str
    elo: dict[str, Any] = field(default_factory=dict)
    xg: dict[str, float] = field(default_factory=dict)
    forma: dict[str, Any] = field(default_factory=dict)
    historico: dict[str, Any] = field(default_factory=dict)
    model: ModelMarkets | None = None
    local_visitante: dict[str, Any] = field(default_factory=dict)


def compute_model_markets(
    lambda_home: float,
    lambda_away: float,
    elo_home: float,
    elo_away: float,
    *,
    blend_poisson: float = 0.6,
    blend_elo: float = 0.4,
    calibrate: bool = True,
) -> ModelMarkets:
    poisson = poisson_predict(lambda_home, lambda_away)
    poisson_1x2 = outcome_probabilities(poisson.score_matrix)
    elo_p = elo_predict(elo_home, elo_away, EloConfig())

    # Blend Poisson + ELO (modelo interno, NO usa odds)
    w_p, w_e = blend_poisson, blend_elo
    home = w_p * poisson_1x2["home_win"] + w_e * elo_p.home_win
    draw = w_p * poisson_1x2["draw"] + w_e * elo_p.draw
    away = w_p * poisson_1x2["away_win"] + w_e * elo_p.away_win
    total = home + draw + away
    home, draw, away = home / total, draw / total, away / total

    if calibrate:
        calibrated = calibrate_model_markets(
            home,
            draw,
            away,
            poisson.over_25,
            poisson.under_25,
            poisson.btts_yes,
            poisson.btts_no,
            factors=get_calibration_factors(),
        )
        home, draw, away = calibrated["home_win"], calibrated["draw"], calibrated["away_win"]
        over_25, under_25 = calibrated["over_25"], calibrated["under_25"]
        btts_yes, btts_no = calibrated["btts_yes"], calibrated["btts_no"]
    else:
        over_25, under_25 = poisson.over_25, poisson.under_25
        btts_yes, btts_no = poisson.btts_yes, poisson.btts_no

    max_p = max(home, draw, away)
    confidence = "high" if max_p >= 0.55 else "medium" if max_p >= 0.42 else "low"

    return ModelMarkets(
        home_win=round(home, 4),
        draw=round(draw, 4),
        away_win=round(away, 4),
        over_25=round(over_25, 4),
        under_25=round(under_25, 4),
        btts_yes=round(btts_yes, 4),
        btts_no=round(btts_no, 4),
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        confidence=confidence,
    )


def find_upcoming_matches(data_2026: dict, days_ahead: int = 7) -> list[dict]:
    d26 = normalize_openfootball(data_2026)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = today + timedelta(days=days_ahead)
    upcoming: list[dict] = []

    for rnd in d26.get("rounds", []):
        for m in rnd.get("matches", []):
            if not m.get("date"):
                continue
            try:
                d = datetime.fromisoformat(m["date"].replace("Z", "+00:00").split("T")[0])
            except ValueError:
                continue
            if today <= d <= end and not m.get("score", {}).get("ft"):
                upcoming.append({**m, "roundName": rnd.get("name")})

    if not upcoming:
        for rnd in d26.get("rounds", []):
            for m in rnd.get("matches", []):
                t1 = m.get("team1", {}).get("name", "")
                if not m.get("score", {}).get("ft") and t1 and not re.match(r"^[0-9WL]", t1):
                    upcoming.append({**m, "roundName": rnd.get("name")})
        upcoming = upcoming[:8]

    return upcoming


def analyze_match(
    match: dict,
    data_2018: dict,
    data_2022: dict,
    fd_matches: list[dict],
    elo_ratings: dict[str, float],
) -> MatchAnalysis:
    from apps.worker.ml.wc_features import build_match_features

    features = build_match_features(match, data_2018, data_2022, fd_matches, elo_ratings)
    t1 = features["team1"]
    t2 = features["team2"]
    lambdas = features["lambdas"]

    elo1 = elo_ratings.get(t1, 1500)
    elo2 = elo_ratings.get(t2, 1500)
    elo_diff = elo1 - elo2
    elo_win1 = round(1 / (1 + 10 ** (-elo_diff / 400)) * 100, 1)

    from apps.shared.config import get_settings

    blend_w = get_settings().market_blend_model_weight
    model = compute_model_markets(
        lambdas.lambda_home,
        lambdas.lambda_away,
        elo1,
        elo2,
        blend_poisson=blend_w,
        blend_elo=round(1.0 - blend_w, 4),
        calibrate=True,
    )

    hf1 = features["home_factor"][t1]
    hf2 = features["home_factor"][t2]

    return MatchAnalysis(
        team1=t1,
        team2=t2,
        fecha=match.get("date", "TBD")[:10],
        ronda=match.get("roundName", ""),
        grupo=match.get("group", ""),
        estadio=match.get("ground", ""),
        elo={
            t1: {"rating": elo1, "win_prob": elo_win1},
            t2: {"rating": elo2, "win_prob": round(100 - elo_win1, 1)},
            "favorito": t1 if elo_diff > 20 else t2 if elo_diff < -20 else "Equilibrado",
        },
        xg={
            t1: lambdas.xg_home,
            t2: lambdas.xg_away,
            "total": round(lambdas.xg_home + lambdas.xg_away, 2),
            "source_home": lambdas.profile_home.source,
            "source_away": lambdas.profile_away.source,
        },
        forma=features["form"],
        historico=features["historico"],
        model=model,
        local_visitante={t1: hf1, t2: hf2},
    )
