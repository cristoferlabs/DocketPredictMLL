"""
StatsBomb Open-Data Ingest — xG real para WC matches.

Descarga SOLO los archivos necesarios de WC2018/WC2022 vía GitHub raw API.
No requiere clonar el repositorio completo.

Fuente: https://github.com/statsbomb/open-data

Cache local en: data/statsbomb_raw/
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SB_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_CACHE_DIR = Path("data/statsbomb_raw")
_WC_COMPETITION_ID = 43  # FIFA World Cup en StatsBomb


# ── Descarga con caché local ──────────────────────────────────────────────────

def _cached_get(url: str, local_path: Path, *, force: bool = False) -> dict | list | None:
    """Descarga JSON con caché en disco. Devuelve None si falla."""
    if local_path.exists() and not force:
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        import urllib.request
        local_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "wc2026-model/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        local_path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:
        logger.warning("statsbomb_ingest: no se pudo descargar %s — %s", url, exc)
        return None


# ── API de StatsBomb open-data ────────────────────────────────────────────────

def fetch_competitions() -> list[dict]:
    url = f"{_SB_RAW_BASE}/competitions.json"
    data = _cached_get(url, _CACHE_DIR / "competitions.json")
    return data or []


def find_wc_seasons() -> list[dict[str, Any]]:
    """Devuelve lista de {competition_id, season_id, season_name} para WC."""
    comps = fetch_competitions()
    return [
        c for c in comps
        if c.get("competition_id") == _WC_COMPETITION_ID
    ]


def fetch_matches_for_season(competition_id: int, season_id: int) -> list[dict]:
    url = f"{_SB_RAW_BASE}/matches/{competition_id}/{season_id}.json"
    path = _CACHE_DIR / "matches" / str(competition_id) / f"{season_id}.json"
    data = _cached_get(url, path)
    return data or []


def fetch_events_for_match(match_id: int) -> list[dict]:
    url = f"{_SB_RAW_BASE}/events/{match_id}.json"
    path = _CACHE_DIR / "events" / f"{match_id}.json"
    data = _cached_get(url, path)
    return data or []


# ── Extracción de xG y shots ──────────────────────────────────────────────────

def extract_team_stats_from_events(
    events: list[dict],
    team_name: str,
) -> dict[str, float]:
    """
    Extrae del event stream de StatsBomb:
      - xg_for:         suma de statsbomb_xg de todos los tiros del equipo
      - shots_for:      total de tiros del equipo
      - shots_on_target: tiros a puerta del equipo
      - xg_against:     xG total concedido (tiros del rival)
      - shots_against:  tiros totales del rival
      - possession_pct: % de posesión (aproximado por eventos de pase)

    Todos los valores son para UN partido.
    """
    xg_for = 0.0
    shots_for = 0
    sot_for = 0
    xg_against = 0.0
    shots_against = 0
    sot_against = 0
    passes_for = 0
    passes_against = 0

    team_name_lower = team_name.lower().strip()

    for ev in events:
        ev_team = (ev.get("team") or {}).get("name", "").lower().strip()
        ev_type = (ev.get("type") or {}).get("name", "")

        is_own_team = ev_team == team_name_lower

        if ev_type == "Shot":
            shot_info = ev.get("shot") or {}
            xg = float(shot_info.get("statsbomb_xg") or 0.0)
            outcome = (shot_info.get("outcome") or {}).get("name", "")
            on_target = outcome in ("Goal", "Saved", "Saved To Post")

            if is_own_team:
                xg_for += xg
                shots_for += 1
                if on_target:
                    sot_for += 1
            else:
                xg_against += xg
                shots_against += 1
                if on_target:
                    sot_against += 1

        elif ev_type == "Pass":
            if is_own_team:
                passes_for += 1
            else:
                passes_against += 1

    total_passes = passes_for + passes_against
    possession_pct = passes_for / total_passes if total_passes > 0 else 0.5

    return {
        "xg_for": round(xg_for, 4),
        "shots_for": shots_for,
        "shots_on_target_for": sot_for,
        "xg_against": round(xg_against, 4),
        "shots_against": shots_against,
        "shots_on_target_against": sot_against,
        "possession_pct": round(possession_pct, 4),
        "shot_quality_for": round(xg_for / shots_for, 4) if shots_for > 0 else 0.0,
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_team_xg_database(
    years: list[int] | None = None,
    *,
    sleep_between_matches: float = 0.1,
    verbose: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Descarga eventos WC y construye base de datos de xG por equipo.

    Retorna:
        {
          "Argentina": [
            {"date": "2018-06-16", "opponent": "Iceland", "match_id": 7529,
             "xg_for": 1.23, "shots_for": 18, "shots_on_target_for": 8,
             "xg_against": 0.45, "possession_pct": 0.62, ...},
            ...
          ],
          ...
        }
    Ordenado por fecha para uso walk-forward.
    """
    target_years = set(years or [2018, 2022])
    seasons = find_wc_seasons()

    if verbose:
        print(f"  Temporadas WC disponibles en StatsBomb: {[(s['season_name'], s['season_id']) for s in seasons]}")

    team_db: dict[str, list[dict]] = {}
    total_matches = 0

    for season in seasons:
        season_id = season["season_id"]
        season_name = season.get("season_name", str(season_id))

        # Filtrar por año si se especifica
        year_match = any(str(y) in season_name for y in target_years)
        if not year_match:
            continue

        matches = fetch_matches_for_season(_WC_COMPETITION_ID, season_id)
        if verbose:
            print(f"  {season_name}: {len(matches)} partidos")

        for match in matches:
            match_id = match.get("match_id")
            match_date = (match.get("match_date") or "")[:10]
            home_team = (match.get("home_team") or {}).get("home_team_name", "")
            away_team = (match.get("away_team") or {}).get("away_team_name", "")

            if not match_id or not home_team or not away_team:
                continue

            events = fetch_events_for_match(match_id)
            if not events:
                if verbose:
                    print(f"    SKIP {home_team} vs {away_team} (sin eventos)")
                continue

            # Extraer stats para cada equipo
            home_stats = extract_team_stats_from_events(events, home_team)
            away_stats = extract_team_stats_from_events(events, away_team)

            home_score = match.get("home_score", 0)
            away_score = match.get("away_score", 0)

            # Registro para local
            home_record = {
                "date": match_date,
                "match_id": match_id,
                "opponent": away_team,
                "is_home": True,
                "season": season_name,
                "goals_for": home_score,
                "goals_against": away_score,
                **home_stats,
            }
            # Registro para visitante
            away_record = {
                "date": match_date,
                "match_id": match_id,
                "opponent": home_team,
                "is_home": False,
                "season": season_name,
                "goals_for": away_score,
                "goals_against": home_score,
                **away_stats,
            }

            team_db.setdefault(home_team, []).append(home_record)
            team_db.setdefault(away_team, []).append(away_record)
            total_matches += 1

            if sleep_between_matches > 0:
                time.sleep(sleep_between_matches)

        if verbose:
            print(f"    -> {total_matches} partidos procesados hasta ahora")

    # Ordenar por fecha (necesario para walk-forward)
    for team_records in team_db.values():
        team_records.sort(key=lambda r: r["date"])

    if verbose:
        print(f"\n  Total: {total_matches} partidos | {len(team_db)} equipos")

    return team_db


def get_team_xg_before_date(
    team_db: dict[str, list[dict]],
    team: str,
    cutoff_date: datetime,
    *,
    max_matches: int = 10,
) -> list[dict]:
    """
    Devuelve historial de xG del equipo ANTES de cutoff_date (walk-forward safe).
    Máximo max_matches partidos más recientes.
    """
    records = team_db.get(team, [])
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    filtered = [r for r in records if r["date"] < cutoff_str]
    return filtered[-max_matches:]  # los más recientes


def compute_team_xg_profile(
    records: list[dict],
    *,
    decay: float = 0.85,
) -> dict[str, float] | None:
    """
    Calcula xG medio ponderado (decay exponencial, más reciente = más peso).

    Retorna dict con:
      xg_per_game, shots_per_game, xg_against_per_game, possession_pct,
      shot_quality, n_matches
    """
    if not records:
        return None

    w_sum = 0.0
    xg_for_w = 0.0
    shots_w = 0.0
    xg_against_w = 0.0
    possession_w = 0.0
    shot_quality_w = 0.0

    for i, rec in enumerate(reversed(records)):
        w = decay ** i
        xg_for_w += rec["xg_for"] * w
        shots_w += rec["shots_for"] * w
        xg_against_w += rec["xg_against"] * w
        possession_w += rec["possession_pct"] * w
        shot_quality_w += rec.get("shot_quality_for", 0.0) * w
        w_sum += w

    if w_sum == 0:
        return None

    return {
        "xg_per_game": round(xg_for_w / w_sum, 4),
        "shots_per_game": round(shots_w / w_sum, 2),
        "xg_against_per_game": round(xg_against_w / w_sum, 4),
        "possession_pct": round(possession_w / w_sum, 4),
        "shot_quality": round(shot_quality_w / w_sum, 4),
        "n_matches": len(records),
    }


# ── Guardado / carga del database ─────────────────────────────────────────────

_TEAM_XG_DB_PATH = Path("artifacts/calibration/statsbomb_team_xg.json")


def save_team_xg_database(db: dict) -> Path:
    _TEAM_XG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TEAM_XG_DB_PATH.write_text(
        json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return _TEAM_XG_DB_PATH


def load_team_xg_database() -> dict[str, list[dict]]:
    if not _TEAM_XG_DB_PATH.exists():
        return {}
    try:
        return json.loads(_TEAM_XG_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
