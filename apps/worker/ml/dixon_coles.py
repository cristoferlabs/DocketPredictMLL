"""
Dixon-Coles contextual — ρ por régimen de partido + λ base vs λ corregido.

Contextos:
  close    — partidos cerrados (λ total bajo, ELO parejo)
  balanced — resto intermedio
  mismatch — favorito fuerte (sin DC; lift λ estructural)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from apps.worker.ml.poisson import (
    build_score_matrix,
    build_score_matrix_dixon_coles,
    poisson_pmf,
)

MatchContext = Literal["close", "balanced", "mismatch"]

RHO_ARTIFACT_PATH = Path("artifacts/calibration/wc_dixon_coles_rho.json")

DEFAULT_RHO_BY_CONTEXT: dict[str, float] = {
    "close": -0.15,
    "balanced": -0.10,
    "mismatch": 0.0,
}


@dataclass
class DixonColesContext:
    match_context: MatchContext
    lambda_base_home: float
    lambda_base_away: float
    lambda_corrected_home: float
    lambda_corrected_away: float
    rho: float
    dixon_coles_applied: bool
    lambda_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_context": self.match_context,
            "lambda_base_home": round(self.lambda_base_home, 4),
            "lambda_base_away": round(self.lambda_base_away, 4),
            "lambda_corrected_home": round(self.lambda_corrected_home, 4),
            "lambda_corrected_away": round(self.lambda_corrected_away, 4),
            "rho": round(self.rho, 4),
            "dixon_coles_applied": self.dixon_coles_applied,
            "lambda_flags": self.lambda_flags,
        }


def classify_match_context(
    lambda_home: float,
    lambda_away: float,
    *,
    elo_home: float | None = None,
    elo_away: float | None = None,
    close_total_lambda: float = 2.50,
    close_elo_diff: float = 80.0,
    mismatch_elo_diff: float = 150.0,
    mismatch_lambda_ratio: float = 2.0,
) -> MatchContext:
    total = lambda_home + lambda_away
    elo_diff = abs((elo_home or 1500) - (elo_away or 1500))
    ratio = max(lambda_home, lambda_away) / max(min(lambda_home, lambda_away), 0.05)

    if elo_diff >= mismatch_elo_diff or ratio >= mismatch_lambda_ratio:
        return "mismatch"
    if total <= close_total_lambda and elo_diff <= close_elo_diff:
        return "close"
    return "balanced"


def load_fitted_rho_by_context() -> dict[str, float]:
    if not RHO_ARTIFACT_PATH.exists():
        return dict(DEFAULT_RHO_BY_CONTEXT)
    try:
        raw = json.loads(RHO_ARTIFACT_PATH.read_text(encoding="utf-8"))
        return {
            "close": float(raw.get("close", DEFAULT_RHO_BY_CONTEXT["close"])),
            "balanced": float(raw.get("balanced", DEFAULT_RHO_BY_CONTEXT["balanced"])),
            "mismatch": float(raw.get("mismatch", DEFAULT_RHO_BY_CONTEXT["mismatch"])),
        }
    except Exception:
        return dict(DEFAULT_RHO_BY_CONTEXT)


def save_fitted_rho_by_context(rho: dict[str, float], *, metrics: dict | None = None) -> Path:
    RHO_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: float(v) for k, v in rho.items()
    }
    payload["metrics"] = metrics or {}
    payload["engine"] = "dixon_coles_context_v1"
    RHO_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RHO_ARTIFACT_PATH


def resolve_rho_for_context(
    context: MatchContext,
    *,
    rho_map: dict[str, float] | None = None,
) -> float:
    mapping = rho_map or load_fitted_rho_by_context()
    if context == "mismatch":
        return 0.0
    return float(mapping.get(context, DEFAULT_RHO_BY_CONTEXT.get(context, -0.10)))


def apply_lambda_structural_correction(
    lambda_home: float,
    lambda_away: float,
    context: MatchContext,
    *,
    elo_home: float,
    elo_away: float,
    close_elo_tight: float = 80.0,
    close_ratio_threshold: float = 1.75,
    mismatch_favorite_lift: float = 1.05,
) -> tuple[float, float, list[str]]:
    """
    λ base → λ corregido por contexto (antes de matriz DC).

    close: blend hacia media (partido parejo mal segmentado)
    mismatch: lift leve al favorito (reduce compresión vs mercado)
    balanced: sin cambio estructural
    """
    flags: list[str] = []
    lh, la = lambda_home, lambda_away
    elo_diff = abs(elo_home - elo_away)
    ratio = max(lh, la) / max(min(lh, la), 0.05)

    if context == "close" and elo_diff < close_elo_tight and ratio > close_ratio_threshold:
        avg = (lh + la) / 2.0
        lh = 0.62 * lh + 0.38 * avg
        la = 0.62 * la + 0.38 * avg
        flags.append("lambda_close_blend")

    if context == "mismatch":
        if lh >= la:
            lh = min(2.25, lh * mismatch_favorite_lift)
            flags.append("lambda_mismatch_fav_lift_home")
        else:
            la = min(2.25, la * mismatch_favorite_lift)
            flags.append("lambda_mismatch_fav_lift_away")

    return max(0.3, lh), max(0.3, la), flags


def prepare_dixon_coles_lambdas(
    lambda_home: float,
    lambda_away: float,
    *,
    elo_home: float | None = None,
    elo_away: float | None = None,
    settings: Any | None = None,
) -> DixonColesContext:
    """Pipeline λ base → contexto → λ corregido + ρ."""
    from apps.shared.config import get_settings

    s = settings or get_settings()
    lh_base, la_base = lambda_home, lambda_away
    eh = elo_home if elo_home is not None else 1500.0
    ea = elo_away if elo_away is not None else 1500.0

    ctx = classify_match_context(
        lh_base,
        la_base,
        elo_home=eh,
        elo_away=ea,
        close_total_lambda=getattr(s, "poisson_dc_close_total_lambda", 2.50),
        close_elo_diff=getattr(s, "poisson_dc_close_elo_diff", 80.0),
        mismatch_elo_diff=getattr(s, "poisson_dc_mismatch_elo_diff", 150.0),
        mismatch_lambda_ratio=getattr(s, "poisson_dc_mismatch_lambda_ratio", 2.0),
    )

    lh, la, flags = apply_lambda_structural_correction(
        lh_base,
        la_base,
        ctx,
        elo_home=eh,
        elo_away=ea,
        close_elo_tight=getattr(s, "poisson_elo_tight_max_diff", 80.0),
        close_ratio_threshold=getattr(s, "poisson_ratio_tight_threshold", 1.75),
        mismatch_favorite_lift=getattr(s, "poisson_mismatch_favorite_lift", 1.05),
    )

    rho = resolve_rho_for_context(ctx)
    use_dc = getattr(s, "poisson_dixon_coles_enabled", False) and ctx != "mismatch" and rho != 0.0

    return DixonColesContext(
        match_context=ctx,
        lambda_base_home=lh_base,
        lambda_base_away=la_base,
        lambda_corrected_home=lh,
        lambda_corrected_away=la,
        rho=rho if use_dc else 0.0,
        dixon_coles_applied=use_dc,
        lambda_flags=flags,
    )


def _log_likelihood_dc(
    scores: list[tuple[int, int, float, float]],
    rho: float,
) -> float:
    ll = 0.0
    for gh, ga, lh, la in scores:
        p = poisson_pmf(gh, lh) * poisson_pmf(ga, la)
        if gh <= 1 and ga <= 1:
            from apps.worker.ml.poisson import dixon_coles_tau

            p *= dixon_coles_tau(gh, ga, lh, la, rho)
        ll += math.log(max(p, 1e-12))
    return ll


def fit_rho_for_context(
    scores: list[tuple[int, int, float, float]],
    *,
    rho_grid: list[float] | None = None,
) -> tuple[float, float]:
    """Grid MLE de ρ para un contexto. Devuelve (best_rho, log_likelihood)."""
    if len(scores) < 8:
        return DEFAULT_RHO_BY_CONTEXT.get("balanced", -0.10), 0.0
    grid = rho_grid or [round(x, 2) for x in np.arange(-0.25, 0.05, 0.01)]
    best_rho, best_ll = grid[0], float("-inf")
    for rho in grid:
        ll = _log_likelihood_dc(scores, rho)
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho, best_ll


def fit_rho_by_context_from_archives(
    archives: dict[int, dict],
    *,
    train_years: list[int] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit ρ por contexto sobre WC histórico."""
    from apps.worker.ml.wc_historical import extract_finished_matches, predict_match_1x2_components

    matches = extract_finished_matches(archives, years=train_years or [2018, 2022])
    buckets: dict[str, list[tuple[int, int, float, float]]] = {
        "close": [],
        "balanced": [],
        "mismatch": [],
    }

    for m in matches:
        comp = predict_match_1x2_components(m, archives)
        if not comp:
            continue
        bundle_rows = _match_lambdas_elo(m, archives)
        if not bundle_rows:
            continue
        lh, la, eh, ea = bundle_rows
        ctx = classify_match_context(lh, la, elo_home=eh, elo_away=ea)
        buckets[ctx].append((m.home_goals, m.away_goals, lh, la))

    fitted: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for ctx, rows in buckets.items():
        if ctx == "mismatch":
            fitted[ctx] = 0.0
            metrics[ctx] = {"n": len(rows), "rho": 0.0, "note": "no DC on mismatch"}
            continue
        rho, ll = fit_rho_for_context(rows)
        fitted[ctx] = rho
        metrics[ctx] = {"n": len(rows), "rho": rho, "log_likelihood": round(ll, 2)}

    return fitted, metrics


def _match_lambdas_elo(match, archives) -> tuple[float, float, float, float] | None:
    from apps.worker.ml.wc_historical import _match_feature_bundle

    bundle = _match_feature_bundle(match, archives)
    if not bundle:
        return None
    lambdas = bundle["lambdas"]
    elo = bundle["elo"]
    return (
        lambdas.lambda_home,
        lambdas.lambda_away,
        elo.get(match.team1, 1500),
        elo.get(match.team2, 1500),
    )
