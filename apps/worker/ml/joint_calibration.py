"""
Joint calibration — objetivo conjunto outcome + mercado + proxy CLV.

Loss = log_loss(WC) + λ·market_divergence + μ·clv_proxy

Capa entre shape y live CAL: mezcla suave P_shape → P_market con β aprendido
por contexto (no reemplaza α régimen; reduce carga sobre CAL/EV clamp).

Artifact: artifacts/calibration/wc_joint_calibration.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MatchContext = Literal["close", "balanced", "mismatch"]

JOINT_ARTIFACT_PATH = Path("artifacts/calibration/wc_joint_calibration.json")

OUTCOMES_1X2 = ("home_win", "draw", "away_win")

DEFAULT_MARKET_BLEND: dict[str, float] = {
    "close": 0.10,
    "balanced": 0.18,
    "mismatch": 0.28,
}


@dataclass
class JointObjectiveWeights:
    lambda_market: float = 0.35
    mu_clv: float = 0.15

    def to_dict(self) -> dict[str, float]:
        return {"lambda_market": self.lambda_market, "mu_clv": self.mu_clv}


@dataclass
class JointCalibrationModel:
    engine: str = "joint_calibration_v1"
    weights: JointObjectiveWeights = field(default_factory=JointObjectiveWeights)
    market_blend_by_context: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MARKET_BLEND))
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "weights": self.weights.to_dict(),
            "market_blend_by_context": self.market_blend_by_context,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JointCalibrationModel:
        w = raw.get("weights") or {}
        return cls(
            engine=str(raw.get("engine", "joint_calibration_v1")),
            weights=JointObjectiveWeights(
                lambda_market=float(w.get("lambda_market", 0.35)),
                mu_clv=float(w.get("mu_clv", 0.15)),
            ),
            market_blend_by_context={
                k: float(v)
                for k, v in (raw.get("market_blend_by_context") or DEFAULT_MARKET_BLEND).items()
            },
            metrics=raw.get("metrics") or {},
        )


def load_joint_calibration_model() -> JointCalibrationModel:
    if not JOINT_ARTIFACT_PATH.exists():
        return JointCalibrationModel()
    try:
        return JointCalibrationModel.from_dict(json.loads(JOINT_ARTIFACT_PATH.read_text(encoding="utf-8")))
    except Exception:
        return JointCalibrationModel()


def save_joint_calibration_model(model: JointCalibrationModel) -> Path:
    JOINT_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOINT_ARTIFACT_PATH.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return JOINT_ARTIFACT_PATH


def _renorm_1x2(h: float, d: float, a: float) -> dict[str, float]:
    total = h + d + a
    if total <= 0:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    return {"home_win": h / total, "draw": d / total, "away_win": a / total}


def log_loss_1x2(probs: dict[str, float], label: str) -> float:
    p = max(probs.get(label, 0.0), 1e-9)
    return -math.log(p)


def market_divergence_penalty(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
) -> float:
    """MSE en probabilidad (0–1) sobre 1X2."""
    return sum(
        (model_probs.get(k, 0.0) - market_probs.get(k, 0.0)) ** 2 for k in OUTCOMES_1X2
    ) / len(OUTCOMES_1X2)


def clv_proxy_penalty(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
) -> float:
    """
  Proxy CLV: penaliza cuando el pick del modelo tiene peor implied que el mercado.
  Sin línea de cierre histórica — usa fair market como proxy del close.
  """
    pick = max(OUTCOMES_1X2, key=lambda k: model_probs.get(k, 0.0))
    pm = max(model_probs.get(pick, 0.0), 1e-9)
    pk = max(market_probs.get(pick, 0.0), 1e-9)
    model_impl = 1.0 / pm
    market_impl = 1.0 / pk
    if model_impl <= market_impl:
        return 0.0
    return ((model_impl - market_impl) / market_impl) ** 2


def joint_objective(
    probs: dict[str, float],
    *,
    label: str | None = None,
    market_probs: dict[str, float] | None = None,
    weights: JointObjectiveWeights | None = None,
) -> float:
    """Loss escalar para una fila."""
    w = weights or JointObjectiveWeights()
    loss = 0.0
    if label:
        loss += log_loss_1x2(probs, label)
    if market_probs:
        loss += w.lambda_market * market_divergence_penalty(probs, market_probs)
        loss += w.mu_clv * clv_proxy_penalty(probs, market_probs)
    return loss


def blend_toward_market(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    beta: float,
) -> dict[str, float]:
    """P_joint = (1-β)·P_model + β·P_market (renormalizado)."""
    beta = max(0.0, min(1.0, beta))
    h = (1 - beta) * model_probs["home_win"] + beta * market_probs.get("home_win", 0)
    d = (1 - beta) * model_probs["draw"] + beta * market_probs.get("draw", 0)
    a = (1 - beta) * model_probs["away_win"] + beta * market_probs.get("away_win", 0)
    return _renorm_1x2(h, d, a)


def resolve_market_blend(
    context: MatchContext,
    model: JointCalibrationModel | None = None,
) -> float:
    m = model or load_joint_calibration_model()
    return float(m.market_blend_by_context.get(context, DEFAULT_MARKET_BLEND.get(context, 0.15)))


def apply_joint_pricing_calibration(
    shape_probs: dict[str, float],
    market_probs: dict[str, float] | None,
    context: MatchContext,
    *,
    model: JointCalibrationModel | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Mezcla pricing-aware post-shape, pre-blend ELO/CAL.
    Sin mercado → pass-through.
    """
    from apps.shared.config import get_settings

    s = get_settings()
    enabled = getattr(s, "joint_calibration_enabled", True)
    meta: dict[str, Any] = {
        "joint_calibration": enabled,
        "context": context,
        "applied": False,
    }
    if not enabled or not market_probs:
        meta["reason"] = "no_market_or_disabled"
        return dict(shape_probs), meta

    m = model or load_joint_calibration_model()
    beta = resolve_market_blend(context, m)
    before_div = market_divergence_penalty(shape_probs, market_probs)
    out = blend_toward_market(shape_probs, market_probs, beta)
    after_div = market_divergence_penalty(out, market_probs)
    if after_div >= before_div - 1e-8:
        meta["reason"] = "no_divergence_improvement"
        meta["market_blend_beta"] = 0.0
        return dict(shape_probs), meta
    meta.update(
        {
            "applied": True,
            "market_blend_beta": round(beta, 4),
            "lambda_market": m.weights.lambda_market,
            "mu_clv": m.weights.mu_clv,
            "divergence_before_pp": round(before_div**0.5 * 100, 2),
            "divergence_after_pp": round(after_div**0.5 * 100, 2),
        }
    )
    return out, meta


def _grid_best_beta(
    rows: list[dict[str, Any]],
    context: MatchContext,
    weights: JointObjectiveWeights,
    *,
    beta_grid: list[float] | None = None,
) -> tuple[float, float]:
    subset = [r for r in rows if r.get("context") == context]
    if not subset:
        return DEFAULT_MARKET_BLEND.get(context, 0.15), float("inf")

    grid = beta_grid or [round(x, 2) for x in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]]
    best_beta, best_loss = grid[0], float("inf")
    for beta in grid:
        total = 0.0
        for r in subset:
            p_base = r.get("p_stat") or r["p_shape"]
            p_mkt = r.get("p_market")
            label = r.get("label")
            if p_mkt:
                p = blend_toward_market(p_base, p_mkt, beta)
            else:
                p = p_base
            total += joint_objective(p, label=label, market_probs=p_mkt, weights=weights)
        avg = total / len(subset)
        if avg < best_loss:
            best_loss, best_beta = avg, beta
    return best_beta, best_loss


def fit_joint_weights_grid(
    rows: list[dict[str, Any]],
    *,
    lambda_grid: list[float] | None = None,
    mu_grid: list[float] | None = None,
) -> tuple[JointObjectiveWeights, dict[str, Any]]:
    """Grid λ, μ sobre filas con mercado; valida con todas las filas."""
    market_rows = [r for r in rows if r.get("p_market")]
    if not market_rows:
        return JointObjectiveWeights(), {"note": "no_market_rows"}

    l_grid = lambda_grid or [0.20, 0.35, 0.50]
    m_grid = mu_grid or [0.10, 0.15, 0.25]
    best_w = JointObjectiveWeights()
    best_loss = float("inf")
    metrics: dict[str, Any] = {}

    for lam in l_grid:
        for mu in m_grid:
            w = JointObjectiveWeights(lambda_market=lam, mu_clv=mu)
            total = sum(
                joint_objective(
                    r.get("p_stat") or r["p_shape"]
                    if not r.get("p_market")
                    else blend_toward_market(
                        r.get("p_stat") or r["p_shape"],
                        r["p_market"],
                        DEFAULT_MARKET_BLEND.get(r["context"], 0.15),
                    ),
                    label=r.get("label"),
                    market_probs=r.get("p_market"),
                    weights=w,
                )
                for r in market_rows
            )
            avg = total / len(market_rows)
            if avg < best_loss:
                best_loss, best_w = avg, w

    metrics["weight_search"] = {
        "lambda_market": best_w.lambda_market,
        "mu_clv": best_w.mu_clv,
        "market_rows_loss": round(best_loss, 4),
        "n_market": len(market_rows),
    }
    return best_w, metrics


def fit_joint_calibration(rows: list[dict[str, Any]]) -> tuple[JointCalibrationModel, dict[str, Any]]:
    """
    Fit β por contexto + pesos λ/μ del objetivo conjunto.
    `rows` requiere: context, p_shape; opcional label, p_market.
    """
    if not rows:
        return JointCalibrationModel(), {"error": "no_rows", "n": 0}

    weights, w_metrics = fit_joint_weights_grid(rows)
    blends: dict[str, float] = {}
    ctx_metrics: dict[str, Any] = {}
    for ctx in ("close", "balanced", "mismatch"):
        beta, loss = _grid_best_beta(rows, ctx, weights)  # type: ignore[arg-type]
        blends[ctx] = round(beta, 4)
        n = len([r for r in rows if r.get("context") == ctx])
        ctx_metrics[ctx] = {"beta": blends[ctx], "joint_loss": round(loss, 4), "n": n}

    # Métricas agregadas antes/después
    mkt_rows = [r for r in rows if r.get("p_market")]
    div_before = 0.0
    div_after = 0.0
    ll_before = 0.0
    ll_after = 0.0
    for r in mkt_rows:
        ctx = r["context"]
        beta = blends.get(ctx, 0.15)
        p_adj = blend_toward_market(r.get("p_stat") or r["p_shape"], r["p_market"], beta)
        div_before += market_divergence_penalty(r.get("p_stat") or r["p_shape"], r["p_market"])
        div_after += market_divergence_penalty(p_adj, r["p_market"])
        if r.get("label"):
            ll_before += log_loss_1x2(r.get("p_stat") or r["p_shape"], r["label"])
            ll_after += log_loss_1x2(p_adj, r["label"])

    n_mkt = len(mkt_rows) or 1
    metrics: dict[str, Any] = {
        "n_total": len(rows),
        "n_market": len(mkt_rows),
        "n_outcome_only": len(rows) - len(mkt_rows),
        "by_context": ctx_metrics,
        **w_metrics,
        "market_divergence_mean_before": round(div_before / n_mkt, 6),
        "market_divergence_mean_after": round(div_after / n_mkt, 6),
        "log_loss_market_rows_before": round(ll_before / n_mkt, 4) if mkt_rows else None,
        "log_loss_market_rows_after": round(ll_after / n_mkt, 4) if mkt_rows else None,
    }

    model = JointCalibrationModel(
        weights=weights,
        market_blend_by_context=blends,
        metrics=metrics,
    )
    return model, metrics


def build_joint_training_rows(
    archives: dict[int, dict],
    odds_events: list[dict] | None = None,
    *,
    train_years: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Filas outcome (WC histórico) + pricing (upcoming con cuotas).
    p_shape = Poisson+DC+shape learned.
    """
    from apps.api.services.odds_context import find_wc_odds_in_events
    from apps.api.services.worldcup_engine import find_upcoming_matches, name_match
    from apps.worker.ml.dixon_coles import classify_match_context
    from apps.worker.ml.odds_math import fair_h2h_market
    from apps.worker.ml.wc_historical import _match_feature_bundle, extract_finished_matches
    from apps.api.services.worldcup_engine import compute_model_markets

    rows: list[dict[str, Any]] = []

    def _stat_row(
        lh: float,
        la: float,
        eh: float,
        ea: float,
        *,
        label: str | None,
        p_market: dict[str, float] | None,
        team1: str = "",
        team2: str = "",
    ) -> dict[str, Any]:
        ctx = classify_match_context(lh, la, elo_home=eh, elo_away=ea)
        raw = compute_model_markets(
            lh,
            la,
            eh,
            ea,
            calibrate=True,
            market_fair_1x2=None,
            apply_joint_calibration=False,
        )
        p_stat = {
            "home_win": raw.home_win,
            "draw": raw.draw,
            "away_win": raw.away_win,
        }
        return {
            "context": ctx,
            "label": label,
            "p_shape": p_stat,
            "p_stat": p_stat,
            "p_market": p_market,
            "team1": team1,
            "team2": team2,
        }

    for match in extract_finished_matches(archives, years=train_years or [2018, 2022]):
        bundle = _match_feature_bundle(match, archives)
        if not bundle:
            continue
        lambdas = bundle["lambdas"]
        elo = bundle["elo"]
        eh = elo.get(match.team1, 1500)
        ea = elo.get(match.team2, 1500)
        g1, g2 = match.home_goals, match.away_goals
        if g1 > g2:
            label = "home_win"
        elif g1 == g2:
            label = "draw"
        else:
            label = "away_win"
        rows.append(
            _stat_row(
                lambdas.lambda_home,
                lambdas.lambda_away,
                eh,
                ea,
                label=label,
                p_market=None,
                team1=match.team1,
                team2=match.team2,
            )
        )

    events = odds_events or []
    d26 = archives.get(2026, {})
    d22 = archives.get(2022, {})
    d18 = archives.get(2018, {})
    from apps.api.services.worldcup_engine import calc_elo_ratings
    from apps.worker.ml.wc_features import build_match_features

    elo_map = calc_elo_ratings(d18, d22, d26)
    for um in find_upcoming_matches(d26, days_ahead=30):
        features = build_match_features(um, d18, d22, [], elo_map)
        t1, t2 = features["team1"], features["team2"]
        if not t1 or not t2:
            continue
        ev = find_wc_odds_in_events(events, t1, t2) if events else None
        if not ev:
            continue
        p_market = market_fair_1x2_from_event(ev, t1, t2)
        if not p_market:
            continue
        lambdas = features["lambdas"]
        eh = elo_map.get(t1, 1500)
        ea = elo_map.get(t2, 1500)
        rows.append(
            _stat_row(
                lambdas.lambda_home,
                lambdas.lambda_away,
                eh,
                ea,
                label=None,
                p_market=p_market,
                team1=t1,
                team2=t2,
            )
        )

    return rows


def market_fair_1x2_from_event(
    odds_event: dict | None,
    team1: str,
    team2: str,
) -> dict[str, float] | None:
    """Fair probs 1X2 alineadas a team1=home_win, team2=away_win."""
    if not odds_event:
        return None
    from apps.api.services.odds_context import _match_odds_event
    from apps.api.services.worldcup_engine import name_match
    from apps.worker.ml.odds_math import fair_h2h_market

    fair = fair_h2h_market(odds_event)
    if not fair:
        return None
    p = {
        "home_win": float(fair.get("home", {}).get("fair_prob", 0) or 0),
        "draw": float(fair.get("draw", {}).get("fair_prob", 0) or 0),
        "away_win": float(fair.get("away", {}).get("fair_prob", 0) or 0),
    }
    if sum(p.values()) <= 0:
        return None
    if _match_odds_event(odds_event, team2, team1):
        p = {
            "home_win": p["away_win"],
            "draw": p["draw"],
            "away_win": p["home_win"],
        }
    return p
