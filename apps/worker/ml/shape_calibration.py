"""
Calibración de forma (shape) — aprendida desde WC histórico.

Aprende:
  - draw_factor(context, λ_total) — uplift de empate vs Poisson+DC
  - favorite_scale(peak) — corrección de cola según fuerza del favorito

Artifact: artifacts/calibration/wc_shape_calibration.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MatchContext = Literal["close", "balanced", "mismatch"]

SHAPE_ARTIFACT_PATH = Path("artifacts/calibration/wc_shape_calibration.json")

# Fallback si no hay artifact (pre-fit)
DEFAULT_DRAW_BY_CONTEXT: dict[str, float] = {
    "close": 1.06,
    "balanced": 1.03,
    "mismatch": 1.02,
}
DEFAULT_PEAK_BINS: list[dict[str, float]] = [
    {"min_peak": 0.0, "max_peak": 0.40, "favorite_scale": 1.0},
    {"min_peak": 0.40, "max_peak": 0.50, "favorite_scale": 0.98},
    {"min_peak": 0.50, "max_peak": 0.60, "favorite_scale": 1.04},
    {"min_peak": 0.60, "max_peak": 1.0, "favorite_scale": 1.08},
]

SHRINK_PRIOR_N = 10


@dataclass
class ShapeFeatures:
    lambda_total: float
    elo_gap: float
    lambda_home: float = 0.0
    lambda_away: float = 0.0

    @classmethod
    def from_match(
        cls,
        lambda_home: float,
        lambda_away: float,
        *,
        elo_home: float,
        elo_away: float,
    ) -> ShapeFeatures:
        return cls(
            lambda_total=lambda_home + lambda_away,
            elo_gap=abs(elo_home - elo_away),
            lambda_home=lambda_home,
            lambda_away=lambda_away,
        )


@dataclass
class ShapeCalibrationModel:
    engine: str = "shape_calibration_v2_learned"
    draw_by_context: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DRAW_BY_CONTEXT))
    lambda_bins: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    peak_bins: list[dict[str, float]] = field(default_factory=lambda: list(DEFAULT_PEAK_BINS))
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "draw_by_context": self.draw_by_context,
            "lambda_bins": self.lambda_bins,
            "peak_bins": self.peak_bins,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShapeCalibrationModel:
        return cls(
            engine=str(raw.get("engine", "shape_calibration_v2_learned")),
            draw_by_context={
                k: float(v) for k, v in (raw.get("draw_by_context") or DEFAULT_DRAW_BY_CONTEXT).items()
            },
            lambda_bins=raw.get("lambda_bins") or {},
            peak_bins=raw.get("peak_bins") or list(DEFAULT_PEAK_BINS),
            metrics=raw.get("metrics") or {},
        )


def load_shape_calibration_model() -> ShapeCalibrationModel:
    if not SHAPE_ARTIFACT_PATH.exists():
        return ShapeCalibrationModel()
    try:
        raw = json.loads(SHAPE_ARTIFACT_PATH.read_text(encoding="utf-8"))
        return ShapeCalibrationModel.from_dict(raw)
    except Exception:
        return ShapeCalibrationModel()


def save_shape_calibration_model(model: ShapeCalibrationModel) -> Path:
    SHAPE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAPE_ARTIFACT_PATH.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return SHAPE_ARTIFACT_PATH


def _shrink_ratio(observed: float, predicted: float, n: int, *, prior: float = 1.0) -> float:
    if predicted <= 0 or n <= 0:
        return prior
    raw = observed / predicted
    return (n * raw + SHRINK_PRIOR_N * prior) / (n + SHRINK_PRIOR_N)


def _renorm_1x2(h: float, d: float, a: float) -> dict[str, float]:
    total = h + d + a
    if total <= 0:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    return {"home_win": h / total, "draw": d / total, "away_win": a / total}


def _lambda_tertiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        mid = sum(values) / len(values) if values else 2.5
        return mid - 0.3, mid + 0.3
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def _resolve_draw_factor(
    model: ShapeCalibrationModel,
    context: MatchContext,
    features: ShapeFeatures | None,
) -> tuple[float, str]:
    base = float(model.draw_by_context.get(context, DEFAULT_DRAW_BY_CONTEXT.get(context, 1.0)))
    if not features or context not in model.lambda_bins:
        return base, "context_base"

    bins = model.lambda_bins[context]
    lt = features.lambda_total
    for b in bins:
        if b["lambda_min"] <= lt < b["lambda_max"] or (
            lt >= b["lambda_min"] and b["lambda_max"] >= 0.99
        ):
            return float(b["draw_factor"]), "lambda_bin"
    return base, "context_base"


def _resolve_favorite_scale(model: ShapeCalibrationModel, peak: float) -> tuple[float, dict[str, float] | None]:
    for b in model.peak_bins:
        if b["min_peak"] <= peak < b["max_peak"] or (peak >= b["min_peak"] and b["max_peak"] >= 0.99):
            return float(b["favorite_scale"]), b
    last = model.peak_bins[-1] if model.peak_bins else None
    return float(last["favorite_scale"]) if last else 1.0, last


def apply_poisson_shape_calibration(
    probs: dict[str, float],
    context: MatchContext,
    *,
    features: ShapeFeatures | None = None,
    model: ShapeCalibrationModel | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Ajuste de forma data-driven (artifact) con fallback a defaults.
    """
    from apps.shared.config import get_settings

    s = get_settings()
    use_learned = getattr(s, "poisson_shape_use_learned", True)
    m = model if model is not None else (load_shape_calibration_model() if use_learned else ShapeCalibrationModel())

    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    peak = max(h, a)
    meta: dict[str, Any] = {
        "shape_calibration": True,
        "context": context,
        "learned": use_learned and SHAPE_ARTIFACT_PATH.exists(),
        "engine": m.engine,
    }
    flags: list[str] = []

    draw_factor, draw_src = _resolve_draw_factor(m, context, features)
    d *= draw_factor
    flags.append(f"draw_{draw_src}")

    fav_scale, peak_bin = _resolve_favorite_scale(m, peak)
    if abs(fav_scale - 1.0) > 0.005:
        if h >= a:
            h *= fav_scale
        else:
            a *= fav_scale
        flags.append("favorite_scale")

    out = _renorm_1x2(h, d, a)
    meta["shape_flags"] = flags
    meta["draw_factor"] = round(draw_factor, 4)
    meta["favorite_scale"] = round(fav_scale, 4)
    meta["favorite_peak"] = round(peak, 4)
    if features:
        meta["lambda_total"] = round(features.lambda_total, 3)
        meta["elo_gap"] = round(features.elo_gap, 1)
    if peak_bin:
        meta["peak_bin"] = {
            "min_peak": peak_bin["min_peak"],
            "max_peak": peak_bin["max_peak"],
        }
    return out, meta


def _collect_shape_rows(
    archives: dict[int, dict],
    *,
    train_years: list[int] | None = None,
) -> list[dict[str, Any]]:
    from apps.worker.ml.dixon_coles import classify_match_context
    from apps.worker.ml.poisson import outcome_probabilities, predict_match as poisson_predict
    from apps.worker.ml.wc_historical import _match_feature_bundle, extract_finished_matches

    rows: list[dict[str, Any]] = []
    for match in extract_finished_matches(archives, years=train_years or [2018, 2022]):
        bundle = _match_feature_bundle(match, archives)
        if not bundle:
            continue
        lambdas = bundle["lambdas"]
        elo = bundle["elo"]
        eh = elo.get(match.team1, 1500)
        ea = elo.get(match.team2, 1500)
        lh, la = lambdas.lambda_home, lambdas.lambda_away
        pred = poisson_predict(lh, la, elo_home=eh, elo_away=ea)
        probs = outcome_probabilities(pred.score_matrix)
        ctx = classify_match_context(lh, la, elo_home=eh, elo_away=ea)
        g1, g2 = match.home_goals, match.away_goals
        is_draw = 1 if g1 == g2 else 0
        fav_home = probs["home_win"] >= probs["away_win"]
        fav_won = (g1 > g2) if fav_home else (g2 > g1)
        if g1 > g2:
            label = "home_win"
        elif g1 == g2:
            label = "draw"
        else:
            label = "away_win"
        rows.append(
            {
                "context": ctx,
                "lambda_total": lh + la,
                "elo_gap": abs(eh - ea),
                "p_draw": probs["draw"],
                "p_home": probs["home_win"],
                "p_away": probs["away_win"],
                "peak": max(probs["home_win"], probs["away_win"]),
                "is_draw": is_draw,
                "fav_won": 1 if fav_won else 0,
                "p_fav": max(probs["home_win"], probs["away_win"]),
                "label": label,
            }
        )
    return rows


def fit_shape_calibration_from_archives(
    archives: dict[int, dict],
    *,
    train_years: list[int] | None = None,
) -> tuple[ShapeCalibrationModel, dict[str, Any]]:
    """MLE con shrinkage: draw por (context, λ) y favorite_scale por peak bin."""
    rows = _collect_shape_rows(archives, train_years=train_years)
    if not rows:
        return ShapeCalibrationModel(), {"error": "no_rows", "n": 0}

    metrics: dict[str, Any] = {"n_matches": len(rows)}

    # --- draw by context (base) ---
    draw_by_context: dict[str, float] = {}
    ctx_metrics: dict[str, Any] = {}
    for ctx in ("close", "balanced", "mismatch"):
        subset = [r for r in rows if r["context"] == ctx]
        if not subset:
            draw_by_context[ctx] = DEFAULT_DRAW_BY_CONTEXT.get(ctx, 1.0)
            continue
        obs = sum(r["is_draw"] for r in subset)
        pred = sum(r["p_draw"] for r in subset)
        factor = _shrink_ratio(obs, pred, len(subset), prior=DEFAULT_DRAW_BY_CONTEXT.get(ctx, 1.0))
        draw_by_context[ctx] = round(max(0.85, min(1.25, factor)), 4)
        ctx_metrics[ctx] = {
            "n": len(subset),
            "draw_rate": round(obs / len(subset), 4),
            "mean_p_draw": round(pred / len(subset), 4),
            "draw_factor": draw_by_context[ctx],
        }
    metrics["by_context"] = ctx_metrics

    # --- draw by context × λ tertiles ---
    lambda_bins: dict[str, list[dict[str, float]]] = {}
    for ctx in ("close", "balanced", "mismatch"):
        subset = [r for r in rows if r["context"] == ctx]
        if len(subset) < 9:
            continue
        lt_values = [r["lambda_total"] for r in subset]
        t1, t2 = _lambda_tertiles(lt_values)
        edges = [
            (0.0, t1),
            (t1, t2),
            (t2, 99.0),
        ]
        bins_out: list[dict[str, float]] = []
        for lo, hi in edges:
            bin_rows = [r for r in subset if lo <= r["lambda_total"] < hi or (hi >= 99 and r["lambda_total"] >= lo)]
            if len(bin_rows) < 4:
                continue
            obs = sum(r["is_draw"] for r in bin_rows)
            pred = sum(r["p_draw"] for r in bin_rows)
            prior = draw_by_context.get(ctx, 1.0)
            factor = _shrink_ratio(obs, pred, len(bin_rows), prior=prior)
            bins_out.append(
                {
                    "lambda_min": round(lo, 3),
                    "lambda_max": round(hi, 3) if hi < 99 else 99.0,
                    "draw_factor": round(max(0.85, min(1.25, factor)), 4),
                    "n": len(bin_rows),
                }
            )
        if bins_out:
            lambda_bins[ctx] = bins_out
    metrics["lambda_bins"] = {
        ctx: [{"n": b["n"], "factor": b["draw_factor"]} for b in bins] for ctx, bins in lambda_bins.items()
    }

    # --- favorite scale by peak bin ---
    peak_edges = [0.0, 0.40, 0.50, 0.60, 1.0]
    peak_bins: list[dict[str, float]] = []
    peak_metrics: list[dict[str, Any]] = []
    for i in range(len(peak_edges) - 1):
        lo, hi = peak_edges[i], peak_edges[i + 1]
        subset = [r for r in rows if lo <= r["peak"] < hi or (hi >= 1.0 and r["peak"] >= lo)]
        if len(subset) < 5:
            default_scale = DEFAULT_PEAK_BINS[min(i, len(DEFAULT_PEAK_BINS) - 1)]["favorite_scale"]
            peak_bins.append({"min_peak": lo, "max_peak": hi, "favorite_scale": default_scale, "n": len(subset)})
            continue
        obs = sum(r["fav_won"] for r in subset)
        pred = sum(r["p_fav"] for r in subset)
        # Si modelo subestima favorito → obs/pred > 1 → scale > 1
        scale = _shrink_ratio(obs, pred, len(subset), prior=1.0)
        scale = max(0.90, min(1.15, scale))
        peak_bins.append(
            {
                "min_peak": lo,
                "max_peak": hi,
                "favorite_scale": round(scale, 4),
                "n": len(subset),
            }
        )
        peak_metrics.append(
            {
                "bin": f"{lo:.2f}-{hi:.2f}",
                "n": len(subset),
                "fav_win_rate": round(obs / len(subset), 4),
                "mean_p_fav": round(pred / len(subset), 4),
                "favorite_scale": round(scale, 4),
            }
        )
    metrics["peak_bins"] = peak_metrics

    # --- log loss before/after (in-sample diagnostic) ---
    def _log_loss(row: dict, adjusted: dict[str, float]) -> float:
        actual = {
            "home_win": 1 if row.get("_gh", 0) > row.get("_ga", 0) else 0,
            "draw": row["is_draw"],
            "away_win": 1 if row.get("_gh", 0) < row.get("_ga", 0) else 0,
        }
        # simplified: use stored outcomes
        return 0.0

    model = ShapeCalibrationModel(
        draw_by_context=draw_by_context,
        lambda_bins=lambda_bins,
        peak_bins=peak_bins,
        metrics=metrics,
    )
    return model, metrics


def evaluate_shape_model(
    rows: list[dict[str, Any]],
    model: ShapeCalibrationModel,
) -> dict[str, float]:
    """Log-loss 1x2 antes/después de shape (diagnóstico in-sample)."""
    ll_before = 0.0
    ll_after = 0.0
    n = 0
    for r in rows:
        probs = {"home_win": r["p_home"], "draw": r["p_draw"], "away_win": r["p_away"]}
        feats = ShapeFeatures(
            lambda_total=r["lambda_total"],
            elo_gap=r["elo_gap"],
        )
        adj, _ = apply_poisson_shape_calibration(
            probs,
            r["context"],  # type: ignore[arg-type]
            features=feats,
            model=model,
        )
        y = r["label"]
        ll_before += -math.log(max(probs[y], 1e-9))
        ll_after += -math.log(max(adj[y], 1e-9))
        n += 1
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "log_loss_before": round(ll_before / n, 4),
        "log_loss_after": round(ll_after / n, 4),
        "delta": round((ll_before - ll_after) / n, 4),
    }
