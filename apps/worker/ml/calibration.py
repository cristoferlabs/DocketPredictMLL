"""Probability calibration — ECE, reliability bins, isotonic regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(
    probs: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (lower is better)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return 0.0

    bins = reliability_bins(p.tolist(), y.tolist(), n_bins=n_bins)
    ece = 0.0
    n = len(p)
    for b in bins:
        count = b["count"]
        if count > 0:
            ece += (count / n) * abs(b["mean_pred"] - b["mean_outcome"])
    return round(float(ece), 6)


def reliability_bins(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability diagram data per bin."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return []

    edges = np.linspace(0, 1, n_bins + 1)
    result: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        count = int(mask.sum())
        if count == 0:
            result.append(
                {
                    "bin": i,
                    "lo": round(float(lo), 3),
                    "hi": round(float(hi), 3),
                    "count": 0,
                    "mean_pred": 0.0,
                    "mean_outcome": 0.0,
                }
            )
            continue
        result.append(
            {
                "bin": i,
                "lo": round(float(lo), 3),
                "hi": round(float(hi), 3),
                "count": count,
                "mean_pred": round(float(p[mask].mean()), 4),
                "mean_outcome": round(float(y[mask].mean()), 4),
            }
        )
    return result


def brier_score_multiclass(
    probs: list[list[float]],
    labels: list[int],
) -> float:
    """Multiclass Brier score for K classes (one-hot labels)."""
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=float)
    k = p.shape[1]
    y = np.zeros((len(labels), k), dtype=float)
    for i, lbl in enumerate(labels):
        if 0 <= lbl < k:
            y[i, lbl] = 1.0
    return round(float(np.mean(np.sum((p - y) ** 2, axis=1))), 6)


@dataclass
class IsotonicCalibrator:
    """Per-market isotonic calibrator."""

    market: str
    _model: IsotonicRegression | None = field(default=None, repr=False)
    sample_size: int = 0
    method: str = "isotonic"

    def fit(self, probs: list[float], outcomes: list[int]) -> "IsotonicCalibrator":
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(outcomes, dtype=float)
        if len(p) < 10:
            self._model = None
            self.sample_size = len(p)
            return self
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(p, y)
        self.sample_size = len(p)
        return self

    def transform(self, prob: float) -> float:
        if self._model is None:
            return prob
        clipped = float(np.clip(prob, 1e-6, 1 - 1e-6))
        return round(float(self._model.predict([clipped])[0]), 6)

    def transform_dict(self, probs: dict[str, float]) -> dict[str, float]:
        return {k: self.transform(v) for k, v in probs.items()}


# Default shrink factors (identity) until fitted from historical data
DEFAULT_CALIBRATION_FACTORS: dict[str, dict[str, float]] = {
    "1X2": {"home_win": 1.0, "draw": 1.0, "away_win": 1.0},
    "over_under_2.5": {"over": 1.0, "under": 1.0},
    "btts": {"yes": 1.0, "no": 1.0},
}

# Bucket calibration — favorito / medio / underdog + dampening empate (paso B)
DEFAULT_BUCKET_CONFIG: dict[str, Any] = {
    "team_win": {"favorite": 1.0, "medium": 1.0, "underdog": 1.0},
    "draw": 1.0,
    "draw_dampen_threshold": 0.48,
    "draw_dampen_factor": 0.90,
    "underdog_cap_max_p": 0.30,
    "underdog_cap_factor": 0.85,
    "compressed_favorite_max_lift": 0.14,
}

FAVORITE_BUCKET_MIN = 0.55
COMPRESSED_FAVORITE_MAX = 0.68
UNDERDOG_BUCKET_MAX = 0.40
MIN_BUCKET_SAMPLES = 12


def classify_team_win_bucket(prob: float, *, match_peak: float | None = None) -> str:
    """Clasifica P de victoria para calibración por bucket (rol real, no slot local)."""
    if prob >= FAVORITE_BUCKET_MIN:
        return "favorite"
    if prob < UNDERDOG_BUCKET_MAX:
        return "underdog"
    if match_peak is not None and prob >= match_peak - 1e-9:
        if match_peak >= 0.52 and prob >= 0.45:
            return "favorite"
        if match_peak >= 0.48 and prob >= 0.42:
            return "medium"
    return "medium"


def _default_factors_with_buckets() -> dict[str, Any]:
    import copy

    base = copy.deepcopy(DEFAULT_CALIBRATION_FACTORS)
    base["1X2_buckets"] = copy.deepcopy(DEFAULT_BUCKET_CONFIG)
    return base


def fit_bucket_1x2_calibration(
    archives: dict[int, dict],
    train_years: list[int] | None = None,
    min_samples: int = MIN_BUCKET_SAMPLES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Paso A — isotonic 1X2 por bucket (favorite / medium / underdog / draw).

    Entrena contra resultados reales WC (no mercado).
    Devuelve (bucket_config, metrics).
    """
    from apps.worker.ml.wc_historical import actual_outcomes, extract_finished_matches, predict_match_historical

    train_years = train_years or [2018, 2022]
    matches = extract_finished_matches(archives, years=train_years)

    buckets: dict[str, tuple[list[float], list[int]]] = {
        "favorite": ([], []),
        "medium": ([], []),
        "underdog": ([], []),
        "draw": ([], []),
    }

    for m in matches:
        probs = predict_match_historical(m, archives)
        actual = actual_outcomes(m)
        for side, p, y in (
            ("home_win", probs["home_win"], actual["home_win"]),
            ("away_win", probs["away_win"], actual["away_win"]),
        ):
            role = classify_team_win_bucket(p)
            buckets[role][0].append(p)
            buckets[role][1].append(y)
        buckets["draw"][0].append(probs["draw"])
        buckets["draw"][1].append(actual["draw"])

    calibrators: dict[str, IsotonicCalibrator] = {}
    team_win_factors: dict[str, float] = {}
    ece: dict[str, dict[str, float]] = {}

    for role in ("favorite", "medium", "underdog"):
        ps, ys = buckets[role]
        cal = IsotonicCalibrator(f"team_win_{role}")
        cal.fit(ps, ys)
        calibrators[role] = cal
        if len(ps) >= min_samples:
            ece[role] = {
                "before": expected_calibration_error(ps, ys),
                "after": expected_calibration_error([cal.transform(p) for p in ps], ys),
                "n": len(ps),
            }
            team_win_factors[role] = factors_from_isotonic(cal, ps, ys)
        else:
            team_win_factors[role] = 1.0

    ps_d, ys_d = buckets["draw"]
    cal_draw = IsotonicCalibrator("draw")
    cal_draw.fit(ps_d, ys_d)
    draw_factor = 1.0
    if len(ps_d) >= min_samples:
        ece["draw"] = {
            "before": expected_calibration_error(ps_d, ys_d),
            "after": expected_calibration_error([cal_draw.transform(p) for p in ps_d], ys_d),
            "n": len(ps_d),
        }
        draw_factor = factors_from_isotonic(cal_draw, ps_d, ys_d)

    # Paso B — dampening empate estructural (sesgo vs mercado detectado en audit)
    bucket_config: dict[str, Any] = {
        "team_win": {
            # Histórico WC puede pedir <1; paso C sube hacia mercado live (>1 lift favoritos)
            "favorite": round(float(np.clip(team_win_factors.get("favorite", 1.0), 0.85, 1.45)), 4),
            "medium": round(float(np.clip(team_win_factors.get("medium", 1.0), 0.85, 1.25)), 4),
            "underdog": round(float(np.clip(team_win_factors.get("underdog", 1.0), 0.55, 1.05)), 4),
        },
        "draw": round(float(np.clip(draw_factor, 0.80, 1.05)), 4),
        "draw_dampen_threshold": DEFAULT_BUCKET_CONFIG["draw_dampen_threshold"],
        "draw_dampen_factor": DEFAULT_BUCKET_CONFIG["draw_dampen_factor"],
        "underdog_cap_max_p": DEFAULT_BUCKET_CONFIG["underdog_cap_max_p"],
        "underdog_cap_factor": DEFAULT_BUCKET_CONFIG["underdog_cap_factor"],
        "compressed_favorite_max_lift": DEFAULT_BUCKET_CONFIG["compressed_favorite_max_lift"],
    }

    metrics = {
        "sample_size": len(matches),
        "ece_by_bucket": ece,
        "team_win_factors": team_win_factors,
        "draw_factor": draw_factor,
    }
    return bucket_config, metrics


def merge_bucket_config(
    factors: dict[str, Any] | None,
    bucket_config: dict[str, Any],
) -> dict[str, Any]:
    """Inyecta bucket config en factors para calibrate_model_markets."""
    import copy

    merged = copy.deepcopy(factors or _default_factors_with_buckets())
    merged.setdefault("1X2_buckets", {})
    merged["1X2_buckets"].update(bucket_config)
    for grp, defaults in DEFAULT_CALIBRATION_FACTORS.items():
        merged.setdefault(grp, {})
        for k, v in defaults.items():
            merged[grp].setdefault(k, v)
    return merged


def apply_bucket_factor(prob: float, factor: float, role: str) -> float:
    """Aplica factor de bucket; lift extra si favorito comprimido (Brasil ~50%)."""
    if role == "favorite" and factor > 1.0:
        if prob < 0.5:
            # Sub-50%: scalar shrink aleja de 0.5 — empujar hacia arriba explícitamente
            gain = (factor - 1.0) * min(0.42, max(0.08, 0.68 - prob) * 0.72)
            return round(min(0.82, prob + gain), 6)
        calibrated = apply_scalar_calibration(prob, factor)
        if prob < COMPRESSED_FAVORITE_MAX:
            gap = max(0.0, 0.62 - prob)
            lift = (factor - 1.0) * min(0.28, gap * 0.62)
            if prob < FAVORITE_BUCKET_MIN:
                lift += (factor - 1.0) * min(0.16, gap * 0.42)
            return round(min(0.82, calibrated + lift), 6)
        return calibrated
    if role == "medium" and factor > 1.05 and prob >= 0.42:
        gain = (factor - 1.0) * min(0.18, max(0.0, 0.58 - prob) * 0.5)
        return round(min(0.78, prob + gain), 6)
    if role == "underdog" and factor < 1.0:
        return apply_scalar_calibration(prob, factor)
    return apply_scalar_calibration(prob, factor)


def _fix_compressed_favorite(
    h: float,
    d: float,
    a: float,
    raw_h: float,
    raw_d: float,
    raw_a: float,
    *,
    max_lift: float = 0.14,
) -> tuple[float, float, float]:
    """
    Tras normalizar buckets, recupera favoritos comprimidos (Brasil ~50% → ~58%).

    Usa probabilidades crudas del modelo para no perder el lift en renormalización.
    """
    raw_peak = max(raw_h, raw_a)
    peak = max(h, a)
    if raw_peak < 0.48 or peak >= 0.62:
        return h, d, a
    target_min = min(0.68, max(0.52, raw_peak + 0.10 + max_lift * 0.25))
    if peak >= target_min:
        return h, d, a
    lift = min(max_lift, target_min - peak, max(0.0, 0.64 - raw_peak) * 0.85)
    if lift <= 0.002:
        return h, d, a
    if raw_h >= raw_a:
        nh = min(0.78, h + lift)
        rem = d + a
        if rem <= 0:
            return nh, d, a
        scale = max(0.0, (1.0 - nh) / rem)
        return nh, d * scale, a * scale
    na = min(0.78, a + lift)
    rem = h + d
    if rem <= 0:
        return h, d, na
    scale = max(0.0, (1.0 - na) / rem)
    return h * scale, d * scale, na


def apply_bucket_1x2(
    home_win: float,
    draw: float,
    away_win: float,
    bucket_config: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Aplica factores por bucket + dampening empate (paso B)."""
    cfg = bucket_config or DEFAULT_BUCKET_CONFIG
    tw = cfg.get("team_win", {})
    peak = max(home_win, away_win)

    h_role = classify_team_win_bucket(home_win, match_peak=peak)
    a_role = classify_team_win_bucket(away_win, match_peak=peak)
    h = apply_bucket_factor(home_win, float(tw.get(h_role, 1.0)), h_role)
    a = apply_bucket_factor(away_win, float(tw.get(a_role, 1.0)), a_role)
    d = apply_scalar_calibration(draw, float(cfg.get("draw", 1.0)))

    threshold = float(cfg.get("draw_dampen_threshold", 0.55))
    if max(home_win, away_win) >= threshold:
        d = apply_scalar_calibration(d, float(cfg.get("draw_dampen_factor", 0.90)))

    cap_p = float(cfg.get("underdog_cap_max_p", 0.30))
    cap_f = float(cfg.get("underdog_cap_factor", 0.85))
    if h < cap_p:
        h = apply_scalar_calibration(h, cap_f)
    if a < cap_p:
        a = apply_scalar_calibration(a, cap_f)

    total = h + d + a
    if total > 0:
        h, d, a = h / total, d / total, a / total
    else:
        return home_win, draw, away_win
    max_lift = float(cfg.get("compressed_favorite_max_lift", 0.14))
    return _fix_compressed_favorite(
        h, d, a, home_win, draw, away_win, max_lift=max_lift
    )


CALIBRATION_ARTIFACT_PATH = Path("artifacts/calibration/wc_bucket_factors.json")
CALIBRATION_APPROVED_PATH = Path("artifacts/calibration/wc_bucket_factors.approved.json")
CALIBRATION_CANDIDATE_PATH = Path("artifacts/calibration/wc_bucket_factors.candidate.json")


def save_fitted_calibration_factors(
    factors: dict[str, Any],
    *,
    approved: bool = True,
) -> Path:
    CALIBRATION_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json

    payload = dict(factors)
    if approved:
        payload.pop("_deploy_blocked", None)
        payload.pop("_deploy_reasons", None)
        CALIBRATION_ARTIFACT_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        CALIBRATION_APPROVED_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return CALIBRATION_ARTIFACT_PATH

    payload["_deploy_blocked"] = True
    CALIBRATION_CANDIDATE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return CALIBRATION_CANDIDATE_PATH


def load_fitted_calibration_factors() -> dict[str, Any]:
    """Carga factors con buckets; ignora candidatos bloqueados por deploy gate."""
    import json

    def _load_path(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    raw = _load_path(CALIBRATION_ARTIFACT_PATH)
    if raw and raw.get("_deploy_blocked"):
        raw = None
    if raw is None:
        raw = _load_path(CALIBRATION_APPROVED_PATH)
    if raw is None:
        return _default_factors_with_buckets()
    try:
        return merge_bucket_config(raw, raw.get("1X2_buckets", DEFAULT_BUCKET_CONFIG))
    except Exception:
        return _default_factors_with_buckets()


def seed_buckets_for_market_bias(buckets: dict[str, Any], audit) -> dict[str, Any]:
    """Salto inicial: isotónico histórico (<1) vs mercado live (>1 en favoritos)."""
    import copy

    b = copy.deepcopy(buckets)
    tw = b.setdefault("team_win", {})
    fav_gap = float(getattr(audit, "favorite_compression_avg", 0.0))
    dog_gap = float(getattr(audit, "underdog_inflation_avg", 0.0))

    if fav_gap > 0.05:
        # factor > 1 activa lift en apply_bucket_factor; <1 comprime más
        tw["favorite"] = round(max(float(tw.get("favorite", 1.0)), 1.08), 4)
        if fav_gap > 0.15:
            tw["favorite"] = round(max(float(tw["favorite"]), 1.18), 4)
        if float(tw.get("medium", 1.0)) < 1.12:
            tw["medium"] = 1.12
        b["compressed_favorite_max_lift"] = round(
            max(float(b.get("compressed_favorite_max_lift", 0.14)), 0.18), 4
        )
    if dog_gap > 0.05:
        tw["underdog"] = round(min(float(tw.get("underdog", 1.0)), 0.72), 4)
    if float(getattr(audit, "draw_inflation_avg", 0.0)) > 0.03:
        b["draw"] = round(max(0.72, float(b.get("draw", 1.0)) - 0.04), 4)
    return b


def propose_bucket_adjustments(
    buckets: dict[str, Any],
    audit,
    *,
    step_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Ajuste dirigido por señales del audit vs mercado live.

    fav_compression > 0 → subir factor favorite (>1 para lift real)
    draw_inflation > 0   → bajar draw / dampening
    underdog_inflation > 0 → bajar factor underdog
    """
    import copy

    b = copy.deepcopy(buckets)
    tw = b.setdefault("team_win", {})
    base = 0.06 * max(0.3, step_scale)

    fav_gap = float(getattr(audit, "favorite_compression_avg", 0.0))
    draw_gap = float(getattr(audit, "draw_inflation_avg", 0.0))
    dog_gap = float(getattr(audit, "underdog_inflation_avg", 0.0))

    fav_f = float(tw.get("favorite", 1.0))
    if fav_gap > 0.012:
        boost = base * min(3.5, fav_gap / 0.03)
        if fav_f < 1.0:
            tw["favorite"] = round(min(1.85, 1.06 + boost), 4)
        else:
            tw["favorite"] = round(min(1.85, fav_f + boost), 4)
        if fav_gap > 0.08:
            tw["medium"] = round(min(1.28, float(tw.get("medium", 1.0)) + boost * 0.35), 4)
            b["compressed_favorite_max_lift"] = round(
                min(0.35, float(b.get("compressed_favorite_max_lift", 0.14)) + base * 0.35),
                4,
            )
    elif fav_gap < -0.012:
        tw["favorite"] = round(max(0.92, fav_f - base * 0.7), 4)

    if draw_gap > 0.012:
        cut = base * min(2.5, draw_gap / 0.025)
        b["draw"] = round(max(0.65, float(b.get("draw", 1.0)) - cut), 4)
        b["draw_dampen_factor"] = round(
            max(0.68, float(b.get("draw_dampen_factor", 0.90)) - cut * 0.6), 4
        )
    elif draw_gap < -0.012:
        b["draw"] = round(min(1.08, float(b.get("draw", 1.0)) + base * 0.45), 4)

    if dog_gap > 0.012:
        cut = base * min(2.8, dog_gap / 0.03)
        tw["underdog"] = round(max(0.48, float(tw.get("underdog", 1.0)) - cut), 4)
        if dog_gap > 0.06:
            b["underdog_cap_factor"] = round(
                max(0.72, float(b.get("underdog_cap_factor", 0.85)) - cut * 0.25), 4
            )

    return b


def propose_scalar_1x2_adjustments(
    factors: dict[str, Any],
    audit,
    *,
    step_scale: float = 1.0,
) -> dict[str, Any]:
    """Ajusta factores escalares 1X2 (pre-bucket) hacia mercado."""
    import copy

    out = copy.deepcopy(factors)
    fav_gap = float(getattr(audit, "favorite_compression_avg", 0.0))
    dog_gap = float(getattr(audit, "underdog_inflation_avg", 0.0))
    if fav_gap <= 0.01 and dog_gap <= 0.01:
        return out
    base = 0.04 * max(0.3, step_scale)
    grp = out.setdefault("1X2", {})
    for key in ("home_win", "away_win"):
        cur = float(grp.get(key, 1.0))
        if fav_gap > 0.012:
            grp[key] = round(min(1.55, max(cur, 1.0) + base * min(3.0, fav_gap / 0.035)), 4)
        elif dog_gap > 0.012:
            grp[key] = round(max(0.88, cur - base * 0.3), 4)
    return out


def propose_market_tune_candidates(
    factors: dict[str, Any],
    audit,
    *,
    step_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Genera candidatos para coordinate descent (elige el que más reduce |score|)."""
    import copy

    buckets = factors.get("1X2_buckets", {})
    primary = copy.deepcopy(factors)
    primary["1X2_buckets"] = propose_bucket_adjustments(buckets, audit, step_scale=step_scale)
    primary = propose_scalar_1x2_adjustments(primary, audit, step_scale=step_scale)

    candidates = [primary]
    b2 = copy.deepcopy(factors)
    tw = b2.setdefault("1X2_buckets", {}).setdefault("team_win", {})
    tw["favorite"] = round(min(1.85, float(tw.get("favorite", 1.0)) + 0.10 * step_scale), 4)
    candidates.append(propose_scalar_1x2_adjustments(b2, audit, step_scale=step_scale))

    b3 = copy.deepcopy(factors)
    bk = b3.setdefault("1X2_buckets", {})
    bk["compressed_favorite_max_lift"] = round(
        min(0.35, float(bk.get("compressed_favorite_max_lift", 0.14)) + 0.03 * step_scale),
        4,
    )
    candidates.append(b3)

    if float(getattr(audit, "underdog_inflation_avg", 0.0)) > 0.03:
        b4 = copy.deepcopy(factors)
        utw = b4.setdefault("1X2_buckets", {}).setdefault("team_win", {})
        utw["underdog"] = round(max(0.48, float(utw.get("underdog", 1.0)) - 0.06 * step_scale), 4)
        candidates.append(b4)

    return candidates


def _bias_objective(audit) -> float:
    """Distancia a mercado — 0 es alineación perfecta."""
    return abs(float(getattr(audit, "favorite_bias_score", 0.0)))


def tune_buckets_for_market_bias(
    factors: dict[str, Any],
    audit_fn,
    *,
    target_score: float = 0.25,
    max_iter: int = 30,
    apply_fn=None,
) -> tuple[dict[str, Any], Any, int]:
    """
    Paso C — hill-climbing guiado por audit live hasta |favorite_bias_score| < target.

    audit_fn: callable() -> FavoriteBiasAudit (o objeto con favorite_bias_score)
  apply_fn: opcional; recibe factors y aplica en runtime (set_calibration_factors)
    """
    import copy

    best_factors = copy.deepcopy(factors)
    if apply_fn:
        apply_fn(best_factors)
    best_audit = audit_fn()
    best_obj = _bias_objective(best_audit)
    step_scale = 1.0
    iterations = 0
    stagnation = 0

    for i in range(max_iter):
        if abs(float(best_audit.favorite_bias_score)) <= target_score:
            break

        candidates = propose_market_tune_candidates(
            best_factors, best_audit, step_scale=step_scale
        )
        accepted = False
        for candidate in candidates:
            if apply_fn:
                apply_fn(candidate)
            trial_audit = audit_fn()
            trial_obj = _bias_objective(trial_audit)
            if trial_obj < best_obj - 1e-5:
                best_factors = candidate
                best_audit = trial_audit
                best_obj = trial_obj
                iterations = i + 1
                stagnation = 0
                step_scale = min(1.5, step_scale * 1.06)
                accepted = True
                break

        if accepted:
            continue
        stagnation += 1
        step_scale *= 0.55
        if apply_fn:
            apply_fn(best_factors)
        if stagnation >= 8:
            break

    if apply_fn:
        apply_fn(best_factors)
    return best_factors, best_audit, iterations


def factors_from_isotonic(
    calibrator: IsotonicCalibrator,
    probs: list[float],
    outcomes: list[int],
) -> float:
    """
    Derive scalar shrink factor for calibrate_model_markets from isotonic fit.
    factor ≈ 1.0 means well calibrated; <1 shrinks toward 50%.
    """
    if not probs:
        return 1.0
    factors: list[float] = []
    for p, _y in zip(probs, outcomes):
        if abs(p - 0.5) < 0.02:
            continue
        cal_p = calibrator.transform(p)
        factors.append((cal_p - 0.5) / (p - 0.5))
    if not factors:
        return 1.0
    return round(float(np.clip(np.median(factors), 0.5, 1.5)), 6)


def fit_calibration_bundle(
    archives: dict[int, dict],
    train_years: list[int] | None = None,
    min_samples: int = 20,
) -> tuple[dict[str, dict[str, float]], dict[str, IsotonicCalibrator], dict[str, Any]]:
    """
    Fit isotonic per outcome on historical WC data + buckets 1X2 (paso A).
    Returns (scalar factors for DB, calibrators, metrics).
    """
    from apps.worker.ml.wc_historical import actual_outcomes, extract_finished_matches, predict_match_historical

    train_years = train_years or [2018, 2022]
    matches = extract_finished_matches(archives, years=train_years)
    buckets: dict[str, tuple[list[float], list[int]]] = {
        "home_win": ([], []),
        "draw": ([], []),
        "away_win": ([], []),
        "over_25": ([], []),
        "under_25": ([], []),
        "btts_yes": ([], []),
    }

    for m in matches:
        probs = predict_match_historical(m, archives)
        actual = actual_outcomes(m)
        for key in buckets:
            buckets[key][0].append(probs[key])
            buckets[key][1].append(actual[key])

    calibrators: dict[str, IsotonicCalibrator] = {}
    ece_before: dict[str, float] = {}
    ece_after: dict[str, float] = {}

    for market, (ps, ys) in buckets.items():
        cal = IsotonicCalibrator(market)
        cal.fit(ps, ys)
        calibrators[market] = cal
        if len(ps) >= min_samples:
            ece_before[market] = expected_calibration_error(ps, ys)
            calibrated = [cal.transform(p) for p in ps]
            ece_after[market] = expected_calibration_error(calibrated, ys)

    factors: dict[str, dict[str, float]] = {
        "1X2": {},
        "over_under_2.5": {},
        "btts": {},
    }
    mapping = {
        "home_win": ("1X2", "home_win"),
        "draw": ("1X2", "draw"),
        "away_win": ("1X2", "away_win"),
        "over_25": ("over_under_2.5", "over"),
        "under_25": ("over_under_2.5", "under"),
        "btts_yes": ("btts", "yes"),
    }
    for market, (ps, ys) in buckets.items():
        if len(ps) < min_samples:
            continue
        f = factors_from_isotonic(calibrators[market], ps, ys)
        grp, outcome = mapping[market]
        factors[grp][outcome] = f

    for grp in factors:
        for default_key in DEFAULT_CALIBRATION_FACTORS.get(grp, {}):
            factors[grp].setdefault(default_key, 1.0)

    bucket_config, bucket_metrics = fit_bucket_1x2_calibration(
        archives, train_years=train_years, min_samples=MIN_BUCKET_SAMPLES
    )
    factors = merge_bucket_config(factors, bucket_config)

    metrics = {
        "sample_size": len(matches),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "factors": factors,
        "bucket_calibration": bucket_metrics,
    }
    return factors, calibrators, metrics


def apply_scalar_calibration(prob: float, factor: float) -> float:
    """Simple linear shrink toward 0.5: p' = 0.5 + (p - 0.5) * factor."""
    factor = float(np.clip(factor, 0.5, 1.5))
    return round(0.5 + (prob - 0.5) * factor, 6)


def apply_win_scalar_calibration(prob: float, factor: float) -> float:
    """Scalar 1X2: factor>1 con p<0.5 debe subir prob, no comprimir más."""
    factor = float(np.clip(factor, 0.5, 1.55))
    if factor > 1.0 and prob < 0.5:
        gain = (factor - 1.0) * min(0.38, max(0.06, 0.66 - prob) * 0.65)
        return round(min(0.85, prob + gain), 6)
    return apply_scalar_calibration(prob, factor)


def calibrate_model_markets(
    home_win: float,
    draw: float,
    away_win: float,
    over_25: float,
    under_25: float,
    btts_yes: float,
    btts_no: float,
    factors: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Apply calibration factors, bucket 1X2, and renormalize."""
    f = factors or _default_factors_with_buckets()
    f1x2 = f.get("1X2", {})
    fou = f.get("over_under_2.5", {})
    fbtts = f.get("btts", {})
    buckets = f.get("1X2_buckets")

    if buckets:
        # Rol por bucket — no aplicar factores escalares por slot home/away (evita sesgo local).
        h, d, a = home_win, draw, away_win
        h, d, a = apply_bucket_1x2(h, d, a, buckets)
    else:
        h = apply_win_scalar_calibration(home_win, f1x2.get("home_win", 1.0))
        d = apply_scalar_calibration(draw, f1x2.get("draw", 1.0))
        a = apply_win_scalar_calibration(away_win, f1x2.get("away_win", 1.0))
        total = h + d + a
        if total > 0:
            h, d, a = h / total, d / total, a / total

    o = apply_scalar_calibration(over_25, fou.get("over", 1.0))
    u = apply_scalar_calibration(under_25, fou.get("under", 1.0))
    ou_total = o + u
    if ou_total > 0:
        o, u = o / ou_total, u / ou_total

    by = apply_scalar_calibration(btts_yes, fbtts.get("yes", 1.0))
    bn = apply_scalar_calibration(btts_no, fbtts.get("no", 1.0))
    bt_total = by + bn
    if bt_total > 0:
        by, bn = by / bt_total, bn / bt_total

    return {
        "home_win": h,
        "draw": d,
        "away_win": a,
        "over_25": o,
        "under_25": u,
        "btts_yes": by,
        "btts_no": bn,
    }
