"""Probability calibration — ECE, reliability bins, isotonic regression."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    Fit isotonic per outcome on historical WC data.
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

    metrics = {
        "sample_size": len(matches),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "factors": factors,
    }
    return factors, calibrators, metrics


def apply_scalar_calibration(prob: float, factor: float) -> float:
    """Simple linear shrink toward 0.5: p' = 0.5 + (p - 0.5) * factor."""
    factor = float(np.clip(factor, 0.5, 1.5))
    return round(0.5 + (prob - 0.5) * factor, 6)


def calibrate_model_markets(
    home_win: float,
    draw: float,
    away_win: float,
    over_25: float,
    under_25: float,
    btts_yes: float,
    btts_no: float,
    factors: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Apply calibration factors and renormalize 1X2."""
    f = factors or DEFAULT_CALIBRATION_FACTORS
    f1x2 = f.get("1X2", {})
    fou = f.get("over_under_2.5", {})
    fbtts = f.get("btts", {})

    h = apply_scalar_calibration(home_win, f1x2.get("home_win", 1.0))
    d = apply_scalar_calibration(draw, f1x2.get("draw", 1.0))
    a = apply_scalar_calibration(away_win, f1x2.get("away_win", 1.0))
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
