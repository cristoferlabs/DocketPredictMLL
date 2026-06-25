"""
Métricas de calibración — Brier, LogLoss, curvas y fit de pesos Poisson/ELO.

Usado offline (scripts/fit_model_weights.py) y en evaluación walk-forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from apps.worker.ml.calibration import (
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
)
from apps.worker.ml.model_combiner import ModelCombinationWeights, Probabilities1X2, combine_poisson_elo

WEIGHTS_ARTIFACT_PATH = Path("artifacts/calibration/wc_model_weights.json")
OUTCOMES_1X2 = ("home_win", "draw", "away_win")
LABEL_TO_KEY = {0: "home_win", 1: "draw", 2: "away_win"}


@dataclass
class CalibrationReport:
    n_samples: int
    brier_1x2: float
    log_loss_1x2: float
    ece_max_prob: float
    hit_rate_1x2: float
    brier_by_outcome: dict[str, float] = field(default_factory=dict)
    reliability_bins: list[dict[str, Any]] = field(default_factory=list)
    underdog_inflation_pp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "brier_1x2": self.brier_1x2,
            "log_loss_1x2": self.log_loss_1x2,
            "ece_max_prob": self.ece_max_prob,
            "hit_rate_1x2": self.hit_rate_1x2,
            "brier_by_outcome": self.brier_by_outcome,
            "reliability_bins": self.reliability_bins,
            "underdog_inflation_pp": self.underdog_inflation_pp,
            "details": self.details,
        }


@dataclass
class WeightFitResult:
    weights: ModelCombinationWeights
    train_report: CalibrationReport
    test_report: CalibrationReport | None
    grid: list[dict[str, Any]] = field(default_factory=list)
    underdog_dampen_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "poisson": self.weights.poisson,
            "elo": self.weights.elo,
            "market": self.weights.market,
            "underdog_dampen_factor": self.underdog_dampen_factor,
            "train": self.train_report.to_dict(),
            "test": self.test_report.to_dict() if self.test_report else None,
            "grid_top5": sorted(self.grid, key=lambda x: x["brier_1x2"])[:5],
        }


def log_loss_binary(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    eps: float = 1e-15,
) -> float:
    """Log loss binario — menor es mejor."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return 0.0
    return round(float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))), 6)


def log_loss_multiclass(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    eps: float = 1e-15,
) -> float:
    """Log loss multiclass (one-hot)."""
    if not probs:
        return 0.0
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    n = len(labels)
    loss = 0.0
    for i, lbl in enumerate(labels):
        if 0 <= lbl < p.shape[1]:
            loss -= np.log(p[i, lbl])
    return round(float(loss / n), 6)


def brier_score_binary(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return 0.0
    return round(float(np.mean((p - y) ** 2)), 6)


def _max_prob_calibration(
    probs_1x2: list[list[float]],
    labels: list[int],
) -> tuple[float, list[dict[str, Any]]]:
    """ECE sobre la probabilidad del outcome predicho (máx prob)."""
    max_probs: list[float] = []
    hits: list[int] = []
    for row, lbl in zip(probs_1x2, labels):
        idx = int(np.argmax(row))
        max_probs.append(row[idx])
        hits.append(1 if idx == lbl else 0)
    ece = expected_calibration_error(max_probs, hits)
    bins = reliability_bins(max_probs, hits, n_bins=8)
    return ece, bins


def _underdog_inflation_pp(
    probs_1x2: list[list[float]],
    labels: list[int],
    *,
    threshold: float = 0.40,
) -> float:
    """
    Inflación underdog histórica: E[p_model | underdog pick] - E[actual | underdog pick].
    Positivo = modelo sobreestima underdogs.
    """
    model_dog: list[float] = []
    actual_dog: list[float] = []
    for row, lbl in zip(probs_1x2, labels):
        away_p, draw_p, home_p = row[2], row[1], row[0]
        peak = max(home_p, away_p)
        if peak >= threshold:
            continue
        dog_side = 2 if away_p >= home_p else 0
        model_dog.append(row[dog_side])
        actual_dog.append(1.0 if lbl == dog_side else 0.0)
    if not model_dog:
        return 0.0
    return round((float(np.mean(model_dog)) - float(np.mean(actual_dog))) * 100.0, 2)


def evaluate_1x2_predictions(
    probs_1x2: list[list[float]],
    labels: list[int],
) -> CalibrationReport:
    """Brier + LogLoss + ECE + bins para predicciones 1X2."""
    n = len(labels)
    if n == 0:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0)

    brier = brier_score_multiclass(probs_1x2, labels)
    ll = log_loss_multiclass(probs_1x2, labels)
    ece, bins = _max_prob_calibration(probs_1x2, labels)

    hits = sum(1 for row, lbl in zip(probs_1x2, labels) if int(np.argmax(row)) == lbl)
    brier_by: dict[str, float] = {}
    for idx, key in LABEL_TO_KEY.items():
        p_bin = [row[idx] for row in probs_1x2]
        y_bin = [1 if lbl == idx else 0 for lbl in labels]
        brier_by[key] = brier_score_binary(p_bin, y_bin)

    dog_inf = _underdog_inflation_pp(probs_1x2, labels)

    return CalibrationReport(
        n_samples=n,
        brier_1x2=brier,
        log_loss_1x2=ll,
        ece_max_prob=ece,
        hit_rate_1x2=round(hits / n, 4),
        brier_by_outcome=brier_by,
        reliability_bins=bins,
        underdog_inflation_pp=dog_inf,
    )


def blend_components(
    poisson: dict[str, float],
    elo: dict[str, float],
    weights: ModelCombinationWeights,
) -> list[float]:
    blended, _ = combine_poisson_elo(poisson, elo, weights=weights)
    d = blended.as_dict()
    return [d["home_win"], d["draw"], d["away_win"]]


def _components_to_probs(
    components: list[dict[str, Any]],
    weights: ModelCombinationWeights,
) -> tuple[list[list[float]], list[int]]:
    probs: list[list[float]] = []
    labels: list[int] = []
    for row in components:
        probs.append(blend_components(row["poisson"], row["elo"], weights))
        labels.append(int(row["label"]))
    return probs, labels


def fit_poisson_elo_weights(
    components: list[dict[str, Any]],
    *,
    market_weight: float = 0.2,
    grid_step: float = 0.05,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> WeightFitResult:
    """
    Grid search Poisson/ELO minimizando Brier en train (2018+2022 holdout por año).

    components: salida de collect_1x2_components (poisson, elo, label, year).
    """
    train_years = train_years or [2018]
    test_years = test_years or [2022]

    train = [c for c in components if c.get("year") in train_years]
    test = [c for c in components if c.get("year") in test_years]
    if not train:
        train = components
        test = []

    grid_results: list[dict[str, Any]] = []
    best_brier = float("inf")
    best_wp = 0.5

    wp_values = np.arange(0.25, 0.76, grid_step)
    for wp in wp_values:
        wp_f = round(float(wp), 4)
        we_f = round(1.0 - wp_f, 4)
        w = ModelCombinationWeights(poisson=wp_f, elo=we_f, market=market_weight)
        train_probs, train_labels = _components_to_probs(train, w)
        report = evaluate_1x2_predictions(train_probs, train_labels)
        grid_results.append(
            {
                "poisson": wp_f,
                "elo": we_f,
                "brier_1x2": report.brier_1x2,
                "log_loss_1x2": report.log_loss_1x2,
            }
        )
        if report.brier_1x2 < best_brier:
            best_brier = report.brier_1x2
            best_wp = wp_f

    best_weights = ModelCombinationWeights(
        poisson=best_wp,
        elo=round(1.0 - best_wp, 4),
        market=market_weight,
    )
    train_probs, train_labels = _components_to_probs(train, best_weights)
    train_report = evaluate_1x2_predictions(train_probs, train_labels)

    test_report: CalibrationReport | None = None
    if test:
        test_probs, test_labels = _components_to_probs(test, best_weights)
        test_report = evaluate_1x2_predictions(test_probs, test_labels)

    dog_pp = train_report.underdog_inflation_pp
    if test_report:
        dog_pp = max(dog_pp, test_report.underdog_inflation_pp)
    dampen = propose_underdog_dampen_factor(dog_pp)

    return WeightFitResult(
        weights=best_weights,
        train_report=train_report,
        test_report=test_report,
        grid=grid_results,
        underdog_dampen_factor=dampen,
    )


def propose_underdog_dampen_factor(underdog_inflation_pp: float) -> float:
    """
    Factor multiplicativo sobre prob underdog (<30%) post-calibración.

    inflation > 8pp → dampen 0.82; > 5pp → 0.88; > 3pp → 0.93; else 1.0
    """
    if underdog_inflation_pp > 8.0:
        return 0.82
    if underdog_inflation_pp > 5.0:
        return 0.88
    if underdog_inflation_pp > 3.0:
        return 0.93
    return 1.0


def apply_underdog_dampening(
    home_win: float,
    draw: float,
    away_win: float,
    *,
    dampen_factor: float = 1.0,
    cap_p: float = 0.30,
) -> tuple[float, float, float]:
    """Reduce probabilidad de outcomes bajo cap_p (underdogs)."""
    if dampen_factor >= 1.0:
        return home_win, draw, away_win

    h, d, a = home_win, draw, away_win
    if h < cap_p:
        h *= dampen_factor
    if a < cap_p:
        a *= dampen_factor

    total = h + d + a
    if total <= 0:
        return home_win, draw, away_win
    return h / total, d / total, a / total


def save_fitted_model_weights(result: WeightFitResult) -> Path:
    WEIGHTS_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["engine"] = "calibration_metrics_v1"
    WEIGHTS_ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return WEIGHTS_ARTIFACT_PATH


def load_fitted_model_weights() -> dict[str, Any] | None:
    if not WEIGHTS_ARTIFACT_PATH.exists():
        return None
    try:
        return json.loads(WEIGHTS_ARTIFACT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_1x2_components(
    archives: dict[int, dict],
    *,
    years: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Recolecta Poisson/ELO por partido (sin blend) para fit de pesos."""
    from apps.worker.ml.wc_historical import extract_finished_matches, predict_match_1x2_components

    years = years or [2018, 2022]
    matches = extract_finished_matches(archives, years=years)
    rows: list[dict[str, Any]] = []
    for m in matches:
        comp = predict_match_1x2_components(m, archives)
        if comp:
            rows.append(comp)
    return rows
