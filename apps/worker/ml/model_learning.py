"""
Fase C — aprendizaje bayesiano online, Brier live y auto-retrain de pesos.

Loop: predicción → CLV → resultado → update bias → (N resultados) → re-fit pesos.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apps.worker.ml.calibration_metrics import (
    collect_1x2_components,
    fit_poisson_elo_weights,
    load_fitted_model_weights,
    save_fitted_model_weights,
)

LEARNING_STATE_PATH = Path("artifacts/calibration/wc_learning_state.json")
LABEL_KEYS = ("home_win", "draw", "away_win")


@dataclass
class LearningState:
    n_updates: int = 0
    logit_bias: dict[str, float] = field(
        default_factory=lambda: {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    )
    rolling_brier_sum: float = 0.0
    rolling_brier_n: int = 0
    rolling_clv_sum: float = 0.0
    rolling_clv_n: int = 0
    results_since_retrain: int = 0
    last_retrain_at: str | None = None
    last_update_at: str | None = None

    @property
    def rolling_brier(self) -> float | None:
        if self.rolling_brier_n <= 0:
            return None
        return round(self.rolling_brier_sum / self.rolling_brier_n, 6)

    @property
    def rolling_clv(self) -> float | None:
        if self.rolling_clv_n <= 0:
            return None
        return round(self.rolling_clv_sum / self.rolling_clv_n, 6)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_learning_state() -> LearningState:
    if not LEARNING_STATE_PATH.exists():
        return LearningState()
    try:
        raw = json.loads(LEARNING_STATE_PATH.read_text(encoding="utf-8"))
        return LearningState(
            n_updates=int(raw.get("n_updates", 0)),
            logit_bias=dict(raw.get("logit_bias") or {}),
            rolling_brier_sum=float(raw.get("rolling_brier_sum", 0.0)),
            rolling_brier_n=int(raw.get("rolling_brier_n", 0)),
            rolling_clv_sum=float(raw.get("rolling_clv_sum", 0.0)),
            rolling_clv_n=int(raw.get("rolling_clv_n", 0)),
            results_since_retrain=int(raw.get("results_since_retrain", 0)),
            last_retrain_at=raw.get("last_retrain_at"),
            last_update_at=raw.get("last_update_at"),
        )
    except Exception:
        return LearningState()


def save_learning_state(state: LearningState) -> Path:
    LEARNING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["rolling_brier"] = state.rolling_brier
    payload["rolling_clv"] = state.rolling_clv
    payload["engine"] = "model_learning_v1"
    LEARNING_STATE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return LEARNING_STATE_PATH


def apply_learning_corrections(
    home_win: float,
    draw: float,
    away_win: float,
    *,
    state: LearningState | None = None,
) -> tuple[float, float, float]:
    """Aplica bias bayesiano acumulado en espacio logit (acotado)."""
    from apps.shared.config import get_settings

    s = state or load_learning_state()
    cap = get_settings().model_logit_bias_cap
    biases = s.logit_bias or {}

    def _adj(prob: float, key: str) -> float:
        bias = float(biases.get(key, 0.0))
        bias = max(-cap, min(cap, bias))
        return _inv_logit(_logit(prob) + bias)

    h = _adj(home_win, "home_win")
    d = _adj(draw, "draw")
    a = _adj(away_win, "away_win")
    total = h + d + a
    if total <= 0:
        return home_win, draw, away_win
    return h / total, d / total, a / total


def bayesian_outcome_update(
    model_probs: dict[str, float],
    actual_label: int,
    state: LearningState,
    *,
    learning_rate: float | None = None,
) -> LearningState:
    """
    Update online de bias logit por outcome (residual one-hot).

    actual_label: 0=home, 1=draw, 2=away
    """
    from apps.shared.config import get_settings

    lr = learning_rate if learning_rate is not None else get_settings().model_learning_rate
    cap = get_settings().model_logit_bias_cap

    for idx, key in enumerate(LABEL_KEYS):
        p = float(model_probs.get(key, 0.0))
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        y = 1.0 if idx == actual_label else 0.0
        residual = y - p
        cur = float(state.logit_bias.get(key, 0.0))
        state.logit_bias[key] = round(max(-cap, min(cap, cur + lr * residual)), 6)

    state.n_updates += 1
    state.results_since_retrain += 1
    state.last_update_at = datetime.now(timezone.utc).isoformat()
    return state


def update_from_wc_evaluation(
    *,
    model_probs: dict[str, float] | None,
    predicted_probability: float,
    predicted_outcome: str,
    team_home: str,
    team_away: str,
    actual_label: int,
    brier_score: float,
    clv_vs_close: float | None = None,
) -> LearningState:
    """Integra evaluación WC + CLV en estado de aprendizaje."""
    state = load_learning_state()

    probs = model_probs or _infer_probs_from_pick(
        predicted_probability,
        predicted_outcome,
        team_home,
        team_away,
    )
    state = bayesian_outcome_update(probs, actual_label, state)
    state.rolling_brier_sum += brier_score
    state.rolling_brier_n += 1
    if clv_vs_close is not None:
        state.rolling_clv_sum += clv_vs_close
        state.rolling_clv_n += 1
    save_learning_state(state)
    return state


def _infer_probs_from_pick(
    probability: float,
    predicted_outcome: str,
    team_home: str,
    team_away: str,
) -> dict[str, float]:
    """Fallback cuando metadata no trae vector 1X2 completo."""
    p = min(max(probability, 0.05), 0.85)
    rem = max(0.05, 1.0 - p)
    draw = rem * 0.35
    tail = rem - draw
    home, away = (p, tail) if predicted_outcome == team_home else (tail, p)
    if predicted_outcome.lower() in ("empate", "draw"):
        return {"home_win": tail / 2, "draw": p, "away_win": tail / 2}
    return {"home_win": home, "draw": draw, "away_win": away}


def evaluate_live_brier_from_db(db) -> float | None:
    """Brier medio sobre wc_predictions ya evaluadas (producción live)."""
    try:
        rows = (
            db.schema("ml")
            .table("wc_predictions")
            .select("brier_score")
            .not_.is_("evaluated_at", "null")
            .not_.is_("brier_score", "null")
            .order("evaluated_at", desc=True)
            .limit(100)
            .execute()
        )
        scores = [float(r["brier_score"]) for r in rows.data or [] if r.get("brier_score") is not None]
        if len(scores) < 3:
            return None
        return round(sum(scores) / len(scores), 6)
    except Exception:
        return None


def deploy_calibration_gate(
    *,
    audit,
    live_brier: float | None,
    historical_brier: float | None,
    max_bias: float | None = None,
    max_live_brier: float | None = None,
    backtest_roi: float | None = None,
    min_roi_backtest: float | None = None,
    backtest_roi_details: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Gate de deploy — bloquea calibración si métricas live o backtest empeoran.

    Returns (approved, reasons).
    """
    from apps.shared.config import get_settings

    settings = get_settings()
    max_bias = max_bias if max_bias is not None else settings.model_max_favorite_bias
    max_live_brier = (
        max_live_brier if max_live_brier is not None else settings.model_max_live_brier_1x2
    )
    min_roi = (
        min_roi_backtest if min_roi_backtest is not None else settings.ev_min_roi_backtest
    )

    reasons: list[str] = []
    bias = abs(float(getattr(audit, "favorite_bias_score", 0.0)))
    if bias > max_bias:
        reasons.append(f"favorite_bias {bias:.3f} > {max_bias}")

    if live_brier is not None and live_brier > max_live_brier:
        reasons.append(f"live_brier {live_brier:.4f} > {max_live_brier}")

    if historical_brier is not None and live_brier is not None:
        if live_brier > historical_brier + 0.08:
            reasons.append(
                f"live_brier {live_brier:.4f} >> hist {historical_brier:.4f} (+0.08)"
            )

    state = load_learning_state()
    if state.rolling_brier is not None and state.rolling_brier > max_live_brier:
        reasons.append(f"rolling_brier {state.rolling_brier:.4f} > {max_live_brier}")

    if backtest_roi is not None and backtest_roi < min_roi:
        bets = (backtest_roi_details or {}).get("bets", 0)
        reasons.append(
            f"backtest_roi {backtest_roi:.3f} < {min_roi} (holdout, n_bets={bets})"
        )

    return len(reasons) == 0, reasons


async def maybe_retrain_wc_weights(db=None) -> dict[str, Any]:
    """
    Re-fit Poisson/ELO cuando hay suficientes resultados WC 2026 nuevos.
    """
    from apps.shared.config import get_settings
    from apps.shared.supabase_client import get_supabase
    from apps.worker.ingest.worldcup_json import fetch_all_worldcup_archives

    settings = get_settings()
    state = load_learning_state()
    min_results = settings.model_retrain_min_wc_results

    if state.results_since_retrain < min_results:
        return {
            "status": "skipped",
            "reason": f"results_since_retrain {state.results_since_retrain} < {min_results}",
        }

    archives = await fetch_all_worldcup_archives()
    components = collect_1x2_components(archives, years=[2018, 2022, 2026])
    n_2026 = sum(1 for c in components if c.get("year") == 2026)
    if n_2026 < 4:
        return {"status": "skipped", "reason": f"wc2026_samples {n_2026} < 4"}

    result = fit_poisson_elo_weights(
        components,
        market_weight=settings.model_calibration_market_weight,
        train_years=[2018, 2022],
        test_years=[2026],
    )

    old = load_fitted_model_weights() or {}
    old_brier = float((old.get("test") or {}).get("brier_1x2", 999.0))
    new_brier = (
        result.test_report.brier_1x2
        if result.test_report
        else result.train_report.brier_1x2
    )

    improved = new_brier <= old_brier + 0.01
    if improved:
        save_fitted_model_weights(result)
        state.results_since_retrain = 0
        state.last_retrain_at = datetime.now(timezone.utc).isoformat()
        save_learning_state(state)

    summary = {
        "status": "retrained" if improved else "rejected",
        "old_brier": old_brier,
        "new_brier": new_brier,
        "weights": {
            "poisson": result.weights.poisson,
            "elo": result.weights.elo,
        },
        "wc2026_samples": n_2026,
        "underdog_dampen": result.underdog_dampen_factor,
    }

    if db is not None:
        try:
            db.schema("ops").table("job_runs").insert(
                {
                    "job_type": "wc_retrain_weights",
                    "status": "completed",
                    "metadata": summary,
                }
            ).execute()
        except Exception:
            pass

    return summary
