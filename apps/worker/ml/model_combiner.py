"""
Model combination — Poisson + ELO (+ mercado solo en capa calibración).

Invariante decisión:
  - EV / SHARP / edge usan probabilidades SIN mezcla de mercado en vivo.
  - El peso de mercado (p.ej. 20%) aplica solo en calibration_layer=True
    y se expone aparte como anchored_probs (telemetría / display).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

OUTCOMES_1X2 = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class Probabilities1X2:
    home_win: float
    draw: float
    away_win: float

    def as_dict(self) -> dict[str, float]:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
        }

    @classmethod
    def from_mapping(cls, probs: Mapping[str, float]) -> Probabilities1X2:
        return cls(
            home_win=float(probs.get("home_win", 0.0)),
            draw=float(probs.get("draw", 0.0)),
            away_win=float(probs.get("away_win", 0.0)),
        )


@dataclass
class ModelCombinationWeights:
    """Pesos ENGINE v3 — Poisson / ELO / mercado (calibración)."""

    poisson: float = 0.5
    elo: float = 0.3
    market: float = 0.2

    def normalized(self, *, include_market: bool = False) -> tuple[float, float, float]:
        """Devuelve (w_poisson, w_elo, w_market) sumando 1."""
        p, e, m = max(0.0, self.poisson), max(0.0, self.elo), max(0.0, self.market)
        if not include_market:
            m = 0.0
        total = p + e + m
        if total <= 0:
            return 0.5, 0.5, 0.0
        return p / total, e / total, m / total

    @classmethod
    def from_settings(cls) -> ModelCombinationWeights:
        from apps.shared.config import get_settings
        from apps.worker.ml.calibration_metrics import load_fitted_model_weights

        fitted = load_fitted_model_weights()
        if fitted and fitted.get("poisson") is not None:
            return cls(
                poisson=float(fitted["poisson"]),
                elo=float(fitted.get("elo", 0.3)),
                market=float(fitted.get("market", 0.2)),
            )

        s = get_settings()
        return cls(
            poisson=s.model_weight_poisson,
            elo=s.model_weight_elo,
            market=s.model_calibration_market_weight,
        )


@dataclass
class CombinationResult:
    """Resultado del combiner con trazabilidad de fuentes."""

    decision: Probabilities1X2
    poisson: Probabilities1X2
    elo: Probabilities1X2
    blended_statistical: Probabilities1X2
    weights: ModelCombinationWeights
    weights_applied: dict[str, float] = field(default_factory=dict)
    market: Probabilities1X2 | None = None
    anchored: Probabilities1X2 | None = None
    blend_applied: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def blend_meta(self) -> dict[str, Any]:
        """Metadatos serializables para ModelMarkets / Telegram."""
        out: dict[str, Any] = {
            "engine": "model_combiner_v1",
            "weights_config": {
                "poisson": self.weights.poisson,
                "elo": self.weights.elo,
                "market": self.weights.market,
            },
            "weights_applied": self.weights_applied,
            "blend_applied": self.blend_applied,
            "poisson": self.poisson.as_dict(),
            "elo": self.elo.as_dict(),
            "blended_statistical": self.blended_statistical.as_dict(),
            "decision": self.decision.as_dict(),
        }
        if self.market is not None:
            out["market_fair"] = self.market.as_dict()
        if self.anchored is not None:
            out["anchored"] = self.anchored.as_dict()
        out.update(self.meta)
        return out


def _normalize(probs: Probabilities1X2) -> Probabilities1X2:
    total = probs.home_win + probs.draw + probs.away_win
    if total <= 0:
        return Probabilities1X2(home_win=1 / 3, draw=1 / 3, away_win=1 / 3)
    return Probabilities1X2(
        home_win=probs.home_win / total,
        draw=probs.draw / total,
        away_win=probs.away_win / total,
    )


def _weighted_blend(
    components: list[tuple[float, Probabilities1X2]],
) -> Probabilities1X2:
    home = draw = away = 0.0
    for weight, probs in components:
        if weight <= 0:
            continue
        home += weight * probs.home_win
        draw += weight * probs.draw
        away += weight * probs.away_win
    return _normalize(Probabilities1X2(home_win=home, draw=draw, away_win=away))


def combine_poisson_elo(
    poisson: Mapping[str, float],
    elo: Mapping[str, float],
    *,
    weights: ModelCombinationWeights | None = None,
) -> tuple[Probabilities1X2, dict[str, float]]:
    """Blend Poisson + ELO — base del modelo (sin mercado)."""
    w = weights or ModelCombinationWeights.from_settings()
    wp, we, _ = w.normalized(include_market=False)
    p = Probabilities1X2.from_mapping(poisson)
    e = Probabilities1X2.from_mapping(elo)
    blended = _weighted_blend([(wp, p), (we, e)])
    return blended, {"poisson": wp, "elo": we, "market": 0.0}


def apply_market_calibration_layer(
    model_probs: Probabilities1X2,
    market_fair: Mapping[str, float],
    *,
    weights: ModelCombinationWeights | None = None,
    alpha: float | None = None,
) -> Probabilities1X2:
    """
    Ancla post-modelo hacia mercado deviggeado.

    alpha explícito (live_calibration) tiene prioridad sobre w.market legacy.
    """
    wm = alpha
    if wm is None:
        w = weights or ModelCombinationWeights.from_settings()
        wm = w.market
    if wm <= 0:
        return model_probs
    wm_model = 1.0 - wm
    m = Probabilities1X2.from_mapping(market_fair)
    return _weighted_blend([(wm_model, model_probs), (wm, m)])


def combine_1x2(
    poisson: Mapping[str, float],
    elo: Mapping[str, float],
    *,
    weights: ModelCombinationWeights | None = None,
    market_fair: Mapping[str, float] | None = None,
    calibration_layer: bool = False,
) -> CombinationResult:
    """
    Pipeline completo de combinación 1X2.

    calibration_layer=False (default):
      decision = blend(Poisson, ELO)
    calibration_layer=True y market_fair presente:
      anchored = blend(decision, market) con w_market
      decision sigue siendo solo Poisson+ELO (invariante EV)
    """
    w = weights or ModelCombinationWeights.from_settings()
    p = Probabilities1X2.from_mapping(poisson)
    e = Probabilities1X2.from_mapping(elo)
    statistical, applied = combine_poisson_elo(poisson, elo, weights=w)

    market_probs: Probabilities1X2 | None = None
    anchored: Probabilities1X2 | None = None
    blend_applied = False

    if calibration_layer and market_fair is not None:
        market_probs = Probabilities1X2.from_mapping(market_fair)
        if w.market > 0:
            anchored = apply_market_calibration_layer(
                statistical, market_fair, weights=w
            )
            blend_applied = True
            applied = {
                **applied,
                "market": w.normalized(include_market=True)[2],
            }

    return CombinationResult(
        decision=statistical,
        poisson=p,
        elo=e,
        blended_statistical=statistical,
        weights=w,
        weights_applied=applied,
        market=market_probs,
        anchored=anchored,
        blend_applied=blend_applied,
        meta={"calibration_layer": calibration_layer},
    )
