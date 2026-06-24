"""Ensemble combining Poisson, ELO, GK and XGBoost into final predictions."""

from dataclasses import dataclass, field
from typing import Any

from apps.worker.ml.elo import EloConfig, EloProbabilities, predict_match as elo_predict
from apps.worker.ml.gk import GKAdjustment, GoalkeeperProfile, apply_gk_adjustment
from apps.worker.ml.poisson import (
    PoissonConfig,
    PoissonPrediction,
    estimate_lambdas,
    outcome_probabilities,
    predict_match as poisson_predict,
)
from apps.worker.ml.xgboost_model import LABEL_TO_OUTCOME, XGBoostModel

MARKET_LABELS_1X2 = list(LABEL_TO_OUTCOME.values())


@dataclass
class MatchInput:
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_xg: float | None = None
    away_xg: float | None = None
    home_goals_avg: float | None = None
    away_goals_avg: float | None = None
    home_gk: GoalkeeperProfile | None = None
    away_gk: GoalkeeperProfile | None = None
    extra_features: dict[str, float] = field(default_factory=dict)


@dataclass
class EnsembleWeights:
    elo: float = 0.25
    poisson: float = 0.35
    gk: float = 0.15
    xgboost: float = 0.25

    def normalized(self) -> "EnsembleWeights":
        total = self.elo + self.poisson + self.gk + self.xgboost
        if total <= 0:
            return EnsembleWeights(elo=0.25, poisson=0.35, gk=0.15, xgboost=0.25)
        return EnsembleWeights(
            elo=self.elo / total,
            poisson=self.poisson / total,
            gk=self.gk / total,
            xgboost=self.xgboost / total,
        )


@dataclass
class MarketPrediction:
    market_type: str
    predicted_outcome: str
    probability: float
    confidence_tier: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BettingCombination:
    priority: str
    expected_value: float | None
    kelly_fraction: float | None
    legs: list[dict[str, Any]]


@dataclass
class EnsembleResult:
    elo: EloProbabilities
    poisson: PoissonPrediction
    gk: GKAdjustment
    xgboost_probs: dict[str, float]
    predictions: list[MarketPrediction]
    combinations: list[BettingCombination]


def _confidence_tier(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def _blend_1x2(
    elo_p: EloProbabilities,
    poisson_p: dict[str, float],
    xgb_p: dict[str, float],
    weights: EnsembleWeights,
) -> dict[str, float]:
    w = weights.normalized()
    blended = {}
    for key in MARKET_LABELS_1X2:
        blended[key] = (
            w.elo * getattr(elo_p, key)
            + w.poisson * poisson_p.get(key, 0.33)
            + w.xgboost * xgb_p.get(key, 0.33)
        )
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return {k: round(v, 6) for k, v in blended.items()}


def _kelly(probability: float, odds: float) -> float:
    from apps.shared.config import get_settings
    from apps.worker.ml.ev_anomaly import fractional_kelly

    return fractional_kelly(probability, odds, get_settings().kelly_fraction)


def predict_match(
    match_input: MatchInput,
    xgb_model: XGBoostModel | None = None,
    weights: EnsembleWeights | None = None,
    odds: dict[str, float] | None = None,
) -> EnsembleResult:
    """Run full ensemble pipeline for a single match."""
    w = weights or EnsembleWeights()
    xgb_model = xgb_model or XGBoostModel()

    elo_result = elo_predict(match_input.home_elo, match_input.away_elo, EloConfig())

    lam_home, lam_away = estimate_lambdas(
        home_xg=match_input.home_xg,
        away_xg=match_input.away_xg,
        home_goals_avg=match_input.home_goals_avg,
        away_goals_avg=match_input.away_goals_avg,
    )

    gk_result = apply_gk_adjustment(
        lam_home, lam_away, match_input.home_gk, match_input.away_gk
    )

    poisson_result = poisson_predict(
        gk_result.home_xga_adjusted,
        gk_result.away_xga_adjusted,
        PoissonConfig(),
    )
    poisson_1x2 = outcome_probabilities(poisson_result.score_matrix)

    poisson_meta = {
        **poisson_1x2,
        "over_25": poisson_result.over_25,
        "under_25": poisson_result.under_25,
        "btts_yes": poisson_result.btts_yes,
        "btts_no": poisson_result.btts_no,
        "lambda_home": poisson_result.lambda_home,
        "lambda_away": poisson_result.lambda_away,
    }

    features = xgb_model.build_feature_vector(
        elo_probs={"home_win": elo_result.home_win, "draw": elo_result.draw, "away_win": elo_result.away_win},
        poisson_probs=poisson_meta,
        gk_adjustment={"home_factor": gk_result.home_factor, "away_factor": gk_result.away_factor},
        extra_features=match_input.extra_features,
    )
    xgb_probs = xgb_model.predict_proba(features)

    blended_1x2 = _blend_1x2(elo_result, poisson_1x2, xgb_probs, w)

    predictions: list[MarketPrediction] = []
    for outcome, prob in blended_1x2.items():
        predictions.append(
            MarketPrediction(
                market_type="1X2",
                predicted_outcome=outcome,
                probability=prob,
                confidence_tier=_confidence_tier(prob),
                metadata={"source": "ensemble"},
            )
        )

    for market, outcome, prob in [
        ("over_under_2.5", "over", poisson_result.over_25),
        ("over_under_2.5", "under", poisson_result.under_25),
        ("btts", "yes", poisson_result.btts_yes),
        ("btts", "no", poisson_result.btts_no),
    ]:
        predictions.append(
            MarketPrediction(
                market_type=market,
                predicted_outcome=outcome,
                probability=prob,
                confidence_tier=_confidence_tier(prob),
                metadata={"source": "poisson"},
            )
        )

    combinations: list[BettingCombination] = []
    best_1x2 = max(blended_1x2.items(), key=lambda x: x[1])
    main_odds = (odds or {}).get(best_1x2[0], 2.0)
    ev = round(best_1x2[1] * main_odds - 1.0, 4)
    kelly = _kelly(best_1x2[1], main_odds)

    combinations.append(
        BettingCombination(
            priority="high" if best_1x2[1] >= 0.55 else "medium",
            expected_value=ev,
            kelly_fraction=kelly,
            legs=[
                {
                    "market_type": "1X2",
                    "selection": best_1x2[0],
                    "odds": main_odds,
                    "probability": best_1x2[1],
                }
            ],
        )
    )

    if poisson_result.over_25 >= 0.58:
        ou_odds = (odds or {}).get("over_25", 1.85)
        combinations.append(
            BettingCombination(
                priority="medium",
                expected_value=round(poisson_result.over_25 * ou_odds - 1.0, 4),
                kelly_fraction=_kelly(poisson_result.over_25, ou_odds),
                legs=[
                    {
                        "market_type": "over_under_2.5",
                        "selection": "over",
                        "odds": ou_odds,
                        "probability": poisson_result.over_25,
                    }
                ],
            )
        )

    return EnsembleResult(
        elo=elo_result,
        poisson=poisson_result,
        gk=gk_result,
        xgboost_probs=xgb_probs,
        predictions=predictions,
        combinations=combinations,
    )
