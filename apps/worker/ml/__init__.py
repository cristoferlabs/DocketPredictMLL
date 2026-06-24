"""ML models package."""

from apps.worker.ml.elo import EloConfig, predict_match as elo_predict
from apps.worker.ml.ensemble import EnsembleResult, MatchInput, predict_match
from apps.worker.ml.gk import GoalkeeperProfile, apply_gk_adjustment
from apps.worker.ml.poisson import PoissonConfig, predict_match as poisson_predict
from apps.worker.ml.xgboost_model import XGBoostModel

__all__ = [
    "EloConfig",
    "elo_predict",
    "PoissonConfig",
    "poisson_predict",
    "GoalkeeperProfile",
    "apply_gk_adjustment",
    "XGBoostModel",
    "MatchInput",
    "EnsembleResult",
    "predict_match",
]
