"""XGBoost meta-learner wrapper (train/predict interfaces)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore


LABEL_TO_OUTCOME = {0: "home_win", 1: "draw", 2: "away_win"}
OUTCOME_TO_LABEL = {"home_win": 0, "draw": 1, "away_win": 2}


@dataclass
class XGBoostConfig:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass
class XGBoostModel:
    config: XGBoostConfig = field(default_factory=XGBoostConfig)
    model: Any = None
    feature_names: list[str] = field(default_factory=list)
    is_trained: bool = False

    def build_feature_vector(
        self,
        elo_probs: dict[str, float],
        poisson_probs: dict[str, float],
        gk_adjustment: dict[str, float],
        extra_features: dict[str, float] | None = None,
    ) -> np.ndarray:
        base = {
            "elo_home": elo_probs.get("home_win", 0.33),
            "elo_draw": elo_probs.get("draw", 0.33),
            "elo_away": elo_probs.get("away_win", 0.33),
            "poisson_home": poisson_probs.get("home_win", 0.33),
            "poisson_draw": poisson_probs.get("draw", 0.33),
            "poisson_away": poisson_probs.get("away_win", 0.33),
            "poisson_over25": poisson_probs.get("over_25", 0.5),
            "poisson_btts": poisson_probs.get("btts_yes", 0.5),
            "gk_home_factor": gk_adjustment.get("home_factor", 1.0),
            "gk_away_factor": gk_adjustment.get("away_factor", 1.0),
            "lambda_home": poisson_probs.get("lambda_home", 1.3),
            "lambda_away": poisson_probs.get("lambda_away", 1.1),
        }
        if extra_features:
            base.update(extra_features)
        if not self.feature_names:
            self.feature_names = sorted(base.keys())
        return np.array([base.get(k, 0.0) for k in self.feature_names], dtype=np.float32)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        self.model = xgb.XGBClassifier(**self.config.__dict__)
        self.model.fit(X, y)
        self.is_trained = True
        acc = float((self.model.predict(X) == y).mean()) if len(y) > 0 else 0.0
        return {
            "train_accuracy": acc,
            "n_samples": len(y),
            "feature_names": self.feature_names,
        }

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        if not self.is_trained or self.model is None:
            return {label: round(1 / 3, 6) for label in LABEL_TO_OUTCOME.values()}

        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        classes = list(self.model.classes_)
        result: dict[str, float] = {label: round(1 / 3, 6) for label in LABEL_TO_OUTCOME.values()}
        for i, cls in enumerate(classes):
            key = LABEL_TO_OUTCOME.get(int(cls), str(cls))
            result[key] = round(float(proba[i]), 6)
        return result

    def save(self, path: Path) -> None:
        if self.model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        if xgb is None or not path.exists():
            return
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        self.is_trained = True
