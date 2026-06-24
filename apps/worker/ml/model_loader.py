"""Load ensemble weights and XGBoost artifacts from Supabase / disk."""

from __future__ import annotations

import logging
from pathlib import Path

from apps.worker.ml.ensemble import EnsembleWeights
from apps.worker.ml.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/xgboost")


def load_ensemble_weights(db, league_id: str) -> EnsembleWeights:
    """Read per-league weights from ml.model_weights."""
    weights = EnsembleWeights()
    try:
        rows = (
            db.schema("ml")
            .table("model_weights")
            .select("model_type, weight")
            .eq("league_id", league_id)
            .eq("market_type", "1X2")
            .execute()
        )
        mapping = {r["model_type"]: float(r["weight"]) for r in rows.data or []}
        if mapping:
            weights = EnsembleWeights(
                elo=mapping.get("elo", 0.25),
                poisson=mapping.get("poisson", 0.35),
                gk=mapping.get("gk", 0.15),
                xgboost=mapping.get("xgboost", 0.25),
            )
    except Exception as exc:
        logger.warning("load_ensemble_weights: %s", exc)
    return weights.normalized()


def load_active_xgboost(db) -> XGBoostModel:
    """Load trained XGBoost from active model_versions artifact_path."""
    model = XGBoostModel()
    try:
        row = (
            db.schema("ml")
            .table("model_versions")
            .select("artifact_path, metrics")
            .eq("model_type", "xgboost")
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        if not row.data:
            return model
        artifact = row.data[0].get("artifact_path")
        metrics = row.data[0].get("metrics") or {}
        if metrics.get("feature_names"):
            model.feature_names = list(metrics["feature_names"])
        if artifact:
            path = Path(artifact)
            model.load(path)
    except Exception as exc:
        logger.warning("load_active_xgboost: %s", exc)
    return model


def load_calibration_factors_from_db(db) -> dict[str, dict[str, float]] | None:
    """Load active calibration factors for worldcup engine."""
    from apps.worker.ml.calibration import DEFAULT_CALIBRATION_FACTORS

    try:
        rows = (
            db.schema("ml")
            .table("calibration_factors")
            .select("market, outcome, factor")
            .eq("competition", "fifa_world_cup")
            .eq("is_active", True)
            .execute()
        )
        if not rows.data:
            return None
        factors: dict[str, dict[str, float]] = {
            "1X2": {},
            "over_under_2.5": {},
            "btts": {},
        }
        market_map = {
            "1X2": "1X2",
            "over_under_2.5": "over_under_2.5",
            "btts": "btts",
        }
        for r in rows.data:
            market = r["market"]
            if market in market_map:
                factors[market][r["outcome"]] = float(r["factor"])
        for grp, defaults in DEFAULT_CALIBRATION_FACTORS.items():
            for k, v in defaults.items():
                factors[grp].setdefault(k, v)
        return factors
    except Exception as exc:
        logger.warning("load_calibration_factors: %s", exc)
        return None
