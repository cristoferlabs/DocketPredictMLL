"""EV anomaly detection and Kelly stake sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.shared.config import Settings, get_settings


@dataclass
class StakeRecommendation:
    kelly_full: float
    kelly_fractional: float
    stake_units: float
    allowed: bool
    flags: list[str]


def kelly_full(probability: float, odds: float) -> float:
    if odds <= 1.0 or probability <= 0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - probability
    kelly = (b * probability - q) / b
    return round(max(0.0, kelly), 4)


def fractional_kelly(
    probability: float,
    odds: float,
    kelly_fraction: float | None = None,
    max_stake: float = 0.25,
) -> float:
    settings = get_settings()
    frac = kelly_fraction if kelly_fraction is not None else settings.kelly_fraction
    full = kelly_full(probability, odds)
    return round(min(max_stake, full * frac), 4)


def check_ev_anomaly(
    *,
    edge_fair: float,
    ev_fair: float,
    model_prob: float,
    fair_implied: float,
    settings: Settings | None = None,
) -> tuple[bool, list[str]]:
    """
    Return (allowed, flags). Block suspiciously high EV/edge.
    """
    settings = settings or get_settings()
    flags: list[str] = []

    max_edge = getattr(settings, "ev_max_edge_fair", 0.12)
    max_ev = getattr(settings, "ev_max_fair", 0.15)
    max_divergence = getattr(settings, "ev_max_model_market_divergence", 0.20)

    if edge_fair > max_edge:
        flags.append(f"edge_fair>{max_edge:.0%}")
    if ev_fair > max_ev:
        flags.append(f"ev_fair>{max_ev:.0%}")
    if abs(model_prob - fair_implied) > max_divergence:
        flags.append(f"model_market_divergence>{max_divergence:.0%}")

    return len(flags) == 0, flags


def evaluate_pick(
    *,
    model_prob: float,
    fair_odds: float,
    edge_fair: float,
    ev_fair: float,
    fair_implied: float | None = None,
    settings: Settings | None = None,
) -> StakeRecommendation:
    settings = settings or get_settings()
    impl = fair_implied if fair_implied is not None else (1.0 / fair_odds if fair_odds > 1 else 0)
    allowed, flags = check_ev_anomaly(
        edge_fair=edge_fair,
        ev_fair=ev_fair,
        model_prob=model_prob,
        fair_implied=impl,
        settings=settings,
    )
    full = kelly_full(model_prob, fair_odds)
    frac = fractional_kelly(model_prob, fair_odds, settings.kelly_fraction)
    return StakeRecommendation(
        kelly_full=full,
        kelly_fractional=frac,
        stake_units=frac,
        allowed=allowed,
        flags=flags,
    )


def log_anomaly_to_db(db, context: str, flags: list[str], metadata: dict[str, Any]) -> None:
    """Log EV blocks separately from batch data audits (context=ev_anomaly)."""
    if not flags:
        return
    try:
        db.schema("ops").table("data_quality_log").insert(
            {
                "context": "ev_anomaly",
                "status": "ev_blocked",
                "completeness_pct": None,
                "flags": [
                    {
                        "level": "warning",
                        "code": f,
                        "message": f,
                        "match": context,
                        **metadata,
                    }
                    for f in flags
                ],
            }
        ).execute()
    except Exception:
        pass
