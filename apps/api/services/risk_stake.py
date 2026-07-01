"""Risk / Stake Layer — asignación de bankroll para SHARP y PARLAY.

Integra:
- Kelly fraccional (diferenciado por edge)
- EV shrinkage (ev_final = shrunk EV)
- Drawdown adjustment (reduce stake en drawdown)
- Portfolio exposure cap (total concurrente)
- Variance posture (underdogs seek variance, favorites minimize)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any

from apps.api.services.bet_decision_tree import BetDecisionResult
from apps.shared.config import get_settings
from apps.worker.ml.ev_anomaly import fractional_kelly

# ── Portfolio state (persistent across calls) ──────────────────────────

_PORTFOLIO_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "artifacts", "portfolio_state.json"
)


@dataclass
class PortfolioState:
    current_drawdown_pct: float = 0.0
    total_exposure_pct: float = 0.0
    n_active_bets: int = 0
    staking_rolling_clv: float | None = None  # CLV vs closing line de apuestas reales
    meta: dict[str, Any] = field(default_factory=dict)


def _load_portfolio() -> PortfolioState:
    try:
        with open(_PORTFOLIO_FILE) as f:
            raw = json.load(f)
        # Migrate: old field name was "rolling_clv"
        if "rolling_clv" in raw and "staking_rolling_clv" not in raw:
            raw["staking_rolling_clv"] = raw.pop("rolling_clv")
        elif "rolling_clv" in raw:
            del raw["rolling_clv"]
        known = {fld.name for fld in dc_fields(PortfolioState)}
        return PortfolioState(**{k: v for k, v in raw.items() if k in known})
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return PortfolioState()


def _save_portfolio(state: PortfolioState) -> None:
    os.makedirs(os.path.dirname(_PORTFOLIO_FILE), exist_ok=True)
    with open(_PORTFOLIO_FILE, "w") as f:
        json.dump(
            {
                "current_drawdown_pct": state.current_drawdown_pct,
                "total_exposure_pct": state.total_exposure_pct,
                "n_active_bets": state.n_active_bets,
                "staking_rolling_clv": state.staking_rolling_clv,
                "meta": state.meta,
            },
            f,
        )


# ── Drawdown factor ────────────────────────────────────────────────────


def _drawdown_adjustment(drawdown_pct: float) -> float:
    """Factor multiplicativo por drawdown: 0-20% → 1.0-0.5 lineal.

    drawdown 0%  → factor 1.0 (sin ajuste)
    drawdown 10% → factor 0.75
    drawdown 20%+ → factor 0.50 (stake reducido a la mitad)
    """
    if drawdown_pct <= 0:
        return 1.0
    adj = max(0.50, 1.0 - drawdown_pct / 0.40)
    return round(adj, 4)


# ── Variance posture ───────────────────────────────────────────────────


def _variance_posture(p_win: float) -> float:
    """Multiplicador de sensibilidad de stake según posture de varianza.

    Underdog (p < 40%): seek variance → boost EV sensitivity
    Neutral  (40-60%): 1.0
    Favorite (p > 60%): minimize variance → dampen EV sensitivity
    """
    if p_win < 0.40:
        # Buscar varianza: boost sensibilidad hasta +20%
        return 1.0 + (0.40 - p_win) * 0.50  # p=30% → 1.05, p=20% → 1.10
    if p_win > 0.60:
        # Minimizar varianza: reducir sensibilidad hasta -20%
        return 1.0 - (p_win - 0.60) * 0.50  # p=70% → 0.95, p=80% → 0.90
    return 1.0


# ── Main allocation ────────────────────────────────────────────────────


def allocate_sharp_stake(
    decision: BetDecisionResult,
    *,
    ev_final: float,
    mds: int,
    confidence_norm: float,
) -> float:
    """Stake % bankroll para single SHARP — proporcional al EV shrink.

    stake ∝ Kelly_fraccional × EV_sensitivity × MDS × conf × drawdown
    El hard cap del 2% es solo el techo absoluto.
    """
    settings = get_settings()
    pick = decision.pick
    portfolio = _load_portfolio()

    # 1. Compute base Kelly from pick's probability and fair odds
    if pick and pick.fair_odds > 1 and pick.model_prob > 0:
        raw_kelly = fractional_kelly(
            probability=pick.model_prob,
            odds=pick.fair_odds,
        )
    else:
        raw_kelly = 0.0

    # 2. If no Kelly available, derive from ev_final
    if raw_kelly <= 0:
        raw_kelly = ev_final * settings.kelly_fraction * 2

    # 3. EV sensitivity: stake ∝ shrunk EV, normalized at 8%
    ev_sensitivity = min(1.0, ev_final / 0.08)

    # 4. Variance posture: underdogs seek variance, favorites minimize
    var_posture = _variance_posture(pick.model_prob if pick else 0.50)
    ev_sensitivity *= var_posture

    # 5. Quality factors
    mds_factor = min(1.0, mds / 85.0)
    conf_factor = min(1.0, confidence_norm / 0.85)

    # 6. Drawdown adjustment
    dd_adj = _drawdown_adjustment(portfolio.current_drawdown_pct)

    # 7. Portfolio exposure cap: reduce if already heavily exposed
    exposure_factor = 1.0
    max_exposure = getattr(settings, "portfolio_max_exposure_pct", 5.0)
    if portfolio.total_exposure_pct >= max_exposure:
        # Escalar lineal: 5% exp → factor 1.0, 10% exp → factor 0.5
        exposure_factor = max(0.25, 1.0 - (portfolio.total_exposure_pct - max_exposure) / max_exposure)

    # 8. Scale: raw_kelly is decimal (0-1), convert to percentage
    stake_pct = (
        raw_kelly
        * 100
        * ev_sensitivity
        * mds_factor
        * conf_factor
        * dd_adj
        * exposure_factor
    )

    # 9. Final safety cap: configurable via sharp_max_stake_pct
    return round(max(0.0, min(settings.sharp_max_stake_pct, stake_pct)), 2)


def update_portfolio_after_bet(
    *,
    stake_pct: float,
    result_pnl: float | None = None,
    staking_clv: float | None = None,
) -> None:
    """Actualiza estado de portafolio.

    Llamar DOS veces por apuesta:
    1. Al COLOCAR (result_pnl=None): registra exposición activa (+stake, +n_bets).
    2. Al RESOLVER (result_pnl=float): libera exposición (-stake, -n_bets) y
       actualiza drawdown y CLV rolling.
    """
    portfolio = _load_portfolio()

    if result_pnl is None:
        # Bet placed — track active exposure
        portfolio.total_exposure_pct = min(100.0, portfolio.total_exposure_pct + stake_pct)
        portfolio.n_active_bets += 1
    else:
        # Bet resolved — release exposure, update drawdown
        portfolio.total_exposure_pct = max(0.0, portfolio.total_exposure_pct - stake_pct)
        portfolio.n_active_bets = max(0, portfolio.n_active_bets - 1)
        if result_pnl < 0:
            portfolio.current_drawdown_pct = min(
                1.0, portfolio.current_drawdown_pct + abs(result_pnl) * 0.5
            )
        else:
            portfolio.current_drawdown_pct = max(
                0.0, portfolio.current_drawdown_pct - result_pnl * 0.3
            )

    if staking_clv is not None:
        if portfolio.staking_rolling_clv is None:
            portfolio.staking_rolling_clv = round(staking_clv, 4)
        else:
            # Exponential moving average (alpha=0.1 → ≈últimas 10 apuestas)
            portfolio.staking_rolling_clv = round(
                0.9 * portfolio.staking_rolling_clv + 0.1 * staking_clv, 4
            )

    _save_portfolio(portfolio)


def allocate_parlay_stake(
    *,
    combined_prob: float,
    ev_parlay: float,
    combo_score: float,
    n_legs: int,
    correlation_penalty: float,
) -> float:
    """Stake % bankroll para combinada (menor que singles)."""
    settings = get_settings()
    if ev_parlay <= 0 or combined_prob <= 0:
        return 0.0
    base = settings.parlay_base_stake_pct
    leg_factor = 0.85 ** max(0, n_legs - 3)
    score_boost = min(1.3, 1.0 + combo_score * 2)
    stake = base * leg_factor * correlation_penalty * score_boost
    if ev_parlay >= 0.15:
        stake *= 1.1
    cap = min(settings.parlay_max_stake_pct, 2.0)  # hard cap: nunca > 2%
    return round(max(0.05, min(cap, stake)), 2)
