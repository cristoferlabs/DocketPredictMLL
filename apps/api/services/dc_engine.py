"""
DC (Doble Oportunidad) evaluation engine.

X2 = Empate + Visitante  (dc_away_draw)
1X = Local + Empate      (dc_home_draw)
12 = Local + Visitante   (dc_home_away)

DC evaluation is intentionally simpler than the 1X2 SHARP engine:
- No full pipeline invariant checks (DC is derivative of h2h fair odds)
- Thresholds tuned for the high-prob, low-odds nature of DC
- Kelly(25%) stake with 2% bankroll cap
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from apps.api.services.odds_context import EvOpportunity
from apps.api.services.worldcup_engine import ModelMarkets

logger = logging.getLogger(__name__)

DC_MARKET = "Doble Oportunidad"

_KELLY_FRACTION = 0.25
_MAX_STAKE_PCT = 2.0

# (short_label, human_template, model_field)
_DC_SLOTS: list[tuple[str, str, str]] = [
    ("X2", "X2 (Empate o {away})",          "dc_away_draw"),
    ("1X", "1X ({home} o Empate)",           "dc_home_draw"),
    ("12", "12 ({home} o {away})",           "dc_home_away"),
]

# Map selection strings in ev_opps back to slot keys
_EV_SLOT_MAP: dict[str, str] = {
    "away_draw": "X2",
    "home_draw": "1X",
    "home_away": "12",
}


@dataclass
class DCPick:
    short_label: str        # "X2", "1X", "12"
    label: str              # "X2 (Empate o Sweden)"
    model_prob: float       # calibrated model prob (sum of 2 outcomes)
    fair_odds: float        # devigged odds (derived from h2h) or model fair
    market_odds: float      # raw book odds from Odds API (0 = no direct quote)
    ev_pct: float           # EV% vs fair odds
    stake_pct: float        # 25%-Kelly, capped at 2.0%
    decision: str           # "STRONG_BET" | "MODERATE_BET" | "WEAK_BET" | "NO_BET"
    risk: str               # "MUY BAJO" | "BAJO" | "MEDIO" | "ALTO"
    risk_emoji: str
    is_primary: bool        # highest value pick


# ─── Decision helpers ─────────────────────────────────────────────────────────

def _risk(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "MUY BAJO", "🟢"
    if prob >= 0.55:
        return "BAJO", "🟡"
    if prob >= 0.45:
        return "MEDIO", "🟠"
    return "ALTO", "🔴"


def _decision(prob: float, ev_pct: float) -> str:
    if prob >= 0.55 and ev_pct >= 3.0:
        return "STRONG_BET"
    if prob >= 0.50 and ev_pct >= 1.5:
        return "MODERATE_BET"
    if prob >= 0.45 and ev_pct >= 0.0:
        return "WEAK_BET"
    return "NO_BET"


def _kelly_stake(model_prob: float, fair_odds: float) -> float:
    """25% fractional Kelly, capped at _MAX_STAKE_PCT% bankroll."""
    if fair_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b = fair_odds - 1.0
    kelly = (model_prob * b - (1.0 - model_prob)) / b
    if kelly <= 0:
        return 0.0
    return round(min(kelly * _KELLY_FRACTION * 100, _MAX_STAKE_PCT), 1)


# ─── Main evaluator ───────────────────────────────────────────────────────────

def evaluate_dc(
    model: ModelMarkets,
    ev_opps: list[EvOpportunity],
    team1: str,
    team2: str,
) -> list[DCPick]:
    """
    Evaluate X2, 1X, 12 as standalone DC picks.

    Returns up to 3 picks sorted by model_prob descending.
    EV uses the fair (devigged) odds from compute_ev_opportunities() when
    available; otherwise falls back to model-implied fair odds (EV = 0).
    The `is_primary` flag marks the highest-value actionable pick.
    """
    # Index ev_opps for DC market by slot short label
    dc_opps: dict[str, EvOpportunity] = {}
    for o in ev_opps:
        if o.market != DC_MARKET:
            continue
        # Match by the selection string fragments
        if f"1X ({team1}" in o.selection or "home_draw" in o.selection:
            dc_opps["1X"] = o
        elif f"X2 (Empate/{team2}" in o.selection or "away_draw" in o.selection:
            dc_opps["X2"] = o
        elif f"12 ({team1}/{team2}" in o.selection or "home_away" in o.selection:
            dc_opps["12"] = o

    picks: list[DCPick] = []
    for short, label_tmpl, field_name in _DC_SLOTS:
        prob = getattr(model, field_name, 0.0)
        if prob <= 0:
            continue

        label = label_tmpl.format(home=team1, away=team2)
        opp = dc_opps.get(short)

        if opp:
            fair_o = opp.fair_odds or round(1.0 / prob, 2)
            raw_o = opp.raw_odds or 0.0
            ev_pct = round(opp.expected_value * 100, 1)
        else:
            fair_o = round(1.0 / prob, 2)
            raw_o = 0.0
            ev_pct = 0.0   # no market data → EV undefined

        stake = _kelly_stake(prob, fair_o)
        dec = _decision(prob, ev_pct)
        risk, risk_emoji = _risk(prob)

        picks.append(DCPick(
            short_label=short,
            label=label,
            model_prob=round(prob, 4),
            fair_odds=fair_o,
            market_odds=raw_o,
            ev_pct=ev_pct,
            stake_pct=stake,
            decision=dec,
            risk=risk,
            risk_emoji=risk_emoji,
            is_primary=False,
        ))

    if not picks:
        return []

    # Mark primary: best STRONG/MODERATE by EV×prob; fallback → highest prob
    actionable = [p for p in picks if p.decision in ("STRONG_BET", "MODERATE_BET")]
    primary_key = (
        max(actionable, key=lambda p: p.ev_pct * p.model_prob)
        if actionable
        else max(picks, key=lambda p: p.model_prob)
    )
    picks = [
        dataclasses.replace(p, is_primary=(p.short_label == primary_key.short_label))
        for p in picks
    ]

    # Sort: X2 first (highest prob in most matches), then 1X, then 12
    order = {"X2": 0, "1X": 1, "12": 2}
    picks.sort(key=lambda p: order.get(p.short_label, 9))
    return picks


def best_dc_pick(picks: list[DCPick]) -> DCPick | None:
    """Return the primary pick or None if no DC picks available."""
    for p in picks:
        if p.is_primary:
            return p
    return picks[0] if picks else None
