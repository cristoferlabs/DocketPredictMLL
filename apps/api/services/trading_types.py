"""Shared types for trading / decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingPick:
    market: str
    selection: str
    model_prob: float
    ev_fair: float = 0.0
    edge_fair: float = 0.0
    fair_odds: float = 0.0
    raw_odds: float = 0.0
    kelly_stake: float = 0.0
    from_ev: bool = False
