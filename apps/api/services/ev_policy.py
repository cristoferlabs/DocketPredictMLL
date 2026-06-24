"""
Política EV — una sola fuente de verdad.

  Raw EV  → informativo (cuota bruta con vig).
  Fair EV → decisión (cuota fair devig, gates SHARP/árbol).
"""

from __future__ import annotations

from typing import Literal

EvKind = Literal["raw", "fair"]


def ev_for_decision(*, ev_fair: float, ev_raw: float | None = None) -> float:
    """EV usado en gates, stake y árbol de decisión."""
    return ev_fair


def edge_for_decision(*, edge_fair: float, edge_raw: float | None = None) -> float:
    """Edge usado en gates y ranking de picks."""
    return edge_fair


def format_ev_display(*, ev_fair_pct: float, ev_raw_pct: float | None = None) -> str:
    """Texto Telegram: fair primero (decisión), raw como referencia."""
    fair = f"EV fair {ev_fair_pct:+.1f}%"
    if ev_raw_pct is not None and abs(ev_raw_pct - ev_fair_pct) > 0.5:
        return f"{fair} (raw {ev_raw_pct:+.1f}% ref.)"
    return fair
