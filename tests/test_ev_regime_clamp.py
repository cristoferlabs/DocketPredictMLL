"""Tests EV clamp estructural por régimen α."""

from apps.api.services.ev_policy import (
    clamp_ev_by_regime,
    ev_for_decision,
    regime_ev_cap,
)
from apps.shared.config import Settings


def test_regime_ev_caps():
    s = Settings()
    assert regime_ev_cap("aligned", settings=s) == 0.10
    assert regime_ev_cap("moderate", settings=s) == 0.12
    assert regime_ev_cap("high", settings=s) == 0.15
    assert regime_ev_cap("extreme", settings=s) == 0.18


def test_ev_for_decision_clamps_extreme():
    s = Settings()
    raw = 0.35
    capped = ev_for_decision(ev_fair=raw, alpha_regime="extreme", settings=s)
    assert capped == 0.18
    assert capped < raw


def test_ev_for_decision_aligned_passes_low_ev():
    s = Settings()
    assert ev_for_decision(ev_fair=0.08, alpha_regime="aligned", settings=s) == 0.08


def test_clamp_ev_by_regime_flag():
    capped, was = clamp_ev_by_regime(0.25, "moderate")
    assert was is True
    assert capped == 0.12
