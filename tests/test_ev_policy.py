"""Tests EV policy — fórmula canónica y gates de display."""

from apps.api.services.ev_policy import (
    ev_decimal,
    ev_percent,
    format_ev_display,
    is_actionable_value,
    is_structural_mismatch,
)


def test_ev_canonical_formula():
    # Scotland ~23.5% @ 9.50
    ev = ev_decimal(0.235, 9.50)
    assert abs(ev - (0.235 * 9.50 - 1.0)) < 1e-6
    assert ev_percent(0.235, 9.50) == round(ev * 100, 2)


def test_structural_mismatch_scotland_brazil():
    assert is_structural_mismatch(0.235, 1 / 9.50)
    assert not is_actionable_value(
        ev_fair_pct=120.0,
        model_prob=0.235,
        market_implied=1 / 9.50,
    )


def test_format_ev_requires_odds_context():
    text = format_ev_display(
        ev_fair_pct=5.2,
        ev_raw_pct=8.0,
        odds_decimal=2.10,
        model_prob=0.52,
        market_implied=0.48,
    )
    assert "@ 2.10" in text
    assert "EV fair" in text


def test_format_ev_flags_structural():
    text = format_ev_display(
        ev_fair_pct=80.0,
        ev_raw_pct=160.0,
        odds_decimal=9.50,
        model_prob=0.235,
        market_implied=1 / 9.50,
        market="1X2",
    )
    assert "estructural" in text or "INVESTIGATE" in text or "NO BET" in text


def test_format_ev_shows_true_value_when_capped():
    text = format_ev_display(
        ev_fair_pct=176.8,
        ev_raw_pct=200.0,
        odds_decimal=6.15,
        model_prob=0.45,
        market_implied=0.163,
        market="1X2",
    )
    assert "EV calc +176.8%" in text
    assert "tope visual" in text
    assert "INVESTIGATE" in text or "estructural" in text
