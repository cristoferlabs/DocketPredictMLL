"""Tests for EV policy — raw informative, fair decisive."""

from apps.api.services.ev_policy import ev_for_decision, format_ev_display


def test_ev_for_decision_uses_fair():
    assert ev_for_decision(ev_fair=0.05, ev_raw=0.12) == 0.05


def test_format_ev_display_shows_both_when_divergent():
    text = format_ev_display(ev_fair_pct=4.2, ev_raw_pct=11.5)
    assert "fair" in text
    assert "raw" in text
    assert "11.5" in text


def test_format_ev_display_fair_only_when_close():
    text = format_ev_display(ev_fair_pct=4.0, ev_raw_pct=4.2)
    assert "raw" not in text
