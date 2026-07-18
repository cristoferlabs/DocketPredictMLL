"""Tests Telegram Betting Terminal v2 — capa UI."""

from apps.api.services.telegram_terminal.formatters import (
    build_ranked_picks,
    format_exploration,
    format_match_dashboard,
    format_opportunities,
)
from apps.api.services.telegram_terminal.keyboards import exploration_keyboard
from apps.api.services.telegram_terminal.states import TerminalState
from apps.api.services.telegram_terminal.terminal import BettingTerminal
from tests.test_trust_arbitration import _czech_mexico
from apps.api.services.odds_context import compute_market_context
from apps.api.services.sharp_engine import run_sharp_engine
from apps.shared.config import get_settings


def test_exploration_has_no_probabilities():
    matches = [
        {"team1": "Morocco", "team2": "Haiti", "fecha": "2026-06-24"},
        {"team1": "Scotland", "team2": "Brazil", "fecha": "2026-06-24"},
    ]
    text = format_exploration(matches)
    assert "PARTIDOS DE HOY" in text
    assert "Morocco vs Haiti" in text
    assert "%" not in text
    assert "EV" not in text


def test_exploration_keyboard_uses_terminal_callbacks():
    matches = [{"team1": "A", "team2": "B", "fecha": "2026-06-01"}]
    kb = exploration_keyboard(matches)
    assert kb["inline_keyboard"][0][0]["callback_data"] == "t:m:0"


def test_dashboard_no_ev_no_stake():
    analysis, model, odds = _czech_mexico()
    ctx = compute_market_context(model, analysis.team1, analysis.team2, odds)
    text = format_match_dashboard(analysis, ctx)
    assert "MODEL" in text
    assert "MARKET CONTEXT" in text
    assert "NO_BET" not in text
    assert "Stake" not in text
    assert "EV:" not in text


def test_opportunities_shows_all_outcomes_with_odds():
    analysis, model, odds = _czech_mexico()
    ctx = compute_market_context(model, analysis.team1, analysis.team2, odds)
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    picks = build_ranked_picks(analysis, [], sharp, ctx)
    text = format_opportunities(analysis, picks)
    # Picks always display model prob and odds (real market @ or fair [f])
    assert "model" in text
    assert "@" in text or "[f" in text or "sin cuota" in text
    assert len(picks) >= 3


def test_terminal_callback_detection():
    assert BettingTerminal.is_terminal_callback("t:m:0")
    assert BettingTerminal.is_terminal_callback("t:o")
    assert not BettingTerminal.is_terminal_callback("wc:A|B")


def test_terminal_states_enum():
    assert TerminalState.EXPLORATION.value == "EXPLORATION"
    assert TerminalState.PARLAY_VIEW.value == "PARLAY_VIEW"
