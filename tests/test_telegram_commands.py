"""Telegram command output shape — ENGINE v2, sin blend legacy."""

from apps.api.services.parlay_engine import format_parlay_message, ParlayBuildResult
from apps.api.services.engine_constants import ENGINE_VERSION_TAG
from apps.api.services.trading_card import format_trading_message
from tests.test_trust_arbitration import _czech_mexico
from apps.api.services.odds_context import compute_market_context
from apps.api.services.trading_card import build_trading_card
from apps.api.services.sharp_engine import run_sharp_engine
from apps.shared.config import get_settings


def test_trading_message_has_engine_v2_no_blend():
    analysis, model, odds = _czech_mexico()
    ctx = compute_market_context(model, analysis.team1, analysis.team2, odds)
    sharp = run_sharp_engine(analysis, market_ctx=ctx, settings=get_settings())
    card = build_trading_card(analysis, [], market_ctx=ctx)
    msg = format_trading_message(card, alta_header=False)
    assert ENGINE_VERSION_TAG in msg
    assert "Modelo ajustado" not in msg
    assert "50% modelo" not in msg
    assert "DISCREPANCIA EXTREMA" not in msg
    assert "Nivel 1 — MODEL" in msg
    if sharp.sharp_allowed:
        assert "PICK PRINCIPAL" in msg


def test_parlay_message_has_engine_v2():
    msg = format_parlay_message(ParlayBuildResult(eligible_picks=[], rejected_picks=[], tickets=[]))
    assert ENGINE_VERSION_TAG in msg
    assert "QUANT ENGINE" in msg
