"""Tests for Telegram update deduplication."""

from apps.api.services.telegram_dedup import claim_update


def test_claim_update_first_wins():
    uid = 9_999_999_001
    assert claim_update(uid) is True
    assert claim_update(uid) is False
