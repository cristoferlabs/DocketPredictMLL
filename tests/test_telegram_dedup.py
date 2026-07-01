"""Tests for Telegram update deduplication."""

import random

from apps.api.services.telegram_dedup import claim_update


def test_claim_update_first_wins():
    # Use a random ID so Redis state from prior runs doesn't cause false failures
    uid = random.randint(10_000_000_000, 99_999_999_999)
    assert claim_update(uid) is True
    assert claim_update(uid) is False
