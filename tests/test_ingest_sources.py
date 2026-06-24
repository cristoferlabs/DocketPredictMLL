"""Test API source configuration."""

from apps.shared.config import get_settings


def test_free_tier_keys_configured():
    s = get_settings()
    assert s.football_data_key
    assert s.odds_api_key


def test_api_football_free_max_season():
    from apps.worker.tasks.ingest import API_FOOTBALL_FREE_MAX_SEASON

    assert API_FOOTBALL_FREE_MAX_SEASON == 2024
