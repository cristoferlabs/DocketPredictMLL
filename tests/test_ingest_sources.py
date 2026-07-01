"""Test API source configuration."""

from apps.shared.config import get_settings


def test_free_tier_keys_configured():
    s = get_settings()
    assert s.football_data_key
    assert s.odds_api_key


def test_api_football_wc_league_configured():
    # Pro plan — no season cap; WC2026 ingested via WC_LEAGUE_ID=1
    from apps.worker.tasks.ingest import WC_LEAGUE_ID, WC_SEASON

    assert WC_LEAGUE_ID == 1
    assert WC_SEASON == 2026
