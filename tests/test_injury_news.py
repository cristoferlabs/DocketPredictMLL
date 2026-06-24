"""Tests for injury news relevance filtering."""

from apps.api.services.injury_news import (
    NewsArticle,
    article_relevant_to_fixture,
    filter_articles_for_fixture,
)


def test_excludes_brazil_haiti_story_for_morocco_haiti():
    art = NewsArticle(
        title="Raphinha injury after Brazil 3-0 win over Haiti",
        summary="Brazil winger limped off during World Cup match vs Haiti",
        date="2026-06-22",
    )
    assert article_relevant_to_fixture(art, "Morocco", "Haiti") is False


def test_keeps_story_when_fixture_team_in_title():
    art = NewsArticle(
        title="Morocco injury concern ahead of Haiti clash",
        summary="Defender doubtful for next match",
        date="2026-06-22",
    )
    assert article_relevant_to_fixture(art, "Morocco", "Haiti") is True


def test_keeps_brazil_story_for_scotland_brazil():
    art = NewsArticle(
        title="Raphinha ruled out for Brazil vs Scotland",
        summary="Hamstring injury for Brazil winger",
        date="2026-06-22",
    )
    assert article_relevant_to_fixture(art, "Scotland", "Brazil") is True


def test_filter_articles_for_fixture():
    articles = [
        NewsArticle(title="Morocco squad update", summary="fitness check", date="2026-06-22"),
        NewsArticle(
            title="Brazil star injured vs Haiti",
            summary="not this fixture",
            date="2026-06-22",
        ),
    ]
    filtered = filter_articles_for_fixture(articles, "Morocco", "Haiti")
    assert len(filtered) == 1
    assert filtered[0].title.startswith("Morocco")
