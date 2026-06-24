"""Injury / suspension news from GNews + NewsAPI (parity with n8n workflows)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from apps.api.services.worldcup_engine import name_match
from apps.shared.config import get_settings

logger = logging.getLogger(__name__)
INJURY_KEYWORDS = [
    "lesion",
    "lesionado",
    "lesiona",
    "baja",
    "suspendi",
    "descart",
    "no jugara",
    "injury",
    "injured",
    "out for",
    "misses",
    "miss match",
    "doubt",
    "ausente",
    "sancion",
    "tarjeta roja",
    "expulsad",
    "ruled out",
]

SUSPENSION_KEYWORDS = ["suspendi", "sancion", "tarjeta roja", "expulsad", "suspended"]

# Selecciones WC frecuentes — para descartar noticias de otro partido.
WC_NATION_ALIASES = [
    "argentina", "france", "brazil", "england", "spain", "germany", "portugal",
    "netherlands", "belgium", "italy", "croatia", "morocco", "usa", "united states",
    "switzerland", "mexico", "denmark", "uruguay", "colombia", "japan", "senegal",
    "scotland", "chile", "australia", "south korea", "canada", "ecuador", "peru",
    "austria", "turkey", "nigeria", "cameroon", "ghana", "costa rica", "serbia",
    "poland", "wales", "haiti", "paraguay", "qatar", "saudi arabia", "iran",
    "tunisia", "algeria", "egypt", "panama", "jamaica", "new zealand",
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _text_mentions_team(text: str, team: str) -> bool:
    if not text or not team:
        return False
    ntext = _norm(text)
    nteam = _norm(team)
    if not nteam:
        return False
    return nteam in ntext


def _is_fixture_team(name: str, team1: str, team2: str) -> bool:
    return name_match(name, team1) or name_match(name, team2)


def article_relevant_to_fixture(article: NewsArticle, team1: str, team2: str) -> bool:
    """
    Mantiene noticias del partido actual y descarta historias de otros encuentros
    (ej. lesión de Raphinha en Brasil vs Haití cuando se analiza Marruecos vs Haití).
    """
    body = f"{article.title} {article.summary}"
    if not (_text_mentions_team(body, team1) or _text_mentions_team(body, team2)):
        return False

    title = article.title or ""
    for nation in WC_NATION_ALIASES:
        if _is_fixture_team(nation, team1, team2):
            continue
        if _text_mentions_team(title, nation):
            return False
    return True


def filter_articles_for_fixture(
    articles: list[NewsArticle],
    team1: str,
    team2: str,
) -> list[NewsArticle]:
    return [a for a in articles if article_relevant_to_fixture(a, team1, team2)]


@dataclass
class NewsArticle:
    title: str
    summary: str
    date: str
    score: int = 0
    source: str = ""


@dataclass
class InjuryReport:
    articles: list[NewsArticle] = field(default_factory=list)
    has_injuries: bool = False
    has_suspensions: bool = False
    has_impact_news: bool = False
    gnews_ok: bool = False
    newsapi_ok: bool = False

    def headline_lines(self, limit: int = 3) -> list[str]:
        lines: list[str] = []
        for a in self.articles[:limit]:
            prefix = f"{a.date}: " if a.date else ""
            line = f"{prefix}{a.title}"
            if a.summary:
                line += f" — {a.summary[:120]}"
            lines.append(line)
        return lines


def _keyword_score(text: str) -> int:
    tx = text.lower()
    return sum(1 for k in INJURY_KEYWORDS if k in tx)


def _normalize_gnews_article(raw: dict) -> NewsArticle:
    return NewsArticle(
        title=(raw.get("title") or "")[:100],
        summary=(raw.get("description") or "")[:140],
        date=(raw.get("publishedAt") or "")[:10],
        source="gnews",
    )


def _normalize_newsapi_article(raw: dict) -> NewsArticle:
    return NewsArticle(
        title=(raw.get("title") or "")[:100],
        summary=(raw.get("description") or "")[:140],
        date=(raw.get("publishedAt") or "")[:10],
        source="newsapi",
    )


def merge_injury_articles(
    gnews_payload: dict | None,
    newsapi_payload: dict | None,
    *,
    max_items: int = 5,
) -> list[NewsArticle]:
    """Merge and dedupe articles (same logic as n8n procNews)."""
    merged: list[NewsArticle] = []
    for raw in (gnews_payload or {}).get("articles") or []:
        art = _normalize_gnews_article(raw)
        art.score = _keyword_score(f"{art.title} {art.summary}")
        if art.score > 0:
            merged.append(art)
    for raw in (newsapi_payload or {}).get("articles") or []:
        art = _normalize_newsapi_article(raw)
        art.score = _keyword_score(f"{art.title} {art.summary}")
        if art.score > 0:
            merged.append(art)

    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for art in sorted(merged, key=lambda x: x.score, reverse=True):
        key = art.title[:40].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(art)
    return unique[:max_items]


def classify_flags(articles: list[NewsArticle]) -> tuple[bool, bool, bool]:
    text = " ".join(f"{a.title} {a.summary}" for a in articles).lower()
    has_injuries = any(k in text for k in ("lesion", "injury", "injured", "lesionado", "out injured"))
    has_suspensions = any(k in text for k in SUSPENSION_KEYWORDS)
    has_impact = len(text.strip()) > 10
    return has_injuries, has_suspensions, has_impact


async def _fetch_gnews(team1: str, team2: str, token: str) -> dict | None:
    q = quote(f'{team1} OR {team2} (lesion OR injury OR baja OR lesionado)')
    url = (
        f"https://gnews.io/api/v4/search?q={q}"
        f"&token={token}&max=5&sortby=publishedAt&in=title,description"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("GNews: %s", exc)
        return None


async def _fetch_newsapi(team1: str, team2: str, api_key: str) -> dict | None:
    q = quote(f'"{team1}" OR "{team2}" (lesion OR injury OR baja OR lesionado)')
    url = (
        f"https://newsapi.org/v2/everything?q={q}"
        f"&apiKey={api_key}&sortBy=publishedAt&pageSize=5&searchIn=title,description"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("NewsAPI: %s", exc)
        return None


async def fetch_injury_report(team1: str, team2: str) -> InjuryReport:
    """Fetch and merge injury news for both teams."""
    settings = get_settings()
    gnews_data = None
    newsapi_data = None

    if settings.gnews_api_key:
        gnews_data = await _fetch_gnews(team1, team2, settings.gnews_api_key)
    if settings.newsapi_key:
        newsapi_data = await _fetch_newsapi(team1, team2, settings.newsapi_key)

    articles = merge_injury_articles(gnews_data, newsapi_data)
    articles = filter_articles_for_fixture(articles, team1, team2)
    has_inj, has_susp, has_impact = classify_flags(articles)

    return InjuryReport(
        articles=articles,
        has_injuries=has_inj,
        has_suspensions=has_susp,
        has_impact_news=has_impact and bool(articles),
        gnews_ok=gnews_data is not None,
        newsapi_ok=newsapi_data is not None,
    )
