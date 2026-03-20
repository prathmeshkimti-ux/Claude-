"""
News Data Fetcher & NLP Pipeline
==================================
Fetches financial news headlines from RSS feeds and the NewsAPI,
then runs a lightweight NLP pipeline to extract:

  - Sentiment score   (positive / neutral / negative, -1 → +1)
  - Topic tags        (earnings, regulations, geopolitics, …)
  - Impact score      (0 = minor, 1 = moderate, 2 = major)

Heavy-weight FinBERT classification happens in src/models/sentiment_model.py.
This module handles the data-ingestion concern only.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public RSS feeds (no API key required)
# ---------------------------------------------------------------------------
RSS_FEEDS: Dict[str, str] = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch":     "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "seeking_alpha":   "https://seekingalpha.com/market_currents.xml",
    "yahoo_finance":   "https://finance.yahoo.com/rss/headline",
    "investing_com":   "https://www.investing.com/rss/news.rss",
    "moneycontrol":    "https://www.moneycontrol.com/rss/latestnews.xml",
    "economic_times":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}

# Keywords that map to topic labels
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "earnings":     ["earnings", "profit", "revenue", "quarterly", "eps", "guidance"],
    "regulations":  ["regulation", "sec", "sebi", "ban", "sanction", "fine", "compliance"],
    "geopolitics":  ["war", "conflict", "sanction", "nato", "tension", "military", "invasion"],
    "supply_shock": ["supply", "shortage", "disruption", "opec", "inventory", "pipeline"],
    "fed_policy":   ["fed", "fomc", "interest rate", "hike", "cut", "tapering", "powell", "rbi"],
    "merger_acq":   ["merger", "acquisition", "takeover", "buyout", "deal"],
    "macro":        ["gdp", "inflation", "unemployment", "cpi", "ppi", "trade deficit"],
}

# Keywords that elevate impact score
HIGH_IMPACT_KEYWORDS = [
    "crash", "collapse", "bankrupt", "default", "recession", "war",
    "sanctions", "rate hike", "rate cut", "earnings miss", "earnings beat",
    "guidance cut", "guidance raise",
]


class NewsDataFetcher:
    """
    Fetches and processes financial news from RSS feeds and NewsAPI.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols used as search filters when querying NewsAPI.
    lookback_hours : int
        Maximum age (in hours) of articles to include.
    use_newsapi : bool
        If True and NEWS_API_KEY is configured, supplement RSS with NewsAPI.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = None,
        use_newsapi: bool = True,
    ) -> None:
        self.symbols = [s.upper() for s in (symbols or [])]
        self.lookback_hours = lookback_hours or config.data.news_lookback_hours
        self.use_newsapi = use_newsapi and bool(config.api.news_api_key)
        self._vader = SentimentIntensityAnalyzer()
        self._seen_ids: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_news(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame of processed news articles.

        Columns: title, source, published_at, url, sentiment_score,
                 sentiment_label, topics, impact_score, symbol_mentions
        """
        articles = self._fetch_rss()
        if self.use_newsapi and symbol:
            articles += self._fetch_newsapi(symbol)

        if not articles:
            logger.warning("No news articles fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df = self._deduplicate(df)
        df = self._filter_recency(df)

        if symbol:
            # Prioritise articles mentioning the symbol
            df["symbol_mentions"] = df["title"].str.contains(
                symbol, case=False, na=False
            ).astype(int)

        return df.sort_values("published_at", ascending=False).reset_index(drop=True)

    def get_aggregated_sentiment(self, symbol: Optional[str] = None) -> Dict:
        """
        Return a compact sentiment summary for use in the feature vector.

        Returns
        -------
        dict with keys:
            avg_sentiment, sentiment_std, bullish_ratio, bearish_ratio,
            article_count, top_topics, avg_impact
        """
        df = self.get_news(symbol)

        if df.empty:
            return self._empty_sentiment_dict()

        avg_sent = df["sentiment_score"].mean()
        sent_std = df["sentiment_score"].std()
        bull_ratio = (df["sentiment_label"] == "positive").mean()
        bear_ratio = (df["sentiment_label"] == "negative").mean()

        all_topics = [t for topics in df["topics"] for t in topics]
        topic_counts: Dict[str, int] = {}
        for t in all_topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        top_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)[:3]

        return {
            "avg_sentiment": round(avg_sent, 4),
            "sentiment_std": round(sent_std, 4),
            "bullish_ratio": round(bull_ratio, 4),
            "bearish_ratio": round(bear_ratio, 4),
            "article_count": len(df),
            "top_topics": top_topics,
            "avg_impact": round(df["impact_score"].mean(), 4),
        }

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------

    def _fetch_rss(self) -> List[Dict]:
        articles = []
        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:  # cap per feed
                    article = self._parse_rss_entry(entry, source)
                    if article:
                        articles.append(article)
            except Exception as exc:
                logger.debug("RSS fetch failed for %s: %s", source, exc)
        return articles

    def _fetch_newsapi(self, query: str) -> List[Dict]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": config.api.news_api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)
            return []

        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "") or ""
            description = item.get("description", "") or ""
            text = f"{title}. {description}".strip()
            pub_str = item.get("publishedAt", "")
            try:
                pub_at = pd.to_datetime(pub_str, utc=True)
            except Exception:
                pub_at = pd.Timestamp.utcnow()

            articles.append(self._process_text(
                title=title,
                text=text,
                source=item.get("source", {}).get("name", "newsapi"),
                published_at=pub_at,
                url=item.get("url", ""),
            ))
        return articles

    # ------------------------------------------------------------------
    # Parsers / processors
    # ------------------------------------------------------------------

    def _parse_rss_entry(self, entry, source: str) -> Optional[Dict]:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        text = f"{title}. {summary}".strip()
        url = getattr(entry, "link", "") or ""

        pub_parsed = getattr(entry, "published_parsed", None)
        if pub_parsed:
            pub_at = pd.Timestamp(*pub_parsed[:6], tzinfo=timezone.utc)
        else:
            pub_at = pd.Timestamp.utcnow()

        return self._process_text(title, text, source, pub_at, url)

    def _process_text(
        self,
        title: str,
        text: str,
        source: str,
        published_at: pd.Timestamp,
        url: str,
    ) -> Dict:
        # VADER sentiment
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]   # -1 → +1
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "title": title,
            "source": source,
            "published_at": published_at,
            "url": url,
            "sentiment_score": round(compound, 4),
            "sentiment_label": label,
            "topics": self._detect_topics(text),
            "impact_score": self._compute_impact(text, compound),
        }

    @staticmethod
    def _detect_topics(text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(topic)
        return found

    @staticmethod
    def _compute_impact(text: str, compound: float) -> float:
        """
        0 = minor, 0.5 = moderate, 1 = major.
        Escalated by strong sentiment and high-impact keywords.
        """
        text_lower = text.lower()
        base = abs(compound)  # 0 → 1
        boost = sum(0.15 for kw in HIGH_IMPACT_KEYWORDS if kw in text_lower)
        return min(1.0, round(base + boost, 4))

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        df["_id"] = df["title"].apply(
            lambda t: hashlib.md5(t.encode()).hexdigest()
        )
        df = df.drop_duplicates(subset=["_id"]).drop(columns=["_id"])
        return df

    def _filter_recency(self, df: pd.DataFrame) -> pd.DataFrame:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=self.lookback_hours)
        return df[df["published_at"] >= cutoff]

    @staticmethod
    def _empty_sentiment_dict() -> Dict:
        return {
            "avg_sentiment": 0.0,
            "sentiment_std": 0.0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "article_count": 0,
            "top_topics": [],
            "avg_impact": 0.0,
        }
