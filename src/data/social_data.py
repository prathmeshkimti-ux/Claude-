"""
Social Sentiment Data Fetcher
==============================
Collects and aggregates social-media signals from Twitter/X and Reddit.

Metrics extracted per symbol:
  - mention_volume         : absolute number of mentions in the window
  - mention_velocity       : rate of change vs previous window
  - sentiment_polarity     : mean VADER compound score (-1 → +1)
  - engagement_spike       : 1 if volume > 2σ above 7-day mean
  - influencer_score       : weighted score from verified / high-follower accounts
  - reddit_wsb_mentions    : WallStreetBets-specific mention count (retail hype proxy)

Twitter: uses Tweepy v4 bearer-token (OAuth 2.0 App-Only).
Reddit:  uses PRAW or raw JSON API (no key required for read-only public posts).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import config

logger = logging.getLogger(__name__)


class SocialDataFetcher:
    """
    Fetches social-media sentiment signals.

    Parameters
    ----------
    lookback_hours : int
        Window length for aggregation.
    max_tweets : int
        Maximum recent tweets to fetch per symbol.
    """

    def __init__(
        self,
        lookback_hours: int = None,
        max_tweets: int = 200,
    ) -> None:
        self.lookback_hours = lookback_hours or config.data.social_lookback_hours
        self.max_tweets = max_tweets
        self._vader = SentimentIntensityAnalyzer()
        self._twitter_client = self._init_twitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_social_sentiment(self, symbol: str) -> Dict:
        """
        Return aggregated social sentiment dict for *symbol*.

        Keys: mention_volume, mention_velocity, sentiment_polarity,
              engagement_spike, reddit_wsb_mentions, composite_score
        """
        tweets = self._fetch_tweets(symbol)
        reddit_posts = self._fetch_reddit(symbol)

        all_texts = [t["text"] for t in tweets] + [r["text"] for r in reddit_posts]
        sentiment_scores = [self._vader.polarity_scores(t)["compound"] for t in all_texts]

        mention_volume = len(all_texts)
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        # Engagement spike: compare to a rolling baseline (mocked here)
        # In production, store historical counts in Redis/DB and compare
        baseline_mentions = self._get_baseline_mentions(symbol)
        engagement_spike = int(mention_volume > baseline_mentions * 2)

        # Influencer score: weight high-follower accounts (Twitter only)
        influencer_score = self._compute_influencer_score(tweets)

        # Reddit WSB mentions
        wsb_mentions = sum(
            1 for r in reddit_posts if "wallstreetbets" in r.get("subreddit", "").lower()
        )

        # Composite: weighted blend of volume signal and sentiment
        volume_norm = min(mention_volume / 500.0, 1.0)
        composite = 0.4 * avg_sentiment + 0.3 * volume_norm + 0.3 * influencer_score

        return {
            "mention_volume": mention_volume,
            "sentiment_polarity": round(avg_sentiment, 4),
            "engagement_spike": engagement_spike,
            "influencer_score": round(influencer_score, 4),
            "reddit_wsb_mentions": wsb_mentions,
            "composite_score": round(composite, 4),
        }

    def get_social_timeseries(self, symbol: str) -> pd.DataFrame:
        """
        Return hourly sentiment timeseries for the look-back window.
        Useful for LSTM input (sequential social signal).
        """
        tweets = self._fetch_tweets(symbol)
        if not tweets:
            return pd.DataFrame()

        records = []
        for tw in tweets:
            score = self._vader.polarity_scores(tw["text"])["compound"]
            records.append({
                "timestamp": tw.get("created_at", pd.Timestamp.utcnow()),
                "sentiment": score,
                "followers": tw.get("followers_count", 0),
            })

        df = pd.DataFrame(records).set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        hourly = df.resample("h").agg({"sentiment": "mean", "followers": "sum"})
        return hourly.fillna(0)

    # ------------------------------------------------------------------
    # Twitter helpers
    # ------------------------------------------------------------------

    def _init_twitter(self):
        """Initialise Tweepy client if bearer token is available."""
        bearer = config.api.twitter_bearer_token
        if not bearer:
            logger.info("Twitter bearer token not set; Twitter fetching disabled.")
            return None
        try:
            import tweepy
            return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
        except ImportError:
            logger.warning("tweepy not installed; Twitter fetching disabled.")
            return None

    def _fetch_tweets(self, symbol: str) -> List[Dict]:
        """Search recent tweets mentioning the ticker."""
        if self._twitter_client is None:
            return self._mock_tweets(symbol)

        query = f"${symbol} OR #{symbol} -is:retweet lang:en"
        try:
            import tweepy
            response = self._twitter_client.search_recent_tweets(
                query=query,
                max_results=min(self.max_tweets, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["public_metrics"],
            )
            if not response.data:
                return []

            user_map = {}
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    user_map[user.id] = user.public_metrics.get("followers_count", 0)

            tweets = []
            for tw in response.data:
                tweets.append({
                    "text": tw.text,
                    "created_at": tw.created_at or pd.Timestamp.utcnow(),
                    "likes": tw.public_metrics.get("like_count", 0) if tw.public_metrics else 0,
                    "retweets": tw.public_metrics.get("retweet_count", 0) if tw.public_metrics else 0,
                    "followers_count": user_map.get(tw.author_id, 0),
                })
            return tweets

        except Exception as exc:
            logger.warning("Twitter fetch failed for %s: %s", symbol, exc)
            return self._mock_tweets(symbol)

    @staticmethod
    def _mock_tweets(symbol: str) -> List[Dict]:
        """Return synthetic tweets for testing when API is unavailable."""
        import random
        templates = [
            f"${symbol} looking bullish! Great support level here. #stocks",
            f"Holding ${symbol} long-term. Solid fundamentals.",
            f"${symbol} just broke resistance. Could run to new highs.",
            f"Not sure about ${symbol} right now, choppy market.",
            f"Sold my ${symbol} position. Taking profits.",
        ]
        return [
            {
                "text": templates[i % len(templates)],
                "created_at": pd.Timestamp.utcnow(),
                "likes": random.randint(0, 200),
                "retweets": random.randint(0, 50),
                "followers_count": random.randint(100, 50000),
            }
            for i in range(20)
        ]

    # ------------------------------------------------------------------
    # Reddit helpers
    # ------------------------------------------------------------------

    def _fetch_reddit(self, symbol: str) -> List[Dict]:
        """
        Fetch Reddit posts via the public JSON API (no auth required).

        Searches r/stocks, r/investing, r/wallstreetbets.
        """
        subreddits = ["stocks", "investing", "wallstreetbets", "IndiaInvestments"]
        posts = []
        import requests

        for sub in subreddits:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {"q": symbol, "sort": "new", "limit": 25, "t": "day"}
            headers = {"User-Agent": "PricePredictionBot/1.0"}
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=8)
                resp.raise_for_status()
                data = resp.json()
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    posts.append({
                        "text": f"{post.get('title', '')}. {post.get('selftext', '')}",
                        "subreddit": sub,
                        "upvotes": post.get("ups", 0),
                        "created_at": pd.Timestamp.fromtimestamp(
                            post.get("created_utc", 0), tz=timezone.utc
                        ),
                    })
            except Exception as exc:
                logger.debug("Reddit fetch failed for %s/%s: %s", sub, symbol, exc)

        return posts

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_baseline_mentions(symbol: str) -> float:
        """
        Retrieve 7-day average mention count from cache/DB.

        Stub: returns 50 as a default baseline.
        Production: query Redis / InfluxDB for stored rolling average.
        """
        return 50.0

    @staticmethod
    def _compute_influencer_score(tweets: List[Dict]) -> float:
        """
        Weight tweet sentiment by author follower count.
        Returns a normalised score in [0, 1].
        """
        if not tweets:
            return 0.0

        weighted = sum(
            min(tw.get("followers_count", 0) / 100_000, 1.0)
            for tw in tweets
        )
        return min(weighted / max(len(tweets), 1), 1.0)
