"""
Sentiment Feature Engineering
================================
Converts raw news / social signal dicts into normalised sentiment features
that can be appended to the model feature vector.

Outputs a compact scored representation:
  - news_sentiment_score   : -1 → +1 (from news NLP pipeline)
  - social_sentiment_score : -1 → +1 (from Twitter/Reddit)
  - news_impact_score      : 0 → 1
  - engagement_spike       : 0/1
  - topic_vector           : one-hot over major topics
  - composite_sentiment    : weighted blend of news + social
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical topic list (order matters for topic_vector encoding)
TOPICS = [
    "earnings",
    "regulations",
    "geopolitics",
    "supply_shock",
    "fed_policy",
    "merger_acq",
    "macro",
]


class SentimentFeatures:
    """
    Combine news and social signals into a model-ready sentiment feature dict.

    Parameters
    ----------
    news_sentiment : dict
        Output of NewsDataFetcher.get_aggregated_sentiment().
    social_sentiment : dict
        Output of SocialDataFetcher.get_social_sentiment().
    news_weight : float
        Relative weight of news in the composite score.
    social_weight : float
        Relative weight of social in the composite score.
    """

    def __init__(
        self,
        news_sentiment: Optional[Dict] = None,
        social_sentiment: Optional[Dict] = None,
        news_weight: float = 0.6,
        social_weight: float = 0.4,
    ) -> None:
        self.news = news_sentiment or {}
        self.social = social_sentiment or {}
        assert abs(news_weight + social_weight - 1.0) < 1e-6, "Weights must sum to 1."
        self.news_weight = news_weight
        self.social_weight = social_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment_vector(self) -> Dict[str, float]:
        """
        Return a flat dict of sentiment features for the model.
        """
        result = {}

        # News features
        result["news_sentiment"] = self.news.get("avg_sentiment", 0.0)
        result["news_sentiment_std"] = self.news.get("sentiment_std", 0.0)
        result["news_bullish_ratio"] = self.news.get("bullish_ratio", 0.0)
        result["news_bearish_ratio"] = self.news.get("bearish_ratio", 0.0)
        result["news_article_count_norm"] = min(self.news.get("article_count", 0) / 50.0, 1.0)
        result["news_impact"] = self.news.get("avg_impact", 0.0)

        # Social features
        result["social_sentiment"] = self.social.get("sentiment_polarity", 0.0)
        result["social_mention_volume_norm"] = min(
            self.social.get("mention_volume", 0) / 500.0, 1.0
        )
        result["social_engagement_spike"] = float(self.social.get("engagement_spike", 0))
        result["social_influencer_score"] = self.social.get("influencer_score", 0.0)
        result["social_wsb_norm"] = min(self.social.get("reddit_wsb_mentions", 0) / 20.0, 1.0)

        # Topic one-hot encoding
        top_topics = self.news.get("top_topics", [])
        for topic in TOPICS:
            result[f"topic_{topic}"] = float(topic in top_topics)

        # Composite score
        result["composite_sentiment"] = self._compute_composite()

        # Sentiment divergence (news vs social)
        result["sentiment_divergence"] = abs(
            result["news_sentiment"] - result["social_sentiment"]
        )

        return result

    def get_sentiment_signal(self) -> str:
        """
        Return a human-readable signal label based on composite score.
        """
        composite = self._compute_composite()
        if composite > 0.15:
            return "bullish"
        elif composite < -0.15:
            return "bearish"
        return "neutral"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_composite(self) -> float:
        news_score = self.news.get("avg_sentiment", 0.0)
        social_score = self.social.get("sentiment_polarity", 0.0)
        # Boost composite when both agree
        agreement_boost = 1.2 if np.sign(news_score) == np.sign(social_score) else 1.0
        composite = (self.news_weight * news_score + self.social_weight * social_score) * agreement_boost
        return round(max(-1.0, min(1.0, composite)), 4)
