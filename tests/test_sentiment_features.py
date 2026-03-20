"""
Unit tests for SentimentFeatures and FinBERTSentiment (VADER fallback).
"""

import pytest

from src.features.sentiment import SentimentFeatures, TOPICS
from src.models.sentiment_model import FinBERTSentiment


class TestSentimentFeatures:

    def test_get_sentiment_vector_returns_dict(self):
        news = {"avg_sentiment": 0.5, "sentiment_std": 0.1, "bullish_ratio": 0.6,
                "bearish_ratio": 0.1, "article_count": 10, "top_topics": ["earnings"], "avg_impact": 0.4}
        social = {"sentiment_polarity": 0.3, "mention_volume": 150, "engagement_spike": 0,
                   "influencer_score": 0.2, "reddit_wsb_mentions": 5, "composite_score": 0.3}
        sf = SentimentFeatures(news_sentiment=news, social_sentiment=social)
        vec = sf.get_sentiment_vector()
        assert isinstance(vec, dict)
        assert "composite_sentiment" in vec
        assert "news_sentiment" in vec
        assert "social_sentiment" in vec

    def test_topic_one_hot_encoding(self):
        news = {"top_topics": ["earnings", "geopolitics"], "avg_sentiment": 0.0,
                "sentiment_std": 0.0, "bullish_ratio": 0.0, "bearish_ratio": 0.0,
                "article_count": 0, "avg_impact": 0.0}
        sf = SentimentFeatures(news_sentiment=news)
        vec = sf.get_sentiment_vector()
        assert vec["topic_earnings"] == 1.0
        assert vec["topic_geopolitics"] == 1.0
        assert vec["topic_fed_policy"] == 0.0

    def test_composite_range(self):
        news = {"avg_sentiment": 0.8, "top_topics": [], "sentiment_std": 0.0,
                "bullish_ratio": 0.9, "bearish_ratio": 0.0, "article_count": 5, "avg_impact": 0.5}
        social = {"sentiment_polarity": 0.7, "mention_volume": 100, "engagement_spike": 1,
                   "influencer_score": 0.5, "reddit_wsb_mentions": 10, "composite_score": 0.6}
        sf = SentimentFeatures(news_sentiment=news, social_sentiment=social)
        vec = sf.get_sentiment_vector()
        assert -1.0 <= vec["composite_sentiment"] <= 1.0

    def test_signal_bullish(self):
        news = {"avg_sentiment": 0.8, "top_topics": [], "sentiment_std": 0.0,
                "bullish_ratio": 1.0, "bearish_ratio": 0.0, "article_count": 10, "avg_impact": 0.6}
        social = {"sentiment_polarity": 0.8, "mention_volume": 200, "engagement_spike": 1,
                   "influencer_score": 0.9, "reddit_wsb_mentions": 20, "composite_score": 0.8}
        sf = SentimentFeatures(news_sentiment=news, social_sentiment=social)
        assert sf.get_sentiment_signal() == "bullish"

    def test_signal_bearish(self):
        news = {"avg_sentiment": -0.8, "top_topics": [], "sentiment_std": 0.0,
                "bullish_ratio": 0.0, "bearish_ratio": 1.0, "article_count": 10, "avg_impact": 0.6}
        social = {"sentiment_polarity": -0.8, "mention_volume": 200, "engagement_spike": 0,
                   "influencer_score": 0.1, "reddit_wsb_mentions": 2, "composite_score": -0.6}
        sf = SentimentFeatures(news_sentiment=news, social_sentiment=social)
        assert sf.get_sentiment_signal() == "bearish"

    def test_weight_validation(self):
        with pytest.raises(AssertionError):
            SentimentFeatures(news_weight=0.5, social_weight=0.3)  # don't sum to 1


class TestFinBERTSentiment:

    def test_score_positive_text(self):
        model = FinBERTSentiment()
        result = model.score_text("Apple reports record-breaking quarterly earnings, stock soars.")
        assert result["label"] in ("positive", "neutral", "negative")
        assert abs(sum([result["positive"], result["neutral"], result["negative"]]) - 1.0) < 0.01

    def test_score_negative_text(self):
        model = FinBERTSentiment()
        result = model.score_text("Company faces massive losses and bankruptcy concerns.")
        # compound should be negative or neutral
        assert "compound" in result
        assert -1.0 <= result["compound"] <= 1.0

    def test_empty_text_returns_neutral(self):
        model = FinBERTSentiment()
        result = model.score_text("")
        assert result["label"] == "neutral"

    def test_aggregate_returns_required_keys(self):
        model = FinBERTSentiment()
        texts = ["Stock market up today.", "Inflation fears weigh on markets."]
        agg = model.aggregate(texts)
        for key in ("avg_compound", "bullish_ratio", "bearish_ratio", "dominant_label"):
            assert key in agg

    def test_get_directional_proba_sums_to_one(self):
        model = FinBERTSentiment()
        texts = ["Strong buy signal.", "Market uncertainty persists."]
        proba = model.get_directional_proba(texts)
        assert abs(sum(proba.values()) - 1.0) < 1e-4

    def test_caching(self):
        model = FinBERTSentiment(use_cache=True)
        text = "Revenue growth beats expectations."
        r1 = model.score_text(text)
        r2 = model.score_text(text)
        assert r1 == r2
