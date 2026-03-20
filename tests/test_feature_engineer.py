"""
Integration tests for FeatureEngineer pipeline (no network required).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import FeatureEngineer


@pytest.fixture
def synthetic_price_df():
    rng = np.random.default_rng(99)
    n = 400
    close = 100 + np.cumsum(rng.normal(0.02, 1, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.002, 0.015, n))
    low = close * (1 - rng.uniform(0.002, 0.015, n))
    volume = rng.integers(500_000, 8_000_000, n).astype(float)
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestFeatureEngineer:

    def test_build_feature_matrix_returns_dataframe(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        X = eng.build_feature_matrix(scale=False)
        assert isinstance(X, pd.DataFrame)
        assert not X.empty

    def test_no_nan_in_output(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        X = eng.build_feature_matrix(scale=False)
        assert not X.isna().any().any(), "Feature matrix must have no NaN after cleanup"

    def test_scaled_output_has_zero_median(self, synthetic_price_df):
        """RobustScaler centres the median to ~0 for non-binary features."""
        eng = FeatureEngineer(price_df=synthetic_price_df)
        X = eng.build_feature_matrix(scale=True)
        # Most columns should have small median (allowing for binary/constant cols)
        medians = X.median()
        # Check that the majority of feature medians are small
        small_median_frac = (medians.abs() < 1.0).mean()
        assert small_median_frac > 0.6

    def test_scaler_stored_after_fit(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        eng.build_feature_matrix(scale=True)
        assert eng.get_scaler() is not None

    def test_supervised_dataset_shape(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        X, y = eng.build_supervised_dataset(horizon=3, scale=False)
        assert len(X) == len(y)
        assert len(X) > 0
        assert X.shape[1] > 0

    def test_target_values_valid(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        _, y = eng.build_supervised_dataset(horizon=3, scale=False)
        assert set(y.unique()).issubset({-1, 0, 1})

    def test_sentiment_features_appended(self, synthetic_price_df):
        news = {"avg_sentiment": 0.3, "sentiment_std": 0.1, "bullish_ratio": 0.5,
                "bearish_ratio": 0.2, "article_count": 8, "top_topics": ["earnings"], "avg_impact": 0.3}
        social = {"sentiment_polarity": 0.2, "mention_volume": 80, "engagement_spike": 0,
                   "influencer_score": 0.1, "reddit_wsb_mentions": 3, "composite_score": 0.2}
        eng = FeatureEngineer(
            price_df=synthetic_price_df,
            news_sentiment=news,
            social_sentiment=social,
        )
        X = eng.build_feature_matrix(scale=False)
        sent_cols = [c for c in X.columns if c.startswith("sent_")]
        assert len(sent_cols) > 0

    def test_get_latest_feature_vector_is_series(self, synthetic_price_df):
        eng = FeatureEngineer(price_df=synthetic_price_df)
        latest = eng.get_latest_feature_vector(scale=False)
        assert isinstance(latest, pd.Series)
        assert not latest.isna().any()
