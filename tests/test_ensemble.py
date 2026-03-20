"""
Unit tests for EnsembleModel.
"""

import pytest

from src.models.ensemble import EnsembleModel


BULL_PROBA = {"bullish": 0.70, "sideways": 0.20, "bearish": 0.10}
BEAR_PROBA = {"bullish": 0.10, "sideways": 0.20, "bearish": 0.70}
NEUTRAL_PROBA = {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}


class TestEnsembleModel:

    def test_probabilities_sum_to_one(self):
        model = EnsembleModel()
        result = model.predict(ml_proba=BULL_PROBA, statistical_proba=NEUTRAL_PROBA)
        proba = result["probabilities"]
        assert abs(sum(proba.values()) - 1.0) < 1e-4

    def test_signal_matches_dominant_proba(self):
        model = EnsembleModel()
        result = model.predict(ml_proba=BULL_PROBA)
        assert result["signal"] == "bullish"

    def test_bearish_signal(self):
        model = EnsembleModel()
        result = model.predict(ml_proba=BEAR_PROBA)
        assert result["signal"] == "bearish"

    def test_confidence_between_zero_and_one(self):
        model = EnsembleModel()
        result = model.predict(ml_proba=BULL_PROBA, statistical_proba=BULL_PROBA)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_macro_risk_dampens_bull(self):
        model = EnsembleModel()
        result_low_risk = model.predict(ml_proba=BULL_PROBA, macro_risk_score=0.0)
        result_high_risk = model.predict(ml_proba=BULL_PROBA, macro_risk_score=1.0)
        # High macro risk should reduce bullish probability
        assert result_high_risk["probabilities"]["bullish"] < result_low_risk["probabilities"]["bullish"]

    def test_fallback_when_no_models(self):
        model = EnsembleModel()
        result = model.predict()
        assert "probabilities" in result
        assert "signal" in result

    def test_multiple_model_breakdown_recorded(self):
        model = EnsembleModel()
        result = model.predict(
            statistical_proba=NEUTRAL_PROBA,
            ml_proba=BULL_PROBA,
            sentiment_proba=BULL_PROBA,
        )
        breakdown = result["model_breakdown"]
        assert "statistical" in breakdown
        assert "ml" in breakdown
        assert "sentiment" in breakdown

    def test_custom_weights(self):
        """Sentiment-dominated ensemble should follow sentiment signal."""
        weights = {"statistical": 0.05, "ml": 0.05, "deep_learning": 0.05, "sentiment": 0.85, "macro": 0.0}
        model = EnsembleModel(weights=weights)
        result = model.predict(
            statistical_proba=BEAR_PROBA,
            ml_proba=BEAR_PROBA,
            sentiment_proba=BULL_PROBA,
        )
        # Sentiment dominates → expect bullish
        assert result["signal"] == "bullish"

    def test_proba_values_are_rounded(self):
        model = EnsembleModel()
        result = model.predict(ml_proba=BULL_PROBA)
        for v in result["probabilities"].values():
            # should have at most 4 decimal places
            assert round(v, 4) == v
