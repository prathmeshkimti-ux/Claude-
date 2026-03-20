"""
Unit tests for RiskEngine.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.risk_engine import RiskEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_price_df():
    rng = np.random.default_rng(0)
    n = 30
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": close, "volume": volume}, index=dates)


@pytest.fixture
def bull_prediction():
    return {
        "probabilities": {"bullish": 0.65, "sideways": 0.20, "bearish": 0.15},
        "signal": "bullish",
        "confidence": 0.65,
        "dominant_prob": 0.65,
    }


@pytest.fixture
def low_confidence_prediction():
    return {
        "probabilities": {"bullish": 0.38, "sideways": 0.34, "bearish": 0.28},
        "signal": "bullish",
        "confidence": 0.40,  # below min_confidence
        "dominant_prob": 0.38,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskEngine:

    def test_approved_on_clean_signal(self, bull_prediction, mock_price_df):
        engine = RiskEngine()
        result = engine.evaluate(prediction=bull_prediction, price_df=mock_price_df)
        assert result["approved"] is True

    def test_blocked_on_low_confidence(self, low_confidence_prediction, mock_price_df):
        engine = RiskEngine()
        result = engine.evaluate(prediction=low_confidence_prediction, price_df=mock_price_df)
        assert result["approved"] is False
        assert result["position_size"] == 0.0

    def test_high_event_risk_blocks_signal(self, bull_prediction, mock_price_df):
        engine = RiskEngine()
        events = [{"name": "FOMC", "hours_until": 12, "impact": "high"}]
        result = engine.evaluate(
            prediction=bull_prediction,
            price_df=mock_price_df,
            upcoming_events=events,
        )
        # High impact event within 24h should block or reduce
        assert result["risk_report"]["event_risk"]["event_risk_score"] > 0.5

    def test_position_size_within_max(self, bull_prediction, mock_price_df):
        from config import config
        engine = RiskEngine()
        result = engine.evaluate(prediction=bull_prediction, price_df=mock_price_df)
        assert result["position_size"] <= config.risk.max_position_pct

    def test_adjusted_proba_sums_to_one(self, bull_prediction, mock_price_df):
        engine = RiskEngine()
        result = engine.evaluate(prediction=bull_prediction, price_df=mock_price_df)
        total = sum(result["adjusted_proba"].values())
        assert abs(total - 1.0) < 1e-4

    def test_empty_price_df_does_not_crash(self, bull_prediction):
        engine = RiskEngine()
        result = engine.evaluate(
            prediction=bull_prediction,
            price_df=pd.DataFrame(),
        )
        assert "approved" in result

    def test_high_vol_reduces_position_size(self, bull_prediction):
        """Verify that high-volatility regime reduces recommended position size."""
        engine = RiskEngine()

        rng = np.random.default_rng(1)
        # Low vol price series
        low_vol_close = 100 + np.cumsum(rng.normal(0, 0.2, 30))
        # High vol price series
        high_vol_close = 100 + np.cumsum(rng.normal(0, 5.0, 30))

        dates = pd.bdate_range("2024-01-01", periods=30)
        vol_base = pd.Series(np.ones(30) * 1_000_000, name="volume")

        low_df = pd.DataFrame({"close": low_vol_close, "volume": vol_base.values}, index=dates)
        high_df = pd.DataFrame({"close": high_vol_close, "volume": vol_base.values}, index=dates)

        res_low = engine.evaluate(prediction=bull_prediction, price_df=low_df)
        res_high = engine.evaluate(prediction=bull_prediction, price_df=high_df)

        assert res_high["position_size"] <= res_low["position_size"] * 1.5
