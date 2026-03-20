"""
Unit tests for TechnicalFeatures.

All tests use synthetic OHLCV data so no network access is required.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_df():
    """Generate 300 bars of synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 5_000_000, n).astype(float)

    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTechnicalFeatures:

    def test_init_with_missing_column_raises(self, ohlcv_df):
        df = ohlcv_df.drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing required columns"):
            TechnicalFeatures(df)

    def test_compute_all_returns_dataframe(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ohlcv_df)

    def test_rsi_range(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        rsi = result["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_bollinger_band_containment(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        upper = result["bb_upper"].dropna()
        lower = result["bb_lower"].dropna()
        mid = result["bb_mid"].dropna()
        # Upper must be >= mid, lower must be <= mid
        assert (upper >= mid).all()
        assert (lower <= mid).all()

    def test_macd_histogram_is_difference(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        hist = result["macd_hist"].dropna()
        macd = result["macd"].dropna()
        signal = result["macd_signal"].dropna()
        # hist = macd - signal
        diff = (macd - signal).reindex(hist.index).dropna()
        hist_aligned = hist.reindex(diff.index)
        np.testing.assert_allclose(hist_aligned.values, diff.values, rtol=1e-4)

    def test_adx_non_negative(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        adx = result["adx"].dropna()
        assert (adx >= 0).all()

    def test_obv_monotone_when_price_always_up(self):
        """OBV should increase monotonically when close always rises."""
        n = 50
        close = pd.Series(np.linspace(100, 150, n))
        volume = pd.Series(np.ones(n) * 1_000_000)
        obv = TechnicalFeatures._obv(close, volume)
        assert (obv.diff().dropna() > 0).all()

    def test_atr_non_negative(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        atr = result["atr"].dropna()
        assert (atr >= 0).all()

    def test_stochastic_range(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        stoch_k, _ = tf._stochastic(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        stoch_k = stoch_k.dropna()
        assert (stoch_k >= 0).all() and (stoch_k <= 100).all()

    def test_no_all_nan_columns(self, ohlcv_df):
        """No column should be entirely NaN (after warm-up period)."""
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        tail = result.tail(100)
        all_nan_cols = [c for c in tail.columns if tail[c].isna().all()]
        assert len(all_nan_cols) == 0, f"All-NaN columns: {all_nan_cols}"

    def test_breakout_flags_binary(self, ohlcv_df):
        tf = TechnicalFeatures(ohlcv_df)
        result = tf.compute_all()
        for col in ("breakout_up", "breakout_down", "volume_spike"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} must be binary"
