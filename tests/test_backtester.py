"""
Unit tests for the Backtester.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester, BacktestResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_up_df():
    """300 bars of steadily rising prices."""
    rng = np.random.default_rng(10)
    n = 300
    close = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    close = np.maximum(close, 1.0)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def always_bullish_signals(trending_up_df):
    return pd.Series("bullish", index=trending_up_df.index)


@pytest.fixture
def always_sideways_signals(trending_up_df):
    return pd.Series("sideways", index=trending_up_df.index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktester:

    def test_run_returns_backtest_results(self, trending_up_df, always_bullish_signals):
        bt = Backtester(trending_up_df, always_bullish_signals, initial_capital=1_000_000)
        results = bt.run()
        assert isinstance(results, BacktestResults)

    def test_equity_curve_length(self, trending_up_df, always_bullish_signals):
        bt = Backtester(trending_up_df, always_bullish_signals, initial_capital=1_000_000)
        results = bt.run()
        assert len(results.equity_curve) == len(trending_up_df)

    def test_bullish_signals_on_uptrend_positive_return(self, trending_up_df, always_bullish_signals):
        bt = Backtester(
            trending_up_df, always_bullish_signals,
            initial_capital=1_000_000, commission_pct=0.001, slippage_pct=0.0005,
        )
        results = bt.run()
        summary = results.summary()
        assert summary["total_return_pct"] > 0

    def test_sideways_signals_zero_trades(self, trending_up_df, always_sideways_signals):
        bt = Backtester(trending_up_df, always_sideways_signals, initial_capital=1_000_000)
        results = bt.run()
        assert results.summary()["total_trades"] == 0

    def test_summary_has_required_keys(self, trending_up_df, always_bullish_signals):
        bt = Backtester(trending_up_df, always_bullish_signals, initial_capital=1_000_000)
        summary = bt.run().summary()
        required_keys = {
            "total_return_pct", "cagr_pct", "sharpe_ratio",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "total_trades", "calmar_ratio",
        }
        assert required_keys.issubset(summary.keys())

    def test_max_drawdown_non_positive(self, trending_up_df, always_bullish_signals):
        bt = Backtester(trending_up_df, always_bullish_signals, initial_capital=1_000_000)
        results = bt.run()
        assert results.summary()["max_drawdown_pct"] <= 0

    def test_sharpe_ratio_finite(self, trending_up_df, always_bullish_signals):
        bt = Backtester(trending_up_df, always_bullish_signals, initial_capital=1_000_000)
        results = bt.run()
        sharpe = results.summary()["sharpe_ratio"]
        import math
        assert math.isfinite(sharpe)

    def test_commission_reduces_returns(self, trending_up_df, always_bullish_signals):
        bt_no_comm = Backtester(
            trending_up_df, always_bullish_signals,
            initial_capital=1_000_000, commission_pct=0.0, slippage_pct=0.0,
        )
        bt_with_comm = Backtester(
            trending_up_df, always_bullish_signals,
            initial_capital=1_000_000, commission_pct=0.005, slippage_pct=0.002,
        )
        r_no = bt_no_comm.run().summary()["total_return_pct"]
        r_with = bt_with_comm.run().summary()["total_return_pct"]
        assert r_no >= r_with
