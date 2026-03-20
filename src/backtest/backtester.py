"""
Backtesting Framework
======================
Walk-forward backtester that simulates the full prediction pipeline
on historical data and computes professional performance metrics.

Key features:
  - Walk-forward cross-validation (no look-ahead bias)
  - Realistic slippage and commission modelling
  - Per-trade log with entry/exit reasons
  - Summary statistics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate

Usage:
    backtester = Backtester(price_df, feature_df, signal_series)
    results = backtester.run()
    print(results.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: str          # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    size: float             # position size as fraction of portfolio
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Container for all backtest output."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    config: Dict

    def summary(self) -> Dict:
        r = self.returns.dropna()
        eq = self.equity_curve.dropna()

        total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
        n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
        cagr = (1 + total_return) ** (1 / n_years) - 1

        sharpe = self._sharpe(r)
        sortino = self._sortino(r)
        max_dd = self._max_drawdown(eq)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / max(len(self.trades), 1)

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "total_trades": len(self.trades),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(win_rate * avg_win + (1 - win_rate) * avg_loss, 2),
        }

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(r: pd.Series, rf: float = 0.05) -> float:
        if r.std() < 1e-9:
            return 0.0
        excess = r - rf / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))

    @staticmethod
    def _sortino(r: pd.Series, rf: float = 0.05) -> float:
        excess = r - rf / 252
        downside = excess[excess < 0].std()
        if downside < 1e-9:
            return 0.0
        return float(excess.mean() / downside * np.sqrt(252))

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / (rolling_max + 1e-9)
        return float(dd.min())


class Backtester:
    """
    Walk-forward backtester for the price prediction system.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV DataFrame.
    signal_series : pd.Series
        Pre-computed directional signals indexed by date.
        Values: "bullish", "sideways", "bearish"
    confidence_series : pd.Series, optional
        Per-signal confidence scores. Used for position sizing.
    initial_capital : float
    commission_pct : float
    slippage_pct : float
    max_holding_days : int
        Force-close position after this many days.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        signal_series: pd.Series,
        confidence_series: Optional[pd.Series] = None,
        initial_capital: float = None,
        commission_pct: float = None,
        slippage_pct: float = None,
        max_holding_days: int = 5,
    ) -> None:
        cfg = config.backtest
        self.price_df = price_df.copy()
        self.signals = signal_series
        self.confidence = confidence_series
        self.capital = initial_capital or cfg.initial_capital
        self.commission_pct = commission_pct or cfg.commission_pct
        self.slippage_pct = slippage_pct or cfg.slippage_pct
        self.max_holding_days = max_holding_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResults:
        """Execute the backtest and return results."""
        aligned = self.price_df.join(
            self.signals.rename("signal"), how="inner"
        )
        if self.confidence is not None:
            aligned = aligned.join(self.confidence.rename("confidence"), how="left")
            aligned["confidence"] = aligned["confidence"].fillna(0.55)
        else:
            aligned["confidence"] = 0.55

        portfolio_value = self.capital
        equity_curve = {}
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        holding_days = 0

        for date, row in aligned.iterrows():
            close = row["close"]
            signal = row.get("signal", "sideways")
            conf = row.get("confidence", 0.55)

            # Update open position
            if open_trade is not None:
                holding_days += 1
                pnl_pct = (close / open_trade.entry_price - 1)
                if open_trade.direction == "short":
                    pnl_pct = -pnl_pct

                should_exit = (
                    holding_days >= self.max_holding_days
                    or (open_trade.direction == "long" and signal == "bearish")
                    or (open_trade.direction == "short" and signal == "bullish")
                )

                if should_exit:
                    exit_price = close * (1 + self.slippage_pct * (-1 if open_trade.direction == "long" else 1))
                    exit_price = close * (1 - self.slippage_pct)  # sell slippage
                    net_pnl_pct = pnl_pct - self.commission_pct - self.slippage_pct
                    pnl_value = portfolio_value * open_trade.size * net_pnl_pct

                    open_trade.exit_date = date
                    open_trade.exit_price = exit_price
                    open_trade.pnl = pnl_value
                    open_trade.return_pct = net_pnl_pct
                    open_trade.exit_reason = "signal_flip" if holding_days < self.max_holding_days else "max_hold"

                    portfolio_value += pnl_value
                    trades.append(open_trade)
                    open_trade = None
                    holding_days = 0

            # Open new position
            if open_trade is None and signal in ("bullish", "bearish") and conf >= 0.55:
                direction = "long" if signal == "bullish" else "short"
                size = min(conf * 0.5, config.risk.max_position_pct)  # size proportional to confidence
                entry_price = close * (1 + self.slippage_pct if direction == "long" else 1 - self.slippage_pct)

                open_trade = Trade(
                    entry_date=date,
                    exit_date=None,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=None,
                    size=size,
                )
                holding_days = 0

            equity_curve[date] = portfolio_value

        # Close any open position at end
        if open_trade is not None and not aligned.empty:
            last_close = aligned["close"].iloc[-1]
            pnl_pct = (last_close / open_trade.entry_price - 1)
            if open_trade.direction == "short":
                pnl_pct = -pnl_pct
            pnl_value = portfolio_value * open_trade.size * (pnl_pct - self.commission_pct)
            open_trade.exit_date = aligned.index[-1]
            open_trade.exit_price = last_close
            open_trade.pnl = pnl_value
            open_trade.return_pct = pnl_pct - self.commission_pct
            open_trade.exit_reason = "end_of_data"
            portfolio_value += pnl_value
            trades.append(open_trade)

        eq_series = pd.Series(equity_curve)
        ret_series = eq_series.pct_change().dropna()

        logger.info(
            "Backtest complete: %d trades | Final equity: %.2f",
            len(trades), portfolio_value,
        )

        return BacktestResults(
            trades=trades,
            equity_curve=eq_series,
            returns=ret_series,
            config={
                "initial_capital": self.capital,
                "commission_pct": self.commission_pct,
                "slippage_pct": self.slippage_pct,
                "max_holding_days": self.max_holding_days,
            },
        )

    # ------------------------------------------------------------------
    # Walk-forward helper
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        n_splits: int = 10,
        min_train_bars: int = 252,
    ) -> List[Dict]:
        """
        Run walk-forward validation across n_splits periods.

        Returns a list of summary dicts, one per period.
        """
        results = []
        n = len(self.price_df)
        step = max((n - min_train_bars) // n_splits, 21)

        for i in range(n_splits):
            test_start = min_train_bars + i * step
            test_end = min(test_start + step, n)

            test_prices = self.price_df.iloc[test_start:test_end]
            test_signals = self.signals.iloc[test_start:test_end]
            test_conf = (
                self.confidence.iloc[test_start:test_end]
                if self.confidence is not None else None
            )

            bt = Backtester(
                price_df=test_prices,
                signal_series=test_signals,
                confidence_series=test_conf,
                initial_capital=self.capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct,
            )
            period_results = bt.run().summary()
            period_results["period"] = i + 1
            period_results["start_date"] = str(test_prices.index[0].date())
            period_results["end_date"] = str(test_prices.index[-1].date())
            results.append(period_results)

        return results
