"""
Market Data Fetcher
===================
Fetches OHLCV price data, order book snapshots, and derived microstructure
metrics from yfinance (and optionally Alpha Vantage for intraday tick data).

Outputs a clean, timezone-aware pandas DataFrame ready for feature engineering.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import config

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches and pre-processes market data for one or more symbols.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols, e.g. ["AAPL", "RELIANCE.NS", "CL=F"]
    interval : str
        Data granularity ("1d", "1h", …). Defaults to config value.
    period : str
        Look-back window ("5y", "2y", …). Ignored when start/end are given.
    start : str | None
        ISO start date "YYYY-MM-DD". Overrides *period*.
    end : str | None
        ISO end date   "YYYY-MM-DD". Defaults to today.
    """

    def __init__(
        self,
        symbols: List[str],
        interval: str = None,
        period: str = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval or config.data.default_interval
        self.period = period or config.data.default_period
        self.start = start
        self.end = end or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Return OHLCV + derived microstructure columns for *symbol*."""
        symbol = symbol.upper()
        if symbol in self._cache:
            return self._cache[symbol]

        df = self._download(symbol)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return df

        df = self._clean(df)
        df = self._add_microstructure(df)
        self._cache[symbol] = df
        logger.info("Fetched %d rows for %s (%s)", len(df), symbol, self.interval)
        return df

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols."""
        return {sym: self.fetch(sym) for sym in self.symbols}

    def get_latest(self, symbol: str) -> pd.Series:
        """Return the most recent bar as a Series."""
        df = self.fetch(symbol)
        return df.iloc[-1] if not df.empty else pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download(self, symbol: str) -> pd.DataFrame:
        """Download raw OHLCV from yfinance."""
        kwargs: Dict = dict(
            tickers=symbol,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )
        if self.start:
            kwargs["start"] = self.start
            kwargs["end"] = self.end
        else:
            kwargs["period"] = self.period

        try:
            df = yf.download(**kwargs, multi_level_index=False)
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitise the raw OHLCV frame."""
        df = df.copy()
        df.dropna(subset=["close"], inplace=True)
        df.sort_index(inplace=True)

        # Ensure all expected columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        # Remove clearly erroneous bars (zero price / volume)
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]
        return df

    @staticmethod
    def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        """Derive VWAP, typical price, and order-flow proxy columns."""
        df = df.copy()

        # Typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        # VWAP (rolling 20-bar approximation when tick data unavailable)
        df["vwap"] = (
            (df["typical_price"] * df["volume"]).rolling(20).sum()
            / df["volume"].rolling(20).sum()
        )

        # Intrabar spread proxy  (high - low) / close
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

        # Daily returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volume z-score (relative volume)
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

        # Price momentum (rate of change)
        for p in [5, 10, 20]:
            df[f"roc_{p}"] = df["close"].pct_change(p)

        # High/Low breakout flags
        df["52w_high"] = df["high"].rolling(252).max()
        df["52w_low"] = df["low"].rolling(252).min()
        df["pct_from_52w_high"] = (df["close"] - df["52w_high"]) / df["52w_high"]
        df["pct_from_52w_low"] = (df["close"] - df["52w_low"]) / df["52w_low"]

        return df

    # ------------------------------------------------------------------
    # Order book snapshot (stub – replace with live WebSocket feed)
    # ------------------------------------------------------------------

    @staticmethod
    def mock_order_book(
        mid_price: float,
        levels: int = 5,
        spread_bps: float = 5.0,
    ) -> Dict:
        """
        Generate a synthetic order-book for testing.

        In production replace with:
            - Zerodha WebSocket (Indian markets)
            - Interactive Brokers TWS API
            - Coinbase / Binance WebSocket (crypto)
        """
        half_spread = mid_price * spread_bps / 10_000
        bids, asks = [], []
        for i in range(levels):
            bid_px = mid_price - half_spread * (i + 1)
            ask_px = mid_price + half_spread * (i + 1)
            size = np.random.randint(100, 2000) * (levels - i)
            bids.append({"price": round(bid_px, 2), "size": size})
            asks.append({"price": round(ask_px, 2), "size": size})

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        total_bid_vol = sum(b["size"] for b in bids)
        total_ask_vol = sum(a["size"] for a in asks)

        return {
            "bids": bids,
            "asks": asks,
            "spread": best_ask - best_bid,
            "mid": (best_bid + best_ask) / 2,
            "order_flow_imbalance": (total_bid_vol - total_ask_vol)
            / (total_bid_vol + total_ask_vol + 1e-9),
        }
