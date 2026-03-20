"""
Macro & Fundamental Data Fetcher
=================================
Pulls macroeconomic and fundamental data from multiple sources.

Sources used (free / publicly available):
  - FRED via pandas-datareader  (interest rates, CPI, DXY proxy)
  - yfinance                    (fundamentals for individual stocks)
  - World Bank API              (GDP, current account – slower, cached)
  - Quandl / Nasdaq Data Link   (optional, requires API key)

All methods return tidy pandas DataFrames indexed by date.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free FRED series IDs that are relevant to equity / commodity prediction
# ---------------------------------------------------------------------------
FRED_SERIES: Dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "10y_yield": "DGS10",
    "2y_yield": "DGS2",
    "yield_curve_10_2": None,          # computed
    "cpi_yoy": "CPIAUCSL",
    "unemployment": "UNRATE",
    "industrial_production": "INDPRO",
    "ism_manufacturing": "MANEMP",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",         # broad dollar index
}


class MacroDataFetcher:
    """
    Provides macroeconomic features aligned to a common date index.

    Parameters
    ----------
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date   "YYYY-MM-DD".
    use_fred : bool
        Whether to attempt fetching from FRED (requires internet).
    """

    def __init__(
        self,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        use_fred: bool = True,
    ) -> None:
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.use_fred = use_fred
        self._cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_macro_features(self) -> pd.DataFrame:
        """
        Return a single DataFrame with all macro features forward-filled to
        business-daily frequency.
        """
        frames: List[pd.DataFrame] = []

        if self.use_fred:
            fred_df = self._fetch_fred()
            if not fred_df.empty:
                frames.append(fred_df)

        # Fallback / supplement with yfinance proxies
        yf_macro = self._fetch_yfinance_macro()
        if not yf_macro.empty:
            frames.append(yf_macro)

        if not frames:
            logger.warning("No macro data fetched; returning empty frame.")
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        df = df.resample("B").last().ffill()
        df = df.loc[self.start : self.end]

        # Derived features
        if "10y_yield" in df.columns and "2y_yield" in df.columns:
            df["yield_curve_slope"] = df["10y_yield"] - df["2y_yield"]

        if "cpi_yoy" in df.columns:
            df["cpi_mom"] = df["cpi_yoy"].pct_change()
            df["real_rate"] = df.get("10y_yield", 0) - df["cpi_yoy"]

        if "dollar_index" in df.columns:
            df["dollar_index_chg"] = df["dollar_index"].pct_change()

        return df.dropna(how="all")

    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch key fundamental ratios for a stock via yfinance.

        Returns a flat dict suitable for merging with the feature vector.
        """
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.info
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
            return {}

        fields = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "roe": info.get("returnOnEquity"),
            "roce": info.get("returnOnAssets"),          # proxy
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "free_cash_flow": info.get("freeCashflow"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "short_ratio": info.get("shortRatio"),
            "52w_change": info.get("52WeekChange"),
            "held_percent_institutions": info.get("heldPercentInstitutions"),
            "insider_percent_held": info.get("heldPercentInsiders"),
        }

        return {k: v for k, v in fields.items() if v is not None}

    def get_sector_performance(self, sector_etfs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return rolling sector performance using ETF proxies.

        Default ETFs cover major SPDR US sectors.
        """
        if sector_etfs is None:
            sector_etfs = [
                "XLK",   # Technology
                "XLF",   # Financials
                "XLE",   # Energy
                "XLV",   # Health Care
                "XLI",   # Industrials
                "XLY",   # Consumer Discretionary
                "XLP",   # Consumer Staples
                "XLU",   # Utilities
                "XLB",   # Materials
                "XLRE",  # Real Estate
            ]

        sector_data = yf.download(
            sector_etfs,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        if sector_data.empty:
            return pd.DataFrame()

        try:
            close = sector_data["Close"]
        except KeyError:
            return pd.DataFrame()

        perf = pd.DataFrame(index=close.index)
        for etf in close.columns:
            perf[f"{etf}_ret_1m"] = close[etf].pct_change(21)
            perf[f"{etf}_ret_3m"] = close[etf].pct_change(63)

        return perf.ffill().dropna(how="all")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_fred(self) -> pd.DataFrame:
        """Download FRED series via pandas-datareader."""
        try:
            import pandas_datareader as pdr  # optional dependency
        except ImportError:
            logger.info("pandas-datareader not installed; skipping FRED fetch.")
            return pd.DataFrame()

        frames = {}
        series_map = {
            k: v for k, v in FRED_SERIES.items() if v is not None
        }
        for name, fred_id in series_map.items():
            try:
                s = pdr.DataReader(fred_id, "fred", self.start, self.end)
                frames[name] = s.squeeze()
            except Exception as exc:
                logger.debug("FRED fetch failed for %s (%s): %s", name, fred_id, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        return df

    def _fetch_yfinance_macro(self) -> pd.DataFrame:
        """
        Use yfinance ETF/index proxies for key macro series.

        Tickers:
            ^TNX  → 10-year US Treasury yield
            ^IRX  → 3-month T-bill
            DX-Y.NYB / UUP → US Dollar Index proxy
            ^VIX  → CBOE Volatility Index
            GC=F  → Gold futures
            CL=F  → Crude oil
        """
        proxies = {
            "tnx_yield": "^TNX",
            "irx_yield": "^IRX",
            "vix": "^VIX",
            "gold": "GC=F",
            "crude_oil": "CL=F",
            "usd_index": "UUP",       # PowerShares DB USD Bull ETF
            "sp500": "^GSPC",
        }

        frames = {}
        for name, ticker in proxies.items():
            try:
                df = yf.download(
                    ticker,
                    start=self.start,
                    end=self.end,
                    auto_adjust=True,
                    progress=False,
                    multi_level_index=False,
                )
                if not df.empty and "Close" in df.columns:
                    frames[name] = df["Close"].rename(name)
            except Exception as exc:
                logger.debug("yfinance macro fetch failed for %s: %s", name, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames.values(), axis=1)
        df.index = pd.to_datetime(df.index)
        return df
