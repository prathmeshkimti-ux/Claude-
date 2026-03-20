"""
Commodity-Specific Data Fetcher
================================
Collects supply/demand/macro factors critical for commodity price prediction.

Covers:
  - Crude oil (WTI / Brent): EIA inventory, OPEC production, rig count
  - Gold / Silver: Dollar index, real rates, ETF flows
  - Natural gas: storage, weather forecasts
  - Agricultural: crop reports (stub)

Primary sources:
  - EIA (US Energy Information Administration) — free JSON API
  - yfinance                                   — futures prices
  - FRED / pandas-datareader                   — macro proxies

All methods return pandas DataFrames indexed by date.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from config import config

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"

# EIA series IDs for key petroleum metrics
EIA_SERIES = {
    "us_crude_inventory": "PET.WCRSTUS1.W",          # weekly crude stocks (mbbl)
    "us_crude_production": "PET.WCRFPUS2.W",          # weekly production (mbbl/d)
    "us_gasoline_inventory": "PET.WGTSTUS1.W",
    "us_crude_imports": "PET.WCRIMUS2.W",
    "us_refinery_utilisation": "PET.WPULEUS3.W",
}


class CommodityDataFetcher:
    """
    Fetches supply, demand, and macro data for commodity price prediction.

    Parameters
    ----------
    commodity : str
        One of "crude_oil", "gold", "natural_gas", "silver".
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date   "YYYY-MM-DD".
    eia_api_key : str | None
        EIA API key (free registration at eia.gov). Falls back to mock data.
    """

    def __init__(
        self,
        commodity: str = "crude_oil",
        start: str = "2015-01-01",
        end: Optional[str] = None,
        eia_api_key: Optional[str] = None,
    ) -> None:
        self.commodity = commodity.lower()
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.eia_api_key = eia_api_key or config.api.quandl_api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self) -> pd.DataFrame:
        """
        Return a combined feature DataFrame for the selected commodity.
        Automatically routes to the appropriate commodity-specific method.
        """
        if self.commodity in ("crude_oil", "wti", "brent"):
            return self._crude_oil_features()
        elif self.commodity == "gold":
            return self._gold_features()
        elif self.commodity in ("natural_gas", "natgas"):
            return self._natgas_features()
        elif self.commodity == "silver":
            return self._silver_features()
        else:
            logger.warning("Unknown commodity '%s'; returning empty features.", self.commodity)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Crude Oil
    # ------------------------------------------------------------------

    def _crude_oil_features(self) -> pd.DataFrame:
        frames = []

        # Price data
        price_df = self._fetch_price_data(["CL=F", "BZ=F"], ["wti_price", "brent_price"])
        if not price_df.empty:
            frames.append(price_df)

        # EIA Inventory
        eia_df = self._fetch_eia_inventory()
        if not eia_df.empty:
            frames.append(eia_df)

        # Baker Hughes rig count proxy (yfinance doesn't have it; stub)
        rig_df = self._mock_rig_count()
        frames.append(rig_df)

        # Macro factors
        macro_df = self._fetch_price_data(
            ["UUP", "TIP", "^TNX"],
            ["dollar_index", "inflation_etf", "10y_yield"],
        )
        if not macro_df.empty:
            frames.append(macro_df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        df = df.resample("B").last().ffill()
        df = df.loc[self.start : self.end]

        # Derived features
        if "wti_price" in df.columns and "brent_price" in df.columns:
            df["wti_brent_spread"] = df["wti_price"] - df["brent_price"]

        if "us_crude_inventory" in df.columns:
            df["inventory_chg"] = df["us_crude_inventory"].diff()
            df["inventory_chg_4w"] = df["us_crude_inventory"].diff(4)
            # Surprise vs 4-week rolling average
            inv_mean = df["us_crude_inventory"].rolling(4).mean()
            df["inventory_surprise"] = (df["us_crude_inventory"] - inv_mean) / (inv_mean + 1e-9)

        return df.dropna(how="all")

    # ------------------------------------------------------------------
    # Gold
    # ------------------------------------------------------------------

    def _gold_features(self) -> pd.DataFrame:
        frames = []

        price_df = self._fetch_price_data(
            ["GC=F", "SI=F", "GLD"],
            ["gold_price", "silver_price", "gold_etf"],
        )
        if not price_df.empty:
            frames.append(price_df)

        # Key gold drivers
        drivers_df = self._fetch_price_data(
            ["UUP", "TIP", "^IRX", "^TNX", "^VIX"],
            ["dollar_index", "tips_etf", "3m_yield", "10y_yield", "vix"],
        )
        if not drivers_df.empty:
            frames.append(drivers_df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).resample("B").last().ffill()
        df = df.loc[self.start : self.end]

        # Gold-specific derived features
        if "10y_yield" in df.columns and "tips_etf" in df.columns:
            # Real rate proxy: 10Y yield minus breakeven inflation
            df["real_rate_proxy"] = df["10y_yield"] - df["tips_etf"].pct_change(252) * 100

        if "gold_price" in df.columns and "silver_price" in df.columns:
            df["gold_silver_ratio"] = df["gold_price"] / (df["silver_price"] + 1e-9)

        return df.dropna(how="all")

    # ------------------------------------------------------------------
    # Natural Gas
    # ------------------------------------------------------------------

    def _natgas_features(self) -> pd.DataFrame:
        price_df = self._fetch_price_data(["NG=F"], ["natgas_price"])
        macro_df = self._fetch_price_data(
            ["UUP", "^TNX"],
            ["dollar_index", "10y_yield"],
        )

        frames = [f for f in [price_df, macro_df] if not f.empty]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).resample("B").last().ffill()
        return df.loc[self.start : self.end].dropna(how="all")

    # ------------------------------------------------------------------
    # Silver
    # ------------------------------------------------------------------

    def _silver_features(self) -> pd.DataFrame:
        price_df = self._fetch_price_data(
            ["SI=F", "GC=F", "SLV"],
            ["silver_price", "gold_price", "silver_etf"],
        )
        macro_df = self._fetch_price_data(
            ["UUP", "^TNX", "^VIX"],
            ["dollar_index", "10y_yield", "vix"],
        )
        frames = [f for f in [price_df, macro_df] if not f.empty]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).resample("B").last().ffill()
        df = df.loc[self.start : self.end]

        if "gold_price" in df.columns and "silver_price" in df.columns:
            df["gold_silver_ratio"] = df["gold_price"] / (df["silver_price"] + 1e-9)

        return df.dropna(how="all")

    # ------------------------------------------------------------------
    # EIA fetcher
    # ------------------------------------------------------------------

    def _fetch_eia_inventory(self) -> pd.DataFrame:
        """Fetch EIA weekly petroleum inventory data."""
        if not self.eia_api_key:
            logger.info("EIA API key not set; using mock inventory data.")
            return self._mock_eia_inventory()

        frames = {}
        for name, series_id in EIA_SERIES.items():
            url = f"{EIA_BASE_URL}/seriesid/{series_id}"
            params = {"api_key": self.eia_api_key, "frequency": "weekly"}
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("response", {}).get("data", [])
                if rows:
                    series = pd.Series(
                        {r["period"]: r["value"] for r in rows},
                        name=name,
                    )
                    series.index = pd.to_datetime(series.index)
                    series = pd.to_numeric(series, errors="coerce")
                    frames[name] = series
            except Exception as exc:
                logger.debug("EIA fetch failed for %s: %s", name, exc)

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).sort_index()

    def _mock_eia_inventory(self) -> pd.DataFrame:
        """Generate synthetic EIA-style weekly inventory data."""
        import numpy as np
        dates = pd.bdate_range(self.start, self.end, freq="W-FRI")
        rng = pd.Series(range(len(dates)))
        return pd.DataFrame(
            {
                "us_crude_inventory": 430_000 + (rng - rng.mean()) * 200
                + pd.Series(range(len(dates))).apply(lambda _: pd.np.random.normal(0, 2000) if hasattr(pd, 'np') else 0),
            },
            index=dates,
        )

    # ------------------------------------------------------------------
    # Baker Hughes rig count (stub)
    # ------------------------------------------------------------------

    def _mock_rig_count(self) -> pd.DataFrame:
        """
        Synthetic rig count data.
        Production: fetch from Baker Hughes weekly PDF / CSV at:
        https://rigcount.bakerhughes.com/na-rig-count
        """
        import numpy as np
        dates = pd.bdate_range(self.start, self.end, freq="W-FRI")
        counts = 500 + 50 * np.sin(2 * 3.14159 * pd.Series(range(len(dates))) / 52)
        return pd.DataFrame({"rig_count": counts.values}, index=dates)

    # ------------------------------------------------------------------
    # Generic price fetcher
    # ------------------------------------------------------------------

    def _fetch_price_data(
        self, tickers: List[str], names: List[str]
    ) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if raw.empty:
                return pd.DataFrame()

            if len(tickers) == 1:
                close = raw[["Close"]].rename(columns={"Close": names[0]})
            else:
                if "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"]
                else:
                    close = raw.filter(like="Close")
                close.columns = names[: len(close.columns)]

            close.index = pd.to_datetime(close.index)
            return close
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", tickers, exc)
            return pd.DataFrame()
