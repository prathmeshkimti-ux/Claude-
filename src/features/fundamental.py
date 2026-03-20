"""
Fundamental Feature Engineering
==================================
Converts raw fundamental and macro data into model-ready normalised features.

Produces:
  - Value scores         : PE rank, PB rank, etc.
  - Quality scores       : ROE, ROCE, margin stability
  - Growth scores        : revenue/earnings momentum
  - Financial health     : debt ratios, current ratio, FCF yield
  - Macro risk score     : composite of yield curve, dollar, VIX, inflation
  - Institutional signal : FII/DII flow proxy
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """
    Transform fundamental ratios and macro data into a normalised feature set.

    Parameters
    ----------
    fundamentals : dict
        Output of MacroDataFetcher.get_stock_fundamentals(symbol).
    macro_df : pd.DataFrame
        Output of MacroDataFetcher.get_macro_features().
    sector_df : pd.DataFrame
        Output of MacroDataFetcher.get_sector_performance().
    """

    def __init__(
        self,
        fundamentals: Optional[Dict] = None,
        macro_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.fundamentals = fundamentals or {}
        self.macro_df = macro_df if macro_df is not None else pd.DataFrame()
        self.sector_df = sector_df if sector_df is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fundamental_vector(self) -> Dict[str, float]:
        """
        Return a flat dict of fundamental features ready for the feature vector.

        All values are in meaningful units (ratios, %-changes, z-scores).
        """
        result = {}
        result.update(self._value_features())
        result.update(self._quality_features())
        result.update(self._growth_features())
        result.update(self._financial_health_features())
        result.update(self._institutional_features())
        return result

    def get_macro_features_today(self) -> Dict[str, float]:
        """
        Return the latest macro feature values as a flat dict.
        Used to append to each prediction request.
        """
        if self.macro_df.empty:
            return {}
        latest = self.macro_df.iloc[-1].to_dict()
        return {k: (float(v) if pd.notna(v) else 0.0) for k, v in latest.items()}

    def compute_macro_risk_score(self) -> float:
        """
        Aggregate macro indicators into a single risk score [0, 1].

        Higher score = more risk (bad for equities / commodities).
        """
        if self.macro_df.empty:
            return 0.5  # neutral default

        latest = self.macro_df.iloc[-1]
        scores = []

        # VIX: normalise to 0–1 over typical range 10–80
        vix = latest.get("vix", np.nan)
        if pd.notna(vix):
            scores.append(min(max((vix - 10) / 70, 0), 1))

        # Yield curve inversion: negative spread = risk
        yc = latest.get("yield_curve_slope", np.nan)
        if pd.notna(yc):
            # slope near 0 or negative → high risk
            scores.append(max(0, min((-yc + 1) / 3, 1)))

        # Dollar strength: sharply rising dollar = headwind for risk assets
        dxy_chg = latest.get("dollar_index_chg", np.nan)
        if pd.notna(dxy_chg):
            scores.append(min(max(dxy_chg * 10, 0), 1))

        # Inflation: high and rising = risky
        cpi = latest.get("cpi_yoy", np.nan)
        if pd.notna(cpi):
            # 2% target; penalise deviations
            scores.append(min(abs(cpi - 2.0) / 8, 1))

        if not scores:
            return 0.5

        return round(float(np.mean(scores)), 4)

    def compute_sector_momentum_score(self, sector_ticker: Optional[str] = None) -> float:
        """
        Return 1-month sector return as a proxy for sector momentum.
        Positive → tailwind, negative → headwind.
        """
        if self.sector_df.empty or sector_ticker is None:
            return 0.0

        col = f"{sector_ticker}_ret_1m"
        if col not in self.sector_df.columns:
            return 0.0

        latest = self.sector_df[col].iloc[-1]
        return float(latest) if pd.notna(latest) else 0.0

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _value_features(self) -> Dict[str, float]:
        f = self.fundamentals
        return {
            "pe_ratio": self._safe(f.get("pe_ratio")),
            "forward_pe": self._safe(f.get("forward_pe")),
            "peg_ratio": self._safe(f.get("peg_ratio")),
            "price_to_book": self._safe(f.get("price_to_book")),
            "price_to_sales": self._safe(f.get("price_to_sales")),
            "ev_to_sales": self._safe(
                f.get("enterprise_value", np.nan) / (f.get("market_cap", 1) or 1)
            ),
        }

    def _quality_features(self) -> Dict[str, float]:
        f = self.fundamentals
        return {
            "roe": self._safe(f.get("roe")),
            "roce": self._safe(f.get("roce")),
            "profit_margin": self._safe(f.get("profit_margin")),
            "operating_margin": self._safe(f.get("operating_margin")),
            "beta": self._safe(f.get("beta"), default=1.0),
        }

    def _growth_features(self) -> Dict[str, float]:
        f = self.fundamentals
        return {
            "revenue_growth": self._safe(f.get("revenue_growth")),
            "earnings_growth": self._safe(f.get("earnings_growth")),
            "52w_change": self._safe(f.get("52w_change")),
        }

    def _financial_health_features(self) -> Dict[str, float]:
        f = self.fundamentals
        return {
            "debt_to_equity": self._safe(f.get("debt_to_equity")),
            "current_ratio": self._safe(f.get("current_ratio")),
            "dividend_yield": self._safe(f.get("dividend_yield")),
            "short_ratio": self._safe(f.get("short_ratio")),
        }

    def _institutional_features(self) -> Dict[str, float]:
        f = self.fundamentals
        return {
            "institutional_ownership": self._safe(f.get("held_percent_institutions")),
            "insider_ownership": self._safe(f.get("insider_percent_held")),
            # High institutional + low insider → more professional ownership
            "inst_to_insider_ratio": self._safe(
                f.get("held_percent_institutions", 0)
                / (f.get("insider_percent_held", 0.01) or 0.01)
            ),
        }

    @staticmethod
    def _safe(value, default: float = 0.0) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
