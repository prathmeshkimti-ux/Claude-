"""
Feature Engineering Pipeline
==============================
Orchestrates all feature sources into a single, model-ready feature matrix.

Steps:
  1. Compute technical indicators from OHLCV data.
  2. Merge macro time-series features (aligned by date).
  3. Append scalar fundamental features (broadcast to all rows).
  4. Append scalar sentiment features (broadcast or daily joined).
  5. Append commodity-specific features (if applicable).
  6. Add target labels for supervised learning.
  7. Normalise and handle missing values.
  8. Return (X, y) split.

The pipeline is stateless: fit a scaler externally when training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import config
from .technical import TechnicalFeatures
from .fundamental import FundamentalFeatures
from .sentiment import SentimentFeatures

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Full-stack feature engineering for the price prediction system.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV DataFrame from MarketDataFetcher.
    macro_df : pd.DataFrame, optional
        Macro time-series from MacroDataFetcher.get_macro_features().
    fundamentals : dict, optional
        Scalar fundamental dict from MacroDataFetcher.get_stock_fundamentals().
    news_sentiment : dict, optional
        Aggregated news sentiment from NewsDataFetcher.get_aggregated_sentiment().
    social_sentiment : dict, optional
        Social sentiment from SocialDataFetcher.get_social_sentiment().
    commodity_df : pd.DataFrame, optional
        Commodity feature frame from CommodityDataFetcher.get_features().
    horizons : list[int], optional
        Prediction horizons in days. Defaults to config value.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict] = None,
        news_sentiment: Optional[Dict] = None,
        social_sentiment: Optional[Dict] = None,
        commodity_df: Optional[pd.DataFrame] = None,
        horizons: Optional[List[int]] = None,
    ) -> None:
        self.price_df = price_df.copy()
        self.macro_df = macro_df
        self.fundamentals = fundamentals or {}
        self.news_sentiment = news_sentiment or {}
        self.social_sentiment = social_sentiment or {}
        self.commodity_df = commodity_df
        self.horizons = horizons or config.model.horizons
        self._scaler: Optional[RobustScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(self, scale: bool = True) -> pd.DataFrame:
        """
        Build the complete feature matrix (no target labels).

        Parameters
        ----------
        scale : bool
            If True, apply RobustScaler. The fitted scaler is stored in
            self._scaler for use during inference.
        """
        df = self._technical_layer()
        df = self._merge_macro(df)
        df = self._append_scalars(df)
        df = self._final_cleanup(df)

        if scale:
            df, self._scaler = self._scale(df)

        return df

    def build_supervised_dataset(
        self, horizon: int = 3, scale: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build (X, y) for a supervised classification task.

        Target y:
            1  = bullish  (close[t+horizon] / close[t] - 1 > +threshold)
            0  = sideways
           -1  = bearish  (return < -threshold)

        Parameters
        ----------
        horizon : int
            Prediction horizon in trading days.
        scale : bool
            Whether to normalise X.
        """
        X = self.build_feature_matrix(scale=scale)
        y = self._make_target(horizon)

        # Align on common index (drop NaN rows from both)
        common_idx = X.index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        logger.info(
            "Dataset built: %d samples, %d features, horizon=%d days",
            len(X), X.shape[1], horizon,
        )
        return X, y

    def get_latest_feature_vector(self, scale: bool = True) -> pd.Series:
        """
        Return the most recent feature row for live inference.
        """
        X = self.build_feature_matrix(scale=scale)
        return X.iloc[-1]

    def get_scaler(self) -> Optional[RobustScaler]:
        """Return the fitted RobustScaler (for persistence / inference)."""
        return self._scaler

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _technical_layer(self) -> pd.DataFrame:
        """Compute all technical features from price data."""
        tech = TechnicalFeatures(self.price_df)
        features = tech.compute_all()

        # Include raw OHLCV + derived microstructure cols from MarketDataFetcher
        passthrough_cols = [
            "close", "volume", "vwap", "spread_proxy",
            "returns", "log_returns", "volume_zscore",
            "roc_5", "roc_10", "roc_20",
            "pct_from_52w_high", "pct_from_52w_low",
        ]
        for col in passthrough_cols:
            if col in self.price_df.columns and col not in features.columns:
                features[col] = self.price_df[col]

        return features

    def _merge_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Left-join macro time-series features onto the price index."""
        if self.macro_df is None or self.macro_df.empty:
            return df

        macro = self.macro_df.reindex(df.index, method="ffill")
        merged = pd.concat([df, macro], axis=1)
        return merged

    def _append_scalars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Broadcast fundamental + sentiment scalar features to all rows."""
        # Fundamental vector
        fund = FundamentalFeatures(
            fundamentals=self.fundamentals,
            macro_df=self.macro_df,
        )
        fund_vec = fund.get_fundamental_vector()
        macro_risk = fund.compute_macro_risk_score()
        for key, val in fund_vec.items():
            df[f"fund_{key}"] = val
        df["macro_risk_score"] = macro_risk

        # Sentiment vector
        sent = SentimentFeatures(
            news_sentiment=self.news_sentiment,
            social_sentiment=self.social_sentiment,
        )
        sent_vec = sent.get_sentiment_vector()
        for key, val in sent_vec.items():
            df[f"sent_{key}"] = val

        # Commodity features (time-series join)
        if self.commodity_df is not None and not self.commodity_df.empty:
            comm = self.commodity_df.reindex(df.index, method="ffill")
            # Prefix columns to avoid collisions
            comm.columns = [f"comm_{c}" for c in comm.columns]
            df = pd.concat([df, comm], axis=1)

        return df

    @staticmethod
    def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows / columns with excessive NaN, fill remaining gaps.
        """
        # Drop columns that are >80% NaN
        thresh = int(0.2 * len(df))
        df = df.dropna(axis=1, thresh=thresh)

        # Forward-fill then backward-fill residual NaN
        df = df.ffill().bfill()

        # Drop rows still containing NaN (typically early warm-up rows)
        df = df.dropna()

        return df

    @staticmethod
    def _scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, RobustScaler]:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

    def _make_target(self, horizon: int) -> pd.Series:
        """Compute multi-class directional label for each row."""
        close = self.price_df["close"]
        fwd_return = close.shift(-horizon) / close - 1.0

        threshold = config.model.bullish_threshold
        y = pd.Series(0, index=fwd_return.index, name=f"target_{horizon}d")
        y[fwd_return > threshold] = 1
        y[fwd_return < -threshold] = -1
        return y
