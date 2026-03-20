"""
PricePredictor — Main Prediction Orchestrator
===============================================
Ties together the entire pipeline:

  Data ingestion → Feature engineering → Model inference → Ensemble → Risk → Output

This is the single public entry point for generating a prediction.

Usage:
    predictor = PricePredictor(symbol="AAPL", horizon=3)
    result = predictor.predict()
    # result contains probabilities, signal, confidence, risk report, etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from src.data.market_data import MarketDataFetcher
from src.data.macro_data import MacroDataFetcher
from src.data.news_data import NewsDataFetcher
from src.data.social_data import SocialDataFetcher
from src.data.commodity_data import CommodityDataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.statistical import StatisticalModel
from src.models.ml_models import MLModel
from src.models.deep_learning import LSTMModel, TransformerModel
from src.models.sentiment_model import FinBERTSentiment
from src.models.ensemble import EnsembleModel
from src.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    End-to-end price prediction pipeline.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL", "RELIANCE.NS", "CL=F").
    horizon : int
        Prediction horizon in trading days.
    asset_type : str
        "equity", "commodity", or "index". Affects which data sources are used.
    train_years : int
        How many years of historical data to use for training.
    use_deep_learning : bool
        Whether to include LSTM / Transformer models (slower).
    use_finbert : bool
        Whether to use FinBERT for news sentiment (requires transformers).
    upcoming_events : list[dict], optional
        Scheduled events for risk engine, e.g. earnings releases.
    """

    def __init__(
        self,
        symbol: str,
        horizon: int = 3,
        asset_type: str = "equity",
        train_years: int = 5,
        use_deep_learning: bool = False,
        use_finbert: bool = False,
        upcoming_events: Optional[List[Dict]] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.horizon = horizon
        self.asset_type = asset_type.lower()
        self.train_years = train_years
        self.use_deep_learning = use_deep_learning
        self.use_finbert = use_finbert
        self.upcoming_events = upcoming_events or []

        # Internal state
        self._price_df: Optional[pd.DataFrame] = None
        self._feature_df: Optional[pd.DataFrame] = None
        self._ml_model: Optional[MLModel] = None
        self._lstm_model: Optional[LSTMModel] = None
        self._transformer_model: Optional[TransformerModel] = None
        self._stat_model: Optional[StatisticalModel] = None
        self._ensemble = EnsembleModel()
        self._risk_engine = RiskEngine()
        self._finbert = FinBERTSentiment() if use_finbert else None
        self._trained = False

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def train(self) -> "PricePredictor":
        """
        Download data, engineer features, and train all models.
        Must be called before predict().
        """
        logger.info("Training pipeline for %s (horizon=%dd)", self.symbol, self.horizon)

        # 1. Fetch data
        price_df = self._fetch_price_data()
        macro_df = self._fetch_macro_data()
        fundamentals = self._fetch_fundamentals()
        news_sent = self._fetch_news_sentiment()
        social_sent = self._fetch_social_sentiment()
        commodity_df = self._fetch_commodity_data()

        # 2. Feature engineering
        eng = FeatureEngineer(
            price_df=price_df,
            macro_df=macro_df,
            fundamentals=fundamentals,
            news_sentiment=news_sent,
            social_sentiment=social_sent,
            commodity_df=commodity_df,
            horizons=[self.horizon],
        )
        X, y = eng.build_supervised_dataset(horizon=self.horizon, scale=True)

        # Cache
        self._price_df = price_df
        self._feature_df = X
        self._scaler = eng.get_scaler()
        self._eng = eng

        # 3. Statistical model (trained on returns)
        returns = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
        self._stat_model = StatisticalModel(horizon=self.horizon)
        self._stat_model.fit(returns)

        # 4. ML model
        self._ml_model = MLModel()
        self._ml_model.fit(X, y)

        # 5. Deep learning models (optional)
        if self.use_deep_learning and len(X) > config.model.lstm_lookback + 50:
            n_features = X.shape[1]
            self._lstm_model = LSTMModel(n_features=n_features, horizon=self.horizon)
            self._lstm_model.fit(X, y)

            self._transformer_model = TransformerModel(n_features=n_features, horizon=self.horizon)
            self._transformer_model.fit(X, y)

        self._trained = True
        logger.info("Training complete for %s.", self.symbol)
        return self

    def predict(self) -> Dict:
        """
        Generate a probability distribution over directional outcomes.

        Returns
        -------
        dict with keys:
            symbol, horizon, timestamp,
            probabilities, signal, confidence,
            risk_approved, position_size,
            risk_report, model_breakdown,
            feature_snapshot (top features)
        """
        if not self._trained:
            logger.info("Models not trained; calling train() first.")
            self.train()

        # Refresh latest data (light-weight re-fetch for live use)
        price_df = self._fetch_price_data(period="3mo")
        news_sent = self._fetch_news_sentiment()
        social_sent = self._fetch_social_sentiment()

        # Re-engineer features for inference
        eng = FeatureEngineer(
            price_df=price_df,
            macro_df=self._fetch_macro_data(start="2023-01-01"),
            fundamentals=self._fetch_fundamentals(),
            news_sentiment=news_sent,
            social_sentiment=social_sent,
            horizons=[self.horizon],
        )
        X_live = eng.build_feature_matrix(scale=True)
        latest_row = X_live.iloc[[-1]]  # keep as DataFrame for predict_proba

        # Collect per-model probabilities
        stat_proba = self._stat_model.predict_proba() if self._stat_model else None
        ml_proba = self._ml_model.predict_proba(latest_row) if self._ml_model else None
        lstm_proba = self._lstm_model.predict_proba(X_live) if self._lstm_model else None
        tf_proba = self._transformer_model.predict_proba(X_live) if self._transformer_model else None

        # FinBERT sentiment proba
        sent_proba = None
        if self._finbert is not None:
            news_df = NewsDataFetcher(symbols=[self.symbol]).get_news(self.symbol)
            headlines = list(news_df["title"].head(20)) if not news_df.empty else []
            if headlines:
                sent_proba = self._finbert.get_directional_proba(headlines)

        # Macro risk score
        from src.features.fundamental import FundamentalFeatures
        fund = FundamentalFeatures(macro_df=self._fetch_macro_data(start="2023-01-01"))
        macro_risk = fund.compute_macro_risk_score()

        # Ensemble
        ensemble_result = self._ensemble.predict(
            statistical_proba=stat_proba,
            ml_proba=ml_proba,
            lstm_proba=lstm_proba,
            transformer_proba=tf_proba,
            sentiment_proba=sent_proba,
            macro_risk_score=macro_risk,
        )

        # Risk engine
        risk_result = self._risk_engine.evaluate(
            prediction=ensemble_result,
            price_df=price_df.tail(20),
            upcoming_events=self.upcoming_events,
        )

        # Feature snapshot (top 10 most important)
        feature_snapshot = {}
        if self._ml_model and self._ml_model.feature_importances() is not None:
            fi = self._ml_model.feature_importances()
            if not fi.empty:
                top = fi.head(10)
                feature_snapshot = dict(zip(top["feature"], top["importance"].round(4)))

        return {
            "symbol": self.symbol,
            "horizon_days": self.horizon,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "asset_type": self.asset_type,

            # Core prediction
            "probabilities": risk_result["adjusted_proba"],
            "raw_probabilities": ensemble_result["probabilities"],
            "signal": risk_result["signal"],
            "confidence": ensemble_result["confidence"],

            # Risk
            "risk_approved": risk_result["approved"],
            "position_size_pct": risk_result["position_size"] * 100,
            "risk_report": risk_result["risk_report"],

            # Interpretability
            "model_breakdown": ensemble_result["model_breakdown"],
            "macro_risk_score": macro_risk,
            "feature_snapshot": feature_snapshot,

            # Market context
            "latest_price": float(price_df["close"].iloc[-1]),
            "realised_vol_20d": float(
                np.log(price_df["close"] / price_df["close"].shift(1))
                .tail(20).std() * np.sqrt(252)
            ),
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def interpret_prediction(self, result: Dict) -> str:
        """Return a human-readable interpretation of the prediction."""
        proba = result["probabilities"]
        signal = result["signal"]
        conf = result["confidence"]
        approved = result["risk_approved"]
        size = result["position_size_pct"]

        lines = [
            f"=== {self.symbol} | {self.horizon}-Day Outlook ===",
            f"Signal  : {signal.upper()}  (confidence: {conf:.1%})",
            f"Proba   : Bullish {proba['bullish']:.1%} | "
            f"Sideways {proba['sideways']:.1%} | "
            f"Bearish  {proba['bearish']:.1%}",
            f"Risk OK : {'YES' if approved else 'NO - SIGNAL BLOCKED'}",
            f"Size    : {size:.1f}% of portfolio",
            f"Vol     : {result['realised_vol_20d']:.1%} annualised",
        ]

        breakdown = result.get("model_breakdown", {})
        if breakdown:
            lines.append("\nModel breakdown:")
            for model_name, p in breakdown.items():
                lines.append(
                    f"  {model_name:15s}: Bull {p['bullish']:.1%} | "
                    f"Side {p['sideways']:.1%} | Bear {p['bearish']:.1%}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data fetch helpers (private)
    # ------------------------------------------------------------------

    def _fetch_price_data(self, period: Optional[str] = None) -> pd.DataFrame:
        fetcher = MarketDataFetcher(
            symbols=[self.symbol],
            period=period or f"{self.train_years}y",
        )
        df = fetcher.fetch(self.symbol)
        if df.empty:
            raise ValueError(f"No price data available for {self.symbol}")
        return df

    def _fetch_macro_data(self, start: Optional[str] = None) -> Optional[pd.DataFrame]:
        try:
            macro = MacroDataFetcher(
                start=start or f"{datetime.now().year - self.train_years}-01-01",
                use_fred=False,  # disabled by default; enable with pandas-datareader
            )
            return macro.get_macro_features()
        except Exception as exc:
            logger.warning("Macro data fetch failed: %s", exc)
            return None

    def _fetch_fundamentals(self) -> Dict:
        if self.asset_type != "equity":
            return {}
        try:
            macro = MacroDataFetcher()
            return macro.get_stock_fundamentals(self.symbol)
        except Exception as exc:
            logger.warning("Fundamentals fetch failed: %s", exc)
            return {}

    def _fetch_news_sentiment(self) -> Dict:
        try:
            news = NewsDataFetcher(symbols=[self.symbol])
            return news.get_aggregated_sentiment(self.symbol)
        except Exception as exc:
            logger.warning("News sentiment fetch failed: %s", exc)
            return {}

    def _fetch_social_sentiment(self) -> Dict:
        try:
            social = SocialDataFetcher()
            return social.get_social_sentiment(self.symbol)
        except Exception as exc:
            logger.warning("Social sentiment fetch failed: %s", exc)
            return {}

    def _fetch_commodity_data(self) -> Optional[pd.DataFrame]:
        if self.asset_type != "commodity":
            return None
        commodity_map = {
            "CL=F": "crude_oil", "BZ=F": "crude_oil",
            "GC=F": "gold", "SI=F": "silver",
            "NG=F": "natural_gas",
        }
        commodity = commodity_map.get(self.symbol, "crude_oil")
        try:
            fetcher = CommodityDataFetcher(
                commodity=commodity,
                start=f"{datetime.now().year - self.train_years}-01-01",
            )
            return fetcher.get_features()
        except Exception as exc:
            logger.warning("Commodity data fetch failed: %s", exc)
            return None
