"""
FastAPI Application — Multi-Layer Price Prediction API
=======================================================
Provides REST endpoints for:

  POST /predict              — generate a prediction for a symbol
  POST /train                — train models for a symbol
  POST /backtest             — run backtesting for a symbol + date range
  GET  /health               — service health check
  GET  /supported_assets     — list of supported asset types
  GET  /feature_importance/{symbol} — top feature importances for trained model

Authentication: Bearer token (set PREDICTION_API_KEY env var).
Rate limiting:  via slowapi (optional).

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from config import config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Layer Price Prediction API",
    description=(
        "Institutional-grade probability-based price prediction for stocks and commodities. "
        "Combines technical, fundamental, NLP sentiment, and deep learning signals."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# In-memory model registry (use Redis / DB in production)
_model_registry: Dict[str, object] = {}

# ------------------------------------------------------------------
# Security
# ------------------------------------------------------------------

API_KEY = os.getenv("PREDICTION_API_KEY", "dev-key-change-in-prod")
bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g. AAPL, RELIANCE.NS, CL=F")
    horizon: int = Field(3, ge=1, le=30, description="Prediction horizon in trading days")
    asset_type: str = Field("equity", description="equity | commodity | index")
    use_deep_learning: bool = Field(False, description="Include LSTM/Transformer models (slower)")
    use_finbert: bool = Field(False, description="Use FinBERT for news sentiment")
    upcoming_events: Optional[List[Dict]] = Field(None, description="Scheduled events for risk engine")

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, v):
        if v not in ("equity", "commodity", "index"):
            raise ValueError("asset_type must be equity, commodity, or index")
        return v


class TrainRequest(BaseModel):
    symbol: str
    horizon: int = Field(3, ge=1, le=30)
    asset_type: str = Field("equity")
    train_years: int = Field(5, ge=1, le=20)
    use_deep_learning: bool = False
    force_retrain: bool = False


class BacktestRequest(BaseModel):
    symbol: str
    horizon: int = Field(3, ge=1, le=30)
    start_date: str = Field("2015-01-01", description="YYYY-MM-DD")
    end_date: Optional[str] = None
    initial_capital: float = Field(1_000_000.0, gt=0)


class PredictionResponse(BaseModel):
    symbol: str
    horizon_days: int
    timestamp: str
    asset_type: str
    probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]
    signal: str
    confidence: float
    risk_approved: bool
    position_size_pct: float
    risk_report: Dict
    model_breakdown: Dict
    macro_risk_score: float
    feature_snapshot: Dict
    latest_price: float
    realised_vol_20d: float


class BacktestResponse(BaseModel):
    symbol: str
    summary: Dict
    config: Dict


class TrainResponse(BaseModel):
    symbol: str
    horizon: int
    status: str
    message: str
    trained_at: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Service health check."""
    return {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/supported_assets", tags=["System"])
def supported_assets():
    """List supported asset types and example symbols."""
    return {
        "equity": {
            "description": "Individual stocks and ETFs",
            "examples": ["AAPL", "GOOGL", "MSFT", "RELIANCE.NS", "TCS.NS"],
        },
        "commodity": {
            "description": "Commodity futures",
            "examples": ["CL=F (Crude Oil)", "GC=F (Gold)", "SI=F (Silver)", "NG=F (Natural Gas)"],
        },
        "index": {
            "description": "Market indices",
            "examples": ["^GSPC (S&P 500)", "^NSEI (Nifty 50)", "^DJI (Dow Jones)"],
        },
    }


@app.post("/train", response_model=TrainResponse, tags=["Model Management"])
async def train_model(
    req: TrainRequest,
    _token: str = Depends(verify_token),
):
    """
    Train all models for a given symbol.
    Models are cached in-memory for subsequent /predict calls.
    """
    from src.prediction.predictor import PricePredictor

    cache_key = f"{req.symbol}_{req.horizon}"

    if not req.force_retrain and cache_key in _model_registry:
        return TrainResponse(
            symbol=req.symbol,
            horizon=req.horizon,
            status="cached",
            message="Model already trained. Use force_retrain=true to retrain.",
            trained_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    try:
        predictor = PricePredictor(
            symbol=req.symbol,
            horizon=req.horizon,
            asset_type=req.asset_type,
            train_years=req.train_years,
            use_deep_learning=req.use_deep_learning,
        )
        predictor.train()
        _model_registry[cache_key] = predictor

        return TrainResponse(
            symbol=req.symbol,
            horizon=req.horizon,
            status="success",
            message=f"Models trained successfully for {req.symbol}.",
            trained_at=datetime.now(tz=timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.exception("Training failed for %s", req.symbol)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(exc)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    req: PredictRequest,
    _token: str = Depends(verify_token),
):
    """
    Generate a probability-based price direction prediction.

    Returns the full prediction including:
    - Directional probabilities (bullish/sideways/bearish)
    - Risk-adjusted signal
    - Position sizing recommendation
    - Model-by-model breakdown
    """
    from src.prediction.predictor import PricePredictor

    cache_key = f"{req.symbol}_{req.horizon}"

    predictor: PricePredictor
    if cache_key in _model_registry:
        predictor = _model_registry[cache_key]
    else:
        # Auto-train if not cached
        try:
            predictor = PricePredictor(
                symbol=req.symbol,
                horizon=req.horizon,
                asset_type=req.asset_type,
                use_deep_learning=req.use_deep_learning,
                use_finbert=req.use_finbert,
                upcoming_events=req.upcoming_events or [],
            )
            predictor.train()
            _model_registry[cache_key] = predictor
        except Exception as exc:
            logger.exception("Auto-training failed for %s", req.symbol)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model training failed: {str(exc)}",
            )

    try:
        result = predictor.predict()
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed for %s", req.symbol)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post("/backtest", response_model=BacktestResponse, tags=["Backtesting"])
async def backtest(
    req: BacktestRequest,
    _token: str = Depends(verify_token),
):
    """
    Run a backtest for a symbol using trained models.

    Requires the model to be trained first via /train.
    Returns Sharpe, Sortino, Win Rate, Max Drawdown, and more.
    """
    from src.prediction.predictor import PricePredictor
    from src.backtest.backtester import Backtester
    from src.data.market_data import MarketDataFetcher

    cache_key = f"{req.symbol}_{req.horizon}"
    if cache_key not in _model_registry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No trained model found for {req.symbol} horizon={req.horizon}. "
                   "Please call /train first.",
        )

    try:
        # Fetch historical price data for the backtest window
        fetcher = MarketDataFetcher(
            symbols=[req.symbol],
            start=req.start_date,
            end=req.end_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        )
        price_df = fetcher.fetch(req.symbol)

        if price_df.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No price data for {req.symbol} in the requested date range.",
            )

        predictor: PricePredictor = _model_registry[cache_key]

        # Generate signal series using the trained ML model + ensemble
        # (simplified: use ML model prediction on rolling window)
        signals, confidences = _generate_signal_series(predictor, price_df)

        bt = Backtester(
            price_df=price_df,
            signal_series=signals,
            confidence_series=confidences,
            initial_capital=req.initial_capital,
        )
        results = bt.run()
        summary = results.summary()

        return BacktestResponse(
            symbol=req.symbol,
            summary=summary,
            config={
                "start_date": req.start_date,
                "end_date": req.end_date,
                "horizon": req.horizon,
                "initial_capital": req.initial_capital,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Backtest failed for %s", req.symbol)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(exc)}",
        )


@app.get("/feature_importance/{symbol}", tags=["Interpretability"])
async def feature_importance(
    symbol: str,
    horizon: int = 3,
    top_n: int = 20,
    _token: str = Depends(verify_token),
):
    """
    Return top feature importances for a trained model.
    """
    cache_key = f"{symbol.upper()}_{horizon}"
    if cache_key not in _model_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trained model for {symbol} horizon={horizon}.",
        )

    from src.prediction.predictor import PricePredictor
    predictor: PricePredictor = _model_registry[cache_key]

    if predictor._ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ML model not trained.",
        )

    fi = predictor._ml_model.feature_importances()
    if fi.empty:
        return {"symbol": symbol, "features": []}

    top = fi.head(top_n).to_dict(orient="records")
    return {"symbol": symbol, "horizon": horizon, "top_features": top}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _generate_signal_series(predictor, price_df: pd.DataFrame):
    """
    Generate a signal series for backtesting by rolling the ML model
    over the historical price DataFrame.

    Uses a simplified approach: re-engineer features once and predict
    on each row using the trained ML model.
    """
    from src.features.feature_engineer import FeatureEngineer

    eng = FeatureEngineer(price_df=price_df, horizons=[predictor.horizon])
    X = eng.build_feature_matrix(scale=True)

    if X.empty or predictor._ml_model is None:
        # Return neutral signals
        neutral = pd.Series("sideways", index=price_df.index)
        conf = pd.Series(0.5, index=price_df.index)
        return neutral, conf

    signals = []
    confidences = []
    for i in range(len(X)):
        row = X.iloc[[i]]
        proba = predictor._ml_model.predict_proba(row)
        signal = max(proba, key=proba.get)
        conf = max(proba.values())
        signals.append(signal)
        confidences.append(conf)

    return (
        pd.Series(signals, index=X.index),
        pd.Series(confidences, index=X.index),
    )
