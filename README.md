# Multi-Layer Price Prediction System

An institutional-grade, **probability-based** price direction prediction engine for stocks and commodities. Rather than predicting exact prices, it outputs a probability distribution over directional outcomes (bullish / sideways / bearish) for configurable horizons (1–30 trading days).

---

## Architecture

```
Data Ingestion
    ├── Market Data      (OHLCV, VWAP, order-book proxy)
    ├── Macro Data       (rates, VIX, dollar index, CPI)
    ├── News Data        (RSS feeds, NewsAPI + NLP pipeline)
    ├── Social Sentiment (Twitter, Reddit)
    └── Commodity Data   (EIA inventory, rig count, gold drivers)
            ↓
Feature Engineering
    ├── Technical        (RSI, MACD, ADX, Bollinger, ATR, OBV, …)
    ├── Fundamental      (P/E, ROE, debt ratios, institutional flow)
    └── Sentiment        (VADER + FinBERT aggregation)
            ↓
Model Layer
    ├── Statistical      (ARIMA + GARCH → conditional distribution)
    ├── ML               (Random Forest + XGBoost + LightGBM)
    ├── Deep Learning    (Bidirectional LSTM + Transformer encoder)
    └── Sentiment Model  (FinBERT / VADER fallback)
            ↓
Ensemble Model  (weighted average + optional meta-learner stacking)
            ↓
Risk Engine
    ├── Volatility regime filter
    ├── Event risk filter (earnings, FOMC, …)
    ├── Liquidity check
    ├── Confidence gate
    └── Kelly-based position sizing
            ↓
Final Signal + Probability Distribution
```

---

## Example Output

```json
{
  "symbol": "AAPL",
  "horizon_days": 3,
  "signal": "bullish",
  "confidence": 0.68,
  "probabilities": {
    "bullish": 0.65,
    "sideways": 0.22,
    "bearish": 0.13
  },
  "risk_approved": true,
  "position_size_pct": 2.3,
  "realised_vol_20d": 0.187,
  "model_breakdown": {
    "statistical": {"bullish": 0.55, "sideways": 0.28, "bearish": 0.17},
    "ml":          {"bullish": 0.69, "sideways": 0.20, "bearish": 0.11},
    "sentiment":   {"bullish": 0.61, "sideways": 0.25, "bearish": 0.14}
  }
}
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional)

```bash
cp .env.example .env
# Edit .env and add your keys (all optional – system works without them)
```

### 3. Run a prediction in Python

```python
from src.prediction.predictor import PricePredictor

predictor = PricePredictor(
    symbol="AAPL",
    horizon=3,           # 3-day ahead prediction
    asset_type="equity",
)
predictor.train()        # downloads data and trains all models
result = predictor.predict()
print(predictor.interpret_prediction(result))
```

### 4. Start the REST API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `GET`  | `/supported_assets` | List supported asset types |
| `POST` | `/train` | Train models for a symbol |
| `POST` | `/predict` | Generate probability prediction |
| `POST` | `/backtest` | Run historical backtest |
| `GET`  | `/feature_importance/{symbol}` | Top feature importances |

All endpoints require `Authorization: Bearer <PREDICTION_API_KEY>`.

### Example cURL

```bash
# Train
curl -X POST http://localhost:8000/train \
  -H "Authorization: Bearer dev-key-change-in-prod" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 3}'

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer dev-key-change-in-prod" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 3, "asset_type": "equity"}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Supported Assets

| Type | Examples |
|------|----------|
| Equity | `AAPL`, `MSFT`, `GOOGL`, `RELIANCE.NS`, `TCS.NS` |
| Commodity | `CL=F` (Crude), `GC=F` (Gold), `SI=F` (Silver), `NG=F` (Nat Gas) |
| Index | `^GSPC`, `^NSEI`, `^DJI` |

---

## Performance Expectations

Even institutional models achieve only **55–65% accuracy**. This system focuses on:
- **Probability calibration** over point predictions
- **Risk-adjusted position sizing** via Kelly criterion
- **Ensemble diversity** to reduce model-specific errors
- **Regime awareness** (low/high volatility, pre-event dampening)

---

## Data Sources

| Layer | Source | Auth Required |
|-------|--------|---------------|
| Market prices | yfinance (Yahoo Finance) | No |
| Macro / rates | yfinance + FRED | No (FRED optional) |
| News sentiment | RSS feeds + NewsAPI | No (NewsAPI optional) |
| Social sentiment | Twitter API + Reddit | Optional |
| Commodity supply | EIA JSON API | Optional (free) |
| Deep NLP | FinBERT (HuggingFace) | No |

---

## Project Structure

```
├── api/                  FastAPI application
├── config/               Centralised configuration
├── src/
│   ├── data/             Data ingestion layer
│   ├── features/         Feature engineering
│   ├── models/           Model implementations
│   ├── risk/             Risk engine
│   ├── backtest/         Backtesting framework
│   └── prediction/       Orchestration pipeline
├── tests/                Unit & integration tests
├── requirements.txt
└── .env.example
```
