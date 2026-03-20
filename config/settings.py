"""
Central configuration for the multi-layer price prediction system.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataConfig:
    # Market data
    default_interval: str = "1d"
    default_period: str = "5y"
    supported_intervals: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d", "1wk"
    ])

    # Technical indicator windows
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14

    # Sentiment
    news_lookback_hours: int = 48
    social_lookback_hours: int = 24


@dataclass
class ModelConfig:
    # Prediction horizons (in days)
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # Classification thresholds
    bullish_threshold: float = 0.015   # +1.5%
    bearish_threshold: float = -0.015  # -1.5%

    # Ensemble weights (sum to 1.0)
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "statistical": 0.10,
        "ml": 0.35,
        "deep_learning": 0.30,
        "sentiment": 0.15,
        "macro": 0.10,
    })

    # Training
    train_test_split: float = 0.8
    validation_split: float = 0.1
    random_state: int = 42

    # LSTM
    lstm_lookback: int = 60
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32

    # XGBoost / LightGBM
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6


@dataclass
class RiskConfig:
    # Volatility regime thresholds (annualised vol)
    low_vol_threshold: float = 0.15
    high_vol_threshold: float = 0.35

    # Maximum allowed event risk score (0-1)
    max_event_risk: float = 0.7

    # Confidence gate: suppress signal if ensemble confidence < this
    min_confidence: float = 0.55

    # Position sizing caps
    max_position_pct: float = 0.05   # 5% of portfolio per trade


@dataclass
class BacktestConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000.0
    commission_pct: float = 0.001     # 0.1% per trade
    slippage_pct: float = 0.0005      # 0.05%


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"

    # External API keys (loaded from environment)
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    quandl_api_key: str = os.getenv("QUANDL_API_KEY", "")

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    cache_ttl_seconds: int = 300


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global singleton
config = AppConfig()
