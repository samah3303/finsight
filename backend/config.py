"""
FinSight Backend Configuration
Loads environment variables and provides app-wide settings.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # DeepSeek API
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = "https://api.deepseek.com/v1/chat/completions"
    deepseek_model: str = "deepseek-chat"

    # Neon PostgreSQL
    database_url: str = os.getenv("DATABASE_URL", "")

    # Forecasting defaults
    default_lookback_days: int = 365
    default_forecast_days: int = 30
    lstm_sequence_length: int = 60
    lstm_epochs: int = 50
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2

    # Sentiment
    sentiment_weight: float = 0.30
    model_weight: float = 0.70
    max_news_headlines: int = 15

    # Portfolio
    risk_free_rate: float = 0.05
    var_confidence: float = 0.95

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


config = Config()
