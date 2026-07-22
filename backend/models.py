"""
Pydantic schemas for request/response validation in FinSight.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─── Request Schemas ───

class ForecastRequest(BaseModel):
    ticker: str = Field(..., description="Stock/crypto ticker symbol (e.g., AAPL, BTC-USD)")
    period: str = Field(default="1y", description="yfinance period string: 1y, 2y, 5y, max")
    forecast_days: int = Field(default=30, ge=7, le=365, description="Days to forecast ahead")
    include_sentiment: bool = Field(default=True, description="Augment with DeepSeek sentiment")


class PortfolioRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="List of ticker symbols")
    weights: Optional[list[float]] = Field(default=None, description="Portfolio weights (must sum to 1)")
    period: str = Field(default="1y", description="yfinance period string")


class SentimentRequest(BaseModel):
    ticker: str = Field(..., description="Ticker to analyze sentiment for")
    headlines: Optional[list[str]] = Field(default=None, description="Optional custom headlines")


class BacktestRequest(BaseModel):
    ticker: str
    period: str = "1y"
    forecast_days: int = 30


# ─── Response Schemas ───

class SentimentResult(BaseModel):
    ticker: str
    headlines: list[str]
    overall_score: float = Field(ge=-1.0, le=1.0)
    classification: str  # bullish / bearish / neutral
    per_headline: list[dict]
    summary: str


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    mape: Optional[float] = None


class ForecastResult(BaseModel):
    ticker: str
    forecast_days: int
    historical_dates: list[date]
    historical_prices: list[float]
    prophet_forecast: list[ForecastPoint]
    lstm_forecast: list[ForecastPoint]
    combined_forecast: list[ForecastPoint]
    prophet_metrics: ModelMetrics
    lstm_metrics: ModelMetrics
    sentiment: Optional[SentimentResult] = None
    current_price: float
    predicted_price_30d: float
    predicted_change_pct: float


class PortfolioMetrics(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    best_day: float
    worst_day: float
    positive_days_pct: float


class PortfolioResult(BaseModel):
    tickers: list[str]
    weights: list[float]
    cumulative_returns: list[float]
    dates: list[date]
    metrics: PortfolioMetrics


class HealthResponse(BaseModel):
    status: str
    version: str
    deepseek_configured: bool
    database_configured: bool
