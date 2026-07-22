"""
FinSight — LLM-Augmented Financial Forecaster
FastAPI backend serving forecasts, sentiment analysis, and portfolio metrics.
"""

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import config
from models import (
    ForecastRequest,
    ForecastResult,
    ForecastPoint,
    PortfolioRequest,
    PortfolioResult,
    PortfolioMetrics,
    SentimentRequest,
    SentimentResult,
    HealthResponse,
)
from data_fetcher import (
    fetch_historical_data,
    fetch_multi_ticker_data,
    fetch_news_headlines,
    get_current_price,
    prepare_train_data,
)
from prophet_model import run_prophet_pipeline
from lstm_model import run_lstm_pipeline
from sentiment import analyze_sentiment, apply_sentiment_adjustment
from portfolio import calculate_portfolio_metrics, calculate_basic_stats
from database import close_pool, save_forecasts, save_sentiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("FinSight API starting up...")
    yield
    await close_pool()
    logger.info("FinSight API shut down")


app = FastAPI(
    title="FinSight API",
    description="LLM-Augmented Financial Forecaster — Prophet + LSTM + DeepSeek",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        deepseek_configured=bool(config.deepseek_api_key),
        database_configured=bool(config.database_url),
    )


# ─── Forecast ───

@app.post("/forecast", response_model=ForecastResult)
async def forecast(req: ForecastRequest):
    """Generate a combined forecast (Prophet + LSTM + sentiment) for a ticker."""
    logger.info("Forecast request: ticker=%s, period=%s, days=%d", req.ticker, req.period, req.forecast_days)

    try:
        df = fetch_historical_data(req.ticker, period=req.period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df) < config.lstm_sequence_length + 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: need at least {config.lstm_sequence_length + 10} trading days, got {len(df)}",
        )

    current_price = float(df["Close"].iloc[-1])
    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]

    # ── Prophet ──
    prophet_points, prophet_metrics, _ = run_prophet_pipeline(df, req.forecast_days)

    # ── LSTM ──
    lstm_points, lstm_metrics, _ = run_lstm_pipeline(
        df["Close"].values.astype(np.float64),
        forecast_days=req.forecast_days,
        last_date=last_date,
        ticker=req.ticker,
    )

    # ── Sentiment ──
    sentiment = None
    if req.include_sentiment:
        try:
            sentiment = await analyze_sentiment(req.ticker)
        except Exception as e:
            logger.warning("Sentiment analysis failed (non-fatal): %s", e)
            sentiment = SentimentResult(
                ticker=req.ticker,
                headlines=[],
                overall_score=0.0,
                classification="neutral",
                per_headline=[],
                summary=f"Sentiment analysis unavailable: {e}",
            )

    # ── Combined Forecast ──
    sentiment_score = sentiment.overall_score if sentiment else 0.0

    # Average model predictions
    avg_forecast = []
    for pp, lp in zip(prophet_points, lstm_points):
        avg_forecast.append((pp.forecast + lp.forecast) / 2.0)

    combined_values = apply_sentiment_adjustment(
        avg_forecast,
        sentiment_score,
        current_price,
        model_weight=config.model_weight,
    )

    # Build combined forecast points with confidence bands
    combined_points = []
    for i, (pp, cv) in enumerate(zip(prophet_points, combined_values)):
        # Use Prophet's confidence intervals where available
        ci_lower = pp.lower_bound if pp.lower_bound else cv * 0.95
        ci_upper = pp.upper_bound if pp.upper_bound else cv * 1.05
        # Blend with sentiment
        combined_points.append(
            ForecastPoint(
                date=pp.date,
                forecast=cv,
                lower_bound=round(ci_lower, 4),
                upper_bound=round(ci_upper, 4),
            )
        )

    predicted_price_30d = combined_values[-1] if combined_values else current_price
    predicted_change_pct = round((predicted_price_30d / current_price - 1) * 100, 2)

    # ── Persist to DB (best effort) ──
    try:
        dates_list = [p.date for p in combined_points]
        prophet_vals = [p.forecast for p in prophet_points]
        lstm_vals = [p.forecast for p in lstm_points]
        await save_forecasts(
            req.ticker, dates_list, prophet_vals, lstm_vals, combined_values, sentiment_score
        )
        if sentiment:
            await save_sentiment(
                req.ticker,
                sentiment.overall_score,
                sentiment.classification,
                sentiment.summary,
                sentiment.headlines,
            )
    except Exception as e:
        logger.warning("Failed to persist forecast: %s", e)

    # ── Historical data for charting ──
    historical_dates = [d.date() if hasattr(d, "date") else d for d in df.index[-90:]]
    historical_prices = [round(float(p), 4) for p in df["Close"].tail(90).tolist()]

    return ForecastResult(
        ticker=req.ticker,
        forecast_days=req.forecast_days,
        historical_dates=historical_dates,
        historical_prices=historical_prices,
        prophet_forecast=prophet_points,
        lstm_forecast=lstm_points,
        combined_forecast=combined_points,
        prophet_metrics=prophet_metrics,
        lstm_metrics=lstm_metrics,
        sentiment=sentiment,
        current_price=round(current_price, 2),
        predicted_price_30d=round(predicted_price_30d, 2),
        predicted_change_pct=predicted_change_pct,
    )


# ─── Quick Forecast (GET) ───

@app.get("/forecast/{ticker}", response_model=ForecastResult)
async def quick_forecast(
    ticker: str,
    period: str = Query(default="1y"),
    forecast_days: int = Query(default=30, ge=7, le=90),
    include_sentiment: bool = Query(default=True),
):
    """Convenience GET endpoint for quick forecasts."""
    req = ForecastRequest(
        ticker=ticker,
        period=period,
        forecast_days=forecast_days,
        include_sentiment=include_sentiment,
    )
    return await forecast(req)


# ─── Sentiment Only ───

@app.post("/sentiment", response_model=SentimentResult)
async def sentiment(req: SentimentRequest):
    """Analyze news sentiment for a ticker."""
    headlines = None
    if req.headlines:
        headlines = [{"title": h} for h in req.headlines]
    try:
        result = await analyze_sentiment(req.ticker, headlines=headlines)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment/{ticker}", response_model=SentimentResult)
async def quick_sentiment(ticker: str):
    """Convenience GET endpoint for quick sentiment analysis."""
    return await sentiment(SentimentRequest(ticker=ticker))


# ─── Portfolio Metrics ───

@app.post("/portfolio", response_model=PortfolioResult)
async def portfolio_metrics(req: PortfolioRequest):
    """Calculate portfolio risk & performance metrics."""
    try:
        prices = fetch_multi_ticker_data(req.tickers, period=req.period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(prices.columns) < 1:
        raise HTTPException(status_code=400, detail="No valid ticker data returned")

    metrics = calculate_portfolio_metrics(
        prices,
        weights=req.weights,
        risk_free_rate=config.risk_free_rate,
    )

    # Cumulative returns for charting
    daily_returns = prices.pct_change().dropna()
    if req.weights is None:
        n = len(prices.columns)
        w = [1.0 / n] * n
    else:
        total = sum(req.weights)
        w = [x / total for x in req.weights]
    if len(w) < len(prices.columns):
        remaining = 1.0 - sum(w)
        w.extend([remaining / (len(prices.columns) - len(w))] * (len(prices.columns) - len(w)))

    port_returns = (daily_returns * w).sum(axis=1)
    cumulative = (1 + port_returns).cumprod()

    dates_list = [d.date() if hasattr(d, "date") else d for d in cumulative.index]
    cum_vals = [round(float(v), 6) for v in cumulative.tolist()]

    return PortfolioResult(
        tickers=req.tickers,
        weights=[round(x, 4) for x in w[: len(req.tickers)]],
        cumulative_returns=cum_vals,
        dates=dates_list,
        metrics=metrics,
    )


# ─── Run ───

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.host, port=config.port, reload=True)
