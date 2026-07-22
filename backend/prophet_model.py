"""
Prophet-based time-series forecasting for FinSight.
Handles trend + seasonality decomposition and 30-day ahead prediction.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from prophet import Prophet

from models import ForecastPoint, ModelMetrics
from config import config

logger = logging.getLogger(__name__)


def train_prophet(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on closing prices.

    Args:
        df: DataFrame with 'ds' (dates) and 'y' (closing prices)

    Returns:
        Trained Prophet model
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    model.fit(df)
    logger.info("Prophet model trained on %d data points", len(df))
    return model


def forecast_prophet(
    model: Prophet, periods: int = 30, freq: str = "D"
) -> pd.DataFrame:
    """
    Generate forecasts from a trained Prophet model.

    Returns DataFrame with: ds, yhat, yhat_lower, yhat_upper
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    logger.info("Prophet forecast generated for %d future periods", periods)
    return forecast


def evaluate_prophet(model: Prophet, test: pd.DataFrame) -> ModelMetrics:
    """
    Evaluate Prophet on a held-out test set.

    Args:
        model: Trained Prophet model
        test: Test DataFrame with 'ds' and 'y' columns

    Returns:
        ModelMetrics (MAE, RMSE, MAPE)
    """
    forecast = model.predict(test[["ds"]])
    predicted = forecast["yhat"].values
    actual = test["y"].values

    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    # MAPE — avoid division by zero
    mask = actual != 0
    mape = (
        float(np.mean(np.abs((predicted[mask] - actual[mask]) / actual[mask])) * 100)
        if mask.any()
        else None
    )

    return ModelMetrics(mae=round(mae, 4), rmse=round(rmse, 4), mape=round(mape, 2) if mape else None)


def prophet_forecast_points(forecast_df: pd.DataFrame, historical_end: pd.Timestamp) -> list[ForecastPoint]:
    """Extract future forecast points from Prophet output."""
    points = []
    # Normalize to date-only for safe comparison
    cutoff = historical_end.normalize() if hasattr(historical_end, 'normalize') else pd.Timestamp(historical_end).normalize()
    future_part = forecast_df[forecast_df["ds"].dt.normalize() > cutoff]
    for _, row in future_part.iterrows():
        points.append(
            ForecastPoint(
                date=row["ds"].date(),
                forecast=round(float(row["yhat"]), 4),
                lower_bound=round(float(row.get("yhat_lower", row["yhat"])), 4),
                upper_bound=round(float(row.get("yhat_upper", row["yhat"])), 4),
            )
        )
    return points


def run_prophet_pipeline(
    df: pd.DataFrame,
    forecast_days: int = 30,
) -> tuple[list[ForecastPoint], ModelMetrics, Prophet]:
    """
    End-to-end Prophet pipeline: train, evaluate, forecast.

    Trains two models:
      - Eval model on 85% of data (for backtest metrics)
      - Forecast model on 100% of data (for actual future forecast)

    Returns:
        (forecast_points, metrics, trained_model)
    """
    # Prepare data in Prophet format
    ts_data = df[["Close"]].copy()
    ts_data["ds"] = ts_data.index
    ts_data["y"] = ts_data["Close"]

    n = len(ts_data)
    train = ts_data.iloc[: int(n * 0.85)]
    test = ts_data.iloc[int(n * 0.85) :]

    if len(test) < 2:
        split = max(int(n * 0.7), 10)
        train = ts_data.iloc[:split]
        test = ts_data.iloc[split:]

    # Evaluate on holdout
    eval_model = train_prophet(train[["ds", "y"]])
    metrics = evaluate_prophet(eval_model, test[["ds", "y"]])

    # Train forecast model on ALL data for actual future predictions
    forecast_model = train_prophet(ts_data[["ds", "y"]])
    full_forecast = forecast_prophet(forecast_model, periods=forecast_days)
    points = prophet_forecast_points(full_forecast, pd.Timestamp(ts_data.index[-1]))

    return points, metrics, forecast_model
