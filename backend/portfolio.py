"""
Portfolio risk metrics calculator for FinSight.
Returns Sharpe ratio, max drawdown, VaR, volatility, and other key metrics.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models import PortfolioMetrics

logger = logging.getLogger(__name__)


def calculate_portfolio_metrics(
    prices: pd.DataFrame,
    weights: Optional[list[float]] = None,
    risk_free_rate: float = 0.05,
) -> PortfolioMetrics:
    """
    Calculate key portfolio risk & performance metrics.

    Args:
        prices: DataFrame with date index and ticker columns (closing prices)
        weights: Portfolio weights (must sum to 1). If None, equal weight.
        risk_free_rate: Annual risk-free rate (default 5%)

    Returns:
        PortfolioMetrics
    """
    if prices.empty:
        raise ValueError("Price data is empty")

    n_assets = len(prices.columns)

    # Calculate returns
    daily_returns = prices.pct_change().dropna()

    if weights is None:
        weights = [1.0 / n_assets] * n_assets
    else:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

    if len(weights) < n_assets:
        # Pad missing weights with equal distribution of remainder
        remaining = 1.0 - sum(weights)
        weights.extend([remaining / (n_assets - len(weights))] * (n_assets - len(weights)))

    # Portfolio daily returns
    portfolio_returns = (daily_returns * weights).sum(axis=1)

    # ─── Total Return ───
    cumulative = (1 + portfolio_returns).cumprod()
    total_return = float((cumulative.iloc[-1] - 1) * 100)

    # ─── Annualized Return ───
    trading_days = 252
    annualized_return = float(
        ((1 + total_return / 100) ** (trading_days / len(portfolio_returns)) - 1) * 100
    )

    # ─── Annualized Volatility ───
    annualized_volatility = float(portfolio_returns.std() * np.sqrt(trading_days) * 100)

    # ─── Sharpe Ratio ───
    excess_returns = portfolio_returns - risk_free_rate / trading_days
    sharpe_ratio = float(
        (excess_returns.mean() / portfolio_returns.std()) * np.sqrt(trading_days)
        if portfolio_returns.std() > 0
        else 0.0
    )

    # ─── Max Drawdown ───
    cumsum = cumulative
    running_max = cumsum.cummax()
    drawdown = (cumsum - running_max) / running_max
    max_drawdown = float(drawdown.min() * 100)

    # ─── Value at Risk (95%) ───
    var_95 = float(np.percentile(portfolio_returns, 5) * 100)

    # ─── Best / Worst Day ───
    best_day = float(portfolio_returns.max() * 100)
    worst_day = float(portfolio_returns.min() * 100)

    # ─── Positive Days % ───
    positive_days_pct = float((portfolio_returns > 0).mean() * 100)

    return PortfolioMetrics(
        total_return=round(total_return, 2),
        annualized_return=round(annualized_return, 2),
        annualized_volatility=round(annualized_volatility, 2),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 2),
        var_95=round(var_95, 4),
        best_day=round(best_day, 4),
        worst_day=round(worst_day, 4),
        positive_days_pct=round(positive_days_pct, 2),
    )


def calculate_basic_stats(prices: pd.Series) -> dict:
    """
    Calculate basic stats for a single asset.
    Returns dict with current price, change, volatility, etc.
    """
    returns = prices.pct_change().dropna()

    return {
        "current_price": round(float(prices.iloc[-1]), 2),
        "mean_price": round(float(prices.mean()), 2),
        "min_price": round(float(prices.min()), 2),
        "max_price": round(float(prices.max()), 2),
        "daily_volatility": round(float(returns.std() * 100), 4),
        "total_return_pct": round(float((prices.iloc[-1] / prices.iloc[0] - 1) * 100), 2),
        "latest_return_pct": round(float(returns.iloc[-1] * 100), 4) if len(returns) > 0 else 0.0,
    }
