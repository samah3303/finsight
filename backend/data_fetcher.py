"""
yfinance data fetcher — historical prices and news headlines.
"""

from datetime import date, datetime, timedelta
from typing import Optional
import logging

import yfinance as yf
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


def fetch_historical_data(
    ticker: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker via yfinance.
    Returns DataFrame with standard columns: Open, High, Low, Close, Volume.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and try again.")

    # Ensure index is timezone-naive for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info("Fetched %d rows for %s (period=%s)", len(df), ticker, period)
    return df


def fetch_news_headlines(ticker: str) -> list[dict]:
    """
    Pull recent news articles from yfinance .news attribute.
    Returns list of dicts with keys: title, publisher, link, providerPublishTime.
    Falls back to empty list if .news is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            logger.warning("No news found for %s", ticker)
            return []

        headlines = []
        for article in news[: config.max_news_headlines]:
            content = article.get("content", {})
            headlines.append(
                {
                    "title": content.get("title", ""),
                    "publisher": content.get("provider", {}).get("displayName", ""),
                    "link": content.get("canonicalUrl", {}).get("url", ""),
                    "published": content.get("pubDate", ""),
                }
            )
        logger.info("Fetched %d headlines for %s", len(headlines), ticker)
        return headlines
    except Exception as e:
        logger.warning("Failed to fetch news for %s: %s", ticker, e)
        return []


def fetch_multi_ticker_data(
    tickers: list[str], period: str = "1y"
) -> pd.DataFrame:
    """
    Fetch closing prices for multiple tickers, returning a combined DataFrame
    with date index and ticker columns.
    """
    closes = {}
    for t in tickers:
        try:
            df = fetch_historical_data(t, period=period)
            closes[t] = df["Close"]
        except ValueError:
            logger.warning("Skipping %s — no data returned", t)

    if not closes:
        raise ValueError("No valid tickers returned data")

    combined = pd.DataFrame(closes).dropna()
    return combined


def get_current_price(ticker: str) -> float:
    """Get the most recent closing price for a ticker."""
    df = fetch_historical_data(ticker, period="5d")
    return float(df["Close"].iloc[-1])


def prepare_train_data(
    df: pd.DataFrame, lookback_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test for time-series modeling.
    Returns (train, val, test) DataFrames with 'ds' (dates) and 'y' (close prices).
    """
    ts = df[["Close"]].copy()
    ts.columns = ["y"]
    ts["ds"] = ts.index

    n = len(ts)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = ts.iloc[:train_end]
    val = ts.iloc[train_end:val_end]
    test = ts.iloc[val_end:]

    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test
