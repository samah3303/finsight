"""
Neon PostgreSQL database layer for FinSight.
Stores forecasts, sentiment history, and backtest results.
"""

import logging
from datetime import date
from typing import Optional

import asyncpg
from asyncpg.pool import Pool

from config import config

logger = logging.getLogger(__name__)

_pool: Optional[Pool] = None


async def get_pool() -> Pool:
    """Return a shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        if not config.database_url:
            raise RuntimeError("DATABASE_URL not configured — set it in .env")
        _pool = await asyncpg.create_pool(
            dsn=config.database_url,
            min_size=2,
            max_size=10,
        )
        await _init_schema(_pool)
    return _pool


async def _init_schema(pool: Pool) -> None:
    """Create tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS forecasts (
        id              SERIAL PRIMARY KEY,
        ticker          TEXT NOT NULL,
        prediction_date DATE NOT NULL,
        prophet_pred    DOUBLE PRECISION,
        lstm_pred       DOUBLE PRECISION,
        combined_pred   DOUBLE PRECISION,
        sentiment_score DOUBLE PRECISION,
        actual_price    DOUBLE PRECISION,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_forecasts_ticker ON forecasts(ticker);
    CREATE INDEX IF NOT EXISTS idx_forecasts_date   ON forecasts(prediction_date);

    CREATE TABLE IF NOT EXISTS sentiment_history (
        id          SERIAL PRIMARY KEY,
        ticker      TEXT NOT NULL,
        score       DOUBLE PRECISION NOT NULL,
        classification TEXT NOT NULL,
        summary     TEXT,
        headlines   JSONB,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_history(ticker);
    """
    async with pool.acquire() as conn:
        await conn.execute(ddl)
    logger.info("Database schema initialized")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ─── Forecast persistence ───

async def save_forecasts(
    ticker: str,
    dates: list[date],
    prophet_preds: list[float],
    lstm_preds: list[float],
    combined_preds: list[float],
    sentiment_score: Optional[float] = None,
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO forecasts
               (ticker, prediction_date, prophet_pred, lstm_pred, combined_pred, sentiment_score)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            [
                (ticker, d, p, l, c, sentiment_score)
                for d, p, l, c in zip(dates, prophet_preds, lstm_preds, combined_preds)
            ],
        )


async def get_forecast_history(ticker: str, limit: int = 30) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM forecasts WHERE ticker = $1 ORDER BY prediction_date DESC LIMIT $2",
            ticker,
            limit,
        )
    return [dict(r) for r in rows]


# ─── Sentiment persistence ───

async def save_sentiment(
    ticker: str,
    score: float,
    classification: str,
    summary: str,
    headlines: list[str],
) -> None:
    import json

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO sentiment_history
               (ticker, score, classification, summary, headlines)
               VALUES ($1, $2, $3, $4, $5)""",
            ticker,
            score,
            classification,
            summary,
            json.dumps(headlines),
        )


async def get_sentiment_history(ticker: str, limit: int = 10) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM sentiment_history WHERE ticker = $1 ORDER BY created_at DESC LIMIT $2",
            ticker,
            limit,
        )
    return [dict(r) for r in rows]
