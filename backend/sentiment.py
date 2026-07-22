"""
DeepSeek-powered news sentiment analysis for FinSight.
Analyzes headlines and returns sentiment scores (-1 bearish to +1 bullish).
"""

import json
import logging
from typing import Optional

import httpx

from config import config
from data_fetcher import fetch_news_headlines
from models import SentimentResult

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = """Analyze the sentiment of the following news headlines for {ticker}. 

For each headline:
1. Classify as BULLISH, BEARISH, or NEUTRAL
2. Assign a sentiment score from -1.0 (extremely bearish) to 1.0 (extremely bullish)

After analyzing all headlines:
- Provide an overall sentiment score from -1.0 to 1.0
- Classify overall sentiment as bullish, bearish, or neutral
- Write a 2-3 sentence summary of the key sentiment drivers

Return your response as a JSON object with this exact structure:
{{
  "headlines": [
    {{"title": "headline text", "sentiment": "bullish/bearish/neutral", "score": 0.5}}
  ],
  "overall_score": 0.3,
  "overall_classification": "bullish",
  "summary": "Brief summary of key sentiment drivers."
}}

Headlines to analyze:
{headlines}"""


def _build_headlines_text(headlines: list[dict]) -> str:
    """Format headlines list into a numbered text block for the prompt."""
    return "\n".join(
        f"{i+1}. {h.get('title', 'N/A')} (via {h.get('publisher', 'unknown')})"
        for i, h in enumerate(headlines)
    )


def _parse_sentiment_response(response_text: str, headlines: list[dict]) -> Optional[dict]:
    """Extract JSON from DeepSeek response, handling markdown code blocks."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence with optional language tag
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


async def analyze_sentiment(
    ticker: str,
    headlines: Optional[list[dict]] = None,
) -> SentimentResult:
    """
    Analyze news sentiment for a ticker using DeepSeek API.

    Args:
        ticker: Stock/crypto ticker symbol
        headlines: Pre-fetched headlines (fetched automatically if None)

    Returns:
        SentimentResult with scores, classifications, and summary
    """
    if not config.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY not set — returning neutral sentiment")
        return SentimentResult(
            ticker=ticker,
            headlines=[],
            overall_score=0.0,
            classification="neutral",
            per_headline=[],
            summary="DeepSeek API key not configured. Sentiment analysis unavailable.",
        )

    # Fetch headlines if not provided
    if headlines is None:
        headlines = fetch_news_headlines(ticker)

    if not headlines:
        return SentimentResult(
            ticker=ticker,
            headlines=[],
            overall_score=0.0,
            classification="neutral",
            per_headline=[],
            summary="No news headlines available for this ticker.",
        )

    headlines_text = _build_headlines_text(headlines)
    prompt = SENTIMENT_PROMPT.format(ticker=ticker, headlines=headlines_text)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            config.deepseek_base_url,
            headers={
                "Authorization": f"Bearer {config.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.deepseek_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyst. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    parsed = _parse_sentiment_response(content, headlines)

    if parsed is None:
        logger.error("Failed to parse DeepSeek response: %s", content[:200])
        return SentimentResult(
            ticker=ticker,
            headlines=[h.get("title", "") for h in headlines],
            overall_score=0.0,
            classification="neutral",
            per_headline=[],
            summary="Failed to parse sentiment analysis response.",
        )

    # Normalize overall score
    overall_score = max(-1.0, min(1.0, float(parsed.get("overall_score", 0.0))))

    classification = parsed.get("overall_classification", "neutral").lower()
    if classification not in ("bullish", "bearish", "neutral"):
        classification = "bullish" if overall_score > 0.1 else "bearish" if overall_score < -0.1 else "neutral"

    per_headline = parsed.get("headlines", [])

    logger.info(
        "Sentiment for %s: score=%.2f, classification=%s, %d headlines",
        ticker,
        overall_score,
        classification,
        len(per_headline),
    )

    return SentimentResult(
        ticker=ticker,
        headlines=[h.get("title", "") for h in headlines],
        overall_score=round(overall_score, 4),
        classification=classification,
        per_headline=per_headline,
        summary=parsed.get("summary", "No summary available."),
    )


def apply_sentiment_adjustment(
    model_forecast: list[float],
    sentiment_score: float,
    current_price: float,
    model_weight: float = 0.70,
) -> list[float]:
    """
    Blend model forecasts with sentiment signal.

    Adjustment logic:
    - If sentiment is strongly bearish (-0.3 or worse) and price is > 10-day MA, weight sentiment more heavily
    - Otherwise, use standard weighted average
    """
    adjusted = []
    for pred in model_forecast:
        # Base weighted average
        sentiment_factor = 1.0 + sentiment_score * 0.05  # ±5% max adjustment from sentiment
        base_adjusted = pred * sentiment_factor

        # Weighted blend
        combined = model_weight * pred + (1 - model_weight) * base_adjusted
        adjusted.append(round(combined, 4))

    return adjusted
