"""Sentiment data pipeline — aggregates signals from free APIs.

Sources (all free tier):
- Alternative.me Fear & Greed Index
- CryptoPanic news API
- CoinGecko market overview

These are fetched asynchronously and combined into a SentimentData snapshot.
"""

from __future__ import annotations

from httpx import AsyncClient, HTTPError
import structlog

from src.agent.sanitize import sanitize_prompt_input
from src.agent.state import SentimentData

logger = structlog.get_logger()

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"


class SentimentPipeline:
    """Collects sentiment data from multiple free sources."""

    def __init__(self, cryptopanic_api_key: str = "") -> None:
        self._cryptopanic_key = cryptopanic_api_key
        self._client: AsyncClient | None = None

    async def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(timeout=15.0)
        return self._client

    async def fetch(self) -> SentimentData:
        """Fetch and aggregate all sentiment sources.

        Failures in individual sources are logged but don't crash the pipeline.
        """
        data = SentimentData()

        # Fetch all sources concurrently
        import asyncio

        results = await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_cryptopanic_headlines(),
            return_exceptions=True,
        )

        # Unpack results (tolerant of failures)
        fear_greed = results[0] if not isinstance(results[0], Exception) else None
        headlines = results[1] if not isinstance(results[1], Exception) else None

        if fear_greed is not None:
            data.fear_greed_index = fear_greed

        if headlines:
            data.news_headlines = headlines
            # Simple sentiment heuristic: ratio of bullish vs bearish keywords
            data.news_sentiment = self._score_headlines(headlines)

        logger.info(
            "sentiment_fetched",
            fear_greed=data.fear_greed_index,
            news_sentiment=data.news_sentiment,
            headline_count=len(data.news_headlines),
        )

        return data

    async def _fetch_fear_greed(self) -> float | None:
        """Fetch Bitcoin Fear & Greed Index (0-100)."""
        try:
            client = await self._get_client()
            resp = await client.get(FEAR_GREED_URL)
            resp.raise_for_status()
            result = resp.json()
            value = float(result["data"][0]["value"])
            logger.debug("fear_greed_fetched", value=value)
            return value
        except (HTTPError, KeyError, IndexError, ValueError) as e:
            logger.warning("fear_greed_failed", error=str(e))
            return None

    async def _fetch_cryptopanic_headlines(self) -> list[str]:
        """Fetch recent crypto news headlines from CryptoPanic."""
        if not self._cryptopanic_key:
            return []

        try:
            client = await self._get_client()
            url = f"https://cryptopanic.com/api/free/v1/posts/?auth_token={self._cryptopanic_key}&currencies=BTC&kind=news"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            headlines = []
            for post in data.get("results", [])[:10]:
                title = post.get("title", "")
                if title:
                    headlines.append(sanitize_prompt_input(title))

            logger.debug("cryptopanic_fetched", count=len(headlines))
            return headlines

        except (HTTPError, KeyError) as e:
            logger.warning("cryptopanic_failed", error=str(e))
            return []

    @staticmethod
    def _score_headlines(headlines: list[str]) -> float:
        """Simple keyword-based sentiment scoring (-1 to 1).

        This is a rough heuristic. For real production use, you'd want
        an LLM to score these, but that costs API calls. This is free.
        """
        bullish_words = {
            "surge", "rally", "bullish", "soar", "gain", "record", "high",
            "pump", "breakout", "moon", "buy", "accumulate", "inflow",
            "adoption", "approval", "etf", "institutional",
        }
        bearish_words = {
            "crash", "dump", "bearish", "plunge", "fall", "drop", "low",
            "sell", "outflow", "ban", "hack", "exploit", "rug",
            "regulation", "sec", "lawsuit", "fraud",
        }

        bull_count = 0
        bear_count = 0
        for headline in headlines:
            words = set(headline.lower().split())
            bull_count += len(words & bullish_words)
            bear_count += len(words & bearish_words)

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        # Score from -1 (all bearish) to +1 (all bullish)
        return (bull_count - bear_count) / total

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
