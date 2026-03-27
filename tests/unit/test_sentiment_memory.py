"""Tests for sentiment pipeline and agent memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.sentiment import SentimentPipeline


class TestSentimentScoring:
    """Test headline scoring heuristic."""

    def test_bullish_headlines(self):
        headlines = [
            "Bitcoin surges past $70K on ETF inflows",
            "Institutional adoption hits record high",
            "BTC rally continues as bulls take control",
        ]
        score = SentimentPipeline._score_headlines(headlines)
        assert score > 0  # Should be positive (bullish)

    def test_bearish_headlines(self):
        headlines = [
            "Bitcoin crashes below support as sell-off intensifies",
            "SEC lawsuit threatens major exchange",
            "Massive hack drains $500M from DeFi protocol",
        ]
        score = SentimentPipeline._score_headlines(headlines)
        assert score < 0  # Should be negative (bearish)

    def test_neutral_headlines(self):
        headlines = [
            "Bitcoin trading volume steady this week",
            "New development team joins project",
        ]
        score = SentimentPipeline._score_headlines(headlines)
        assert -0.5 <= score <= 0.5  # Should be roughly neutral

    def test_empty_headlines(self):
        score = SentimentPipeline._score_headlines([])
        assert score == 0.0

    def test_score_range(self):
        """Score should always be between -1 and 1."""
        headlines = ["pump pump pump rally moon surge breakout"]
        score = SentimentPipeline._score_headlines(headlines)
        assert -1.0 <= score <= 1.0


class TestSentimentFetch:
    """Test sentiment pipeline with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_fetch_returns_sentiment_data(self):
        pipeline = SentimentPipeline()

        mock_fear_greed_resp = MagicMock()
        mock_fear_greed_resp.json.return_value = {"data": [{"value": "75"}]}
        mock_fear_greed_resp.raise_for_status = MagicMock()

        async def mock_get(url):
            return mock_fear_greed_resp

        with patch.object(pipeline, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client_fn.return_value = mock_client

            data = await pipeline.fetch()

        assert data.fear_greed_index == 75.0

    @pytest.mark.asyncio
    async def test_fetch_handles_api_failure(self):
        """Pipeline should return empty data on failure, not crash."""
        pipeline = SentimentPipeline()

        async def mock_get(url):
            raise Exception("API down")

        with patch.object(pipeline, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client_fn.return_value = mock_client

            data = await pipeline.fetch()

        # Should return empty but valid SentimentData
        assert data.fear_greed_index is None
        assert data.news_sentiment is None
        assert data.news_headlines == []


class TestAgentMemory:
    """Test agent memory system."""

    @pytest.mark.asyncio
    async def test_get_context_formats_entries(self):
        from src.agent.memory import AgentMemory

        mock_db = MagicMock()
        mock_db.get_recent_memories = AsyncMock(return_value=[
            {"entry_type": "lesson", "content": "Wait for volume confirmation"},
            {"entry_type": "pattern", "content": "RSI divergence precedes reversals"},
        ])

        memory = AgentMemory(mock_db)
        context = await memory.get_context()

        assert len(context) == 2
        assert "[lesson]" in context[0]
        assert "Wait for volume" in context[0]
        assert "[pattern]" in context[1]

    @pytest.mark.asyncio
    async def test_add_lesson_stores_to_db(self):
        from src.agent.memory import AgentMemory

        mock_db = MagicMock()
        mock_db.add_memory = AsyncMock()

        memory = AgentMemory(mock_db)
        await memory.add_lesson("Don't trade on low volume")

        mock_db.add_memory.assert_called_once_with(
            entry_type="lesson",
            content="Don't trade on low volume",
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty_list(self):
        from src.agent.memory import AgentMemory

        mock_db = MagicMock()
        mock_db.get_recent_memories = AsyncMock(return_value=[])

        memory = AgentMemory(mock_db)
        context = await memory.get_context()

        assert context == []
